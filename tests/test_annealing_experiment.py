#!/usr/bin/env python3
import torch
import os
import math
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt
from pytest import mark, fixture

import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
from combinators.densities.kernels import MultivariateNormalKernel, MultivariateNormalLinearKernel, NormalLinearKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl, nvo_avo, mb0, mb1, _estimate_mc, eval_nrep
from combinators import Forward, Reverse, Propose, Condition, RequiresGrad, Resample
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat
from tests.utils import is_smoketest, seed
import combinators.debug as debug

import experiments.annealing.visualize as V
from experiments.annealing.models import mk_model, sample_along, paper_model

def report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards):
    with torch.no_grad():
        # loss
        writer.add_scalar('loss', loss_scalar, i)

        # ESS
        for step, x in zip(range(1,len(ess)+1), ess):
            writer.add_scalar(f'ess/step-{step}', x, i)

        # logZhat
        for step, x in zip(range(1,len(lzh)+1), lzh):
            writer.add_scalar(f'log_Z_hat/step-{step}', x, i)

        # show samples
        if i % eval_break == 0:
            samples = sample_along(targets[0], forwards)
            fig = V.scatter_along(samples)
            writer.add_figure('overview', fig, global_step=i, close=True)


def experiment_runner(is_smoketest, trainer):
    debug.seed()
    eval_break=50
    num_iterations=3 if is_smoketest else 10000
    # Setup
    # num_samples = 5 if is_smoketest else 256
    num_samples=5
    sample_shape=(num_samples,)

    # Models
    out = paper_model()
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    optimizer = optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-3)

    with trange(num_iterations) as bar:
        for i in bar:

            lvss, loss = trainer(i, targets, forwards, reverses, sample_shape)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            # ---------------------------------------
            with torch.no_grad():
                lvs = torch.stack(lvss, dim=0)
                lws = torch.cumsum(lvs, dim=1)
                ess = effective_sample_size(lws, sample_dims=-1)
                lzh = log_Z_hat(lws, sample_dims=-1)

                loss_scalar = loss.detach().cpu().mean().item()

                report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards)

                loss_ct += 1
                loss_sum += loss_scalar
                # Update progress bar
                if i % 10 == 0:
                    loss_avg = loss_sum / loss_ct
                    loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
                    logZh_template = 'logZhat[-1]={:.4f}'.format(lzh[-1].cpu().item())
                    ess_template = 'ess[-1]={:.4f}'.format(ess[-1].cpu().item())
                    loss_ct, loss_sum  = 0, 0.0
                    bar.set_postfix_str("; ".join([loss_template, ess_template, logZh_template]))



def nvi_eager(i, targets, forwards, reverses, sample_shape):
    q0 = targets[0]
    p_prv_tr, _, _ = q0(sample_shape=sample_shape)

    loss = torch.zeros(1, **kw_autodevice())
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []

    for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
        q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO), _step=k)
        p_ext = Reverse(p, rev, _step=k)
        extend = Propose(target=p_ext, proposal=q_ext, _step=k)
        state = extend(sample_shape=sample_shape, sample_dims=0)
        lv = state.weights

        p_prv_tr = state.trace

        lw += lv

        # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
        objective_loss = nvo_avo(lv)
        loss += objective_loss

        lvss.append(lv)

    return lvss, loss

def nvi_eager_resample(i, targets, forwards, reverses, sample_shape):
    q0 = targets[0]
    p_prv_tr, _, _ = q0(sample_shape=sample_shape)

    loss = torch.zeros(1, **kw_autodevice())
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []

    for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
        q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO), _step=k)
        p_ext = Reverse(p, rev, _step=k)
        extend = Resample(Propose(target=p_ext, proposal=q_ext, _step=k))

        state = extend(sample_shape=sample_shape, sample_dims=0)
        lv = state.weights

        p_prv_tr = state.trace

        lw += lv

        # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
        objective_loss = nvo_avo(lv)
        loss += objective_loss

        lvss.append(lv)

    return lvss, loss

def test_annealing_eager(is_smoketest):
    experiment_runner(is_smoketest, nvi_eager)

def test_annealing_eager_resample(is_smoketest):
    experiment_runner(is_smoketest, nvi_eager_resample)

def _weights(out, ret=[])->[Tensor]:
    _ret = ret + ([out.weights.detach().cpu()] if out.weights is not None else [])
    if 'proposal' not in out:
        return _ret
    else:
        return _weights(out.proposal.program, _ret)


def print_and_sum_loss(lv, loss):
    out = loss + nvo_avo(lv)
    print(out)
    return out


def nvi_declarative(i, targets, forwards, reverses, sample_shape):
    def mk_step(q, p, fwd, rev, k)->Propose:
        q_ext = Forward(fwd, q)
        p_ext = Reverse(p, rev)
        return Propose(target=p_ext, proposal=q_ext, loss_fn=print_and_sum_loss, _debug=True, _step=k, loss0=torch.zeros(1, **kw_autodevice()))

    proposal = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        proposal = mk_step(proposal, p, fwd, rev, k)

    out = proposal(sample_shape=sample_shape, sample_dims=0)

    return _weights(out), out.loss

def test_annealing_declarative():
    is_smoketest = True
    experiment_runner(is_smoketest, nvi_declarative)
