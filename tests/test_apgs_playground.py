#!/usr/bin/env python3

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
import math

from typing import Tuple

from collections import namedtuple
from combinators.resampling.strategies import APGStrat
from combinators.inference import _dispatch
from combinators import Forward, Reverse, Propose, Condition, Resample, RequiresGrad
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.stochastic import Trace
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
import combinators.debug as debug
import experiments.apgs_gmm.hao as hao
from experiments.apgs_gmm.models import mk_target, Enc_rws_eta, Enc_apg_z, Enc_apg_eta, GenerativeOriginal, ix
from experiments.apgs_gmm.objectives import resample_variables, apg_update_z

from torch.distributions.one_hot_categorical import OneHotCategorical as cat

if debug.runtime() == 'jupyter':
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm


def apg_objective(enc_rws_eta, enc_apg_z, enc_apg_eta, generative, og, x, num_sweeps, sample_size, compare=True):
    """
    Amortized Population Gibbs objective in GMM problem
    ==========
    abbreviations:
    K -- number of clusters
    D -- data dimensions (D=2 in GMM)
    S -- sample size
    B -- batch size
    N -- number of data points in one (GMM) dataset
    ==========
    variables:
    ob  :  S * B * N * D, observations, as data points
    tau :  S * B * K * D, cluster precisions, as global variables
    mu  :  S * B * K * D, cluster means, as global variables
    eta : ={tau, mu} global block
    z   :  S * B * N * K, cluster assignments, as local variables
    ==========
    """
    # # ##(enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    # # #if compare:
    # # #    debug.seed(1)
    # # #    _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)
    # # #
    # # #    from combinators.resampling.strategies import APGResampler
    # # #    q_eta_z = resample_variables(resampler, _q_eta_z_out.trace, log_weights=_log_w)
    # # #    debug.seed(1)
    # # #
    # # ## otherwise, eager combinators looks like:
    # # #prp  = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    # # #out = prp(x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)
    # # #
    # # #log_w = out.log_omega.detach()
    # # #w = F.softmax(log_w, 0)
    # # #loss = (w * (- out.proposal.weights)).sum(0).mean()
    og2, og2k = generative
    kwargs = dict(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=True)

    assert num_sweeps == 2
    from combinators.resampling.strategies import APGSResamplerOriginal
    resampler = APGSResamplerOriginal(sample_size)
    # ================================================================
    # sweep 1:
    debug.seed(1)
    loss1, log_w1, q_eta_z_out1 = hao.oneshot(enc_rws_eta, enc_apg_z, og, x)
    q_eta_z1 = q_eta_z_out1.trace
    q_eta_z2 = q_eta_z1 # resample_variables(resampler, q_eta_z1, log_weights=log_w1)
    log_wr2 = q_eta_z2.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    # forward
    debug.seed(4)
    q_eta_z_f = enc_apg_eta(q_eta_z2, None, x=x, prior_ng=og.prior_ng, ix="") ## forward kernel
    q_eta_z_f = q_eta_z_f.trace
    print("q_eta_z_f.keys()", set(q_eta_z_f.keys()))
    p_f = og.x_forward(q_eta_z_f, x)
    print("p_f.keys()", set(p_f.keys()))
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    print(tensor_utils.show(log_w1))
    # print(tensor_utils.show(log_wr2))
    # print(tensor_utils.show(log_w_f))

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


    # ================================================================
    # sweep 1:
    debug.seed(1)
    prp1 = Propose(ix=ix(fr=1,to=1),
            target=og, # og == original generator
            #          p(x,η_1,z_1)
            # ---------------------------------- [prp1]
            #          q(z_1 | η_1, x) q_0(η_1|x)
            proposal=Forward(enc_apg_z, enc_rws_eta))

    out1 = prp1(**kwargs)
    try:
        print(trace_utils.valeq(out1.trace, q_eta_z1))
    except Exception as e:
        print(e)
    breakpoint();


    print(torch.equal(out1.log_omega, log_w1))
    print("out1 results ^^^^^")
    import torch.nn.functional as F

    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    # debug.seed(1)
    # fwd2 = Forward(kernel=enc_apg_eta, program=Resample(prp1, strategy=APGStrat(sample_size)))
    # prp2 = Propose(proposal=fwd2, target=Reverse(og2, og2k))

    # debug.seed(1)
    print("1 still working?", torch.equal(out1.log_omega, log_w1))
    # prp2 = Propose(
    #         target=Reverse(og, enc_apg_eta), # og == original generator
    #         #          p(x,η,z_2) q(η_1 | η, x)                 #
    #         # ---------------------------------------
    #         #          q(z_2 | η, x) prp_1(z_1,x,η)             # <<< (τ_prp1:{η,z})
    #         proposal=Forward(enc_apg_eta, Condition(prp1, out1.proposal.trace)))

    print("with z2s", q_eta_z_f)
    print("lprobs", log_q_f.sum(), log_q_f.mean())

    debug.seed(4)

    pro = Forward(enc_apg_eta, prp1, ix=ix(fr=1, to=2), _exclude={'lls'})
    of2 = pro(**kwargs)

    def compare(x, getter, expected):
        # print(torch.equal(getter(q_eta_z_f[x]), getter(of2.kernel.trace[x])))
        print("GOT:", torch.equal(getter(out1.trace[x]), getter(of2.program.trace[x])), "EXPECTED:", expected)

    # compare('states1', lambda x: x.dist.probs, True)
    # compare('states1', lambda x: x.value, True)
    # compare('lls', lambda x: x.value, True)
    # compare('lls', lambda x: x.dist.loc, True)
    # compare('lls', lambda x: x.dist.scale, True)
    #
    # print(tensor_utils.show(of2.trace['precisions'].dist.rate))
    # compare('precisions', lambda x: x.dist.rate, False)
    # print(tensor_utils.show(of2.trace['precisions'].dist.concentration))
    # compare('precisions', lambda x: x.dist.concentration, False)
    # Need to confirm that: q(η_2 | z_1 x) p_{prp1}(x η_1 z_1)
    breakpoint();

    tar = Condition(Reverse(og, enc_apg_eta, ix=ix(fr=2, to=1), trace=of2.trace, _exclude={'lls'}))
    or2 = tar(**kwargs)

    # =======================================================================================
    # =======================================================================================
    # =======================================================================================
    # =======================================================================================
    # =======================================================================================

    prp21 = Propose(
        target=Reverse(og, enc_apg_eta, ix=sweepix(sweep=2, stage=1)),
        #       p(x η_2 z_1) q(η_1 | z_1 x)
        # --------------------------------------- [prp21]
        #    q(η_2 | z_1 x) p_{prp1}(x η_1 z_1)
        proposal=Forward(enc_apg_eta, prp1, ix=sweepix(sweep=2, stage=2)))


    prp22 = Propose(
        ix=sweepix(sweep=2, stage=2),
        target=Reverse(og, enc_apg_eta),
        #       p(x η_2 z_2) q(z_1 | η_2 x)
        # --------------------------------------- [prp22]
        #    q(z_2 | η_2 x) p_{prp22}(x η_2 z_1)
        proposal=Forward(enc_apg_eta, prp21))

    def compare(x, getter):
        # print(torch.equal(getter(q_eta_z_f[x]), getter(of2.kernel.trace[x])))
        print(torch.equal(getter(out1.trace[x]), getter(of2.program.trace[x])))
    compare('precisions', lambda x: x.dist.concentration)
    compare('precisions', lambda x: x.dist.rate)
    compare('precisions', lambda x: x.value)
    compare('means',  lambda x: x.dist.loc)
    compare('means',  lambda x: x.dist.scale)
    compare('means',  lambda x: x.value)
    compare('states', lambda x: x.dist.probs)
    compare('states', lambda x: x.value)

    print(torch.equal(q_eta_z_f['precisions'].dist.concentration, of2.program.trace['precisions'].dist.concentration))
    print(log_q_f.sum(), log_q_f.mean())
    print(of2.log_omega.sum(), of2.log_omega.mean())
    # print(log_p_f.sum(), log_p_f.mean())
    # print(of2.target.log_omega.sum(), of2.target.log_omega.mean())
    # print(log_w_f.sum(), log_w_f.mean())
    # print(of2.log_omega.sum(), of2.log_omega.mean())

    breakpoint();
    out2 = prp2(**kwargs)

    def debug_print(desc, a, b):
        with torch.no_grad():
            eq = torch.equal(a, b)
            print(desc, eq, *[t.detach().sum().cpu().item() for t in [a, b] if not eq])

    print("(Proposal>Forward>Resample).trace == q_eta_z_f?", trace_utils.valeq(out2.proposal.program.trace, q_eta_z_f, strict=True))
    print("log_wr2", log_wr2.sum())
    debug_print("(Proposal>Forward>Resample).log_ω == log_q_f?", log_q_f, out2.proposal.program.log_omega) # log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    print("(Proposal>Forward>Resample).log_ω", out2.proposal.program.log_omega.sum())
    print("(Proposal>Forward>EncAPG_η).log_ω", out2.proposal.kernel.log_omega.sum() if out2.proposal.kernel.log_omega is not None else None)
    print("(Proposal>Forward).log_ω", out2.proposal.log_omega.sum())
    print("log_q_f", log_q_f.sum())
    # debug_print("(Proposal>Forward>Resample).log_ω == log_p_f?", log_p_f, out2.target.log_omega)
    # debug_print("(Proposal).log_ω == log_w_f?", log_w_f, out2.log_omega)
    #
    # print(tensor_utils.show(log_w_f), tensor_utils.show(out2.log_omega))
    breakpoint();

    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f

    breakpoint();

    debug.seed(2)
    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z, None, x=x, prior_ng=generative.prior_ng) ## forward kernel
    q_eta_z_f = q_eta_z_f.trace
    p_f = og.x_forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_eta(q_eta_z, None, x=x, prior_ng=generative.prior_ng)
    q_eta_z_b = q_eta_z_b.trace
    p_b = og.x_forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    breakpoint();
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_eta)
    log_w_z, q_eta_z, metrics = apg_update_z(enc_apg_z, og, q_eta_z, x)
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_z)

    return metrics

def mk_metrics(loss, out, num_sweeps=1, ess_required=True, mode_required=True, density_required=True, mode_required_with_z=False, ix=None):
    with torch.no_grad():
        metrics = dict()
        metrics['loss'] = loss.mean().cpu().item()
        if ess_required:
            w = F.softmax(out.log_omega.detach(), 0)
            metrics['ess'] = (1. / (w**2).sum(0)).mean().cpu().item()
        if mode_required:
            assert ix != None
            q_eta_z = out.proposal.trace
            metrics['E_tau'] = (q_eta_z[f'precisions{ix.to}'].dist.concentration / q_eta_z[f'precisions{ix.to}'].dist.rate).mean(0).detach().mean().cpu().item()
            metrics['E_mu'] = q_eta_z[f'means{ix.to}'].dist.loc.mean(0).detach().mean().cpu().item()
            if mode_required_with_z:
                # this is stable at 1/3
                metrics['E_z'] = q_eta_z[f'states{ix.to}'].dist.probs.mean(0).detach().mean().cpu().item()
        if density_required:
            metrics['density'] = out.target.log_omega.detach().mean().cpu().item()

        if num_sweeps > 1:
            pass
            # exc_kl, inc_kl= kls_eta(models, x, z_true)
            # if 'inc_kl' in metrics:
            #     metrics['inc_kl'] += inc_kl
            # else:
            #     metrics['inc_kl'] = inc_kl
            # if 'exc_kl' in metrics:
            #     metrics['exc_kl'] += exc_kl
            # else:
            #     metrics['exc_kl'] = exc_kl
        return metrics

def train(objective, models, target, og, og2, data, assignments, num_epochs, sample_size, batch_size, normal_gamma_priors, with_tensorboard=False, lr=1e-3, seed=1, eval_break=50, is_smoketest=False, num_sweeps=None):
    # (num_clusters, data_dim, normal_gamma_priors, num_iterations=10000, num_samples = 600batches=50)
    """ data size  S * B * N * 2 """
    # Setup
    debug.seed(seed)
    # writer = debug.MaybeWriter(enable=with_tensorboard)
    loss_ct, loss_sum = 0, 0.0

    [enc_rws_eta, enc_apg_z, enc_apg_eta], generative = models, target

    assert all([len(list(k.parameters())) >  0 for k in models])
    optimizer = optim.Adam([dict(params=x.parameters()) for x in models], lr=lr)
    num_batches = int((data.shape[0] / batch_size))
    # sample_shape = (batch_size, sample_size)
    epochs = range(1) if is_smoketest else trange(num_epochs, desc='Epochs', position=1)
    for e in epochs:
        data, assignments = shuffler(data, assignments)

        num_batches = 3 if is_smoketest else num_batches
        with trange(num_batches, desc=f'Batch {{:{int(math.log(num_epochs, 10))}d}}'.format(e+1), position=0) as batches:
            for bix, b in enumerate(batches):
                optimizer.zero_grad()
                x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)

                loss, metrics = objective(enc_rws_eta=enc_rws_eta, enc_apg_z=enc_apg_z, enc_apg_eta=enc_apg_eta, generative=og2, og=og, x=x, sample_size=sample_size, num_sweeps=num_sweeps)

                loss.backward()
                optimizer.step()

                # REPORTING
                # ---------------------------------------
                with torch.no_grad():
                    batches.set_postfix_str(";".join([f'{k}={{: .2f}}'.format(v) for k, v in metrics.items()])) # loss=loss.detach().cpu().mean().item(), **metrics)

                if is_smoketest and bix > 3:
                    return None


def shuffler(data, assignments):
    """
    shuffle the GMM datasets by both permuting the order of GMM instances (w.r.t. DIM1) and permuting the order of data points in each instance (w.r.t. DIM2)
    """
    concat_var = torch.cat((data, assignments), dim=-1)
    DIM1, DIM2, DIM3 = concat_var.shape
    indices_DIM1 = torch.randperm(DIM1)
    concat_var = concat_var[indices_DIM1]
    indices_DIM2 = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)], dim=0)
    concat_var = torch.gather(concat_var, 1, indices_DIM2.unsqueeze(-1).repeat(1, 1, DIM3))
    return concat_var[:,:,:2], concat_var[:,:,2:]


def main(is_smoketest, num_sweeps, simulate, objective):
    import subprocess
    gitroot = subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode("utf-8").rstrip()

    debug.seed(1)

    data_path=f'{gitroot}/data/gmm/'
    if simulate:
        from experiments.apgs_gmm.simulator import Sim_GMM
        simulator = Sim_GMM(N=60, K=3, D=2, alpha=2.0, beta=2.0, mu=0.0, nu=0.1)
        simulator.sim_save_data(num_seqs=10000, data_path=data_path)

    data = torch.from_numpy(np.load(f'{data_path}ob.npy')).float()
    assignments = torch.from_numpy(np.load(f'{data_path}assignment.npy')).float()

    # hyperparameters
    num_epochs=1 if is_smoketest else 100
    batch_size=50 if is_smoketest else 10
    budget=100
    lr=2e-4
    num_clusters=K=3
    data_dim=D=2
    num_hidden=30
    is_smoketest=debug.is_smoketest()

    normal_gamma_priors = dict(
        mu=torch.zeros((num_clusters, data_dim)),
        nu=torch.ones((num_clusters, data_dim)) * 0.1,
        alpha=torch.ones((num_clusters, data_dim)) * 2.0,
        beta=torch.ones((num_clusters, data_dim)) * 2.0,
    )
    # computable params
    num_batches = (data.shape[0] // batch_size)
    sample_size = budget // num_sweeps

    # Models
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden=num_hidden)
    enc_apg_eta = Enc_apg_eta(K, D)
    # generative = Generative(K, normal_gamma_priors)
    og = GenerativeOriginal(K, D, False, 'cpu')

    og2 = None
    og2k = None

    config = dict(objective=objective, num_sweeps=num_sweeps, num_epochs=num_epochs, batch_size=batch_size, sample_size=sample_size, is_smoketest=is_smoketest)
    print(";".join([f'{k}={v}' for k, v in config.items()]))

    train(
        models=[enc_rws_eta, enc_apg_z, enc_apg_eta],
        target=og,
        og=og,
        og2=(og2, og2k),
        data=data,
        assignments=assignments,
        normal_gamma_priors=normal_gamma_priors,
        **config
    )
    print('tada!')

# ==================================================================================================================== #

def rws_objective_eager(enc_rws_eta, enc_apg_z, generative, og, x, enc_apg_eta=None, compare=True, sample_size=None, num_sweeps=1):
    """ One-shot for eta and z, like a normal RWS """
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    assert num_sweeps == 1
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out, _ = hao.oneshot(enc_rws_eta, enc_apg_z, og, x, metrics=metrics)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og, ix=ix(1,1))
    out = prp(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_omega.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.log_omega)).sum(0).mean()
    if compare:
        assert torch.allclose(loss, _loss)
        assert torch.allclose(log_w, _log_w)
        assert trace_utils.valeq(out.trace, _q_eta_z_out.trace)

    return loss, mk_metrics(loss, out, ix=ix(1,1))

def rws_objective_declarative(enc_rws_eta, enc_apg_z, generative, og, x, enc_apg_eta=None, compare=True, sample_size=None, num_sweeps=1):
    assert num_sweeps == 1
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out, _  = hao.oneshot(enc_rws_eta, enc_apg_z, og, x, metrics=metrics)
        debug.seed(1)

    def loss_fn(out, loss):
        log_w = out.log_omega.detach()
        w = F.softmax(log_w, 0)
        return (w * (- out.proposal.log_omega)).sum(0).mean() + loss

    # otherwise, eager combinators looks like:
    prp = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og, ix=ix(1,1), loss_fn=loss_fn)
    out = prp(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    if compare:
        assert torch.allclose(out.loss, _loss)
        assert torch.allclose(out.log_omega, _log_w)
        assert trace_utils.valeq(out.trace, _q_eta_z_out.trace)

    return out.loss, mk_metrics(out.loss, out, ix=ix(1,1))

def test_rws_vae_eager():
    main(objective=rws_objective_eager, num_sweeps=1, is_smoketest=debug.is_smoketest(), simulate=False)

def test_rws_vae_declarative():
    main(objective=rws_objective_declarative, num_sweeps=1, is_smoketest=debug.is_smoketest(), simulate=False)



def apg_objective_eager(enc_rws_eta, enc_apg_z, enc_apg_eta, generative, og, x, sample_size, num_sweeps, compare=True):
    if compare:
        assert num_sweeps == 2
        from combinators.resampling.strategies import APGSResamplerOriginal
        resampler = APGSResamplerOriginal(sample_size)
        sweeps, metrics = hao.apg_objective((enc_rws_eta, enc_apg_z, enc_apg_eta, og), x, num_sweeps, resampler)
        breakpoint();

        return loss, metrics

def test_apg_2sweep_eager():
    main(objective=apg_objective_eager, num_sweeps=2, is_smoketest=debug.is_smoketest(), simulate=False)
