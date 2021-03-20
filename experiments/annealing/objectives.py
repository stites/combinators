import torch
import math
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
import combinators.stochastic as probtorch
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.inference import copytraces
from combinators.trace.utils import valeq


def _estimate_mc(values: Tensor, log_weights: Tensor, sample_dims: Tuple[int], reducedims: Tuple[int], keepdims: bool) -> Tensor:
    nw = F.softmax(log_weights, dim=sample_dims)
    return (nw * values).sum(dim=reducedims, keepdim=keepdims)


def _eval_detached(rv):
    if not isinstance(rv, RandomVariable):
        raise ValueError("Node type not supported")
    dist = rv.dist
    param_dict = {k: dist.__dict__[k].detach() for k, _ in dist.arg_constraints.items() if k in dist.__dict__}
    dist = dist.__class__(**param_dict)
    rv_detached = RandomVariable(dist, rv.value, rv.reparameterized)
    assert torch.equal(rv.log_prob, rv_detached.log_prob)
    return rv_detached


def stl_lv(out):
    """ TODO """
    ix = out.ix
    # Need do this to compute sticking (stl) the landing gradient
    q_stl_trace = copytraces(out.q_out.trace, exclude_nodes='g{}'.format(ix+1))
    q_stl_trace.append(_eval_detached(out.q_out.trace['g{}'.format(ix+1)]), name='g{}'.format(ix+1))
    lu_1 = q_stl_trace.log_joint(nodes=out.nodes, **shape_kwargs)


def nvo_avo(out, sample_dims=0) -> Tensor:
    reducedims = (sample_dims,)
    lw = torch.zeros_like(out.lv)
    f = -out.lv
    loss = _estimate_mc(f,
                        lw,
                        sample_dims=sample_dims,
                        reducedims=reducedims,
                        keepdims=False,
                        )
    return loss

def nvo_rkl(
    out,
    sample_dims=0,
    **kwargs,
) -> Tensor:
    reducedims = (sample_dims,)

    lw = out.lw.detach()
    lv = out.lv
    proposal_trace = out.proposal_trace
    target_trace = out.target_trace
    rv_proposal = proposal_trace['g{}'.format(out.ix)]
    rv_target = target_trace['g{}'.format(out.ix+1)]
    rv_fwd = proposal_trace['g{}'.format(out.ix+1)]
    rv_rev = target_trace['g{}'.format(out.ix)]

    # Tests
    # lv = rv_target.log_prob + rv_rev.log_prob - (rv_proposal.log_prob + rv_fwd.log_prob)
    try:
        assert(len(proposal_trace) == 2)
        assert(len(target_trace) == 2)
        assert valeq(proposal_trace, target_trace)

        assert (torch.equal(target_trace.log_joint(sample_dims=sample_dims, batch_dim=1)\
                            - proposal_trace.log_joint(sample_dims=sample_dims, batch_dim=1),
                            lv))
        assert (rv_fwd.log_prob.grad_fn is not None)
        assert (rv_proposal.value.grad_fn is None)
        assert (rv_target.log_prob.grad_fn is not None)
        assert (rv_rev.log_prob.grad_fn is not None)
    except:
        breakpoint();


    ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
    f = -(lv - ldZ)

    grad_log_Z1_term = _estimate_mc(rv_proposal._log_prob,
                                    lw,
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_proposal, probtorch.RandomVariable) else torch.tensor(0.)
    grad_log_Z2_term = _estimate_mc(_eval_nrep(rv_target)._log_prob,
                                    lw+lv.detach(),
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_target, probtorch.RandomVariable) else torch.tensor(0.)

    baseline = _estimate_mc(f.detach(),
                            lw,
                            sample_dims=sample_dims,
                            reducedims=reducedims,
                            keepdims=False,
                            )

    kl_term = _estimate_mc((f - baseline) * _eval1(rv_proposal._log_prob - grad_log_Z1_term),
                           lw,
                           sample_dims=sample_dims,
                           reducedims=reducedims,
                           keepdims=False,
                           )

    loss = kl_term + _eval0(grad_log_Z2_term) + baseline
    return loss


def _eval0(e):
    return e - e.detach()


def _eval1(e):
    return torch.exp(_eval0(e))


def _eval_nrep(rv):
    value = rv.value.detach()
    if isinstance(rv, RandomVariable):
        return RandomVariable(value=value, dist=rv.dist, provenance=rv.provenance, reparameterized=False)
    elif isinstance(rv, ImproperRandomVariable):
        return ImproperRandomVariable(value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance)
    else:
        raise NotImplementedError("Only supports RandomVariable and ImproperRandomVariable")
