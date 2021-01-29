#!/usr/bin/env python3
import torch
from torch import Tensor
from combinators.stochastic import Trace, Provenance, RandomVariable, ImproperRandomVariable
from torch import distributions as D
from combinators.types import TraceLike
from typing import Callable, Any, Tuple, Optional, Set, Union, Dict
from copy import deepcopy
from typeguard import typechecked
from itertools import chain
import combinators.tensor.utils as tensor_utils
from enum import Enum, auto
from combinators.types import check_passable_kwarg

@typechecked
def maybe_sample(trace:Trace, sample_shape:Union[Tuple[int], None, tuple]) -> Callable[[D.Distribution, str], Tuple[Tensor, Provenance]]:
    def curried(dist:D.Distribution, name:str) -> Tuple[Tensor, Provenance]:
        if name in trace:
            return trace[name].value, Provenance.OBSERVED
        elif sample_shape is None:
            return (dist.rsample() if dist.has_rsample else dist.sample(), Provenance.SAMPLED)
        else:
            return (dist.rsample(sample_shape) if dist.has_rsample else dist.sample(sample_shape), Provenance.SAMPLED)
    return curried

@typechecked
def is_valid_subtype(_subtr: Trace, _super: Trace)->bool:
    """ sub trace <: super trace """
    super_keys = frozenset(_super.keys())
    subtr_keys = frozenset(_subtr.keys())

    @typechecked
    def check(key:str)->bool:
        try:
            return _subtr[key].value.shape == _super[key].value.shape
        except:  # FIXME better capture
            return False

    # FIXME: add better error message about dimension checks
    return len(super_keys - subtr_keys) == 0 and all([check(key) for key in super_keys])

@typechecked
def assert_valid_subtrace(tr1:Trace, tr2:Trace) -> None:
    assert is_valid_subtype(tr1, tr2), "{} is not a subtype of {}".format(tr1, tr2)


@typechecked
def valeq(t1:Trace, t2:Trace, nodes:Optional[Dict[str, Any]]=None, check_exist:bool=True, strict:bool=False)->bool:
    """
    Two traces are value-equivalent for a set of names iff the values of the corresponding names in both traces exist
    and are mutually equal.

    check_exist::boolean    Controls if a name must exist in both traces, if set to False one values of variables that
                            exist in both traces are checks for 'value equivalence'
    """
    t1nodes:Set[str] = set(t1._nodes.keys())
    t2nodes:Set[str] = set(t2._nodes.keys())
    if nodes is None:
        _nodes = t1nodes.union(t2nodes) if strict else t1nodes.intersection(t2nodes)
    else:
        _nodes = nodes

    if not all(name in t1 and name in t2 for name in _nodes):
        str_t1nodes = "{" + ", ".join(t1nodes) + "}"
        str_t2nodes = "{" + ", ".join(t2nodes) + "}"
        cmpr_string = "are not in {" + ", ".join(_nodes) + "}" if nodes is not None \
            else "are not equal"
        raise Exception("trace's keys {}:\n  trace 1: {}\n  trace 2: {}".format(cmpr_string, str_t1nodes, str_t2nodes))

    invalid = list(filter(lambda name: not trace_eq(t1, t2, name), _nodes))
    if len(invalid) > 0:
        str_nodes = "{" + ", ".join(invalid) + "}"
        raise Exception("RV nodes {} are not equal in:\n  trace 1: {}\n  trace 2: {}".format(str_nodes, t1, t2))

    return True


@typechecked
def show(tr:TraceLike, fix_width=False):
    get_value = lambda v: v if isinstance(v, Tensor) else v.value
    ten_show = lambda v: tensor_utils.show(get_value(v), fix_width=fix_width)
    return "{" + "; ".join([f"'{k}'-➢{ten_show(v)}" for k, v in tr.items()]) + "}"

@typechecked
def trace_eq(t0:Trace, t1:Trace, name:str):
    return name in t0 and name in t1 and torch.equal(t0[name].value, t1[name].value)

@typechecked
def log_importance_weight(proposal_trace:Trace, target_trace:Trace, batch_dim:int=1, sample_dims:int=0, check_exist:bool=True)->Tensor:
    """
    Computes an importance weight by substractiong log joint distibution of the proposal trace from the log joint
    distributin of the target trace. The log joint is computed based on all nodes inside the target trace.

    NOTE: Important weights can only be computes for value-equivalent traces.
    NOTE: Implicit proposals are not supported, i.e. all names in the target trace must exist in the proposal trace.
    TODO: Fix batching
    """
    assert valeq(proposal_trace, target_trace, nodes=target_trace._nodes, check_exist=check_exist)
    return target_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes) - \
        proposal_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)


def copysubtrace(tr: Trace, subset: Optional[Set[str]]):
    # FIXME: need to verify that this does the expected thing
    out = Trace()
    for key, node in tr.items():
        if subset is None or key in subset:
            out[key] = node
    return out


class RequiresGrad(Enum):
    YES=auto()
    NO=auto()
    DEFAULT=auto()

def get_requires_grad(ten, is_detach_set, global_requires_grad:RequiresGrad=RequiresGrad.DEFAULT):
    if is_detach_set:
        return False
    else:
        return ten.requires_grad if global_requires_grad==RequiresGrad.DEFAULT else global_requires_grad == RequiresGrad.YES

@typechecked
def detach(tr:Trace):
    def detachrv(rv):
        rv._value = rv._value.detach()
        return rv

    return {k: detachrv(rv) for k, rv in tr.items()}


@typechecked
def copytraces(*traces: Trace, requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), overwrite=False, mapper=(lambda x: x))->Trace:
    """
    shallow-copies nodes from many traces into a new trace.
    unless overwrite is set, there is a first-write presidence.
    """
    newtr = Trace()
    for tr in traces:
        for k, rv in tr.items():
            if k in newtr:
                if overwrite:
                    raise NotImplementedError("")
                else:
                    pass

            newrv = copyrv(rv, requires_grad=get_requires_grad(rv.value, k in detach, requires_grad), mapper=mapper)
            newtr.append(newrv, name=k)
    return newtr


@typechecked
def copytrace(tr: Trace, **kwargs)->Trace:
    # sugar
    return copytraces(tr, **kwargs)

@typechecked
def copyrv(rv:Union[RandomVariable, ImproperRandomVariable], requires_grad: bool=True, provenance:Optional[Provenance]=None, deepcopy_value=False, mapper=(lambda x: x)):
    RVClass = type(rv)
    value = tensor_utils.copy(rv.value, requires_grad=requires_grad, deepcopy=deepcopy_value)
    provenance = provenance if provenance is not None else rv.provenance

    if RVClass is RandomVariable:
        return RVClass(**mapper(dict(dist=rv.dist, use_pmf=rv._use_pmf, value=value, provenance=provenance, mask=rv.mask, log_prob=rv.log_prob)))
    elif RVClass is ImproperRandomVariable:
        return RVClass(**mapper(dict(log_density_fn=rv._log_density_fn, value=value, provenance=provenance, mask=rv.mask, log_prob=rv.log_prob)))
    else:
        raise NotImplementedError()


@typechecked
def mapvalues(*traces: Trace, mapper=None, **kwargs):
    assert mapper is not None
    def rvmapper(kwargs):
        shared_kwargs = dict(value=mapper(kwargs['value']), provenance=kwargs['provenance'], mask=kwargs['mask'], log_prob=kwargs['log_prob'])
        if 'log_density_fn' in kwargs:
            # have an improper random variable
            return dict(log_density_fn=kwargs['log_density_fn'], **shared_kwargs)
        else:
            return dict(dist=kwargs['dist'],  use_pmf=kwargs['use_pmf'], **shared_kwargs)

    return copytraces(*traces, mapper=rvmapper, **kwargs)
