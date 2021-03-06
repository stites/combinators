import torch

from abc import ABC
from torch import Tensor
from typing import Any, Tuple, Optional, Union, Set, Callable, NamedTuple

import combinators.resamplers as resamplers

from combinators.out import Out
from combinators.stochastic import Trace, Provenance, RandomVariable, ImproperRandomVariable
from combinators.program import Conditionable, Program, dispatch, check_passable_kwarg
from combinators.metrics import effective_sample_size


# FIXME: move to probtorch
def copytraces(*traces, exclude_node=None):
    """
    merge traces together. domains should be disjoint otherwise last-write-wins.
    """
    newtrace = Trace()
    if exclude_node is None:
        exclude_node = {}

    for tr in traces:
        for k, rv in tr.items():
            if k in exclude_node:
                break
            newtrace.append(rv, name=k)
    return newtrace


# FIXME: move to probtorch
def rerun_with_detached_values(trace:Trace):
    """
    Rerun a trace with detached values, recomputing the computation graph so that
    value do not cause a gradient leak.
    """
    newtrace = Trace()

    def rerun_rv(rv):
        value = rv.value.detach()
        if isinstance(rv, RandomVariable):
            return RandomVariable(value=value, dist=rv.dist, provenance=rv.provenance, reparameterized=rv.reparameterized)
        elif isinstance(rv, ImproperRandomVariable):
            return ImproperRandomVariable(value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance)
        else:
            raise NotImplementedError("Only supports RandomVariable and ImproperRandomVariable")

    for k, v in trace.items():
        newtrace.append(rerun_rv(v), name=k)

    return newtrace


class Inf(ABC):
    """
    Superclass of Inference combinators. This class serves two purposes:

    1. Typechecking. While not enforced at present, we can add @typechecked
       annotations to make this code type-safe. Arguably, typeguard's typeerrors
       are, at times, impenetrable and assert statements might be more
       user-friendly.

    2. Global variables for the inference state. The inference state is
       fragmented across Trace (for \rho, \tau, sample and observe statements),
       combinators.Out for runtime state found in `extras`, and in this
       superclass for global state.
    """
    def __init__(
            self,
            loss_fn:Callable[[Out, Tensor], Tensor]=(lambda _, fin: fin),
            loss0=None,
            ix:Union[Tuple[int], NamedTuple, None]=None,
            sample_dims=None,
            batch_dim=None,
            _debug=False
    ):
        self.loss0 = 0.0 if loss0 is None else loss0
        self.foldr_loss = loss_fn
        self.ix = ix
        self._out = Out(None, None, None)
        self._debug = _debug
        self.batch_dim = batch_dim
        self.sample_dims = sample_dims

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        raise NotImplementedError("@abstractproperty but python's type system doesn't maintain the Callable signature.")


class Condition(Inf):
    """ Conditioned evaluation. """
    def __init__(
            self,
            program: Conditionable,
            cond_trace: Optional[Trace]=None,
            ix=None,
            loss_fn=(lambda _, fin: fin),
            loss0=None,
            _debug=False
    ) -> None:
        Inf.__init__(self, ix=ix, loss_fn=loss_fn, loss0=loss0, _debug=_debug)
        self.program = program

        # FIXME: do we actually need a copy of the trace? Might cause a dangling ref.
        self.conditioning_trace = copytraces(cond_trace)

    def __call__(self, c:Any, _debug=False, **kwargs:Any) -> Out:
        """ Conditioned evaluation """
        debugging = _debug or self._debug
        self.program._cond_trace = self.conditioning_trace
        out = dispatch(self.program)(c, _debug=_debug, **kwargs)
        out['type']=type(self)

        if debugging:
            """ NOTE: holding on to traces like this is a good way to cause space leaks """
            out['cond_trace']=self.conditioning_trace

        self.program._cond_trace = None
        # Also clear cond_trace reference in trace to keep memory profile slim.
        out.trace._cond_trace = None

        return out


class Resample(Inf):
    """
    Compute importance weight of the proposal program's trace under the target
    program's trace, considering the incomming log weight lw of the proposal's
    trace
    """
    def __init__(
            self,
            q: Union[Program, Inf],
            ix=None,
            loss0=None,
            resampler=None,
            normalize_weights=False,
            _debug=False
    ):
        Inf.__init__(self, ix=ix, loss0=loss0, _debug=_debug)
        self.q = q
        self.resampler = resamplers.Systematic(normalize_weights=normalize_weights) if resampler is None else resampler
        self.normalize_weights = normalize_weights

    def __call__(self, c, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Resample Combinator """
        debugging = _debug or self._debug
        ix = self.ix if self.ix is not None else ix

        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        q_out = self.q(c, **inf_kwargs, **shared_kwargs)

        passable_kwargs = {k: v for k, v in shape_kwargs.items() if check_passable_kwarg(k, self.resampler)}

        tr_2, lw_2 = self.resampler(q_out.trace, q_out.log_weight, **passable_kwargs)

        # If we resample, we still need to  corresponding outputs from
        # kernel programs. We enforce that they follow the convention of always
        # passing a dict as output with addresses as keys.
        #
        # Better, long-term fix: Bring back "kernels" and have them be "programs
        # that output {addresses:values}" Or, say "outputs are always dicts"
        c1 = q_out.output
        assert isinstance(c1, dict)
        c2 = {k: v for k, v in c1.items()}
        rs_out_addrs = set(c1.keys()).intersection(set(tr_2.keys()))
        for rs_out_addr in rs_out_addrs:
            assert isinstance(c2[rs_out_addr], torch.Tensor)
            c2[rs_out_addr] = tr_2[rs_out_addr].value

        self._out = Out(
            extras=dict(
                type=type(self),
                ix=ix,
                ),
            trace=tr_2,
            log_weight=lw_2,
            output=c2,
        )

        self._out['loss'] = self.foldr_loss(self._out, self.loss0 if 'loss' not in q_out else q_out['loss'])

        if debugging:
            self._out['q_out'] = q_out

        return self._out


class Extend(Inf, Conditionable):
    def __init__(
            self,
            p: Program, # TODO: Maybe make this :=  p | extend (p, f)? This type annotation is not supported until python 3.10 (IIRC)
            f: Program,
            loss_fn=(lambda _, fin: fin),
            loss0=None,
            ix=None,
            _debug=False,
    ) -> None:
        Conditionable.__init__(self)
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, ix=ix, _debug=_debug)
        self.p = p
        self.f = f

    def __call__(self, c:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs:Any) -> Out:
        """ Extend Combinator """
        debugging = _debug or self._debug
        ix = self.ix if self.ix is not None else ix

        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        if self._cond_trace is None:
            p_out = dispatch(self.p)(c, **inf_kwargs, **shared_kwargs)

            f_out = dispatch(self.f)(p_out.output, **inf_kwargs, **shared_kwargs)

            assert (f_out.log_weight == 0.0)
            assert len({k for k, v in f_out.trace.items() if v.provenance == Provenance.OBSERVED or v.provenance == Provenance.REUSED}) == 0

        else:
            """ conditioned evaluation """
            p_out = dispatch(Condition(self.p, self._cond_trace))(c, **inf_kwargs, **shared_kwargs)

            f_out = dispatch(Condition(self.f, self._cond_trace))(p_out.output, **inf_kwargs, **shared_kwargs)

            assert len({k for k, v in f_out.trace.items() if v.provenance == Provenance.OBSERVED}) == 0

        assert len(set(f_out.trace.keys()).intersection(set(p_out.trace.keys()))) == 0

        log_u2 = f_out.trace.log_joint(**shape_kwargs, nodes={k for k,v in f_out.trace.items() if v.provenance != Provenance.OBSERVED})

        self._out = Out(
            trace=p_out.trace,
            log_weight=p_out.log_weight + log_u2, # $w_1 \cdot u_2$
            output=p_out.output,
            extras=dict(
                # FIXME: switch with marginal semantics
                trace_star = f_out.trace,
                type=type(self),
                ix=ix,
                ))

        self._out['loss'] = self.foldr_loss(self._out, self.loss0 if 'loss' not in p_out else p_out['loss'])

        if debugging:
            self._out['p_out'] = p_out
            self._out['f_out'] = f_out

        return self._out


class Compose(Inf):
    def __init__(
            self,
            q1: Union[Program, Condition, Resample, Inf],
            q2: Program,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            ix=None,
            _debug=False,
    ) -> None:
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, ix=ix, _debug=_debug)
        self.q1 = q1
        self.q2 = q2

    def __call__(self, c:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Compose """
        debugging = _debug or self._debug
        ix = self.ix if self.ix is not None else ix

        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        q1_out = dispatch(self.q1)(c, **inf_kwargs, **shared_kwargs)

        q2_out = dispatch(self.q2)(q1_out.output, **inf_kwargs, **shared_kwargs)

        assert len(set(q2_out.trace.keys()).intersection(set(q1_out.trace.keys()))) == 0, "addresses must not overlap"

        self._out = Out(
            trace=copytraces(q2_out.trace, q1_out.trace),
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
            extras=dict(
                type=type(self),
                ix=ix,
                ))

        # FIXME: be a fold over both losses?
        self._out['loss'] = self.foldr_loss(self._out, self.loss0 if 'loss' not in q1_out else q1_out['loss'])

        if debugging:
            self._out['q1_out'] = q1_out
            self._out['q2_out'] = q2_out

        return self._out



class Propose(Inf):
    def __init__(self,
            p: Union[Program, Extend],
            q: Union[Program, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            ix=None,
            _debug=False,
            _no_reruns:bool=True):
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, ix=ix, _debug=_debug)
        assert not isinstance(p, Compose)
        self.p = p
        self.q = q
        # APG, needs documentation
        self._no_reruns = _no_reruns

    def __call__(self, c, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Propose """
        debugging = _debug or self._debug
        ix = self.ix if self.ix is not None else ix

        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        q_out = dispatch(self.q)(c, **inf_kwargs, **shared_kwargs)

        p_out = dispatch(Condition(self.p, q_out.trace))(c, **inf_kwargs,  **shared_kwargs)

        rho_1 = set(q_out.trace.keys())
        tau_1 = set({k for k, v in q_out.trace.items() if v.provenance != Provenance.OBSERVED})
        tau_2 = set({k for k, v in p_out.trace.items() if v.provenance != Provenance.OBSERVED})
        nodes = rho_1 - (tau_1 - tau_2)

        lu_1 = q_out.trace.log_joint(nodes=nodes, **shape_kwargs)
        # τ*, by definition, can't have OBSERVE or REUSED random variables
        # FIXME: precision errors when converting python into pytorch, see tests.
        lu_star = 0.0 if 'trace_star' not in p_out else q_out.trace.log_joint(nodes=set(p_out.trace_star.keys()), **shape_kwargs)

        lw_1 = q_out.log_weight
        # We call that lv because its the incremental weight in the IS sense
        # In the semantics this corresponds to lw_2 - (lu + [lu_star])
        lv = p_out.log_weight - (lu_1 + lu_star)
        lw_out = lw_1 + lv

        # =============================================== #
        # detach c                                        #
        # =============================================== #
        new_out = None
        if isinstance(p_out.output, torch.Tensor):
            new_out = p_out.output.detach()
        elif isinstance(p_out.output, dict):
            new_out = {}
            for k, v in p_out.output.items():
                if isinstance(v, torch.Tensor):
                    new_out[k] = v.detach()
                else:
                    new_out[k] = v
        else:
            new_out = p_out.output
        # =============================================== #

        self._out = Out(
            trace=p_out.trace if self._no_reruns else rerun_with_detached_values(p_out.trace),
            log_weight=lw_out.detach(),
            output=new_out,
            extras=dict(
                ## stl api ##
                nodes=nodes,
                ## objectives api ##
                lv=lv,
                lw=lw_1.detach() if isinstance(lw_1, torch.Tensor) else torch.tensor(lw_1),
                proposal_trace=copytraces(q_out.trace),
                target_trace=copytraces(p_out.trace, p_out.trace_star) if "trace_star" in p_out else p_out.trace,
                ## apg ##
                forward_trace = q_out.q2_out.trace if q_out.type == "Compose" else None,
                #########
                type=type(self),
                ix=ix,
                ),
        )

        self._out['loss'] = self.foldr_loss(self._out, self.loss0 if 'loss' not in q_out else q_out['loss'])

        if debugging:
            self._out['q_out'] = q_out
            self._out['p_out'] = p_out

        return self._out
