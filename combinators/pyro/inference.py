from torch import Tensor, tensor
from typing import Any, Callable
from typeguard import typechecked
from pyro import poutine
from pyro.poutine import Trace
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from typing import NamedTuple
from torch import Tensor, tensor
from typing import NamedTuple, Any, Callable, Union, Optional, Tuple
from typeguard import typechecked
from pyro import poutine
from pyro.poutine import Trace
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.trace_messenger import (
    TraceMessenger,
    TraceHandler,
    identify_dense_edges,
)
from pyro.poutine.messenger import Messenger
from pyro.poutine.handlers import _make_handler

from combinators.pyro.traces import (
    concat_traces, assert_no_overlap,
    is_substituted,
    is_observed,
    is_auxiliary,
    not_observed,
    _and, _not, _or,
)

@typechecked
class Out(NamedTuple):
    output: Any
    log_weight: Tensor
    trace: Trace

class WithSubstitutionMessenger(ReplayMessenger):
    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        if self.trace is not None and msg["name"] in self.trace and not msg["is_observed"]:
            msg["infer"]['substituted'] = True

_handler_name, _handler = _make_handler(WithSubstitutionMessenger)
_handler.__module__ = __name__
locals()[_handler_name] = _handler

class AuxiliaryMessenger(Messenger):
    def __init__(self) -> None:
        super().__init__()

    def _pyro_sample(self, msg):
        msg['infer']['is_auxiliary'] = True
        return None

_handler_name, _handler = _make_handler(AuxiliaryMessenger)
_handler.__module__ = __name__
locals()[_handler_name] = _handler


@typechecked
def get_marginal(trace:Trace)->Trace:
    return concat_traces(trace, site_filter=lambda _, n: not is_auxiliary(n))

@typechecked
class inference:  # Callable[..., Out]
    pass


# FIXME evaluation in context of loss
@typechecked
class primitive(inference):
    def __init__(self, program: Callable[..., Any]):
        self.program = program

    def __call__(self, *args, **kwargs) -> Out:
        with TraceMessenger() as tracer:
            out = self.program(*args, **kwargs)
            tr: Trace = tracer.trace

            rho_addrs = {k for k in tr.nodes.keys()}
            tau_addrs = {k for k, rv in tr.nodes.items() if not_observed(rv)}

            tau_prime_addrs = {k for k, rv in tr.nodes.items()
                               if _and(is_substituted, not_observed)(rv)}

            lp = tr.log_prob_sum(site_filter=lambda name, _: name in rho_addrs - (
                tau_addrs - tau_prime_addrs
            ))
            lp = lp if isinstance(lp, Tensor) else tensor(lp)
            return Out(output=out, log_weight=lp, trace=tr)


class targets(inference):
    def __call__(self, *args, **kwargs) -> Out:
        ...


class extend(targets):
    def __init__(self, p: Union[primitive, targets], f: Union[primitive, targets]):
        self.p, self.f = p, f

    def __call__(self, *args, **kwargs) -> Out:
        p_out: Out = self.p(*args, **kwargs)

        f_out: Out = auxiliary(self.f)(p_out.output, *args, **kwargs)

        p_nodes, f_nodes = p_out.trace.nodes, f_out.trace.nodes

        under_substitution = any(
            [is_substituted(n) for _, n in {**p_nodes, **f_nodes}.items()]
        )

        if not under_substitution:
            assert f_out.log_weight == 0.0
            node_is_not = _or(is_observed, is_substituted)
        else:
            node_is_not = is_observed

        assert len({k for k, n in f_nodes.items() if node_is_not(n)}) == 0

        assert (
            len(set(f_nodes.keys()).intersection(set(p_nodes.keys()))) == 0
        ), f"{type(self)}: addresses must not overlap"

        log_u2 = f_out.trace.log_prob_sum(site_filter=lambda _, n: not_observed(n)) # nothing observed, checked above!

        return Out(
            trace=concat_traces(p_out.trace, f_out.trace),
            log_weight=p_out.log_weight + log_u2,  # $w_1 \cdot u_2$
            output=f_out.output,
        )


# FIXME
class proposals(inference):
    def __call__(self, *args, **kwargs) -> Out:
        ...


#@typechecked

@typechecked
class compose(inference):
    def __init__(
        self, q2: Union[primitive, proposals], q1: Union[primitive, proposals]
    ) -> None:
        self.q1, self.q2 = q1, q2
        self.substitution_trace

    def __call__(self, *args, **kwargs) -> Out:
        q1_out = self.q1(*args, **kwargs)
        q2_out = self.q2(*args, **kwargs)
        out_trace = concat_traces(q2_out.trace, q1_out.trace)
        assert_no_overlap(q2_out.trace, q1_out.trace, location=type(self))

        return Out(
            trace=out_trace,
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
        )


@typechecked
class propose(inference):
    def __init__(
        self, p: Union[primitive, extend], q: inference, loss_fn=(lambda x, fin: fin)
    ):
        self.p, self.q = p, q

    # @classmethod
    # def marginalize(cls, p, p_out):
    #     if isinstance(p, extend):
    #         assert "marginal_keys" in p_out
    #         return p_out.marginal_out, concat_traces(
    #             p_out.trace, site_filter=lambda name, _: name not in (set(p_out.trace.keys()) - p_out.marginal_keys)
    #         )
    #     else:
    #         return p_out.output, p_out.trace

    def __call__(self, *args, **kwargs) -> Out:
        q_out = self.q(*args, **kwargs)

        with WithSubstitution(self.p, q_out.trace), MarginalizeMessenger() as marg:
            p_out = self.p(*args, **kwargs)

        rho_1 = set(q_out.trace.nodes.keys())
        tau_filter = _and(not_observed, is_random_variable)
        tau_1 = set({k for k, v in q_out.trace.items() if tau_filter(v)})
        tau_2 = set({k for k, v in p_out.trace.items() if tau_filter(v)})

        nodes = rho_1 - (tau_1 - tau_2)

        # FIXME this hook exists to reshape NVI for stl
        # q_trace = dispatch(self.transf_q_trace, q_out.trace, **inf_kwargs)
        lu_1 = q_out.trace.log_joint(site_filter=lambda _, site: site in nodes)
        lw_1 = q_out.log_weight

        # We call that lv because its the incremental weight in the IS sense
        lv = p_out.log_weight - lu_1
        lw_out = lw_1 + lv

        m_trace = marg.get_marginal() # equivalent of tracer.get_trace()
        m_output = m_trace["_RETURN"]

        return Out(
            # FIXME local gradient computations
            # trace=m_trace if self._no_reruns else rerun_with_detached_values(m_trace),
            trace=m_trace,
            log_weight=lw_out.detach(),
            output=m_output,
        )
