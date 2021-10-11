from torch import Tensor, tensor
from typing import Any, Callable
from typeguard import typechecked
from pyro.poutine import Trace
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from typing import NamedTuple
from torch import Tensor, tensor
from typing import NamedTuple, Any, Callable, Union
from typeguard import typechecked
from pyro.poutine import Trace
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from pyro.poutine.handlers import _make_handler

from combinators.pyro.traces import (
    concat_traces,
    assert_no_overlap,
    is_substituted,
    is_observed,
    is_auxiliary,
    not_observed,
    not_substituted,
    node_filter,
    membership_filter,
    _and,
    _or,
)


@typechecked
class Out(NamedTuple):
    output: Any
    log_weight: Tensor
    trace: Trace


class WithSubstitutionMessenger(ReplayMessenger):
    def _pyro_sample(self, msg):
        orig_infer = msg["infer"]
        yes_substitute = (
            self.trace is not None
            and msg["name"] in self.trace
            and not msg["is_observed"]
        )
        super()._pyro_sample(msg)
        if yes_substitute:
            new_infer = msg["infer"]
            orig_infer.update(new_infer)
            orig_infer["substituted"] = True
            msg["infer"] = orig_infer


_handler_name, _handler = _make_handler(WithSubstitutionMessenger)
_handler.__module__ = __name__
locals()[_handler_name] = _handler


class AuxiliaryMessenger(Messenger):
    def __init__(self) -> None:
        super().__init__()

    def _pyro_sample(self, msg):
        msg["infer"]["is_auxiliary"] = True


_handler_name, _handler = _make_handler(AuxiliaryMessenger)
_handler.__module__ = __name__
locals()[_handler_name] = _handler


@typechecked
def get_marginal(trace: Trace) -> Trace:
    return concat_traces(trace, site_filter=lambda _, n: not is_auxiliary(n))


class inference(object):
    pass


Inference = Union[inference, Callable[..., Out]]


class targets(inference):
    pass


Targets = Union[targets, Callable[..., Out]]


class proposals(inference):
    pass


Proposals = Union[proposals, Callable[..., Out]]

# FIXME evaluation in context of loss
@typechecked
class primitive(targets, proposals):
    def __init__(self, program: Callable[..., Any]):
        self.program = program

    def __call__(self, *args, **kwargs) -> Out:
        with TraceMessenger() as tracer:
            out = self.program(*args, **kwargs)
            tr: Trace = tracer.trace

            lp = tr.log_prob_sum(node_filter(_or(is_substituted, is_observed)))
            lp = lp if isinstance(lp, Tensor) else tensor(lp)
            return Out(output=out, log_weight=lp, trace=tr)


Primitive = Union[Callable[..., Out], primitive]


@typechecked
class extend(targets):
    def __init__(self, p: Targets, f: Primitive):
        self.p, self.f = p, f

    def __call__(self, *args, **kwargs) -> Out:
        p_out: Out = self.p(*args, **kwargs)
        f_out: Out = auxiliary(self.f)(p_out.output, *args, **kwargs)
        p_trace, f_trace = p_out.trace, f_out.trace

        under_substitution = any(
            map(is_substituted, [*p_trace.nodes.values(), *f_trace.nodes.values()])
        )

        assert_no_overlap(p_trace, f_trace, location=type(self))
        assert all(map(not_observed, f_trace.nodes.values()))
        if not under_substitution:
            assert f_out.log_weight == 0.0
            assert all(map(not_substituted, f_trace.nodes.values()))

        log_u2 = f_trace.log_prob_sum()

        return Out(
            trace=concat_traces(p_out.trace, f_out.trace),
            log_weight=p_out.log_weight + log_u2,
            output=f_out.output,
        )


@typechecked
class compose(proposals):
    def __init__(self, q2: Proposals, q1: Proposals):
        self.q1, self.q2 = q1, q2

    def __call__(self, *args, **kwargs) -> Out:
        q1_out = self.q1(*args, **kwargs)
        q2_out = self.q2(*args, **kwargs)
        assert_no_overlap(q2_out.trace, q1_out.trace, location=type(self))

        return Out(
            trace=concat_traces(q2_out.trace, q1_out.trace),
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
        )


@typechecked
class propose(proposals):
    def __init__(
        self,
        p: Targets,
        q: Proposals,
        loss_fn: Callable[[Out, Tensor], Tensor] = (lambda x, fin: fin),
    ):
        self.p, self.q = p, q

    def __call__(self, *args, **kwargs) -> Out:
        q_out = self.q(*args, **kwargs)
        p_out = with_substitution(self.p, trace=q_out.trace)(*args, **kwargs)

        rho_1 = set(q_out.trace.nodes.keys())
        # tau_filter = _and(not_observed, is_random_variable)
        tau_filter = not_observed
        tau_1 = set({k for k, v in q_out.trace.nodes.items() if tau_filter(v)})
        tau_2 = set({k for k, v in p_out.trace.nodes.items() if tau_filter(v)})

        # FIXME this hook exists to reshape NVI for stl
        # q_trace = dispatch(self.transf_q_trace, q_out.trace, **inf_kwargs)
        lu_1 = q_out.trace.log_prob_sum(membership_filter(rho_1 - (tau_1 - tau_2)))
        lw_1 = q_out.log_weight

        # We call that lv because its the incremental weight in the IS sense
        lv = p_out.log_weight - lu_1
        lw_out = lw_1 + lv

        m_trace = get_marginal(p_out.trace)
        # FIXME: this is not accounted for -- will return the final kernel output, not the initial output
        # should be something like: m_output = m_trace["_RETURN"]
        m_output = p_out.output

        return Out(
            # FIXME local gradient computations
            # trace=m_trace if self._no_reruns else rerun_with_detached_values(m_trace),
            trace=m_trace,
            log_weight=lw_out.detach(),
            output=m_output,
        )
