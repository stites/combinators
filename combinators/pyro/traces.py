from torch import Tensor, tensor
from typing import Any, Callable, OrderedDict
from typeguard import typechecked
from pyro import poutine
from pyro.poutine import Trace
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from typing import NamedTuple
from torch import Tensor, tensor
from typing import NamedTuple, Any, Callable, Union, Optional, Tuple
from typeguard import typechecked
from pyro.poutine import Trace

# type aliases
Node = dict
Predicate = Callable[[Any], bool]

def true(*args, **kwargs):
    return True

@typechecked
def concat_traces(
    *traces: Trace, site_filter: Callable[[str, Any], bool] = true
)->Trace:
    newtrace = Trace()
    for tr in traces:
        drop_edges = []
        for name, node in tr.nodes.items():
            if site_filter(name, node):
                newtrace.add_node(name, **node)
            else:
                drop_edges.append(name)
        for p, s in zip(tr._pred, tr._succ):
            if p not in drop_edges and s not in drop_edges:
                newtrace.add_edge(p, s)
    return newtrace


def assert_no_overlap(t0, t1, location=""):
    assert (
        len(set(t0.keys()).intersection(set(t1.keys()))) == 0
    ), f"{location}: addresses must not overlap"

@typechecked
def is_observed(node: Node) -> bool:
    return node["is_observed"]


@typechecked
def not_observed(node: Node) -> bool:
    return not is_observed(node)

@typechecked
def is_substituted(node: Node) -> bool:
    INFER, SUBSTITUTED = 'infer', 'substituted'
    return SUBSTITUTED in node[INFER] and node[INFER][SUBSTITUTED]

@typechecked
def is_random_variable(node: Node) -> bool:
    # FIXME as opposed to "is improper random variable"
    raise NotImplementedError()

@typechecked
def is_improper_random_variable(node: Node) -> bool:
    raise NotImplementedError()

@typechecked
def _and(p0:Predicate, p1:Predicate)->Predicate:
    return lambda x: p0(x) and p1(x)

@typechecked
def _or(p0:Predicate, p1:Predicate)->Predicate:
    return lambda x: p0(x) or p1(x)

@typechecked
def _not(p:Predicate)->Predicate:
    return lambda x: not p(x)

# FIXME: I think not using these will cause a bug
class provenance:
    @staticmethod
    def observed(cls, node):
        return is_observed(node) and not is_substituted(node)

    @staticmethod
    def substituted(cls, node):
        return is_substituted(node)

    @staticmethod
    def sampled(cls, node):
        return not (is_observed(node) or is_substituted(node))
