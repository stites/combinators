from typing import Any, Callable, TypeVar
from typeguard import typechecked
from pyro import poutine
from pyro.poutine import Trace
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from typing import NamedTuple
from torch import Tensor, tensor
from typing import NamedTuple, Any, Callable, Union, Optional, Tuple, Set
from typeguard import typechecked
from pyro.poutine import Trace

# type aliases
Node = dict
T = TypeVar("T")
Predicate = Callable[[Any], bool]
SiteFilter = Callable[[str, Node], bool]


def true(*args, **kwargs):
    return True


@typechecked
def concat_traces(
    *traces: Trace, site_filter: Callable[[str, Any], bool] = true
) -> Trace:
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


@typechecked
def assert_no_overlap(t0: Trace, t1: Trace, location=""):
    assert (
        len(set(t0.nodes.keys()).intersection(set(t1.nodes.keys()))) == 0
    ), f"{location}: addresses must not overlap"


@typechecked
def is_observed(node: Node) -> bool:
    return node["is_observed"]


@typechecked
def not_observed(node: Node) -> bool:
    return not is_observed(node)


@typechecked
def _check_infer_map(k: str) -> Callable[[Node], bool]:
    return lambda node: k in node["infer"] and node["infer"][k]


is_substituted = _check_infer_map("substituted")
is_auxiliary = _check_infer_map("is_auxiliary")


@typechecked
def not_substituted(node: Node) -> bool:
    return not is_substituted(node)


@typechecked
def is_random_variable(node: Node) -> bool:
    # FIXME as opposed to "is improper random variable"
    raise NotImplementedError()


@typechecked
def is_improper_random_variable(node: Node) -> bool:
    raise NotImplementedError()


@typechecked
def _and(p0: Predicate, p1: Predicate) -> Predicate:
    return lambda x: p0(x) and p1(x)


@typechecked
def _or(p0: Predicate, p1: Predicate) -> Predicate:
    return lambda x: p0(x) or p1(x)


@typechecked
def _not(p: Predicate) -> Predicate:
    return lambda x: not p(x)


@typechecked
def node_filter(p: Callable[[Node], bool]) -> SiteFilter:
    return lambda _, node: p(node)


@typechecked
def addr_filter(p: Callable[[str], bool]) -> SiteFilter:
    return lambda name, _: p(name)


@typechecked
def membership_filter(members: Set[str]) -> SiteFilter:
    return lambda name, _: name in members
