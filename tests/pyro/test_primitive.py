from combinators.program import WithSubstitution
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

import torch
import pytest
import pyro
import pyro.distributions as dist

from combinators.pyro import primitive, with_substitution

# ===========================================================================
# Models
@pytest.fixture
def model0():
    def model():
        cloudy = pyro.sample("cloudy", dist.Bernoulli(0.3))
        cloudy = "cloudy" if cloudy.item() == 1.0 else "sunny"
        mean_temp = {"cloudy": 55.0, "sunny": 75.0}[cloudy]
        scale_temp = {"cloudy": 10.0, "sunny": 15.0}[cloudy]
        temp = pyro.sample("temp", dist.Normal(mean_temp, scale_temp))
        return cloudy, temp.item()
    yield model

@pytest.fixture
def model1():
    def model():
        flip = pyro.sample("flip", dist.Bernoulli(0.3))
        return "heads" if flip.item() == 1.0 else "tails"
    yield model

# ===========================================================================
# Asserts
def assert_no_observe(model):
    assert isinstance(model, primitive)

    trace = poutine.trace(model).get_trace()
    for name, node in trace.nodes.items():
        if node["type"] == "sample":
            assert not node["is_observed"], f"node {name} is observed!"

def assert_no_overlap(primitive_model, non_overlapping_primitive):
    tr0, tr1 = primitive_model().trace, non_overlapping_primitive().trace
    tr0_names = {name for name, _ in tr0.nodes.items()}
    for name, _ in tr1.nodes.items():
        assert name not in tr0_names, f"{name} is in both traces!"


def assert_log_weight_zero(primitive_output):
    lw = primitive_output.log_weight
    assert isinstance(lw, Tensor) and lw == 0.0

# ===========================================================================
# tests
def test_constructor(model0):
    m = primitive(model0)
    assert_no_observe(m)
    out = m()
    assert_no_observe(m)
    assert_log_weight_zero(out)

    o = out.output[0]
    assert isinstance(o, str) and o in {"cloudy", "sunny"}
    assert isinstance(out.trace, Trace)

def test_no_overlapping_variables(model0, model1):
    m0 = primitive(model0)
    m1 = primitive(model1)
    assert_no_observe(m0)
    assert_no_observe(m1)
    assert_no_overlap(m0, m1)

def test_with_substitution(model0):
    p = primitive(model0)
    q = primitive(model0)
    p_out = p()
    q_out = with_substitution(q, trace=p_out.trace)()

    p_addrs = set(p_out.trace.nodes.keys())
    q_addrs = set(q_out.trace.nodes.keys())
    assert p_addrs.intersection(q_addrs) == p_addrs.union(q_addrs)

    valueat = lambda o, a: o.trace.nodes[a]['value']

    for a in p_addrs:
        assert q_out.trace.nodes[a]['value'] == p_out.trace.nodes[a]['value']
        assert valueat(q_out, a) == valueat(p_out, a)
        assert q_out.trace.nodes[a]["infer"]['substituted'] == True

    assert p_out.output == q_out.output
    assert q_out.log_weight != 0.

# import tests from test_inference.py
def test_run_a_primitive_program(simple1):
    s1_out = primitive(simple1)()
    assert set(s1_out.trace.nodes.keys()) == {"z_1", "z_2", "x_1", "x_2"}
    assert s1_out.log_weight == s1_out.trace.log_prob_sum(site_filter=lambda name, _: name in {"x_1", "x_2"})

def test_simple_substitution(simple1, simple2):
    s1, s2 = primitive(simple1), primitive(simple2)
    s1_out = s1()
    s2_out = with_substitution(s2, trace=s1_out.trace)()

    rho_f_addrs = {"x_2", "x_3", "z_2", "z_3"}
    tau_f_addrs = {"z_2", "z_3"}
    tau_p_addrs = {"z_1", "z_2"}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    lw_out = s2_out.trace.log_prob_sum(site_filter=lambda name, _: name in nodes)

    assert (lw_out == s2_out.log_weight).all()
    assert (
        len(set({"z_1", "x_1"}).intersection(set(s2_out.trace.nodes.keys()))) == 0
    )
