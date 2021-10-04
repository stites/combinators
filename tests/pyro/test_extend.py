from combinators.pyro.inference import get_marginal
from combinators.pyro.traces import is_auxiliary
from pyro import poutine

import torch
import pytest
import pyro
import pyro.distributions as dist

from combinators.pyro import primitive, extend


def test_extend_unconditioned(simple2, simple4):
    s2, s4 = primitive(simple2), primitive(simple4)

    p_out = s2()
    replay_s2 = poutine.replay(s2, trace=p_out.trace)
    assert set(p_out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3"}
    assert p_out.log_weight == p_out.trace.log_prob_sum(lambda n, _: n[0] == 'x')

    f_out = s4(p_out.output)
    replay_s4 = poutine.replay(s4, trace=f_out.trace)
    assert set(f_out.trace.nodes.keys()) == {"z_1"}
    assert f_out.log_weight == 0.

    out = extend(p=replay_s2, f=replay_s4)()
    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1"}
    assert out.log_weight == p_out.log_weight + f_out.trace.log_prob_sum()

    p_nodes = list(filter(lambda kv: kv[0] in p_out.trace.nodes.keys(), out.trace.nodes.items()))
    assert len(p_nodes) > 0
    assert all(list(map(lambda kv: not is_auxiliary(kv[1]), p_nodes)))

    f_nodes = list(filter(lambda kv: kv[0] in f_out.trace.nodes.keys(), out.trace.nodes.items()))
    assert len(f_nodes) > 0
    assert all(list(map(lambda kv: is_auxiliary(kv[1]), f_nodes)))


def test_nested_marginal(simple2, simple4, simple5):
    s2, s4, s5 = primitive(simple2), primitive(simple4), primitive(simple5)

    out = extend(p=extend(p=s2, f=s4), f=s5)()
    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1", "z_5"}

    p_nodes = list(filter(lambda kv: kv[0] not in {"z_1", "z_5"}, out.trace.nodes.items()))
    assert len(p_nodes) > 0

    marg = get_marginal(out.trace)
    assert set(marg.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3"}
