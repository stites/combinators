#!/usr/bin/env python3

import torch
from combinators.trace.utils import copytraces
from combinators.stochastic import Trace

def test_copytraces():
    t = Trace()
    x = copytraces(t)
    assert x is not t
    assert set(x.keys()) == set(t.keys())

def test_copytraces_single():
    t = Trace()
    t.normal(loc=torch.ones(2), scale=torch.ones(2), name="x")
    assert 'x' in t and len(t.keys()) ==1

    x = copytraces(t)
    assert x is not t
    assert set(x.keys()) == set(t.keys())
