from typing import NamedTuple
from torch import Tensor, tensor
from typing import NamedTuple, Any
from pyro import poutine
from typeguard import typechecked

import pytest
from torch.distributions import Bernoulli, Normal

from pytest_bdd import scenarios, given, when, then, parsers

# Constants

def model0():
    cloudy = Bernoulli(0.3).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = Normal(mean_temp, scale_temp).rsample()
    return cloudy, temp.item()


# Scenarios

scenarios('../features/primitive.feature')

# Fixtures

@pytest.fixture
def model():
    yield model0

# Given Steps

@given('a pyro model')
def a_pyro_model(model):
    return model

# When Steps

@when("it has no observe statements")
def confirm_no_observe(model):
    trace = poutine.trace(model).get_trace()
    for name, node in trace.nodes.items():
        if node["type"] == "sample":
            assert not node["is_observed"], f"node {name} is observed!"

@typechecked
class Out(NamedTuple):
    output : Any
    log_weight : Tensor
    trace : poutine.Trace

class primitive:
    def __init__(self, program):
        self.program = program

    def __call__(self, *args, **kwargs):
        tr = poutine.trace(self.program).get_trace(*args, **kwargs)
        out = poutine.replay(self.program, trace=tr)(*args, **kwargs)
        lp = tr.log_prob_sum()
        lp = lp if isinstance(lp, Tensor) else tensor(lp)
        return Out(output=out, log_weight=lp, trace=tr)

# Then Steps

@then('I get a primitive inference combinator')
def make_primitive(model):
    out = primitive(model)()
    assert out.log_weight == 0.
    assert isinstance(out.log_weight, Tensor)
