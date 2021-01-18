#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
import ast
import weakref

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike, get_shape_kwargs
import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad

from combinators.traceable import TraceModule, Conditionable

class Cond:
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    FIXME: can't condition a conditioned model at the moment
    """
    def __init__(self, process: Conditionable, trace: Optional[Trace], requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), _step=None) -> None:
        self.process = process
        self.conditioning_trace = trace_utils.copytrace(trace, requires_grad=requires_grad, detach=detach) if trace is not None else Trace()
        self._requires_grad = requires_grad
        self._detach = detach

    def __call__(self, *args:Any, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        self.process._cond_trace = self.conditioning_trace
        out = self.process(*args, **kwargs)
        self.process._cond_trace = Trace()
        return out

# class Cond2:
#     def __init__(self, program:Cond, trace:Trace = Trace()) -> None:
#         super().__init__()
#         self._conditioning_trace: Trace = trace_utils.copytrace(trace)
#         self.program = program
#
#     def __call__(self, *args, **kwargs):
#         self.program._cond_trace = self._conditioning_trace
#         out = self.program(*args, **kwargs)
#         self.program._cond_trace = Trace()
#         return out

class Program(TraceModule):
    """ superclass of a program? """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def model(self, trace: Trace, *args:Any, **kwargs:Any) -> Output:
        raise NotImplementedError()

    def forward(self, *args:Any, sample_dims=None, **kwargs:Any) -> Tuple[Trace, Output]:
        trace = self.get_trace()

        out = self.model(trace, *args, **get_shape_kwargs(self.model, sample_dims=sample_dims), **kwargs)

        # TODO: enforce purity?
        self._trace = trace

        return trace, out

    @classmethod
    def factory(cls, fn, name:str = ""):
        def generic_model(self, *args, **kwargs):
            return fn(*args, **kwargs)
            # import ipdb; ipdb.set_trace();
            #
            # if not isinstance(out, (tuple, list)):
            #     raise TypeError("ad-hoc models are expected to return a tuple or list with the input trace as the first return.")
            # elif not isinstance(out[0], Trace):
            #     # just being lazy here
            #     raise TypeError("ad-hoc models are expected to return a tuple or list with the input trace as the first return.")
            # elif len(out) == 1:
            #     # okay not about to think about this part very hard...
            #     return out[0], None
            # elif len(out) == 2:
            #     return out
            # else:
            #     # let users slide here, but this seems pretty painful
            #     trace = out[0]
            #     final = out[1:]
            #     return trace, final

        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(model=generic_model)
        )

        return AProgram()

    def copy(self):
        def generic_model(self, *args, **kwargs):
            return fn(*args, **kwargs)

        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(model=generic_model)
        )

        return AProgram()

PROGRAM_REGISTRY = dict()


def model(name:str = ""):
    def wrapper(fn):
        global PROGRAM_REGISTRY
        model_key = name + repr(fn)
        instance = Program.factory(fn, name)
        if model_key not in PROGRAM_REGISTRY:
            PROGRAM_REGISTRY[model_key] = instance
        return PROGRAM_REGISTRY[model_key]
    return wrapper

