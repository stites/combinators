#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
from typing import Iterable

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs
from combinators.trace.utils import RequiresGrad, copytrace
from combinators.traceable import TraceModule, Conditionable
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Observable
from inspect import signature
import inspect
from combinators.objectives import nvo_avo

@typechecked
class State(Iterable):
    def __init__(self, trace:Trace, weights:Optional[Tensor], output:Output):
        self.trace = trace
        self.mweights = weights
        self.output = output

    def __repr__(self):
        return "; ".join([
            f'lw: {tensor_utils.show(self.mweights) if self.mweights is not None else self.mweights}',
            f'tr: {trace_utils.show(self.trace)}',
            f'out: {tensor_utils.show(self.output) if isinstance(self.output, Tensor) else self.output}',
        ])

    def __iter__(self):
        '''
        NOTE: only returns arguments, not the log probs
        '''
        for x in [self.trace, self.output]:
            yield x


class property_dict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        max_len = max(map(len, self.keys()))
        return "\n  ".join([
            f"<property_dict>:",
            *[("{:>"+ str(max_len)+ "}: {}").format(k, tensor_utils.show(v) if isinstance(v, Tensor) else v) for k, v in self.items()]
        ])

@typechecked
class KCache(property_dict):
    def __init__(self, program:Optional[State], kernel:Optional[State]):
        self.program = program
        self.kernel = kernel
    def __repr__(self):
        return "Kernel Cache:" + \
            "\n  program: {}".format(self.program) + \
            "\n  kernel:  {}".format(self.kernel)

@typechecked
class PCache:
    def __init__(self, target:Optional[State], proposal:Optional[State]):
        self.target = target
        self.proposal = proposal
    def __repr__(self):
        return "Propose Cache:" + \
            "\n  proposal: {}".format(self.proposal) + \
            "\n  target:   {}".format(self.target)

class Inf(ABC):
    pass

class Condition(Inf, nn.Module):
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    """
    def __init__(self, process: Conditionable, trace: Trace, requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), _step=None) -> None:
        nn.Module.__init__(self)
        self.process = process
        self.requires_grad = requires_grad
        self.detach = detach
        self.conditioning_trace = trace

    def forward(self, *args:Any, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        self.process._cond_trace = copytrace(self.conditioning_trace, requires_grad=self.requires_grad, detach=self.detach)

        out = self.process(*args, **kwargs)

        self.process._cond_trace = None

        return out

class KernelInf(nn.Module, Conditionable):
    def __init__(self,
                 # program:Union[Program, KernelInf],
                 program:Union[Program, Any],
                 kernel:Kernel, _step:Optional[int]=None, _permissive_arguments:bool=True):
        nn.Module.__init__(self)
        Conditionable.__init__(self)
        self._cache = KCache(None, None)
        self._step = _step
        self._permissive_arguments = _permissive_arguments
        self.program = program
        self.kernel = kernel

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))

    def _program_args(self, fn, *args):
        if self._permissive_arguments:
            assert args is None or len(args) == 0, "need to filter this list, but currently don't have an example"
            # return [v for k,v in args.items() if check_passable_arg(k, fn)]
            return args
        else:
            return args

    def _program_kwargs(self, fn, **kwargs):
        if self._permissive_arguments and isinstance(fn, Program):
            return {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.model)}
        else:
            return kwargs

    def _run_program(self, *program_args:Any, sample_dims=None, **program_kwargs:Any):
        runnable = Condition(self.program, self._cond_trace) if self._cond_trace is not None else self.program

        return runnable(
            *self._program_args(self.program, *program_args),
            sample_dims=sample_dims,
            **self._program_kwargs(self.program, **program_kwargs))

    def _run_kernel(self, program_trace: Trace, program_output:Output, sample_dims=None):
        runnable = Condition(self.kernel, self._cond_trace) if self._cond_trace is not None else self.kernel
        return runnable(program_trace, program_output, sample_dims=sample_dims)

class Reverse(KernelInf, Inf):
    def __init__(self, program: Union[Program, KernelInf], kernel: Kernel, _step=None, _permissive=True) -> None:
        super().__init__(program, kernel, _step, _permissive)

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        program_state = State(*self._run_program(*program_args, sample_dims=sample_dims, **program_kwargs))

        kernel_state = State(*self._run_kernel(*program_state, sample_dims=sample_dims))

        log_aux = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims, nodes=kernel_state.trace._nodes)

        return program_state.trace, log_aux, program_state.output

class Forward(KernelInf, Inf):
    def __init__(self, kernel: Kernel, program: Union[Program, KernelInf], _step=None, _permissive=True) -> None:
        super().__init__(program, kernel, _step, _permissive)

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs) -> Tuple[Trace, Optional[Trace], Output]:
        program_state = State(*self._run_program(*program_args, sample_dims=sample_dims, **program_kwargs))

        kernel_state = State(*self._run_kernel(*program_state, sample_dims=sample_dims))

        log_joint = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims, nodes=kernel_state.trace._nodes)

        self._cache = property_dict(program=program_state, kernel=kernel_state)

        return kernel_state.trace, log_joint, kernel_state.output

ProposeTraces = namedtuple("ProposeTraces", ["proposal", "target"])

class Propose(nn.Module, Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Inf, Program, Condition], loss_fn:Callable[[Tensor], Tensor], validate:bool=True, _debug=False, _step=None):
        nn.Module.__init__(self)
        self.target = target
        self.proposal = proposal
        self.loss_fn = loss_fn
        self.validate = validate
        self._cache = PCache(None, None)
        self._step = _step # used for debugging
        self._debug = _debug # used for debugging

    def forward(self, *shared_args, sample_dims=None, **shared_kwargs) -> Tuple[ProposeTraces, Optional[Trace], Output]:
        # FIXME: target and proposal args can / should be separated
        proposal_state = State(*self.proposal(*shared_args, sample_dims=sample_dims, **shared_kwargs))

        conditioned_target = Condition(self.target, proposal_state.trace)

        target_state = State(*conditioned_target(*shared_args, sample_dims=sample_dims, **shared_kwargs))

        joint_proposal_trace = proposal_state.trace
        joint_target_trace = target_state.trace

        self._cache = property_dict(target=target_state, proposal=proposal_state)
        state = self._cache

        lv = Propose.log_weights(joint_target_trace, joint_proposal_trace, validate=self.validate, sample_dims=sample_dims)
        def nvo_avo(lv: Tensor, sample_dims=0) -> Tensor:
            # values = -lv
            # log_weights = torch.zeros_like(lv)

            # nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
            # loss = (nw * values).sum(dim=(sample_dims,), keepdim=False)
            loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
            return loss


        # if self.proposal._cache is not None:
        #     # FIXME: this is a hack for the moment and should be removed somehow.
        #     # NOTE: this is unnecessary for the e2e/test_1dgaussians.py, but I am a bit nervous about double-gradients
        #     if isinstance(self.proposal._cache, PCache):
        #         raise NotImplemented("If this is the case (which it can be) i will actually need a way to propogate these detachments in a smarter way")
        #
        #     # can we always assume we are in NVI territory?
        #     for k, rv in self.proposal._cache.program.trace.items():
        #         joint_proposal_trace[k]._value = rv.value.detach()
        #
        #     proposal_keys = set(self.proposal._cache.program.trace.keys())
        #     target_keys = set(self.target._cache.kernel.trace.keys()) - proposal_keys
        #     state = PCache(
        #         proposal=State(trace=trace_utils.copysubtrace(proposal_state.trace, proposal_keys), output=proposal_state.output),
        #         target=State(trace=trace_utils.copysubtrace(target_state.trace, target_keys), output=target_state.output),
        #     )
        breakpoint();

        return ProposeTraces(target=state.target.trace, proposal=state.proposal.trace), lv, state.target.output

    @classmethod
    @typechecked
    def log_weights(cls, target_trace:Trace, proposal_trace:Trace, sample_dims:int=None, validate:bool=True):
        if validate:
            assert trace_utils.valeq(proposal_trace, target_trace, nodes=target_trace._nodes, check_exist=True)

        assert sample_dims != -1, "seems to be a bug in probtorch which blocks this behavior"

        batch_dim=None # TODO

        q_tar = target_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)
        p_tar = proposal_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)
        if validate:
            arv = list(target_trace.values())[0]
            dim = 0 if sample_dims is None else sample_dims
            lp_shape = q_tar.shape[0] if len(q_tar.shape) > 0 else 1
            rv_shape = arv.value.shape[dim] if len(arv.value.shape) > 0 else 1
            if rv_shape != lp_shape:
                raise RuntimeError("shape mismatch between log weight and elements in trace, you are probably missing sample_dims")

        return q_tar - p_tar
