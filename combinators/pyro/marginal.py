from pyro.poutine.messenger import Messenger
from pyro.poutine.trace_messenger import (
    TraceMessenger,
    TraceHandler,
)

class MarginalMessenger(TraceMessenger):  # TODO
    """really just "keep track of this trace for later" state"""

    def __init__(self, graph_type=None, param_only=None):
        super().__init__(graph_type=graph_type, param_only=param_only)
        self.marginal = self.trace

    def __enter__(self):
        o = super().__enter__()
        self.marginal = self.trace
        return o

    def __exit__(self, *args, **kwargs):
        o = super().__exit__(*args, **kwargs)
        self.marginal = self.trace
        return o

    def __call__(self, fn):
        return MarginalHandler(self, fn)

    def get_marginal(self):
        return self.marginal.copy()

    def _reset(self):
        raise RuntimeError()

    def _pyro_post_sample(self, msg):
        o = super()._pyro_post_sample(msg)
        self.marginal = self.trace
        return o

    def _pyro_post_param(self, msg):
        super()._pyro_post_sample(msg)
        self.marginal = self.trace


marginal = MarginalMessenger


class MarginalHandler(TraceHandler):
    def __init__(self, msngr, fn):
        super().__init__(msngr, fn)

    def __call__(self, *args, **kwargs):
        o = super().__call__(*args, **kwargs)
        self.msngr.marginal = self.msngr.trace
        return o

    @property
    def marginal(self):
        return self.msngr.marginal

    def get_marginal(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.msngr.get_marginal()

        # site


class MarginalizeMessenger(Messenger):
    def __init__(self, graph_type=None, param_only=None):
        """
        :param string graph_type: string that specifies the type of graph
            to construct (currently only "flat" or "dense" supported)
        :param param_only: boolean that specifies whether to record sample sites
        """
        super().__init__()
        if graph_type is None:
            graph_type = "flat"
        if param_only is None:
            param_only = False
        assert graph_type in ("flat", "dense")
        self.graph_type = graph_type
        self.param_only = param_only
        self.trace = Trace(graph_type=self.graph_type)

    def __enter__(self):
        self.marginal = Trace(graph_type=self.graph_type)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        """
        Adds appropriate edges based on cond_indep_stack information
        upon exiting the context.
        """
        if self.param_only:
            for node in list(self.trace.nodes.values()):
                if node["type"] != "param":
                    self.trace.remove_node(node["name"])
        if self.graph_type == "dense":
            identify_dense_edges(self.trace)
        return super().__exit__(*args, **kwargs)

    def __call__(self, fn):
        """
        TODO docs
        """
        return TraceHandler(self, fn)


## TODO
#def MarginalizeHandler(cls, p, p_out) -> Tuple[Any, Trace]:
#    return p_out.output, p_out.trace
#
