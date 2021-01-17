#!/usr/bin/env python3
import math
import torch
import operator
from functools import partial, reduce
from torch import Tensor, distributions, Size, nn
import torch.distributions as D
from typing import Optional, Dict, Union, Callable
from combinators.tensor.utils import kw_autodevice, autodevice

from combinators import Program
from combinators.embeddings import CovarianceEmbedding
from combinators.stochastic import Trace, ImproperRandomVariable, RandomVariable, Provenance
import combinators.trace.utils as trace_utils

class Distribution(Program):
    """ Normalized version of Density but trying to limit overly-complex class heirarchies """

    def __init__(self, name:str, dist:distributions.Distribution):
        super().__init__()
        self.name = name
        self.dist = dist
        self.RandomVariable = RandomVariable

    def model(self, trace, sample_shape=torch.Size([1,1]), validate=True):
        trace_utils.update_RV_address(trace, self.name, self.dist, sample_shape=sample_shape, validate=validate)
        return trace[self.name].value

    def __repr__(self):
        return f'[{self.name}]' + super().__repr__()

class Normal(Distribution):
    def __init__(self, loc, scale, name, reparam=True, device=None):
        as_tensor = lambda x: x.to(autodevice(device)) if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float, requires_grad=reparam, **kw_autodevice(device))

        self.loc, self.scale = as_tensor(loc), as_tensor(scale)
        self._loc, self._scale = self.loc.cpu().item(), self.scale.cpu().item()

        self._dist = distributions.Normal(loc=self.loc, scale=self.scale)
        super().__init__(name, self._dist)

    def __repr__(self):
        return f"Normal(name={self.name}, loc={self._loc}, scale={self._scale})"

    def as_dist(self, as_multivariate=False):
        return self._dist if not as_multivariate else \
            distributions.MultivariateNormal(loc=self._dist.loc.unsqueeze(0), covariance_matrix=torch.eye(1, device=self._dist.loc.device))

class MultivariateNormal(Distribution):
    def __init__(self, loc, cov, name, reparam=True, device=None):
        as_tensor = lambda x: x.to(autodevice(device)) if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float, requires_grad=reparam, **kw_autodevice(device))
        self.loc, self.cov = as_tensor(loc), as_tensor(cov)
        dist = distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
        super().__init__(name, dist)

class Categorical(Distribution):
    def __init__(self, name, probs=None, logits=None, validate_args=None): #, num_samples=100):
        self.probs = probs
        self.logits = logits
        self.validate_args = validate_args
        super().__init__(name, distributions.Categorical(probs, logits, validate_args))

class Density(Program):
    """ A program that represents a single unnormalized distribution that you can query logprobs on. """

    def __init__(self, name, log_density_fn:Callable[[Tensor], Tensor]):
        super().__init__()
        self.name = name
        self.log_density_fn = log_density_fn
        self.RandomVariable = ImproperRandomVariable # might be useful

    def model(self, trace, **kwargs):
        assert self.name in trace, "an improper RV can only condition on values in an existing trace"
        rv = ImproperRandomVariable(log_density_fn=self.log_density_fn, value=trace[self.name].value, provenance=Provenance.OBSERVED)
        trace.append(rv, name=self.name)
        return None

    def __repr__(self):
        return f'[{self.name}]' + super().__repr__()

class Tempered(Density):
    def __init__(self, name, d1:Union[Distribution, Density], d2:Union[Distribution, Density], beta:Tensor, optimize=False):
        assert torch.all(beta > 0.) and torch.all(beta < 1.), \
            "tempered densities are β=(0, 1) for clarity. Use model directly for β=0 or β=1"
        super().__init__(name, self.log_density_fn)
        self.beta = beta
        self.density1 = d1
        self.density2 = d2
        # FIXME: needed for NVI*/NVIR*
        if optimize:
            raise NotImplementedError("Also need to torch.sigmoid to get beta back")
            self.logit = nn.Parameter(torch.logit(beta))

    def log_density_fn(self, value:Tensor) -> Tensor:
        def log_prob(g, value):
            return g.log_density_fn(value) if isinstance(g, Density) else g.dist.log_prob(value)

        # with torch.no_grad(): # you can't learn anything about this density for the moment
        t = self.beta
        return log_prob(self.density1, value)*(1-t) + \
               log_prob(self.density2, value)*t

    def __repr__(self):
        return "[β={:.4f}]".format(self.beta.item()) + super().__repr__()


class GMM(Density):
    def __init__(self, locs, covs, name="GMM"):
        assert len(locs) == len(covs)
        self.K = K = len(locs)
        super().__init__(name, self.log_density_fn)
        # FIXME: self.components = nn.ParameterList([distributions.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k]) for k in range(K)])
        self.components = [distributions.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k]) for k in range(K)]
        self.assignments = distributions.Categorical(torch.ones(K)) # take this off the trace
        # self.assignments = Categorical("assignments", probs=torch.ones(K))

    def sample(self, sample_shape=torch.Size([1])):
        """ only used to visualize samples """
        # NOTE: no trace being used here
        trace = Trace()
        zs = self.assignments.sample(sample_shape=sample_shape)

        # trace.update(a_trace)
        cluster_shape = (1, *zs.shape[1:-1])
        xs = []
        values, indicies = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            x_k = self.components[k].sample(sample_shape=(n_k, *zs.shape[1:-1]))
            xs.append(x_k)

        xs = torch.cat(xs)[indicies]

        rv = ImproperRandomVariable(log_density_fn=self.log_density_fn, value=xs, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace, xs

    def log_density_fn(self, value): # , log_weights, cond_set, param_set):
        lds = []
        for i, comp in enumerate(self.components):
            ld_i = comp.log_prob(value) # + log_weights[i]
            lds.append(ld_i)
        lds_ = torch.stack(lds, dim=0)
        ld = torch.logsumexp(lds_, dim=0)
        return ld

class RingGMM(GMM):
    def __init__(self, name="RingGMM", loc_scale=5, scale=1, count=8, device=None):
        angles = list(range(0, 360, 360//count))[:count] # integer division may give +1
        position = lambda radians: [math.cos(radians), math.sin(radians)]
        locs = torch.tensor([position(a*math.pi/180) for a in angles], **kw_autodevice(device)) * loc_scale
        covs = [torch.eye(2, **kw_autodevice(device)) * scale for _ in range(count)]
        super().__init__(name=name, locs=locs, covs=covs)
