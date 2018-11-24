#!/usr/bin/env python3

import collections
import logging

from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
import torch.distributions as dists
from torch.nn.functional import log_softmax

import combinators
import trace_tries
import utils

def index_select_rv(rv, dim, indices):
    result = rv
    if not rv.observed:
        value = rv.value.index_select(dim, indices)
        result = RandomVariable(rv.dist, value, rv.observed, rv.mask,
                                rv.reparameterized)
    return result

class PopulationResampler(combinators.Population):
    def __init__(self, sampler, particle_shape):
        super(PopulationResampler, self).__init__(sampler, particle_shape,
                                                  before=True)

    def sample_hook(self, results, trace):
        results, trace = super(PopulationResampler, self).sample_hook(results,
                                                                      trace)
        weights = trace.normalized_log_weight()
        resampler = dists.Categorical(logits=weights)
        ancestor_indices = resampler.sample(self.particle_shape)
        results = [val.index_select(0, ancestor_indices) for val in results]
        trace_resampler = lambda k, rv: index_select_rv(rv, 0, ancestor_indices)
        return tuple(results), trace.map(trace_resampler)

def smc(stepwise, particle_shape, step_generator, initializer=None):
    resampler = PopulationResampler(stepwise, particle_shape)
    return combinators.Reduce(resampler, step_generator, initializer)

def variational_importance(sampler, num_iterations, data, use_cuda=True,
                           lr=1e-6, inclusive_kl=False, patience=50):
    optimizer = torch.optim.Adam(list(sampler.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-6, patience=patience, verbose=True,
        mode='min' if inclusive_kl else 'max',
    )

    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    bounds = list(range(num_iterations))
    for t in range(num_iterations):
        optimizer.zero_grad()

        trace = trace_tries.HierarchicalTrace()
        _, inference, _ = sampler.simulate(data=data, trace=trace)

        bound = -inference.marginal_log_likelihood()
        bound_name = 'EUBO' if inclusive_kl else 'ELBO'
        signed_bound = bound if inclusive_kl else -bound
        logging.info('%s=%.8e at epoch %d', bound_name, signed_bound, t + 1)
        bound.backward()
        optimizer.step()
        bounds[t] = bound if inclusive_kl else -bound
        scheduler.step(bounds[t])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    trained_params = sampler.args_vardict()

    return inference, trained_params, bounds
