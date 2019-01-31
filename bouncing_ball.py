#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.distributions import Categorical, Dirichlet, MultivariateNormal
from torch.distributions import Normal
from torch.distributions.transforms import LowerCholeskyTransform

import combinators
import foldable
import mcmc
import utils

def reflect_directions(dir_loc):
    dir_locs = dir_loc.unsqueeze(-2).repeat(1, 4, 1)
    dir_locs[:, 1, 1] *= -1
    dir_locs[:, 2, :] *= -1
    dir_locs[:, 3, 0] *= -1
    return dir_locs / (dir_locs**2).sum(dim=-1).unsqueeze(-1).sqrt()

def init_bouncing_ball(params=None, trace=None, data={}):
    initial_alpha = trace.param_sample(Dirichlet, params, name='alpha_0')
    pi = trace.sample(Dirichlet, initial_alpha, name='Pi')
    initial_z = trace.variable(Categorical, pi, name='direction_0')

    transition_alpha = torch.stack([
        trace.param_sample(Dirichlet, params, name='alpha_%d' % (d+1))
        for d in range(4)
    ], dim=1)
    transition = torch.stack([
        trace.sample(Dirichlet, transition_alpha[:, d], name='A_%d' % (d+1))
        for d in range(4)
    ], dim=1)

    dir_locs = trace.param_sample(Normal, params, name='directions__loc')
    dir_locs = reflect_directions(dir_locs)
    dir_covs = trace.param_sample(Normal, params, name='directions__cov')

    initial_position = trace.param_observe(
        Normal, params, name='position_0',
        value=data['position_0'].expand(params['position_0']['loc'].shape)
    )

    return initial_position, initial_z, transition, dir_locs, dir_covs

def bouncing_ball_step(theta, t, trace=None, data={}):
    position, z_prev, transition, dir_locs, dir_covs = theta
    directions = {
        'loc': dir_locs,
        'covariance_matrix': dir_covs,
    }
    t += 1

    transition_prev = utils.particle_index(transition, z_prev)
    z_current = trace.variable(Categorical, transition_prev,
                               name='direction_%d' % t)
    direction = utils.vardict_particle_index(directions, z_current)
    direction_covariance = direction['covariance_matrix']
    velocity = trace.observe(
        MultivariateNormal, data.get('displacement_%d' % t), direction['loc'],
        scale_tril=LowerCholeskyTransform()(direction_covariance),
        name='displacement_%d' % t,
    )
    position = position + velocity

    return position, z_current, transition, dir_locs, dir_covs

def identity_step(theta, t, trace=None, data={}):
    return theta

class ProposalStep(nn.Module):
    def __init__(self):
        super(ProposalStep, self).__init__()
        self.direction_predictor = nn.Sequential(
            nn.Linear(2, 4),
            nn.Softsign(),
            nn.Linear(4, 4),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, theta, t, trace=None, data={}):
        position, _, transition, dir_locs, dir_covs = theta
        directions = {
            'loc': dir_locs,
            'covariance_matrix': dir_covs,
        }
        t += 1

        direction_predictions = self.direction_predictor(
            data.get('displacement_%d' % t)
        )
        direction_predictions = direction_predictions.expand(
            position.shape[0], 4
        )
        z_prev = Categorical(logits=direction_predictions).sample()
        transition_prev = utils.particle_index(transition, z_prev)
        z_current = trace.variable(Categorical, transition_prev,
                                   name='direction_%d' % t)

        direction = utils.vardict_particle_index(directions, z_current)
        direction_covariance = direction['covariance_matrix']
        velocity = trace.sample(
            MultivariateNormal, direction['loc'],
            scale_tril=LowerCholeskyTransform()(direction_covariance),
            name='displacement_%d' % t,
        )
        position = position + velocity
        return position, z_current, transition, dir_locs, dir_covs

def generative_model(data, params, particle_shape, step_generator):
    params['position_0']['loc'] = data['position_0']
    init_population = combinators.hyper_population(
        combinators.PrimitiveCall(init_bouncing_ball), particle_shape,
        hyper=params
    )
    return mcmc.reduce_resample_move_smc(
        combinators.PrimitiveCall(bouncing_ball_step), particle_shape,
        step_generator, initializer=init_population
    )

def proposal_step():
    return combinators.PrimitiveCall(ProposalStep(), name='bouncing_ball_step')

def proposal_model(data, params, particle_shape, step_generator):
    params['position_0']['loc'] = data['position_0']
    init_proposal = combinators.hyper_population(
        combinators.PrimitiveCall(init_bouncing_ball), particle_shape,
        trainable=params
    )
    return foldable.Reduce(foldable.Foldable(proposal_step(),
                                             initializer=init_proposal),
                           step_generator)
