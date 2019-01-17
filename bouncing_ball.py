#!/usr/bin/env python3

import torch
from torch.distributions import Categorical, Dirichlet, MultivariateNormal
from torch.distributions import Normal
from torch.distributions.transforms import LowerCholeskyTransform

import utils

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
    dir_covs = trace.param_sample(Normal, params, name='directions__scale')

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
