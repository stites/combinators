#!/usr/bin/env python3

import numpy as np
import torch
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators
import utils

class InitBallDynamics(combinators.Primitive):
    def __init__(self, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'dynamics': {
                'loc': torch.cat((torch.zeros(2, 2),
                                  torch.ones(2, 1) / np.sqrt(2)), dim=-1),
                'scale': torch.ones(2, 3),
            },
            'uncertainty': {
                'loc': torch.eye(2),
                'scale': torch.ones(2, 2),
            },
            'noise': {
                'loc': torch.eye(2),
                'scale': torch.ones(2, 2),
            },
            'position_0': {
                'loc': torch.ones(2),
                'covariance_matrix': torch.eye(2),
            },
        } if not params else params
        super(InitBallDynamics, self).__init__(params, trainable, batch_shape,
                                               q)
        self.transform = LowerCholeskyTransform()

    def _forward(self, data={}):
        dynamics = self.param_sample(Normal, name='dynamics')
        uncertainty = self.param_sample(Normal, name='uncertainty')
        uncertainty = self.transform(uncertainty)
        noise = self.transform(self.param_sample(Normal, name='noise'))
        pos_params = self.args_vardict()['position_0']
        pos_scale = LowerCholeskyTransform()(pos_params['covariance_matrix'])
        position = self.sample(MultivariateNormal, loc=pos_params['loc'],
                               scale_tril=pos_scale, name='position_0')
        return dynamics, uncertainty, noise, position

def reflect_on_boundary(position, dynamics, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = position[:, d] - sign * boundary
    overage = torch.where(torch.sign(overage) == sign, overage,
                          torch.zeros(*overage.shape))
    position = list(torch.unbind(position, 1))
    position[d] = position[d] - 2 * overage
    position = torch.stack(position, dim=1)

    overage = overage.unsqueeze(-1).expand(dynamics[:, d].shape)
    dynamics = list(torch.unbind(dynamics, 1))
    dynamics[d] = torch.where(overage != 0.0, -dynamics[d], dynamics[d])
    dynamics = torch.stack(dynamics, dim=1)
    return position, dynamics

class StepBallDynamics(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        dynamics, uncertainty, noise, position = theta

        # Our dynamics here are actually an affine transformation, so one-extend
        # the position.
        velocity = utils.particle_matmul(
            dynamics, torch.cat((position, torch.ones(*self.batch_shape, 1)),
                                dim=-1)
        )
        proposal = position + self.sample(MultivariateNormal, velocity,
                                          scale_tril=uncertainty,
                                          name='velocity_%d' % t)
        for i in range(2):
            for pos in [True, False]:
                proposal, dynamics = reflect_on_boundary(
                    proposal, dynamics, 6.0, d=i, positive=pos
                )
        position = self.observe('position_%d' % (t+1),
                                data.get('position_%d' % (t+1), None),
                                MultivariateNormal, loc=proposal,
                                scale_tril=noise)

        return dynamics, uncertainty, noise, position

class StepBallGuide(combinators.Primitive):
    def __init__(self, T, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'velocities': {
                'loc': torch.zeros(T, 2),
                'scale': torch.ones(T, 2),
            }
        } if not params else params
        self._num_timesteps = T
        super(StepBallGuide, self).__init__(params, trainable, batch_shape, q)

    @property
    def name(self):
        return 'StepBallDynamics'

    def cond(self, qs):
        return StepBallGuide(self._num_timesteps, self.args_vardict(False),
                             self._hyperparams_trainable, self.batch_shape,
                             qs[self.name])

    def _forward(self, theta, t, data={}):
        params = self.args_vardict()['velocities']

        self.sample(Normal, params['loc'][:, t],
                    softplus(params['scale'][:, t]), name='velocity_%d' % t)
        return theta
