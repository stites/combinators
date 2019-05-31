#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import Bernoulli, MultivariateNormal, Normal
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from torch.distributions.transforms import LowerCholeskyTransform
import torch.nn as nn
from torch.nn.functional import softplus

import combinators.model as model

class NormalInterval(nn.Module):
    def __init__(self, loc, scale, num_scales):
        super(NormalInterval, self).__init__()
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)
        self.num_scales = num_scales

    def forward(self, observation):
        p = Normal(self.loc, self.scale).cdf(observation)
        p = torch.where(p > 0.5, 1. - p, p)
        return 2 * p

class GenerativeStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        goal = kwargs.pop('goal')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'state_uncertainty': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
            }
            if self._discrete_actions:
                kwargs['params']['control'] = {
                    'probs': torch.ones(self._action_dim)
                }
            else:
                kwargs['params']['control'] = {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
        super(GenerativeStep, self).__init__(*args, **kwargs)
        self.goal = goal
        self.state_transition = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim),
        )
        self.predict_observation = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._observation_dim),
        )

    def _forward(self, theta, t, env=None):
        if theta is None:
            state = self.param_sample(Normal, 'state_0')
            control = self.param_sample(Normal, 'control')
        else:
            prev_state, prev_control = theta
            state_uncertainty = self.param_sample(Normal,
                                                  name='state_uncertainty')

            if self._discrete_actions:
                control = self.param_sample(OneHotCategorical, name='control')
            else:
                control = prev_control + self.param_sample(Normal,
                                                           name='control')

            state = self.state_transition(torch.cat((prev_state, control),
                                                    dim=-1))
            state = state + state_uncertainty

        if isinstance(control, torch.Tensor):
            action = torch.tanh(control[0]).cpu().detach().numpy()
        else:
            action = control
        observation, _, done, _ = env.retrieve_step(t, action,
                                                    override_done=True)
        if observation is not None:
            observation = torch.Tensor(observation).to(state).expand(
                self.batch_shape + observation.shape
            )

        prediction = self.predict_observation(state)
        observation_noise = self.param_sample(Normal,
                                              name='observation_noise')
        observation_scale = LowerCholeskyTransform()(observation_noise)
        if observation is not None:
            self.observe('observation', observation, MultivariateNormal,
                         prediction, scale_tril=observation_scale)
        if not done:
            goal_prob = self.goal(prediction, torch.diagonal(observation_scale,
                                                             dim1=-2, dim2=-1))
            self.observe('goal', torch.ones(self.batch_shape).to(prediction),
                         Bernoulli, probs=goal_prob)

        return state, control

class RecognitionStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        self._name = kwargs.pop('name')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
                'control': {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
            }
        super(RecognitionStep, self).__init__(*args, **kwargs)
        if self._discrete_actions:
            self.decode_policy = nn.Sequential(
                nn.Linear(self._state_dim + self._action_dim,
                          self._state_dim * 4),
                nn.PReLU(),
                nn.Linear(self._state_dim * 4, self._action_dim),
                nn.Softmax(dim=-1),
            )
        else:
            self.decode_policy = nn.Sequential(
                nn.Linear(self._state_dim + self._action_dim,
                          self._state_dim * 4),
                nn.PReLU(),
                nn.Linear(self._state_dim * 4, self._action_dim * 16),
                nn.PReLU(),
                nn.Linear(self._action_dim * 16, self._action_dim * 2),
                nn.Tanh(),
            )
        self.encode_uncertainty = nn.Sequential(
            nn.Linear(self._observation_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim * 2),
        )

    @property
    def name(self):
        return self._name

    def _forward(self, theta, t, env=None):
        if theta is None:
            self.param_sample(Normal, 'state_0')
            control = self.param_sample(Normal, 'control')
        else:
            prev_state, prev_control = theta
            self.param_sample(Normal, name='observation_noise')

            control = self.decode_policy(
                torch.cat((prev_state, prev_control), dim=-1)
            )
            if self._discrete_actions:
                control = self.sample(OneHotCategorical, probs=control,
                                      name='control')
            else:
                control = control.reshape(-1, self._action_dim, 2)
                control = prev_control + self.sample(Normal, control[:, :, 0],
                                                     softplus(control[:, :, 1]),
                                                     name='control')

        if isinstance(control, torch.Tensor):
            action = torch.tanh(control[0]).cpu().detach().numpy()
        else:
            action = control
        observation, _, _, _ = env.retrieve_step(t, action, override_done=True)
        if observation is not None and theta is not None:
            observation = torch.Tensor(observation).to(control).expand(
                self.batch_shape + observation.shape
            )
            state_uncertainty = self.encode_uncertainty(observation).reshape(
                -1, self._state_dim, 2
            )
            self.sample(Normal, state_uncertainty[:, :, 0],
                        softplus(state_uncertainty[:, :, 1]),
                        name='state_uncertainty')

class MountainCarCredibleInterval(NormalCredibleInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0.5]).expand(*batch_shape, 1)
        scale = torch.tensor([0.05]).expand(*batch_shape, 1)
        super(MountainCarCredibleInterval, self).__init__(loc, scale, 1)

    def forward(self, loc, scale):
        return super(MountainCarCredibleInterval, self).forward(loc[:, 0],
                                                                scale[:, 0])

class MountainCarStep(GenerativeStep):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['action_dim'] = 1
        kwargs['observation_dim'] = 2
        kwargs['goal'] = MountainCarCredibleInterval(kwargs['batch_shape'])
        super(MountainCarStep, self).__init__(*args, **kwargs)

class CartpoleCredibleInterval(NormalCredibleInterval):
    def __init__(self, batch_shape):
        loc = torch.zeros(*batch_shape, 1)
        scale = torch.tensor([np.pi / (15 * 2)]).expand(*batch_shape, 1)
        super(CartpoleCredibleInterval, self).__init__(loc, scale, 1)

    def forward(self, loc, scale):
        return super(CartpoleCredibleInterval, self).forward(loc[:, 2],
                                                             scale[:, 2])

class CartpoleStep(GenerativeStep):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = True
        kwargs['observation_dim'] = 4
        kwargs['goal'] = CartpoleCredibleInterval(kwargs['batch_shape'])
        super(CartpoleStep, self).__init__(*args, **kwargs)

class BipedalWalkerCredibleInterval(NormalCredibleInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0, 1]).expand(*batch_shape, 2)
        scale = torch.ones(*batch_shape, 2) * 0.0025
        super(BipedalWalkerCredibleInterval, self).__init__(loc, scale, 1)

    def forward(self, loc, scale):
        return super(BipedalWalkerCredibleInterval, self).forward(loc[:, 1:3],
                                                                  scale[:, 1:3])

class BipedalWalkerStep(GenerativeStep):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['observation_dim'] = 24
        kwargs['action_dim'] = 4
        kwargs['goal'] = BipedalWalkerCredibleInterval(kwargs['batch_shape'])
        super(BipedalWalkerStep, self).__init__(*args, **kwargs)
