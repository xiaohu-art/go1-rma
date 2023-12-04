
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation


class ActorEstimator(nn.Module):
    def __init__(self, num_obs, 
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation='elu'):
        super(ActorEstimator, self).__init__()
        
        activation = get_activation(activation)
        
        actor_obs_dim = num_obs + 3
        estimator_obs_dim = num_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(actor_obs_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        estimator_layers = []
        estimator_layers.append(nn.Linear(estimator_obs_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], 3))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        self.estimator = nn.Sequential(*estimator_layers)
    def forward(self, observation):
        with torch.no_grad():
            est_vel = self.estimator(observation)
        action_mean = self.actor(torch.cat((est_vel, observation), dim=1))
        return action_mean
    def estimate(self, observation):
        return self.estimator(observation)
