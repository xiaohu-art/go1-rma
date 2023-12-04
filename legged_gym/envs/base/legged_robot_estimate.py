# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class LeggedRobotEstimate(LeggedRobot):
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.dof_pos_hist_1[env_ids] = 0
        self.dof_pos_hist_2[env_ids] = 0
        self.dof_pos_hist_3[env_ids] = 0
        self.dof_vel_hist_1[env_ids] = 0
        self.dof_vel_hist_2[env_ids] = 0
        self.dof_vel_hist_3[env_ids] = 0
        self.action_hist_1[env_ids] = 0
        self.action_hist_2[env_ids] = 0
        self.action_hist_3[env_ids] = 0

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.continuous_push:
                self._continuous_push()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def compute_observations(self):
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, 
            self.base_ang_vel * self.obs_scales.ang_vel, 
            self.projected_gravity, 
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_1 - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_2 - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_3 - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.dof_vel_hist_1 * self.obs_scales.dof_vel,
            self.dof_vel_hist_2 * self.obs_scales.dof_vel,
            self.dof_vel_hist_3 * self.obs_scales.dof_vel,
            self.actions,
            self.action_hist_1,
            self.action_hist_2,
            self.action_hist_3
        ), dim=-1)
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel, 
            self.projected_gravity, 
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_1 - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_2 - self.default_dof_pos) * self.obs_scales.dof_pos,
            (self.dof_pos_hist_3 - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.dof_vel_hist_1 * self.obs_scales.dof_vel,
            self.dof_vel_hist_2 * self.obs_scales.dof_vel,
            self.dof_vel_hist_3 * self.obs_scales.dof_vel,
            self.actions,
            self.action_hist_1,
            self.action_hist_2,
            self.action_hist_3
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.actor_noise_scale_vec
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.critic_noise_scale_vec
        self.dof_pos_hist_3 = self.dof_pos_hist_2
        self.dof_pos_hist_2 = self.dof_pos_hist_1
        self.dof_pos_hist_1 = self.dof_pos
        self.dof_vel_hist_3 = self.dof_vel_hist_2
        self.dof_vel_hist_2 = self.dof_vel_hist_1
        self.dof_vel_hist_1 = self.dof_vel
        self.action_hist_3 = self.action_hist_2
        self.action_hist_2 = self.action_hist_1
        self.action_hist_1 = self.actions

    def _init_buffers(self):
        super()._init_buffers()
        self.dof_pos_hist_1 = torch.zeros_like(self.dof_pos)
        self.dof_pos_hist_2 = torch.zeros_like(self.dof_pos)
        self.dof_pos_hist_3 = torch.zeros_like(self.dof_pos)
        self.dof_vel_hist_1 = torch.zeros_like(self.dof_vel)
        self.dof_vel_hist_2 = torch.zeros_like(self.dof_vel)
        self.dof_vel_hist_3 = torch.zeros_like(self.dof_vel)
        self.action_hist_1 = torch.zeros_like(self.actions)
        self.action_hist_2 = torch.zeros_like(self.actions)
        self.action_hist_3 = torch.zeros_like(self.actions)
        self.actor_noise_scale_vec = self._get_actor_noise_scale_vec(self.cfg)
        self.critic_noise_scale_vec = self._get_critic_noise_scale_vec(self.cfg)

    def _get_actor_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:int(9+self.cfg.env.num_actions*4)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[int(9+self.cfg.env.num_actions*4):int(9+self.cfg.env.num_actions*8)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[int(9+self.cfg.env.num_actions*8):int(9+self.cfg.env.num_actions*12)] = 0. # previous actions
        return noise_vec

    def _get_critic_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0. # velocity
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:int(12+self.cfg.env.num_actions*4)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[int(12+self.cfg.env.num_actions*4):int(12+self.cfg.env.num_actions*8)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[int(12+self.cfg.env.num_actions*8):int(12+self.cfg.env.num_actions*12)] = 0. # previous actions
        return noise_vec
