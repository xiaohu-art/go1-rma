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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, class_to_dict, task_registry
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import StudentRunner
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(    name=args.task, 
                                        args=args, 
                                        env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(  env=env, 
                                                            env_cfg=env_cfg,
                                                            name=args.task,
                                                            args=args,
                                                            train_cfg=train_cfg)
    teacher = ppo_runner.alg.actor_critic.to(env.device)

    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict["encoder"]["is_teacher"] = False
    train_cfg_dict["encoder"]["mlp_input_dim"] = env_cfg_dict["env"]["num_base_obs"] * env_cfg_dict["env"]["num_history"]
    train_cfg_dict["encoder"]["mlp_output_dim"] = env_cfg_dict["env"]["num_latent"]
    train_cfg_dict["encoder"]["mlp_hidden_dims"] = [1024, 512, 256, 128]

    student: ActorCritic = ActorCritic( env_cfg_dict,
                                        env_cfg_dict["env"]["num_actor_obs"],
                                        env_cfg_dict["env"]["num_privileged_obs"],
                                        env_cfg_dict["env"]["num_actions"],
                                        **train_cfg_dict["policy"],
                                        **train_cfg_dict["encoder"]).to(env.device)
    
    train_cfg.runner.experiment_name = "rma-student"
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    student_runner = StudentRunner( env=env,
                                    env_cfg=env_cfg_dict,
                                    train_cfg=train_cfg_dict,
                                    teacher=teacher,
                                    student=student,
                                    log_dir=log_dir,
                                    device=env.device)

    student_runner.learn(   num_learning_iterations=train_cfg.runner.max_iterations, 
                            init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
