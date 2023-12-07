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
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, class_to_dict, task_registry, Logger
from rsl_rl.modules import ActorCritic

import numpy as np
import torch

import cv2
from tqdm import tqdm


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.terrain_proportions = [0., 0.1, 0.35, 0.35, 0.2]
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torques = False
    train_cfg.seed = 101

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
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
    
    loaded_dict = torch.load("/home/gymuser/go1-legged_gym/logs/student/model_2750.pt")
    student.load_state_dict(loaded_dict["model_state_dict"])

    obs_history = torch.zeros(  (env_cfg.env.num_envs, env_cfg.env.num_history * env_cfg.env.num_base_obs), 
                                dtype=torch.float32,
                                device=env.device)
    obs_history = torch.cat((obs_history[:, 45:], obs[:, :45]), dim=-1)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(student, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    video_width = 1920
    video_height = 1080

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = video_width
    camera_properties.height = video_height
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(1, -1, 0.5)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                  np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_POSITION)

    img_idx = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    from datetime import datetime
    video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
    experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
    dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'.mp4')
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    video = cv2.VideoWriter(dir, fourcc, 50.0, (video_width, video_height))

    for i in tqdm(range(2*int(env.max_episode_length))):
        # print(obs)
        actions, _ = student.act_student(obs_history.detach())
        if FIX_COMMAND:
            env.commands[:, 0] = 1.
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            name = str(img_idx).zfill(4)
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', name + ".png")
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.write_camera_image_to_file(env.sim, env.envs[0], h1,gymapi.IMAGE_COLOR, filename)
            print(filename)
            img_idx += 1 
        env_ids = dones.nonzero(as_tuple=False).flatten()
        obs_history[env_ids, :] = 0.
        obs_history = torch.cat((obs_history[:, 45:], obs[:, :45]), dim=-1)

        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1,gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img[..., :3])

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    
    video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
