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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import torch

from ..algorithms import PPO
from ..modules import ActorCritic
import numpy as np

import rsl_rl
from rsl_rl.utils import store_code_state
from rsl_rl.modules import EmpiricalNormalization
# import wandb
from torchinfo import summary

class OnPolicyRunner:

    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
    
        self.cfg=train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        print(env)
        actor_critic_class = eval(self.policy_cfg.pop("class_name")) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.policy_cfg["num_prop"],
                                                        self.policy_cfg["num_prop"],
                                                        self.env.num_actions,
                                                        **self.policy_cfg, 
                                                        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name")) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]        

        summary(self.alg.actor_critic)

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[self.env.num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        
        _, _ = self.env.reset()  

        # prepare obs
        self.prepare_obs()


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init metrics   

        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_arm_torques_loss = 0.
        value_mixing_ratio = 0.
        torque_supervision_weight = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        # initialize writer

        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, _ = self.env.get_observations()
        obs = self.change_obs_order(obs)
        privileged_obs = None ##### unnecessary
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        armrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            
            # self.env.update_command_curriculum()

            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    obs, rewards, arm_rewards, dones, infos = self.env.step(actions)
                    rewards = torch.clamp(rewards, min=-5.0)
                    arm_rewards = torch.clamp(arm_rewards, min=-5.0)
                    obs = self.change_obs_order(obs)
                    critic_obs = obs 

                    obs, critic_obs, rewards, arm_rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), arm_rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, arm_rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_arm_reward_sum += arm_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        armrewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

                self.alg.compute_returns(critic_obs)
            
            # self.alg.storage.clear()
            
            # mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
                
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            # wandb-compatible time-based logging
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                    )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)
        
    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)

        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def prepare_obs(self):
        self.total = np.zeros((self.policy_cfg["num_hist"], self.policy_cfg["num_prop"])) 
        self.obs_new = torch.zeros(self.env.num_envs, self.policy_cfg["num_prop"] * self.policy_cfg["num_hist"]).to(self.env.device)

        lst, length = self.env.get_obs_list_length()
        lst = [item for item in lst if not item.startswith("policy-priv_")]
        
        result_dict = {}
        for i in range(len(lst)):
            c = np.array(list(range( sum(length[: i + 1 ]) - int(length[i] / self.policy_cfg["num_hist"]), sum(length[: i + 1]) )))
            result_dict[lst[i]] = c

        key_list = list(result_dict.keys())
        a1_list = []
        for i in range(self.policy_cfg["num_hist"]):
            for j in range(len(lst)):
                a1 = np.concatenate([
                    result_dict[key_list[j]] - (i) * result_dict[key_list[j]].shape[0]])
                a1_list.append(a1)
                if j == len(lst) - 1:
                    a1_list = np.concatenate(a1_list)
                    self.total[i, :] = a1_list
                    a1_list = []

    def change_obs_order(self, obs):
        """
        Modify the order of observations containing historical information for easier network reading. Try to avoid modifications if possible!

        Args:
            obs(num_envs, num_prop * num_hist): The input to the actor and critic network
            obs_new(num_envs, num_prop * num_hist): Modified observation order indices corresponding to the observations
            total(num_hist, num_prop): Indices of the modified observation order

        Returns:
            obs(num_envs, num_prop * num_hist): The input to the actor and critic network

        Notes:
            Try to avoid modifications if possible!

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_hist) + joint_pos(18) * 10(num_hist) + joint_vel(18) * 10(num_hist)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
        """ 

        for i in range(self.policy_cfg["num_hist"]):
            obs_1 = obs[:, self.total[i, :]]
            self.obs_new = torch.cat([self.obs_new, obs_1], dim=-1)
        obs = torch.cat([self.obs_new[:, self.policy_cfg["num_prop"] * self.policy_cfg["num_hist"]:], 
                            obs[:, self.policy_cfg["num_prop"] * self.policy_cfg["num_hist"]:]], dim=-1)                    
        obs = self.obs_normalizer(obs)
        self.obs_new = torch.zeros(self.env.num_envs, self.policy_cfg["num_prop"] * self.policy_cfg["num_hist"]).to(self.env.device)
        return obs
