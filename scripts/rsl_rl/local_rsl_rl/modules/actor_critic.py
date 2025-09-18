# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from local_rsl_rl.utils import resolve_nn_activation


# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))

        output = self.linear_output(output)
        return output

class  ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,  
        num_actor_obs,
        num_critic_obs,
        num_priv,
        num_actions,
        num_hist = 10,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        priv_encoder_dims=[64, 18],
        activation='elu',
        activation_out='tanh',
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        # self.dual_heads = kwargs['dual_heads']
        leg_control_head_hidden_dims = kwargs['leg_control_head_hidden_dims']
        arm_control_head_hidden_dims = kwargs['arm_control_head_hidden_dims']
        critic_leg_control_head_hidden_dims = kwargs['critic_leg_control_head_hidden_dims']
        critic_arm_control_head_hidden_dims = kwargs['critic_arm_control_head_hidden_dims']
        self.num_leg_actions = kwargs['num_leg_actions']
        self.num_arm_actions = kwargs['num_arm_actions']
        num_prop = num_actor_obs
        print("num_hist",num_hist)
        print("num_priv",num_priv)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        super(ActorCritic, self).__init__()
        activation = resolve_nn_activation(activation)
        activation_out = resolve_nn_activation(activation_out)
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        class Actor(nn.Module):
            def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation,activation_out, 
                leg_control_head_hidden_dims, arm_control_head_hidden_dims, 
                num_leg_actions, num_arm_actions, num_priv, num_hist, num_prop, priv_encoder_dims):
                super().__init__()
                self.num_arm_actions = num_arm_actions

                # Policy
                if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):

                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
                else:
                    self.priv_encoder = nn.Identity()
                    priv_encoder_output_dim = num_priv

                self.num_priv = num_priv
                self.num_hist = num_hist
                self.num_prop = num_prop

                self.history_encoder = StateHistoryEncoder(activation, mlp_input_dim_a, num_hist, priv_encoder_output_dim)

                # Policy
                if len(actor_hidden_dims) > 0:
                    actor_layers = []
                    actor_layers.append(nn.Linear(mlp_input_dim_a + priv_encoder_output_dim, actor_hidden_dims[0]))
                    actor_layers.append(activation)
                    for l in range(len(actor_hidden_dims) - 1):
                        # if l == len(actor_hidden_dims) - 1:
                        #     actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                        #     # actor_layers.append(nn.Tanh())
                        # else:
                        actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                        actor_layers.append(activation)
                    self.actor_backbone = nn.Sequential(*actor_layers)
                    actor_backbone_output_dim = actor_hidden_dims[-1]
                else:
                    self.actor_backbone = nn.Identity()
                    actor_backbone_output_dim = mlp_input_dim_a + priv_encoder_output_dim

                actor_leg_layers = []
                actor_leg_layers.append(nn.Linear(actor_backbone_output_dim, leg_control_head_hidden_dims[0]))
                actor_leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], num_leg_actions))
                        actor_leg_layers.append(activation_out)
                    else:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                        actor_leg_layers.append(activation)
                self.actor_leg_control_head = nn.Sequential(*actor_leg_layers)

                actor_arm_layers = []
                actor_arm_layers.append(nn.Linear(actor_backbone_output_dim, arm_control_head_hidden_dims[0]))
                actor_arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], num_arm_actions))
                        actor_arm_layers.append(activation_out)
                    else:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                        actor_arm_layers.append(activation)
                self.actor_arm_control_head = nn.Sequential(*actor_arm_layers)
            
            def forward(self, obs, hist_encoding: bool = True):
                obs_prop = obs[:, :self.num_prop]
                if hist_encoding:
                    latent = self.infer_hist_latent(obs)
                else:
                    latent = self.infer_priv_latent(obs)
                backbone_input = torch.cat([obs_prop, latent], dim=1)
                backbone_output = self.actor_backbone(backbone_input)
                leg_output = self.actor_leg_control_head(backbone_output)
                arm_output = self.actor_arm_control_head(backbone_output)

                return torch.cat([leg_output, arm_output], dim=-1)
            
            def infer_priv_latent(self, obs):
                priv = obs[:, (self.num_prop * self.num_hist): ]

                return self.priv_encoder(priv)
            
            def infer_hist_latent(self, obs):
                hist = obs[:, :(self.num_hist * self.num_prop)]
                hist = hist.view(-1, self.num_hist, self.num_prop)
                hist = hist[:, :, :self.num_prop]
                return self.history_encoder(hist)
        
        # Value function
        class Critic(nn.Module):
            def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation,
                         critic_leg_control_head_hidden_dims, critic_arm_control_head_hidden_dims,
                         num_priv, num_hist, num_prop):
                super().__init__()

                self.num_priv = num_priv
                self.num_hist = num_hist
                self.num_prop = num_prop

                # Value
                if len(critic_hidden_dims) > 0:
                    critic_layers = []
                    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                    critic_layers.append(activation)
                    for l in range(len(critic_hidden_dims) - 1):
                        # if l == len(critic_hidden_dims) - 1:
                        #     critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                        # else:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                        critic_layers.append(activation)
                    self.critic_backbone = nn.Sequential(*critic_layers)
                    critic_backbone_output_dim = critic_hidden_dims[-1]
                else:
                    self.critic_backbone = nn.Identity()
                    critic_backbone_output_dim = mlp_input_dim_c

                critic_leg_layers = []
                critic_leg_layers.append(nn.Linear(critic_backbone_output_dim, critic_leg_control_head_hidden_dims[0]))
                critic_leg_layers.append(activation)
                for l in range(len(critic_leg_control_head_hidden_dims)):
                    if l == len(critic_leg_control_head_hidden_dims) - 1:
                        critic_leg_layers.append(nn.Linear(critic_leg_control_head_hidden_dims[l], 1))
                    else:   
                        critic_leg_layers.append(nn.Linear(critic_leg_control_head_hidden_dims[l], critic_leg_control_head_hidden_dims[l + 1]))
                        critic_leg_layers.append(activation)
                self.critic_leg_control_head = nn.Sequential(*critic_leg_layers)

                critic_arm_layers = []
                critic_arm_layers.append(nn.Linear(critic_backbone_output_dim, critic_arm_control_head_hidden_dims[0]))
                critic_arm_layers.append(activation)
                for l in range(len(critic_arm_control_head_hidden_dims)):
                    if l == len(critic_arm_control_head_hidden_dims) - 1:
                        critic_arm_layers.append(nn.Linear(critic_arm_control_head_hidden_dims[l], 1))
                    else:
                        critic_arm_layers.append(nn.Linear(critic_arm_control_head_hidden_dims[l], critic_arm_control_head_hidden_dims[l + 1]))
                        critic_arm_layers.append(activation)
                self.critic_arm_control_head = nn.Sequential(*critic_arm_layers)
            
            def forward(self, obs):
                prop_obs = obs[:, :self.num_prop]
                priv_obs = obs[:, -self.num_priv:]
                prop_and_priv = torch.cat([prop_obs, priv_obs], dim=-1)
                backbone_output = self.critic_backbone(prop_and_priv)
                leg_output = self.critic_leg_control_head(backbone_output)
                arm_output = self.critic_arm_control_head(backbone_output)
                return torch.cat([leg_output, arm_output], dim=-1)

        self.actor = Actor(mlp_input_dim_a, actor_hidden_dims, activation,activation_out, leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
            self.num_leg_actions, self.num_arm_actions, 
            num_priv, num_hist, num_prop, priv_encoder_dims)

        self.critic = Critic(mlp_input_dim_c + num_priv, critic_hidden_dims, activation, critic_leg_control_head_hidden_dims, critic_arm_control_head_hidden_dims, 
                             num_priv, num_hist, num_prop)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            print(self.std.shape)
            self.std = nn.Parameter((init_noise_std * torch.ones(num_actions)).unsqueeze(0)) #TODO
            print("new",self.std.shape)

        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        entropy = self.distribution.entropy()
        leg_entropy_sum = entropy[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_entropy_sum = entropy[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_entropy_sum, arm_entropy_sum], dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def act(self, observations, hist_encoding, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        leg_log_prob_sum = log_prob[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_log_prob_sum = log_prob[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_log_prob_sum, arm_log_prob_sum], dim=-1)

    def act_inference(self, observations, hist_encoding=True):
        actions_mean = self.actor(observations, hist_encoding)
        return actions_mean 

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
