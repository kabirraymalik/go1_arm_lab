# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

@configclass
class LocoManiRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    activation_out: str = MISSING
    """The activation function for the output layer of the actor networks."""
    
    leg_control_head_hidden_dims : list[int] = MISSING
    """The hidden dimensions of the leg control head network."""

    arm_control_head_hidden_dims : list[int] = MISSING
    """The hidden dimensions of the arm control head network."""

    critic_leg_control_head_hidden_dims : list[int] = MISSING
    """The hidden dimensions of the critic leg control head network."""

    critic_arm_control_head_hidden_dims : list[int] = MISSING
    """The hidden dimensions of the critic arm control head network."""

    priv_encoder_dims: list[int] = MISSING
    """The dimensions of the privileged encoder network."""    

    num_leg_actions : int = MISSING
    """The number of leg actions."""

    num_arm_actions : int = MISSING
    """The number of arm actions."""

    num_priv : int = MISSING
    """The number of privileged observations."""

    num_hist : int = MISSING
    """The number of historical observations."""

    num_prop : int = MISSING
    """The number of proprioceptive observations."""

    init_std : list = MISSING
    """The initial noise standard deviation for the policy."""

@configclass
class LocoManiRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    dagger_update_freq : int = MISSING
    """The frequency of dagger update."""

    priv_reg_coef_schedual: list = MISSING
    """The schedule of the privileged regularization coefficient."""

    mixing_schedule : list = MISSING
    """The schedule of the mixing coefficient."""

    eps : float = MISSING
    """The epsilon value for numerical stability."""