from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

@configclass
class Go1ArmRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
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

    init_noise_std : float = MISSING

@configclass
class Go1ArmRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dagger_update_freq : int = MISSING
    """The frequency of dagger update."""

    priv_reg_coef_schedual: list = MISSING
    """The schedule of the privileged regularization coefficient."""

    mixing_schedule : list = MISSING
    """The schedule of the mixing coefficient."""

    eps : float = MISSING
    """The epsilon value for numerical stability."""


@configclass
class Go1ArmRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: Go1ArmRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: Go1ArmRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""


@configclass
class Go1ArmFlatPPORunnerCfg(Go1ArmRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 1000
    experiment_name = "widowgo1_flat"
    empirical_normalization = False
    policy = Go1ArmRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        activation_out="elu",
        leg_control_head_hidden_dims = [256, 128],
        arm_control_head_hidden_dims = [256, 128],
        critic_leg_control_head_hidden_dims = [256, 128, 64],
        critic_arm_control_head_hidden_dims = [256, 128, 64],
        priv_encoder_dims = [32, 18],
        num_leg_actions = 12,
        num_arm_actions = 6,
    )
    
    algorithm = Go1ArmRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 1500, 5000],
        mixing_schedule=[1.0, 0, 4000] ,
        eps = 1e-5,
    )
 
@configclass
class Go1ArmRoughPPORunnerCfg(Go1ArmRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "widowgo1_rough"
    empirical_normalization = False
    policy = Go1ArmRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        activation_out="elu",
        leg_control_head_hidden_dims = [256, 128],
        arm_control_head_hidden_dims = [256, 128],
        critic_leg_control_head_hidden_dims = [256, 128, 64],
        critic_arm_control_head_hidden_dims = [256, 128, 64],
        priv_encoder_dims = [32, 18],
        num_leg_actions = 12,
        num_arm_actions = 6,
    )

    algorithm = Go1ArmRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 1500, 4000],
        mixing_schedule=[1.0, 0, 3000] ,
        eps = 1e-5,
    )
