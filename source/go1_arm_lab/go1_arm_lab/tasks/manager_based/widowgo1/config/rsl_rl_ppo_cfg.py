# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

@configclass
class Widowgo1RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    """---find in 'rsl_rl_ppo_cfg.py'---"""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
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

@configclass
class Widowgo1RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    dagger_update_freq : int = MISSING
    """The frequency of dagger update."""

    priv_reg_coef_schedual: list = MISSING
    """The schedule of the privileged regularization coefficient."""

    mixing_schedule : list = MISSING
    """The schedule of the mixing coefficient."""

    eps : float = MISSING
    """The epsilon value for numerical stability."""

@configclass
class Widowgo1RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: Widowgo1RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: Widowgo1RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class Widowgo1FlatPPORunnerCfg(Widowgo1RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "widowgo1_flat"
    empirical_normalization = False
    policy = Widowgo1RslRlPpoActorCriticCfg(
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
        num_priv = 1 + 1  + 18 + 3 + 4,
        num_hist = 10,
        num_prop =
                  + 3  # base_ang_vel
                  + 18 # joint_pos
                  + 18 # joint_vel
                  + 18 # actions
                  + 3  # velocity_commands
                  + 7  # pose_command
                  + 3, # projected_gravity
        init_std = [[1.0, 1.0, 1.0] * 4 + [1.0] * 6],
    )

    algorithm = Widowgo1RslRlPpoAlgorithmCfg(
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



@configclass
class Widowgo1RoughPPORunnerCfg(Widowgo1RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "unitree_Go2arm_rough"
    empirical_normalization = False
    policy = Widowgo1RslRlPpoActorCriticCfg(
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
        num_priv = 1 + 1  + 18 + 3 + 4,
        num_hist = 10,
        num_prop =
                  + 3  # base_ang_vel
                  + 18 # joint_pos
                  + 18 # joint_vel
                  + 18 # actions
                  + 3  # velocity_commands
                  + 7  # pose_command
                  + 3, # projected_gravity
        init_std = [[1.0, 1.0, 1.0] * 4 + [1.0] * 6],
    )

    algorithm = Widowgo1RslRlPpoAlgorithmCfg(
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