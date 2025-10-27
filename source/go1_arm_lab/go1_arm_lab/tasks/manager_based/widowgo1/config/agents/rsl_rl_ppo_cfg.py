# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from .rsl_cfg import LocoManiRslRlPpoActorCriticCfg, LocoManiRslRlPpoAlgorithmCfg

@configclass
class Widowgo1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "widowgo1_flat"
    empirical_normalization = False
    policy = LocoManiRslRlPpoActorCriticCfg(
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

    algorithm = LocoManiRslRlPpoAlgorithmCfg(
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