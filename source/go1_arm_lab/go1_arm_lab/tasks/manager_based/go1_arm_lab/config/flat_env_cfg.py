# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


from go1_arm_lab.tasks.manager_based.go1_arm_lab.go1_arm_lab_env_cfg import LocomotionVelocityEnvCfg
from go1_arm_lab.assets.widowgo1 import WIDOW_GO1_CFG


@configclass
class Go1ArmFlatEnvCfg(LocomotionVelocityEnvCfg):
    # params for new rewards here
    foot_link_name = ".*_foot"
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = WIDOW_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # event
        self.events.push_robot = None

        # flat terrain 
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # velocity command
        self.commands.base_velocity.curriculum_coeff = 4000
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        # position command 
        self.commands.ee_pose.curriculum_coeff = 5000
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.45, 0.5)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.4, 0.65)
        self.commands.ee_pose.ranges_final.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.ranges_final.pos_z = (0.15, 0.6)

        # reward weight
        # arm
        self.rewards.end_effector_position_tracking.weight = 3.0
        self.rewards.end_effector_orientation_tracking.weight = -2.0
        self.rewards.end_effector_action_rate.weight = -0.01 #-0.005 
        self.rewards.end_effector_action_smoothness.weight = -0.04 #-0.02
        
        # leg
        self.rewards.tracking_lin_vel_x_l1.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        
        self.rewards.feet_air_time.weight = 0.0 #0.5
        self.rewards.F_feet_air_time.weight = 1.0 #0.5
        self.rewards.R_feet_air_time.weight = 1.0 #0.5

        self.rewards.feet_height.weight = -0.0 #TODO
        self.rewards.feet_height_body.weight = -3.0 #TODO
        self.rewards.foot_contact.weight = 0.003 #0.003
        self.rewards.hip_deviation.weight = -0.2
        self.rewards.joint_deviation.weight = -0.01
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.height_reward.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -1.0

        # new rewards added: 
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))


class Go1ArmFlatEnvCfg_PLAY(Go1ArmFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.reset_base = None

        self.commands.ee_pose.is_Go1Arm = False
        self.commands.base_velocity.is_Go1Arm = False
        
        self.commands.ee_pose.is_Go1Arm_Play = True
        
        self.commands.base_velocity.resampling_time_range = (5.0,5.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
       
        self.commands.ee_pose.resampling_time_range = (4.0,4.0)
        self.commands.ee_pose.ranges.pos_x = (0.45, 0.6)
        self.commands.ee_pose.ranges.pos_y = (-0.25, 0.25)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.5)