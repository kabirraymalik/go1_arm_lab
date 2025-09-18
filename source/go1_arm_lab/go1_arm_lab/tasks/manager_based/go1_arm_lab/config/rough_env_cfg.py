from isaaclab.utils import configclass

from go1_arm_lab.tasks.manager_based.go1_arm_lab.go1_arm_lab_env_cfg import LocomotionVelocityEnvCfg
from go1_arm_lab.assets.widowgo1 import WIDOW_GO1_CFG



@configclass
class Go1ArmRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = WIDOW_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # event
        self.events.push_robot = None

        # command
        self.commands.ee_pose.is_Go1Arm_Flat = False #TODO
        # velocity command
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        # position command 
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.35, 0.4)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_final.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_final.pos_z = (0.35, 0.4)


        # reward weight
        # arm
        self.rewards.end_effector_position_tracking.weight = 2.5
        self.rewards.end_effector_orientation_tracking.weight = -1.5
        self.rewards.end_effector_action_rate.weight = -0.005
        self.rewards.end_effector_action_smoothness.weight = -0.02
        # leg
        self.rewards.tracking_lin_vel_x_l1.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.foot_contact.weight = 0.003
        self.rewards.hip_deviation.weight = -0.4
        self.rewards.joint_deviation.weight = -0.04
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.height_reward.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -1.0


@configclass
class Go1ArmRoughEnvCfg_PLAY(Go1ArmRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # command
        self.commands.ee_pose.is_Go1Arm_Flat = False #TODO

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