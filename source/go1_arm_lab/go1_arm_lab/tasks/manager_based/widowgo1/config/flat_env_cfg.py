
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass


from ..manager_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##

from go1_arm_lab.assets.widowgo1 import WIDOW_GO1_CFG

@configclass
class widowgo1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = WIDOW_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
class widowgo1FlatEnvCfg_PLAY(widowgo1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.reset_base = None
        self.commands.ee_pose.enable_curriculum = False
        self.commands.base_velocity.enable_curriculum = False
        self.commands.ee_pose.is_play = True
        
        # self.commands.base_velocity.resampling_time_range = (5.0,10.5)
        # self.commands.base_velocity.rel_standing_envs = 0.2
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
       
        # self.commands.ee_pose.resampling_time_range = (2.0,5.0)
        # self.commands.ee_pose.ranges.pos_x = (0.45, 0.45)
        # self.commands.ee_pose.ranges.pos_y = (0.0, 0.0)
        # self.commands.ee_pose.ranges.pos_z = (0.3, 0.3)