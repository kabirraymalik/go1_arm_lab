# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
from go1_arm_lab.tasks.manager_based.go1_arm_lab.config.agents.rsl_rl_ppo_cfg import Go1ArmFlatPPORunnerCfg, Go1ArmRoughPPORunnerCfg
from dataclasses import MISSING
from isaaclab.utils.math import subtract_frame_transforms
from ..util.torch_utils import euler_from_quat_wxyz, quat_apply_wxyz, sphere2cart, quat_mul_wxyz

if TYPE_CHECKING:
    from go1_arm_lab.env.manager_env import ManagerBasedRLEnv
    from cfg.command_cfg import UniformPoseCommandCfg
    from cfg.command_cfg import HemispherePoseCommandCfg

class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        if getattr(self.cfg, "debug_vis", False):
            self.set_debug_vis(True)
        self.curr_ee_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.env = env
        if self.cfg.is_Go1Arm_Flat == True:
            cfg_runner = Go1ArmFlatPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env
        else:
            cfg_runner = Go1ArmRoughPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env



    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        if self.cfg.is_Go1Arm or self.cfg.is_Go1Arm_Play == True:
            self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0] 

        # compute the error    
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        # print("_update_metrics")
        # print("pose_command",self.pose_command_w)
        # print("pose",self.robot.data.body_state_w[:, self.body_idx, :3])

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # print("_resample_command")
        # -- position


        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])        
        r = torch.empty(len(env_ids), device=self.device)
        r_1 = torch.empty(1, device=self.device)

        if self.cfg.is_Go1Arm == True:

            count = torch.tensor(self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff)

            self.pose_command_b[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1)
            self.pose_command_b[env_ids, 1] = (r.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
            self.pose_command_w_z[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)
            self.pose_command_b[env_ids, 2] = self.pose_command_w_z[env_ids, 0]  - self.robot.data.root_pos_w[env_ids, 2] 
                                              
            for _, i in enumerate(env_ids):
                length_arm = torch.norm(torch.stack([self.pose_command_b[i, 0], 
                                                     self.pose_command_b[i, 1], 
                                                     self.pose_command_b[i, 2] 
                                                    ])) 
                while((length_arm > 0.7) or (length_arm < 0.3) or (self.pose_command_b[i, 0] < 0.45 and torch.abs(self.pose_command_b[i, 1]) < 0.2)):
                        self.pose_command_b[i, 0] = (r_1.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1) 
                        self.pose_command_b[i, 1] = (r_1.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
                        self.pose_command_w_z[i, 0] = (r_1.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)
                        self.pose_command_b[i, 2] = self.pose_command_w_z[i, 0] - self.robot.data.root_pos_w[i, 2]                     
                        length_arm = torch.norm(torch.stack([self.pose_command_b[i, 0], 
                                                        self.pose_command_b[i, 1], 
                                                         self.pose_command_b[i, 2]
                                                        ])) 
            
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges_init.roll) * torch.clamp((1 - count), 0, 1) + \
                                 r.uniform_(*self.cfg.ranges_final.roll) * torch.clamp((count), 0, 1)
                                 
            delta_x = self.pose_command_b[env_ids, 0] 
            delta_y = self.pose_command_b[env_ids, 1] 
            delta_z = self.pose_command_b[env_ids, 2]

            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + \
                                    r.uniform_(*self.cfg.ranges.pitch) * torch.clamp((1 - count), 0, 1) + \
                                    r.uniform_(*self.cfg.ranges_final.pitch) * torch.clamp((count), 0, 1)   
                                    
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + \
                                r.uniform_(*self.cfg.ranges_init.yaw) * torch.clamp((1 - count), 0, 1) + \
                                r.uniform_(*self.cfg.ranges_final.yaw) * torch.clamp((count), 0, 1) 
            # roll useless 
          
        elif self.cfg.is_Go1Arm_Play == True:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_w_z[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_z)
            self.pose_command_b[env_ids, 2] =  self.pose_command_w_z[env_ids, 0] - self.robot.data.root_pos_w[env_ids, 2] 

            delta_x = self.pose_command_b[env_ids, 0] 
            delta_y = self.pose_command_b[env_ids, 1] 
            delta_z = self.pose_command_b[env_ids, 2]         
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + r.uniform_(*self.cfg.ranges.yaw)
        else:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)    
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat     


    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

class HemispherePoseCommand(CommandTerm):
    """
    Command generator for generating 6D pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space, and orientations within specified ranges.
    """
    cfg:HemispherePoseCommandCfg
    
    """Configuration of the Command Generator"""

    def __init__(self, cfg: "HemispherePoseCommandCfg", env: "ManagerBasedRLEnv"):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """

        # initialize the base class
        super().__init__(cfg, env)
        
        # store env as local var
        self.env = env

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create command buffers: (x, y, z, qw, qx, qy, qz) in base frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # spherical center offsets (per env)
        self.arm_base_offset = torch.tensor([
            cfg.sphere_center.x_offset, 
            cfg.sphere_center.y_offset, 
            cfg.sphere_center.z_invariant_offset
            ], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)

        # metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        # for z invariance wrt body link height
        self._goal_world_z = self._goal_world_z = torch.zeros(self.num_envs, device=self.device)

        # get curriculum cfg information
        self._num_steps_default = int(getattr(self.cfg, "num_steps_per_env", 24))

    def __str__(self) -> str:
        msg = "HemispherePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    # optional curriculum helpers
    def _progress_scalar(self) -> torch.Tensor:
        coeff = float(getattr(self.cfg, "curriculum_coeff", 0.0) or 0.0)
        if coeff <= 0.0:
            return torch.tensor(0.0, device=self.device)
        steps_total = float(getattr(self.env, "common_step_counter", 0))
        steps_per_env = float(getattr(self.cfg, "num_steps_per_env", self._num_steps_default))
        s = steps_total / (steps_per_env * coeff) if steps_per_env > 0 else 0.0
        return torch.clamp(torch.tensor(s, device=self.device), 0.0, 1.0)

    def _sample_blended_uniform(self, r: torch.Tensor, name: str) -> torch.Tensor:
        base = getattr(self.cfg.ranges, name)
        rin = getattr(getattr(self.cfg, "ranges_init", None) or object(), name, None)
        rfi = getattr(getattr(self.cfg, "ranges_final", None) or object(), name, None)
        coeff = float(getattr(self.cfg, "curriculum_coeff", 0.0) or 0.0)

        def _u(lohi):
            lo, hi = lohi
            return r.uniform_(lo, hi)

        if coeff <= 0.0 or (rin is None) or (rfi is None):
            return _u(base)

        s = self._progress_scalar()
        v_init = _u(rin)
        v_final = _u(rfi)
        return (1.0 - s) * v_init + s * v_final

    # core helpers
    def _update_metrics(self):
        # transform command from base frame to simulation world frame (root pose)
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # set z at constant env height, invariant to base link z motion
        self.pose_command_w[:, 2] = self._goal_world_z
        # compute the error using body_pos_w/body_quat_w (matches reference)
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)

        # spherical position (l, pitch, yaw) with optional curriculum
        sphere_coords = torch.zeros(len(env_ids), 3, device=self.device)
        sphere_coords[:, 0] = self._sample_blended_uniform(r, "pos_l")
        sphere_coords[:, 1] = self._sample_blended_uniform(r, "pos_p")
        sphere_coords[:, 2] = self._sample_blended_uniform(r, "pos_y")

        cart_coords = sphere2cart(sphere_coords)

        base_yaw_quat = self._get_base_yaw_quat()[env_ids]
        sphere_center_world = self._get_ee_goal_spherical_center()[env_ids]
        cart_coords_world_rotated = quat_apply_wxyz(base_yaw_quat, cart_coords)
        cart_coords_world = sphere_center_world + cart_coords_world_rotated
        
        # set env z to be invariant to current body link z
        self._goal_world_z[env_ids] = cart_coords_world[:, 2]

        # world -> base (position)
        robot_base_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_base_quat_w = self.robot.data.root_quat_w[env_ids]
        self.pose_command_b[env_ids, :3], _ = subtract_frame_transforms(
            robot_base_pos_w, 
            robot_base_quat_w, 
            cart_coords_world, 
            torch.zeros(len(env_ids), 4, device=self.device)
        )

        # spherical orientation (alpha, beta, gamma) with optional curriculum
        euler_sphere = torch.zeros(len(env_ids), 3, device=self.device)
        euler_sphere[:, 0] = self._sample_blended_uniform(r, "orn_alpha")
        euler_sphere[:, 1] = self._sample_blended_uniform(r, "orn_beta")
        euler_sphere[:, 2] = self._sample_blended_uniform(r, "orn_gamma")

        quat_sphere = quat_from_euler_xyz(euler_sphere[:, 0], euler_sphere[:, 1], euler_sphere[:, 2])
        quat_world = quat_mul_wxyz(base_yaw_quat, quat_sphere)

        # world -> base (orientation)
        _, self.pose_command_b[env_ids, 3:] = subtract_frame_transforms(
            robot_base_pos_w, robot_base_quat_w, torch.zeros_like(robot_base_pos_w), quat_world
        )

        if self.cfg.make_quat_unique:
            self.pose_command_b[env_ids, 3:] = quat_unique(self.pose_command_b[env_ids, 3:])

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # goal pose (world): use already-updated pose_command_w for consistency with metrics
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # current body pose (world)
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

    def _update_command(self):
        pass

    def _get_ee_goal_spherical_center(self):
        center = torch.cat(
            [self.robot.data.root_pos_w[:, :2], torch.full((self.num_envs, 1), self.arm_base_offset[0, 2], device=self.device)],
            dim=1,
        )
        base_yaw_quat = self._get_base_yaw_quat()
        xy_offset = torch.cat([self.arm_base_offset[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        return center + quat_apply_wxyz(base_yaw_quat, xy_offset)

    def _get_base_yaw_quat(self):
        _, _, yaw = euler_from_quat_wxyz(self.robot.data.root_quat_w)
        return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
