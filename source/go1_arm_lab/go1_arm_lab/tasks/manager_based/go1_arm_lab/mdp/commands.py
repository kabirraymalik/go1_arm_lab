# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, subtract_frame_transforms
from ..util.torch_utils import euler_from_quat_wxyz, quat_apply_wxyz, sphere2cart, quat_mul_wxyz
from ..util.markers import SPHERE_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# class HemispherePoseCommand(CommandTerm):

#     cfg: HemispherePoseCommandCfg
#     """Configuration for the command generator."""

#     def __init__(self, cfg: HemispherePoseCommandCfg, env: ManagerBasedRLEnv):
#         """Initialize the command generator class.

#         Args:
#             cfg: The configuration parameters for the command generator.
#             env: The environment object.
#         """
#         # initialize the base class
#         super().__init__(cfg, env)
        
#         # extract the robot and body index for which the command is generated
#         self.robot: Articulation = env.scene[cfg.asset_name]

#         self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

#         # create buffers
#         # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
#         self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
#         self.pose_command_b[:, 3] = 1.0
#         self.pose_command_w = torch.zeros_like(self.pose_command_b)
        
#         # configuration constants for spherical coordinate system
#         self.arm_base_offset = torch.tensor([
#             cfg.sphere_center.x_offset, 
#             cfg.sphere_center.y_offset, 
#             cfg.sphere_center.z_invariant_offset
#         ], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
#         # -- metrics
#         self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
#         self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

#     def __str__(self) -> str:
#         msg = "HemispherePoseCommand:\n"
#         msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
#         msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
#         return msg

#     """
#     Properties
#     """

#     @property
#     def command(self) -> torch.Tensor:
#         """The desired pose command. Shape is (num_envs, 7).

#         The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
#         """
#         return self.pose_command_b

#     """
#     Implementation specific functions.
#     """

#     def _update_metrics(self):
#         # transform command from base frame to simulation world frame
#         self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
#             self.robot.data.root_pos_w,
#             self.robot.data.root_quat_w,
#             self.pose_command_b[:, :3],
#             self.pose_command_b[:, 3:],
#         )
#         # compute the error
#         pos_error, rot_error = compute_pose_error(
#             self.pose_command_w[:, :3],
#             self.pose_command_w[:, 3:],
#             self.robot.data.body_pos_w[:, self.body_idx],
#             self.robot.data.body_quat_w[:, self.body_idx],
#         )
#         self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
#         self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

#     def _resample_command(self, env_ids: Sequence[int]):
#         """Sample new pose targets in spherical coordinates and convert to cartesian."""
#         # Create random generator for sampling
#         r = torch.empty(len(env_ids), device=self.device)
        
#         # Sample spherical coordinates (l, pitch, yaw)
#         sphere_coords = torch.zeros(len(env_ids), 3, device=self.device)
#         sphere_coords[:, 0] = r.uniform_(*self.cfg.ranges.pos_l)  # radial distance
#         sphere_coords[:, 1] = r.uniform_(*self.cfg.ranges.pos_p)  # polar angle (pitch)
#         sphere_coords[:, 2] = r.uniform_(*self.cfg.ranges.pos_y)  # azimuthal angle (yaw)
        
#         # Convert spherical coordinates to cartesian coordinates
#         cart_coords = sphere2cart(sphere_coords)
        
#         # Get the spherical coordinate system center in base frame
#         base_yaw_quat = self._get_base_yaw_quat()[env_ids]
#         sphere_center_world = self._get_ee_goal_spherical_center()[env_ids]
        
#         # Transform cartesian coordinates from sphere frame to world frame
#         cart_coords_world_rotated = quat_apply_wxyz(base_yaw_quat, cart_coords)
#         cart_coords_world = sphere_center_world + cart_coords_world_rotated
        
#         # Transform from world frame to robot base frame
#         robot_base_pos_w = self.robot.data.root_pos_w[env_ids]
#         robot_base_quat_w = self.robot.data.root_quat_w[env_ids]
        
#         # Convert world coordinates to base frame coordinates
#         self.pose_command_b[env_ids, :3], _ = subtract_frame_transforms(
#             robot_base_pos_w,
#             robot_base_quat_w,
#             cart_coords_world,
#             torch.zeros(len(env_ids), 4, device=self.device)  # dummy quaternion
#         )
        
#         # Sample orientation in spherical coordinate frame (alpha, beta, gamma correspond to roll, pitch, yaw)
#         euler_angles_sphere = torch.zeros(len(env_ids), 3, device=self.device)
#         euler_angles_sphere[:, 0] = r.uniform_(*self.cfg.ranges.orn_alpha)  # roll
#         euler_angles_sphere[:, 1] = r.uniform_(*self.cfg.ranges.orn_beta)   # pitch  
#         euler_angles_sphere[:, 2] = r.uniform_(*self.cfg.ranges.orn_gamma)  # yaw
        
#         # Convert euler angles to quaternion in spherical coordinate frame
#         quat_sphere = quat_from_euler_xyz(euler_angles_sphere[:, 0], euler_angles_sphere[:, 1], euler_angles_sphere[:, 2])
        
#         # Transform orientation from spherical coordinate frame to world frame
#         # The spherical coordinate frame is aligned with robot base XY projection on ground
#         # and rotates with robot base yaw angle
#         base_yaw_quat = self._get_base_yaw_quat()[env_ids]  # [w, x, y, z]
        
#         # Apply the base yaw rotation to transform from spherical coordinate frame to world frame
#         # quat_world = base_yaw_quat * quat_sphere (rotate quat_sphere by base_yaw_quat)
#         quat_world = quat_mul_wxyz(base_yaw_quat, quat_sphere)
        
#         # Transform from world frame to robot base frame
#         robot_base_pos_w = self.robot.data.root_pos_w[env_ids]
#         robot_base_quat_w = self.robot.data.root_quat_w[env_ids]
        
#         # Convert world orientation to base frame orientation
#         _, self.pose_command_b[env_ids, 3:] = subtract_frame_transforms(
#             robot_base_pos_w,
#             robot_base_quat_w,
#             torch.zeros_like(robot_base_pos_w),  # dummy position
#             quat_world
#         )
        
#         # Make sure the quaternion has real part as positive
#         if self.cfg.make_quat_unique:
#             self.pose_command_b[env_ids, 3:] = quat_unique(self.pose_command_b[env_ids, 3:])     


#     def _update_command(self):
#         pass
    
#     def _get_ee_goal_spherical_center(self):
#         """Get the center of the spherical coordinate system in world frame."""
#         # Get robot base position (x, y) and use z_invariant_offset as absolute world Z coordinate
#         center = torch.cat([
#             self.robot.data.root_pos_w[:, :2], 
#             torch.full((self.num_envs, 1), self.arm_base_offset[0, 2], device=self.device)
#         ], dim=1)
#         base_yaw_quat = self._get_base_yaw_quat()
#         # Apply only x and y offsets, z is already set to absolute world coordinate
#         xy_offset = torch.cat([
#             self.arm_base_offset[:, :2],
#             torch.zeros(self.num_envs, 1, device=self.device)
#         ], dim=1)
#         center = center + quat_apply_wxyz(base_yaw_quat, xy_offset)
#         return center

#     def _get_base_yaw_quat(self):
#         """Get base yaw-only quaternion."""
#         roll, pitch, yaw = euler_from_quat_wxyz(self.robot.data.root_quat_w)
#         return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

#     def _set_debug_vis_impl(self, debug_vis: bool):
#         # create markers if necessary for the first tome
#         if debug_vis:
#             if not hasattr(self, "goal_pose_visualizer"):
#                 # -- goal pose
#                 self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
#                 # -- current body pose
#                 self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
#             # set their visibility to true
#             self.goal_pose_visualizer.set_visibility(True)
#             self.current_pose_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "goal_pose_visualizer"):
#                 self.goal_pose_visualizer.set_visibility(False)
#                 self.current_pose_visualizer.set_visibility(False)

#     def _debug_vis_callback(self, event):
#         # check if robot is initialized
#         # note: this is needed in-case the robot is de-initialized. we can't access the data
#         if not self.robot.is_initialized:
#             return
#         # update the markers
#         # -- goal pose
#         self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
#         # -- current body pose
#         body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
#         self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

# @configclass
# class HemispherePoseCommandCfg(CommandTermCfg):
#         """Configuration for the end-effector trajectory command generator."""

#         class_type: type = HemispherePoseCommand

#         asset_name: str = MISSING
#         """Name of the asset in the environment for which the commands are generated."""

#         body_name: str = MISSING
#         """Name of the end-effector body in the asset for which the commands are generated."""

#         @configclass
#         class Ranges:
#             """Uniform distribution ranges for the trajectory commands."""

#             # Spherical coordinate ranges
#             pos_l: tuple[float, float] = MISSING
#             """Range for the spherical radial coordinate (in m)."""

#             pos_p: tuple[float, float] = MISSING
#             """Range for the spherical polar angle (in rad)."""

#             pos_y: tuple[float, float] = MISSING
#             """Range for the spherical azimuthal angle (in rad)."""

#             # Orientation ranges
#             orn_alpha: tuple[float, float] = MISSING
#             """Range for the roll orientation (in rad)."""

#             orn_beta: tuple[float, float] = MISSING
#             """Range for the pitch orientation (in rad)."""

#             orn_gamma: tuple[float, float] = MISSING
#             """Range for the yaw orientation (in rad)."""

#         ranges: Ranges = MISSING
#         """Distribution ranges for the trajectory commands."""

#         @configclass
#         class SphereCenter:
#             """Configuration for the spherical coordinate system center."""

#             x_offset: float = 0.0
#             """X offset from robot base to sphere center (in m)."""

#             y_offset: float = 0.0
#             """Y offset from robot base to sphere center (in m)."""

#             z_invariant_offset: float = 0.37
#             """Absolute Z coordinate of the sphere center in world frame (in m)."""

#         sphere_center: SphereCenter = SphereCenter()
#         """Configuration for the spherical coordinate system center."""

#         arm_induced_pitch: float = 0.0
#         """Additional pitch angle induced by arm configuration (in rad)."""
        
#         make_quat_unique: bool = True
#         """Whether to make the quaternion unique by ensuring positive real part."""

#         goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
#             prim_path="/Visuals/Command/ee_goal_pose"
#         )
#         """The configuration for the goal pose visualization marker."""

#         current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
#             prim_path="/Visuals/Command/ee_current_pose"
#         )
#         """The configuration for the current pose visualization marker."""

#         # Set the scale of the visualization markers
#         goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
#         current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

class HemispherePoseCommand(CommandTerm):

    cfg: "HemispherePoseCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "HemispherePoseCommandCfg", env: "ManagerBasedRLEnv"):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.env = env
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # buffers: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # spherical coord system constants
        self.arm_base_offset = torch.tensor(
            [
                cfg.sphere_center.x_offset,
                cfg.sphere_center.y_offset,
                cfg.sphere_center.z_invariant_offset,
            ],
            device=self.device,
            dtype=torch.float,
        ).repeat(self.num_envs, 1)

        # curriculum bookkeeping (kept optional and non-invasive)
        # if env exposes a step counter and you want parity with your uniform command,
        # you can read it during sampling; we store a default horizon if needed.
        self._num_steps_per_env_default = int(getattr(self.cfg, "num_steps_per_env", 24))

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HemispherePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def command(self) -> torch.Tensor:
        """Desired pose command: (num_envs, 7) in base frame (x, y, z, qw, qx, qy, qz)."""
        return self.pose_command_b

    # ----------------------------
    # Internal helpers (non-invasive)
    # ----------------------------
    def _progress_scalar(self) -> torch.Tensor:
        """Return a scalar s in [0, 1] for curriculum blending; 0 disables."""
        coeff = float(getattr(self.cfg, "curriculum_coeff", 0.0) or 0.0)
        if coeff <= 0.0:
            return torch.tensor(0.0, device=self.device)

        steps_total = float(getattr(self.env, "common_step_counter", 0))
        steps_per_env = float(
            getattr(self, "num_env_step", getattr(self.cfg, "num_steps_per_env", self._num_steps_per_env_default))
        )
        s = steps_total / (steps_per_env * coeff) if steps_per_env > 0 else 0.0
        return torch.clamp(torch.tensor(s, device=self.device), 0.0, 1.0)

    def _sample_blended_uniform(self, r: torch.Tensor, name: str) -> torch.Tensor:
        """Sample a parameter by blending init/final ranges using the progress scalar.
        Falls back cleanly to self.cfg.ranges if init/final not provided or coeff==0.
        """
        base = getattr(self.cfg.ranges, name)
        rin = getattr(getattr(self.cfg, "ranges_init", None) or object(), name, None)
        rfi = getattr(getattr(self.cfg, "ranges_final", None) or object(), name, None)
        coeff = float(getattr(self.cfg, "curriculum_coeff", 0.0) or 0.0)

        # sample helper
        def _u(lohi):
            lo, hi = lohi
            return r.uniform_(lo, hi)

        # if curriculum disabled or missing init/final → sample base range
        if coeff <= 0.0 or (rin is None) or (rfi is None):
            return _u(base)

        # curriculum enabled: sample each then blend (matches your UniformPose pattern)
        s = self._progress_scalar()
        v_init = _u(rin)
        v_final = _u(rfi)
        return (1.0 - s) * v_init + s * v_final

    # ----------------------------
    # Implementation-specific functions
    # ----------------------------
    def _update_metrics(self):
        # transform command from base to world
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: "Sequence[int]"):
        """Sample new pose targets in spherical coordinates and convert to cartesian."""
        # random generator
        r = torch.empty(len(env_ids), device=self.device)

        # ---- sample spherical position (l, pitch, yaw) with optional curriculum blending
        sphere_coords = torch.zeros(len(env_ids), 3, device=self.device)
        sphere_coords[:, 0] = self._sample_blended_uniform(r, "pos_l")
        sphere_coords[:, 1] = self._sample_blended_uniform(r, "pos_p")
        sphere_coords[:, 2] = self._sample_blended_uniform(r, "pos_y")

        # to cartesian (sphere frame)
        cart_coords = sphere2cart(sphere_coords)

        # sphere center + base-yaw frame
        base_yaw_quat = self._get_base_yaw_quat()[env_ids]
        sphere_center_world = self._get_ee_goal_spherical_center()[env_ids]

        # rotate by base yaw, then translate to world
        cart_coords_world_rotated = quat_apply_wxyz(base_yaw_quat, cart_coords)
        cart_coords_world = sphere_center_world + cart_coords_world_rotated

        # world → base
        robot_base_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_base_quat_w = self.robot.data.root_quat_w[env_ids]
        self.pose_command_b[env_ids, :3], _ = subtract_frame_transforms(
            robot_base_pos_w,
            robot_base_quat_w,
            cart_coords_world,
            torch.zeros(len(env_ids), 4, device=self.device),  # dummy quaternion
        )

        # ---- sample orientation in spherical frame (alpha, beta, gamma) with optional blending
        euler_sphere = torch.zeros(len(env_ids), 3, device=self.device)
        euler_sphere[:, 0] = self._sample_blended_uniform(r, "orn_alpha")
        euler_sphere[:, 1] = self._sample_blended_uniform(r, "orn_beta")
        euler_sphere[:, 2] = self._sample_blended_uniform(r, "orn_gamma")

        quat_sphere = quat_from_euler_xyz(euler_sphere[:, 0], euler_sphere[:, 1], euler_sphere[:, 2])

        # spherical frame → world: rotate by base yaw
        quat_world = quat_mul_wxyz(base_yaw_quat, quat_sphere)

        # world → base
        _, self.pose_command_b[env_ids, 3:] = subtract_frame_transforms(
            robot_base_pos_w,
            robot_base_quat_w,
            torch.zeros_like(robot_base_pos_w),  # dummy position
            quat_world,
        )

        # normalize quaternion sign (optional)
        if self.cfg.make_quat_unique:
            self.pose_command_b[env_ids, 3:] = quat_unique(self.pose_command_b[env_ids, 3:])

    def _update_command(self):
        pass

    def _get_ee_goal_spherical_center(self) -> torch.Tensor:
        """Get the center of the spherical coordinate system in world frame."""
        # base (x, y) + absolute z from config
        center = torch.cat(
            [
                self.robot.data.root_pos_w[:, :2],
                torch.full((self.num_envs, 1), self.arm_base_offset[0, 2], device=self.device),
            ],
            dim=1,
        )
        # apply only XY offsets in base-yaw frame
        base_yaw_quat = self._get_base_yaw_quat()
        xy_offset = torch.cat([self.arm_base_offset[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply_wxyz(base_yaw_quat, xy_offset)
        return center

    def _get_base_yaw_quat(self) -> torch.Tensor:
        """Get base yaw-only quaternion (w, x, y, z)."""
        _, _, yaw = euler_from_quat_wxyz(self.robot.data.root_quat_w)
        return quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # goal pose (world)
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # current body pose (world)
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class HemispherePoseCommandCfg(CommandTermCfg):
    """Configuration for the end-effector trajectory command generator."""

    class_type: type = HemispherePoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the end-effector body in the asset for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the trajectory commands."""
        # Spherical coordinate ranges
        pos_l: tuple[float, float] = MISSING
        """Range for the spherical radial coordinate (in m)."""

        pos_p: tuple[float, float] = MISSING
        """Range for the spherical polar angle (in rad)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the spherical azimuthal angle (in rad)."""

        # Orientation ranges
        orn_alpha: tuple[float, float] = MISSING
        """Range for the roll orientation (in rad)."""

        orn_beta: tuple[float, float] = MISSING
        """Range for the pitch orientation (in rad)."""

        orn_gamma: tuple[float, float] = MISSING
        """Range for the yaw orientation (in rad)."""

    # main (default) ranges (required)
    ranges: Ranges = MISSING
    """Distribution ranges for the trajectory commands."""

    # Optional initial/final ranges for curriculum (non-invasive if unset)
    ranges_init: "HemispherePoseCommandCfg.Ranges | None" = None
    ranges_final: "HemispherePoseCommandCfg.Ranges | None" = None

    # Simple curriculum control:
    #   - curriculum_coeff <= 0.0  → disabled (use `ranges` only)
    #   - > 0.0 → blend init→final using env.common_step_counter and num_steps_per_env
    curriculum_coeff: float = 0.0
    num_steps_per_env: int = 24

    @configclass
    class SphereCenter:
        """Configuration for the spherical coordinate system center."""
        x_offset: float = 0.0
        """X offset from robot base to sphere center (in m)."""
        y_offset: float = 0.0
        """Y offset from robot base to sphere center (in m)."""
        z_invariant_offset: float = 0.37
        """Absolute Z coordinate of the sphere center in world frame (in m)."""

    sphere_center: SphereCenter = SphereCenter()
    """Configuration for the spherical coordinate system center."""

    arm_induced_pitch: float = 0.0
    """Additional pitch angle induced by arm configuration (in rad)."""

    make_quat_unique: bool = True
    """Whether to make the quaternion unique by ensuring positive real part."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_goal_pose"
    )
    """The configuration for the goal pose visualization marker."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pose"
    )
    """The configuration for the current pose visualization marker."""

    # Set the scale of the visualization markers
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        
class EndEffectorTrajectoryCommand(CommandTerm):
    """Command generator for generating end-effector trajectory commands.
    
    This command generator creates smooth trajectories for the end-effector by sampling
    start and end poses in spherical coordinates, then interpolating between them.
    It includes collision checking, orientation control, and visualization support.
    
    The trajectory is generated in a height-roll-pitch-invariant coordinate system
    centered at the arm base position.
    """

    cfg: EndEffectorTrajectoryCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: EndEffectorTrajectoryCommandCfg, env: ManagerBasedRLEnv):
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
        
        # import utility functions
        from util.torch_utils import torch_rand_float, sphere2cart, quat_apply_wxyz, quat_from_euler_xyz, euler_from_quat_wxyz

        # store references to utility functions
        self.torch_rand_float = torch_rand_float
        self.sphere2cart = sphere2cart 
        self.quat_apply_wxyz = quat_apply_wxyz
        self.quat_from_euler_xyz = quat_from_euler_xyz
        self.euler_from_quat_wxyz = euler_from_quat_wxyz

        # create buffers for commands
        # -- pose command: (x, y, z, qw, qx, qy, qz) in base frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0  # initialize quaternion w component
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # trajectory buffers in spherical coordinates
        self.ee_start_pos_sphere_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_pos_sphere_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_ee_goal_pos_sphere_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_ee_goal_pos_cart_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_ee_goal_pos_cart_w = torch.zeros(self.num_envs, 3, device=self.device)

        # orientation buffers
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_ee_goal_orn_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_ee_goal_orn_quat_w[:, 0] = 1.0  # initialize quaternion w component

        # timing buffers
        self.traj_timesteps = torch.zeros(self.num_envs, device=self.device)
        self.traj_total_timesteps = torch.zeros(self.num_envs, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        # configuration constants
        self.arm_base_offset = torch.tensor([
            cfg.sphere_center.x_offset, 
            cfg.sphere_center.y_offset, 
            cfg.sphere_center.z_invariant_offset
        ], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        self.collision_lower_limits_b = torch.tensor(cfg.collision_lower_limits_b, device=self.device, dtype=torch.float)
        self.collision_upper_limits_b = torch.tensor(cfg.collision_upper_limits_b, device=self.device, dtype=torch.float)
        self.collision_check_t = torch.linspace(0, 1, cfg.num_collision_check_samples, device=self.device)[None, None, :]

        # initialize positions
        self.init_ee_start_pos_sphere_b = torch.tensor(cfg.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_ee_end_pos_sphere_b = torch.tensor(cfg.ranges.init_pos_end, device=self.device).unsqueeze(0)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["trajectory_progress"] = torch.zeros(self.num_envs, device=self.device)

        # get step time from environment
        self.step_dt = env.step_dt
        self.is_first_resample = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "EndEffectorTrajectoryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTeleop mode: {self.cfg.teleop_mode}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.is_first_resample[env_ids] = True

        return super().reset(env_ids)

    def _update_metrics(self):
        """Update metrics for trajectory tracking performance."""
        # transform command from base frame to simulation world frame
        arm_base_world_pos = self._get_ee_goal_spherical_center()
        base_yaw_quat = self._get_base_yaw_quat()
        
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            arm_base_world_pos,
            base_yaw_quat,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        
        # trajectory progress (0 to 1)
        progress = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.metrics["trajectory_progress"] = progress

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.teleop_mode:
            self.ee_start_pos_sphere_b[env_ids] = self.init_ee_start_pos_sphere_b[:]
            self.ee_goal_pos_sphere_b[env_ids] = self.init_ee_end_pos_sphere_b[:]
            self.ee_goal_orn_delta_rpy[env_ids, :] = 0
        else:
            is_init = self.is_first_resample[env_ids].any()
            self._resample_ee_goal(env_ids, is_init=is_init)
            
            self.is_first_resample[env_ids] = False

        # Resample timing
        self.traj_timesteps[env_ids] = self.torch_rand_float(
            self.cfg.ranges.traj_time[0], self.cfg.ranges.traj_time[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze() / self.step_dt
        
        self.traj_total_timesteps[env_ids] = self.traj_timesteps[env_ids] + self.torch_rand_float(
            self.cfg.ranges.hold_time[0], self.cfg.ranges.hold_time[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze() / self.step_dt

        # Reset timers
        self.goal_timer[env_ids] = 0.0

    def _update_command(self):
        """Update the current trajectory command based on timing."""
        if not self.cfg.teleop_mode:
            # Update trajectory interpolation
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            self.traj_ee_goal_pos_sphere_b[:] = torch.lerp(
                self.ee_start_pos_sphere_b, self.ee_goal_pos_sphere_b, t[:, None]
            )

        # Convert from spherical to cartesian coordinates
        self.traj_ee_goal_pos_cart_b[:] = self.sphere2cart(self.traj_ee_goal_pos_sphere_b)
        
        # Transform to world frame
        base_yaw_quat = self._get_base_yaw_quat()
        ee_goal_cart_yaw_global = self.quat_apply_wxyz(base_yaw_quat, self.traj_ee_goal_pos_cart_b)
        self.traj_ee_goal_pos_cart_w = self._get_ee_goal_spherical_center() + ee_goal_cart_yaw_global

        # Compute orientation
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.traj_ee_goal_pos_sphere_b[:, 1] + self.cfg.arm_induced_pitch
        self.traj_ee_goal_orn_quat_w = self.quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + torch.pi / 2,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + default_yaw
        )

        # Update pose command in base frame
        arm_base_world_pos = self._get_ee_goal_spherical_center()
        base_yaw_quat = self._get_base_yaw_quat()
        
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            arm_base_world_pos,
            base_yaw_quat,
            self.traj_ee_goal_pos_cart_w,
            self.traj_ee_goal_orn_quat_w,
        )

        # Increment timer and check for resampling
        self.goal_timer += 1
        resample_ids = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_ee_goal(resample_ids)

    def _resample_ee_goal(self, env_ids: Sequence[int], is_init: bool = False):
        """Resample end-effector goal positions and orientations."""
        if len(env_ids) == 0:
            return

        init_env_ids = env_ids.clone()

        if is_init:
            # Initialize with default positions
            self.ee_goal_orn_delta_rpy[env_ids, :] = 0
            self.ee_start_pos_sphere_b[env_ids] = self.init_ee_start_pos_sphere_b[:]
            self.ee_goal_pos_sphere_b[env_ids] = self.init_ee_end_pos_sphere_b[:]
        else:
            # Sample new orientation deltas
            self._resample_ee_goal_delta_orientation_once(env_ids)
            # Previous goal becomes new start
            self.ee_start_pos_sphere_b[env_ids] = self.ee_goal_pos_sphere_b[env_ids].clone()
            
            # Sample new goal with collision checking
            for i in range(10):  # Maximum 10 attempts
                self._resample_ee_goal_position_once(env_ids)
                collision_mask = self._collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break

        # Reset goal timer for all initially specified environments
        self.goal_timer[init_env_ids] = 0.0

    def _resample_ee_goal_position_once(self, env_ids: Sequence[int]):
        """Sample new goal positions in spherical coordinates."""
        self.ee_goal_pos_sphere_b[env_ids, 0] = self.torch_rand_float(
            self.cfg.ranges.pos_l[0], self.cfg.ranges.pos_l[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.ee_goal_pos_sphere_b[env_ids, 1] = self.torch_rand_float(
            self.cfg.ranges.pos_p[0], self.cfg.ranges.pos_p[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.ee_goal_pos_sphere_b[env_ids, 2] = self.torch_rand_float(
            self.cfg.ranges.pos_y[0], self.cfg.ranges.pos_y[1], 
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

    def _resample_ee_goal_delta_orientation_once(self, env_ids: Sequence[int]):
        """Sample new orientation deltas."""
        ee_goal_delta_orn_r = self.torch_rand_float(
            self.cfg.ranges.delta_orn_r[0], self.cfg.ranges.delta_orn_r[1], 
            (len(env_ids), 1), device=self.device
        )
        ee_goal_delta_orn_p = self.torch_rand_float(
            self.cfg.ranges.delta_orn_p[0], self.cfg.ranges.delta_orn_p[1], 
            (len(env_ids), 1), device=self.device
        )
        ee_goal_delta_orn_y = self.torch_rand_float(
            self.cfg.ranges.delta_orn_y[0], self.cfg.ranges.delta_orn_y[1], 
            (len(env_ids), 1), device=self.device
        )
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([
            ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y
        ], dim=-1)

    def _collision_check(self, env_ids: Sequence[int]):
        """Check if trajectory collides with obstacles or goes underground."""
        # Interpolate along trajectory
        ee_targets_sphere_b = torch.lerp(
            self.ee_start_pos_sphere_b[env_ids, ..., None], 
            self.ee_goal_pos_sphere_b[env_ids, ..., None], 
            self.collision_check_t
        ).squeeze(-1)
        
        # Convert to cartesian coordinates
        ee_targets_cart_b = self.sphere2cart(
            torch.permute(ee_targets_sphere_b, (2, 0, 1)).reshape(-1, 3)
        ).reshape(self.cfg.num_collision_check_samples, -1, 3)
        
        # Check collision with bounding box
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_targets_cart_b < self.collision_upper_limits_b, dim=-1),
                torch.all(ee_targets_cart_b > self.collision_lower_limits_b, dim=-1)
            ), dim=0
        )
        
        # Check underground collision
        underground_mask = torch.any(ee_targets_cart_b[..., 2] < self.cfg.underground_limit_b, dim=0)
        
        return collision_mask | underground_mask

    def _get_ee_goal_spherical_center(self):
        """Get the center of the spherical coordinate system in world frame."""
        # Get robot base position (x, y) and use z_invariant_offset as absolute world Z coordinate
        center = torch.cat([
            self.robot.data.root_pos_w[:, :2], 
            torch.full((self.num_envs, 1), self.arm_base_offset[0, 2], device=self.device)
        ], dim=1)
        base_yaw_quat = self._get_base_yaw_quat()
        # Apply only x and y offsets, z is already set to absolute world coordinate
        xy_offset = torch.cat([
            self.arm_base_offset[:, :2],
            torch.zeros(self.num_envs, 1, device=self.device)
        ], dim=1)
        center = center + self.quat_apply_wxyz(base_yaw_quat, xy_offset)
        return center

    def _get_base_yaw_quat(self):
        """Get base yaw-only quaternion."""
        _, _, base_yaw_euler = self.euler_from_quat_wxyz(self.robot.data.root_quat_w)
        return self.quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw_euler)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # Goal pose marker
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # Current pose marker
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                # Trajectory marker
                self.trajectory_visualizer = VisualizationMarkers(self.cfg.trajectory_visualizer_cfg)
            
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.trajectory_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.trajectory_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if not self.robot.is_initialized:
            return

        # Visualize goal pose
        self.goal_pose_visualizer.visualize(
            self.traj_ee_goal_pos_cart_w, self.traj_ee_goal_orn_quat_w
        )
        
        # Visualize current pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        
        # Visualize trajectory
        # Generate trajectory points for visualization
        t_viz = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_targets_sphere = torch.lerp(
            self.ee_start_pos_sphere_b[..., None], 
            self.ee_goal_pos_sphere_b[..., None], 
            t_viz
        ).squeeze(0)  # (num_envs, 3, 10)

        ee_targets_cart_local = torch.stack([
            self.sphere2cart(ee_targets_sphere[..., i]) for i in range(10)
        ], dim=-1)  # (num_envs, 3, 10)

        base_yaw_quat = self._get_base_yaw_quat()
        ee_targets_cart_rot = torch.stack([
            self.quat_apply_wxyz(base_yaw_quat, ee_targets_cart_local[..., i]) for i in range(10)
        ], dim=-1)  # (num_envs, 3, 10)

        center = self._get_ee_goal_spherical_center()  # (num_envs, 3)
        ee_targets_cart_world = ee_targets_cart_rot + center[:, :, None]  # (num_envs, 3, 10)

        points = ee_targets_cart_world.permute(0, 2, 1).reshape(-1, 3)  # (num_envs*10, 3)
        pos_np = points.detach().cpu().numpy()
        self.trajectory_visualizer.visualize(pos_np, None)

@configclass
class EndEffectorTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the end-effector trajectory command generator."""

    class_type: type = EndEffectorTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the end-effector body in the asset for which the commands are generated."""

    teleop_mode: bool = False
    """Whether to use teleoperation mode (no automatic goal resampling)."""

    num_collision_check_samples: int = 10
    """Number of samples to check for collision along the trajectory."""

    underground_limit_b: float = 0.0
    """Lower limit for z-coordinate in base frame to prevent underground goals."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the trajectory commands."""

        # Spherical coordinate ranges
        pos_l: tuple[float, float] = MISSING
        """Range for the spherical radial coordinate (in m)."""

        pos_p: tuple[float, float] = MISSING
        """Range for the spherical polar angle (in rad)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the spherical azimuthal angle (in rad)."""

        # Orientation delta ranges
        delta_orn_r: tuple[float, float] = MISSING
        """Range for the roll orientation delta (in rad)."""

        delta_orn_p: tuple[float, float] = MISSING
        """Range for the pitch orientation delta (in rad)."""

        delta_orn_y: tuple[float, float] = MISSING
        """Range for the yaw orientation delta (in rad)."""

        # Timing ranges
        traj_time: tuple[float, float] = MISSING
        """Range for trajectory execution time (in s)."""

        hold_time: tuple[float, float] = MISSING
        """Range for goal holding time (in s)."""

        # Initial position ranges for initialization
        init_pos_start: tuple[float, float, float] = (0.5, 0.0, 0.0)
        """Initial start position in spherical coordinates."""

        init_pos_end: tuple[float, float, float] = (0.6, 0.0, 0.0)
        """Initial end position in spherical coordinates."""

    ranges: Ranges = MISSING
    """Distribution ranges for the trajectory commands."""

    @configclass
    class SphereCenter:
        """Configuration for the spherical coordinate system center."""

        x_offset: float = 0.03
        """X offset from robot base to sphere center (in m)."""

        y_offset: float = 0.0
        """Y offset from robot base to sphere center (in m)."""

        z_invariant_offset: float = 0.37
        """Absolute Z coordinate of the sphere center in world frame (in m)."""

    sphere_center: SphereCenter = SphereCenter()
    """Configuration for the spherical coordinate system center."""

    collision_lower_limits_b: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Lower collision limits in base frame (x, y, z) in meters."""

    collision_upper_limits_b: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Upper collision limits in base frame (x, y, z) in meters."""

    arm_induced_pitch: float = 0.0
    """Additional pitch angle induced by arm configuration (in rad)."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_goal_pose"
    )
    """The configuration for the goal pose visualization marker."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pose"
    )
    """The configuration for the current pose visualization marker."""

    trajectory_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_trajectory"
    )
    """The configuration for the trajectory visualization marker."""

    # Set the scale of the visualization markers
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    trajectory_visualizer_cfg.markers["sphere"].radius = 0.005 