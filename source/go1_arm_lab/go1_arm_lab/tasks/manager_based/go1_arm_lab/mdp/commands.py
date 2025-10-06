# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, subtract_frame_transforms
from ..util.torch_utils import euler_from_quat_wxyz, quat_apply_wxyz, sphere2cart, quat_mul_wxyz
from ..util.markers import SPHERE_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
        
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