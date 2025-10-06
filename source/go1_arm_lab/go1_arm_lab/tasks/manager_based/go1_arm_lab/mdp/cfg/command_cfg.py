from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
# from .. import pose_command , velocity_command
from ..pose_command import UniformPoseCommand, HemispherePoseCommand
from ..velocity_command import UniformVelocityCommand

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg as UniformPoseCommandBaseCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg as UniformVelocityCommandBaseCfg

@configclass
class HemispherePoseCommandCfg(CommandTermCfg):
    """Base config for hemispherical pose command (spherical sampling)"""
    class_type: type = HemispherePoseCommand

    debug_vis: bool = False

    asset_name: str = MISSING
    body_name: str = MISSING

    is_Go1Arm: bool = False
    """ is_Go1Arm flag. Mirror of UniformPoseCommandCfg for consistency. """

    is_Go1Arm_Play: bool = False
    """ is_Go1Arm_Play flag. """

    is_Go1Arm_Flat: bool = True  
    """ Kept to mirror UniformPoseCommandCfg; not used by sampler. """

    @configclass
    class Ranges:
        # spherical position
        pos_l: tuple[float, float] = MISSING    # radius (m)
        pos_p: tuple[float, float] = MISSING    # polar angle/pitch (rad)
        pos_y: tuple[float, float] = MISSING    # azimuth/yaw (rad)
        # spherical orientation (roll/pitch/yaw)
        orn_alpha: tuple[float, float] = MISSING
        orn_beta: tuple[float, float] = MISSING
        orn_gamma: tuple[float, float] = MISSING

    # main ranges (required)
    ranges: Ranges = MISSING

    # optional curriculum
    ranges_init: "HemispherePoseCommandCfg.Ranges | None" = None
    ranges_final: "HemispherePoseCommandCfg.Ranges | None" = None
    curriculum_coeff: float = 0.0
    """When <= 0.0, no blending; use `ranges` directly."""
    num_steps_per_env: int = 24

    @configclass
    class SphereCenter:
        x_offset: float = 0.0
        y_offset: float = 0.0
        z_invariant_offset: float = 0.37

    sphere_center: SphereCenter = SphereCenter()

    make_quat_unique: bool = True

    # debug vis markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

@configclass
class UniformPoseCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type = UniformPoseCommand

    is_Go1Arm: bool = False
    """ is_Go1Arm flag.
    check pose_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the pose command to linearly increase from range_init to range_final. 
    valid when is_Go1Arm = True
    check pose_command.py for more details
    """

    is_Go1Arm_Play: bool = False 
    """ is_Go1Arm_Play flag.
    check pose_command.py for more details
    """

    is_Go1Arm_Flat: bool = True #TODO
    """ 
    check pose_command.py for more details
    """

    ranges_init: UniformPoseCommandBaseCfg.Ranges = None
    """The initial range"""

    ranges_final: UniformPoseCommandBaseCfg.Ranges = None
    """The maximum range"""



@configclass
class UniformVelocityCommandCfg(UniformVelocityCommandBaseCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    is_Go1Arm: bool = False
    """ is_Go1Arm flag.
    check velocity_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the velocity command to linearly increase from range_init to range_final. 
    valid when is_Go1Arm = True
    check velocity_command.py for more details
    """    
    is_Go1Arm_Flat: bool = True #TODO
    """ 
    check velocity_command.py for more details
    """

    ranges_init: UniformVelocityCommandBaseCfg.Ranges = None
    """The initial range
    check velocity_command.py for more details"""
    
    ranges_final: UniformVelocityCommandBaseCfg.Ranges = None
    """The maximum range
    check velocity_command.py for more details"""

