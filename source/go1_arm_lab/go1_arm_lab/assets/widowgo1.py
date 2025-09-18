import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
usd_file_path = os.path.join(current_dir, "widowGo1.usd")

robot_usd = usd_file_path

GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)
"""Configuration of Go1 actuators using MLP model.

Actuator specifications: https://shop.unitree.com/products/go1-motor

This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""

WIDOW_GO1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robot_usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0,
            fix_root_link=False
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35), # initial position of the robot base height
        joint_pos={
            # dog leg
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # arm
            "widow_waist": 0.0,
            "widow_shoulder": 0.0,
            "widow_elbow": 0.0,
            "widow_forearm_roll": 0.0,
            "widow_wrist_angle": 0.0,
            "widow_wrist_rotate": 0.0,
            "widow_left_finger": 0.037,
            "widow_right_finger": -0.037,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "base_legs": GO1_ACTUATOR_CFG,
        # "arm": ImplicitActuatorCfg(
        #     joint_names_expr=["widow_waist", "widow_shoulder", "widow_elbow", "widow_forearm_roll", 
        #                       "widow_wrist_angle", "widow_wrist_rotate", "widow_left_finger", "widow_right_finger"],
        #     velocity_limit=10.0,
        #     effort_limit=87.0,
        #     stiffness={
        #         "widow_waist": 400,
        #         "widow_shoulder": 400,
        #         "widow_elbow": 400,
        #         "widow_forearm_roll": 400,
        #         "widow_wrist_angle": 400,
        #         "widow_wrist_rotate": 400,
        #         "widow_left_finger": 400,
        #         "widow_right_finger": 400,
        #     },
        #     damping= {
        #         "widow_waist": 40,
        #         "widow_shoulder": 40,
        #         "widow_elbow": 40,
        #         "widow_forearm_roll": 40,
        #         "widow_wrist_angle": 40,
        #         "widow_wrist_rotate": 40,
        #         "widow_left_finger": 40,
        #         "widow_right_finger": 40,
        #     }
        # ),
        "arm": DCMotorCfg(
            joint_names_expr=["widow_.*"],
            velocity_limit=30.0,
            effort_limit=40.5,
            saturation_effort=23.5,
            stiffness=80,
            damping=4,
            friction=0.0,
            armature=0.01,
        ),
    },
)