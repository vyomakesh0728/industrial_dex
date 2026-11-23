"""UR5 with Robotiq 2F-85 gripper configuration matching WireCraft."""

from __future__ import annotations

import os

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB

from ..common.env_cfg import PRIM_PATH, RobotCfg

PRIM_NAME = "ur5_robot"
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _default_ur5_usd() -> str:
    env_dir = os.environ.get("WIRECRAFT_ROBOT_ASSET_DIR", "")
    if env_dir:
        return os.path.join(env_dir, "ur5_with_gripper_camera_v2.usd")
    return "ur5_with_gripper_camera_v2.usd"


def build_robot_cfg() -> RobotCfg:
    articulation = None
    if HAS_ISAACLAB:
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import ArticulationCfg  # type: ignore[import-not-found]

        articulation = ArticulationCfg(
            prim_path=f"{PRIM_PATH}/{PRIM_NAME}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_default_ur5_usd(),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True, disable_gravity=True, max_depenetration_velocity=5.0
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True,
                    enabled_self_collisions=True,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.05, 0.0, 0.7825),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={
                    "shoulder_pan_joint": 3.14159,
                    "shoulder_lift_joint": -1.85,
                    "elbow_joint": 0.90,
                    "wrist_1_joint": 1.658,
                    "wrist_2_joint": 1.571,
                    "wrist_3_joint": 1.571,
                    "finger_joint": 0.0,
                },
            ),
            actuators={
                "ur5_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    effort_limit_sim=150.0,
                    stiffness=1008.43994,
                    damping=40.03376,
                ),
                "ur5_wrist": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_[1-3]_joint"],
                    effort_limit_sim=70.0,
                    stiffness=1000.0,
                    damping=10.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["finger_joint"],
                    effort_limit_sim=100.0,
                    stiffness=1.7,
                    damping=0.2,
                ),
                "gripper_passive": ImplicitActuatorCfg(
                    joint_names_expr=[".*inner_finger_joint.*"],
                    effort_limit_sim=2.0,
                    stiffness=0.2,
                    damping=0.01,
                ),
                "gripper_connection_only": ImplicitActuatorCfg(
                    joint_names_expr=[".*outer_finger_joint.*", ".*knuckle.*"],
                    effort_limit_sim=0.0,
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )

    return RobotCfg(
        name="ur5",
        adapter="idb_tasks.direct.idb.single_arm.ur5.robot:UR5Adapter",
        usd_path=_default_ur5_usd(),
        articulation=articulation,
        arm_joints=list(ARM_JOINTS),
        gripper_joint="finger_joint",
        ee_body="position_approximation",
        gripper_open_pos=0.0,
        gripper_closed_pos=0.785,
        gripper_release_step=None,
        grasp_opening_range=(0.108, 0.745),
        release_opening=0.49,
        fully_closed_opening=0.108,
        finger_bodies=(".*gripper_left_inner_finger", ".*gripper_right_inner_finger"),
        grasp_rest_quat=(1.0, 0.0, 0.0, 0.0),
        ft_sensor_body="gripper_base_link",
        ik_joint_reference="control",
        max_joint_step=0.05,
        joint_lower=[-6.28] * 6,
        joint_upper=[6.28] * 6,
    )
