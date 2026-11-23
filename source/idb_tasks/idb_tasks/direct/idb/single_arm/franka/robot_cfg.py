"""Franka Panda/FR3 configuration with wrist F/T wrench and D405 wrist camera."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB

from ..common.env_cfg import PRIM_PATH, RobotCfg

PRIM_NAME = "franka"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def build_robot_cfg() -> RobotCfg:
    articulation = None
    if HAS_ISAACLAB:
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # type: ignore[import-not-found]

        articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=f"{PRIM_PATH}/{PRIM_NAME}").copy()
        articulation.init_state.pos = (0.0, 0.0, 0.7825)
        articulation.spawn.articulation_props.solver_position_iteration_count = 32
        articulation.spawn.activate_contact_sensors = True
        articulation.actuators["panda_hand"].effort_limit_sim = 70.0

    return RobotCfg(
        name="franka",
        adapter="idb_tasks.direct.idb.single_arm.franka.robot:FrankaAdapter",
        articulation=articulation,
        arm_joints=list(ARM_JOINTS),
        gripper_joint="panda_finger_joint1",
        ee_body="grasp_center",
        tool_parent="panda_hand",
        tool_offset=(0.0, 0.0, 0.1034),
        gripper_open_pos=0.04,
        gripper_closed_pos=0.0,
        gripper_max_step=0.0004,
        gripper_release_step=0.00008,
        grasp_opening_range=(0.03, 0.90),
        release_opening=0.90,
        fully_closed_opening=0.03,
        finger_bodies=("panda_leftfinger", "panda_rightfinger"),
        # Approach horizontal connectors along +X facing Board #1.
        grasp_rest_quat=(0.5, 0.5, 0.5, 0.5),
        ft_sensor_body="panda_hand",
        ik_joint_reference="control",
        max_joint_step=0.05,
    )
