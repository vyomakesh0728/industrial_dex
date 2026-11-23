"""Franka Panda/FR3 adapter: wrist D405 RGB/depth camera, null-space step, mimic finger."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB

from ..common.env_cfg import PRIM_PATH
from ..common.robot import RobotAdapter
from .robot_cfg import PRIM_NAME

WRIST_POS = (0.06, 0.0, 0.04)
WRIST_ROT = (-0.70614, 0.03701, 0.03701, -0.70614)
NULL_GAIN = 0.1
NULL_STEP = 0.01
DAMPING = 0.01


class FrankaAdapter(RobotAdapter):
    prim_name = PRIM_NAME

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        size = cfg.sensors.camera_resolution
        if not HAS_ISAACLAB:
            return {
                "wrist_d405_rgb": {"prim_path": f"{PRIM_PATH}/{PRIM_NAME}/panda_hand/wrist_d405_rgb", "data_type": "rgb", "resolution": size},
                "wrist_d405_depth": {"prim_path": f"{PRIM_PATH}/{PRIM_NAME}/panda_hand/wrist_d405_depth", "data_type": "distance_to_camera", "resolution": size},
            }
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        from isaaclab.sensors import TiledCameraCfg  # type: ignore[import-not-found]

        def _cam(name: str, dtype: str) -> TiledCameraCfg:
            return TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/panda_hand/{name}",
                offset=TiledCameraCfg.OffsetCfg(pos=WRIST_POS, rot=WRIST_ROT, convention="ros"),
                data_types=[dtype],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.04, 1.5),
                ),
                width=size,
                height=size,
            )

        return {
            "wrist_d405_rgb": _cam("wrist_d405_rgb", "rgb"),
            "wrist_d405_depth": _cam("wrist_d405_depth", "distance_to_camera"),
        }

    def initialize(self) -> None:
        super().initialize()
        self.mimic_id = self.robot.find_joints("panda_finger_joint2")[0]

    def null_space_step(self, jacobian: torch.Tensor) -> torch.Tensor:
        arm = self.joint_pos[:, : len(self.arm_ids)]
        toward = (NULL_GAIN * (0.5 * (self.arm_lower + self.arm_upper) - arm)).clamp(-NULL_STEP, NULL_STEP)
        jt = jacobian.transpose(1, 2)
        eye = torch.eye(6, device=jacobian.device)
        pinv = jt @ torch.linalg.solve(jacobian @ jt + (DAMPING**2) * eye, jacobian)
        return toward - (pinv @ toward.unsqueeze(-1)).squeeze(-1)

    def apply_gripper_extras(self, gripper: torch.Tensor) -> None:
        self.robot.set_joint_position_target(self.targets[:, -1:], joint_ids=self.mimic_id)
