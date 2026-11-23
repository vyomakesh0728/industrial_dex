"""UR5 + Robotiq 2F-85 adapter: passive inner fingers, wrist F/T, and wrist camera."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB

from ..common.env_cfg import PRIM_PATH
from ..common.robot import RobotAdapter
from .robot_cfg import PRIM_NAME

PASSIVE_TARGETS = {"right_inner_finger_joint": 0.7854, "left_inner_finger_joint": -0.7854}


class UR5Adapter(RobotAdapter):
    prim_name = PRIM_NAME

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        size = cfg.sensors.camera_resolution
        if not HAS_ISAACLAB:
            return {
                "wrist_d405_rgb": {"prim_path": f"{PRIM_PATH}/{PRIM_NAME}/RSD455/Camera_OmniVision_OV9782_Color", "data_type": "rgb", "resolution": size},
                "wrist_d405_depth": {
                    "prim_path": f"{PRIM_PATH}/{PRIM_NAME}/wrist_3_link/wrist_d405_depth",
                    "data_type": "distance_to_camera",
                    "resolution": size,
                    "clipping_range": (0.04, 1.5),
                },
            }
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        from isaaclab.sensors import TiledCameraCfg  # type: ignore[import-not-found]

        return {
            "wrist_d405_rgb": TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/RSD455/Camera_OmniVision_OV9782_Color",
                spawn=None,
                data_types=["rgb"],
                width=size,
                height=size,
            ),
            # Isaac Lab TiledCamera requires spawn.clipping_range for distance_to_camera.
            "wrist_d405_depth": TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/wrist_3_link/wrist_d405_depth",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.06, 0.0, 0.04), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
                ),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.04, 1.5),
                ),
                data_types=["distance_to_camera"],
                width=size,
                height=size,
            ),
        }

    def post_clone(self) -> None:
        super().post_clone()
        if not HAS_ISAACLAB:
            return
        from pxr import UsdPhysics  # type: ignore[import-not-found]
        from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]

        stage = get_current_stage()
        for env_id in range(self.env.scene.cfg.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{PRIM_NAME}/{self.cfg.ee_body}")
            if prim.IsValid():
                mass = UsdPhysics.MassAPI(prim)
                mass.CreateMassAttr().Set(0.01)
                mass.CreateDiagonalInertiaAttr().Set((0.0001, 0.0001, 0.0001))

    def initialize(self) -> None:
        super().initialize()
        self.passive_ids = self.robot.find_joints(list(PASSIVE_TARGETS), preserve_order=True)[0]
        self.passive_targets = torch.tensor(list(PASSIVE_TARGETS.values()), device=self.env.device).repeat(
            self.env.num_envs, 1
        )

    def apply_gripper_extras(self, gripper: torch.Tensor) -> None:
        self.robot.set_joint_position_target(self.passive_targets, joint_ids=self.passive_ids)
