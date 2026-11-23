"""Single-arm Isaac Lab DirectRLEnv for all IDB Board #1 datacenter tasks.

Mirrors WireCraft's SingleArmWireEnv structure:
- One shared DirectRLEnv coordinates the robot adapter, wire backend, IDB Board #1 runtime,
  and modular task object.
- Strictly simulation-only: returns zero reward (no RL reward shaping, training loop, or policy).
"""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, require_isaaclab

from ...wire_base.articulation import ArticulationWire
from ...wire_base.connectors import ConnectorGeometry
from ...wire_base.deformable import DeformableWire
from .board import IDBBoardRuntime
from .env_cfg import SingleArmIDBEnvCfg, _import_symbol, resolve_env_cfg

if HAS_ISAACLAB:
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    from isaaclab.assets import RigidObject  # type: ignore[import-not-found]
    from isaaclab.envs import DirectRLEnv  # type: ignore[import-not-found]
    from isaaclab.sensors import TiledCamera  # type: ignore[import-not-found]
    from isaaclab.terrains import TerrainImporterCfg  # type: ignore[import-not-found]
else:
    DirectRLEnv = object  # type: ignore[assignment,misc]


class SingleArmIDBEnv(DirectRLEnv):  # type: ignore[misc]
    """Modular Isaac Lab simulation environment for IDB datacenter cable-management tasks."""

    cfg: SingleArmIDBEnvCfg

    def __init__(self, cfg: SingleArmIDBEnvCfg, render_mode: str | None = None, **kwargs):
        require_isaaclab("SingleArmIDBEnv")
        resolve_env_cfg(cfg)
        self.cfg = cfg
        self.connector = ConnectorGeometry.from_family(cfg.connector.family, cfg.connector.socket_clearance)
        self.far_connector = ConnectorGeometry.from_family(cfg.connector.resolved_far_family, cfg.connector.socket_clearance)
        self.arm = _import_symbol(cfg.robot.adapter)(self)
        self.board = IDBBoardRuntime(self)
        self.wire_backend = (ArticulationWire if cfg.wire.kind == "articulation" else DeformableWire)(self)
        self.task = cfg.task.task_class(self)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.safe_low = torch.tensor(cfg.ee_safe_zone[0::2], device=self.device)
        self.safe_high = torch.tensor(cfg.ee_safe_zone[1::2], device=self.device)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.out_of_bounds = zeros.clone()
        self.joint_limit = zeros.clone()
        self.port_drop = zeros.clone()
        self.faulted = zeros.clone()
        self.episode_done = zeros.clone()
        self.fault_count = 0
        self.fault_log: list[dict] = []

        self.arm.initialize()
        self.board.initialize()
        self.wire_backend.initialize()
        self.wire_backend.pin_far_end()
        self.task.initialize()
        self._update_state()

    def _setup_scene(self) -> None:
        self.arm.spawn()
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            num_envs=self.scene.cfg.num_envs,
            env_spacing=self.scene.cfg.env_spacing,
        )
        self.terrain = terrain_cfg.class_type(terrain_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table
        self.board.spawn()
        self.wire_backend.spawn()
        self.scene.clone_environments(copy_from_source=False)
        self.arm.post_clone()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.85, 0.85, 0.85))
        light.func("/World/Light", light)
        self._light_prim_path = "/World/Light"
        self.tiled_cameras = {}
        for name, cam_cfg in self.cfg.camera_cfgs.items():
            self.tiled_cameras[name] = self.scene.sensors[name] = TiledCamera(cam_cfg)

    def _update_state(self) -> None:
        self.arm.update()
        self.wire_state = self.wire_backend.state()
        origins = self.scene.env_origins
        self.ee_local_pos = self.arm.ee_pos - origins
        self.ee_local_quat = self.arm.ee_quat
        self.port_local_pos = self.wire_state.insertion_end_pos - origins
        self.tip_local_pos = self.wire_state.tip_pos - origins
        self.task.update()

    def _fault_mask(self) -> torch.Tensor:
        joints_ok = self.arm.joint_speed() < self.cfg.fault_joint_speed
        ports_ok = self.wire_backend.port_speed() < self.cfg.fault_port_speed
        return ~(joints_ok & ports_ok)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.arm.latch_control_state()

    def _apply_action(self) -> None:
        self.wire_backend.hold_far_end()
        self.board.hold_neighbor_cables()
        self.arm.apply_action(self.actions, self.cfg.action_scale, self.cfg.fix_ee_orient, self.cfg.control_mode)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_state()
        obs: dict[str, torch.Tensor] = {
            "policy": torch.nan_to_num(self.task.observations(), nan=0.0, posinf=0.0, neginf=0.0)
        }
        if self.cfg.sensors.enable_wrist_ft:
            obs["ft"] = torch.nan_to_num(self.arm.wrench.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.sensors.enable_polishing_contact_sensor:
            obs["polishing_force_n"] = self.board.polishing_contact_force_n().unsqueeze(-1)
        for name, camera in self.tiled_cameras.items():
            for data_type, data in camera.data.output.items():
                obs[name] = (data / 255.0) if data_type == "rgb" else data.clone()
        return obs

    def _get_rewards(self) -> torch.Tensor:
        # Strictly simulation-only: no RL reward shaping or training objective.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_state()
        self.faulted = self._fault_mask()
        if self.faulted.any():
            self.fault_count += int(self.faulted.sum())
            self.fault_log.append({
                "step": int(self.common_step_counter),
                "envs": self.faulted.nonzero().flatten().tolist(),
                "joint_speed": self.arm.joint_speed().tolist(),
                "port_speed": self.wire_backend.port_speed().tolist(),
            })
        rubric = self.task.evaluate()
        success = rubric.passed & (~self.faulted)
        damaged = self.task.damaged()
        self.extras["is_success"] = success
        self.extras["faulted"] = self.faulted
        self.extras["damaged"] = damaged
        self.extras["rubric_score"] = rubric.total_score
        self.extras["rubric_max"] = rubric.max_score
        self.extras["rubric_steps"] = rubric.step_scores
        self.extras["phase_flags"] = rubric.phase_flags

        terminated = self.faulted.clone()
        if self.cfg.enable_terminations:
            xy = self.ee_local_pos[:, :2]
            self.out_of_bounds = ((xy < self.safe_low[:2]) | (xy > self.safe_high[:2])).any(dim=-1)
            self.joint_limit = self.arm.joint_limit_mask()
            low = self.cfg.port_drop_height
            self.port_drop = (self.wire_state.insertion_end_pos[:, 2] < low) | (self.wire_state.far_end_pos[:, 2] < low)
            terminated |= self.out_of_bounds | self.joint_limit | self.port_drop
        if self.cfg.terminate_on_damage:
            terminated |= damaged
        if self.cfg.terminate_on_success:
            terminated |= success
        time_out = (
            self.episode_length_buf >= self.max_episode_length - 1
            if self.cfg.terminate_on_time_out
            else torch.zeros_like(terminated)
        )
        self.episode_done = terminated | time_out
        return terminated, time_out

    def _randomize_dome_light(self) -> None:
        if not self.cfg.variation.randomize_lighting:
            return
        from pxr import UsdLux  # type: ignore[import-not-found]
        from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]

        lo, hi = self.cfg.variation.light_intensity_range
        intensity = float(lo + torch.rand(1).item() * (hi - lo))
        prim = get_current_stage().GetPrimAtPath(self._light_prim_path)
        if prim.IsValid():
            UsdLux.DomeLight(prim).GetIntensityAttr().Set(intensity)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.arm.reset(env_ids)
        self._randomize_dome_light()
        nominal_neighbors = self.board.reset(env_ids)
        far_pos, far_quat, act_pos, act_quat = self.task.reset(env_ids, nominal_neighbors)
        self.wire_backend.reset_plugged(env_ids, far_pos, far_quat, act_pos, act_quat)
        self.scene.update(dt=0.0)
        self.arm.update()
        self.wire_state = self.wire_backend.state()
        origins = self.scene.env_origins
        self.ee_local_pos = self.arm.ee_pos - origins
        self.ee_local_quat = self.arm.ee_quat
        self.port_local_pos = self.wire_state.insertion_end_pos - origins
        self.tip_local_pos = self.wire_state.tip_pos - origins
        # Refresh target/source seating geometry without triggering step-0 latch rip damage.
        self.task.refresh_after_reset(env_ids)
