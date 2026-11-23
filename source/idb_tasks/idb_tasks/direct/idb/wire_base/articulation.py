"""Articulated capsule-chain wire backend for IDB datacenter cables."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import require_isaaclab

from .math_utils import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, resample_polyline
from .wire_state import PluggedCableArc, WireState, tip_from_root

HOLD_STIFFNESS = 0.5
HOLD_DAMPING = 0.5
GRAVITY = 9.81


class ArticulationWire:
    """Ports and cable segments are links of one PhysX articulation joined by fixed end joints."""

    def __init__(self, env, prim_path: str | None = None, assembly_cfg=None, tip_length: float | None = None, name: str = "wire"):
        self.env = env
        self.cfg = env.cfg.wire
        self.prim_path = prim_path or env.cfg.wire_prim_path
        self.assembly_cfg = assembly_cfg or env.cfg.wire_spawn
        self.tip_length = env.connector.tip_length if tip_length is None else tip_length
        self.scene_key = name
        self._pinned = False

    def spawn(self) -> None:
        require_isaaclab("ArticulationWire.spawn")
        from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]

        assembly = self.assembly_cfg
        assembly.func(self.prim_path, assembly, translation=tuple(self.cfg.root_pos))
        art_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(self.cfg.root_pos)),
            actuators={
                "bending": ImplicitActuatorCfg(
                    joint_names_expr=["joint_seg_.*"],
                    stiffness=assembly.articulation.stiffness,
                    damping=assembly.articulation.damping,
                )
            },
        )
        self.wire = Articulation(art_cfg)
        self.env.scene.articulations[self.scene_key] = self.wire

    def pin_far_end(self) -> None:
        self._pinned = True

    def hold_far_end(self) -> None:
        """Hold the far connector (Panel B) at its seated pose with a deadbeat mass-scaled spring."""
        if not self._pinned:
            return
        data = self.wire.data
        dt = self.env.physics_dt
        error = self._held_far - data.body_pos_w[:, self._start]
        velocity = data.body_lin_vel_w[:, self._start]
        force = self._far_mass * (HOLD_STIFFNESS * error / (dt * dt) - HOLD_DAMPING * velocity / dt)
        force[:, 2] += self._far_mass[:, 0] * GRAVITY
        force_b = quat_apply_inverse(data.body_quat_w[:, self._start], force).unsqueeze(1)
        self.wire.set_external_force_and_torque(force_b, torch.zeros_like(force_b), body_ids=[self._start])

    def initialize(self) -> None:
        names = self.wire.body_names
        self._start = self.wire.find_bodies("start_port")[0][0]
        self._end = self.wire.find_bodies("end_port")[0][0]
        count = self.assembly_cfg.articulation.num_segments
        self._chain = [self._start, *[names.index(f"seg_{i}") for i in range(count)], self._end]
        self._held_far = self.wire.data.body_pos_w[:, self._start].clone()
        self._far_mass = self.wire.root_physx_view.get_masses()[:, self._start : self._start + 1].to(self.env.device)

    def state(self) -> WireState:
        data = self.wire.data
        pos, quat = data.body_pos_w, data.body_quat_w
        end_pos, end_quat = pos[:, self._end], quat[:, self._end]
        return WireState(
            insertion_end_pos=end_pos,
            insertion_end_quat=end_quat,
            far_end_pos=pos[:, self._start],
            far_end_quat=quat[:, self._start],
            tip_pos=tip_from_root(end_pos, end_quat, self.tip_length),
            centerline=resample_polyline(pos[:, self._chain[1:-1]], self.cfg.centerline_points),
        )

    def cable_contact_paths(self) -> list[str]:
        return [f"{self.prim_path}/seg_{i}" for i in range(self.assembly_cfg.articulation.num_segments)]

    def port_speed(self) -> torch.Tensor:
        data = self.wire.data
        ports = [self._start, self._end]
        speed = data.body_ang_vel_w[:, ports].norm(dim=-1).amax(dim=-1)
        finite = torch.isfinite(data.body_pos_w[:, ports]).all(-1).all(-1)
        return torch.where(finite, speed, torch.nan)

    def reset_plugged(
        self,
        env_ids: torch.Tensor,
        far_port_pos: torch.Tensor,
        far_port_quat: torch.Tensor,
        active_port_pos: torch.Tensor,
        active_port_quat: torch.Tensor,
    ) -> None:
        """Initialize the cable along a smooth 3D PluggedCableArc between two board ports."""
        n = len(env_ids)
        device = self.env.device
        # Cable exits rear of both connector housings along local -X.
        rear = torch.tensor([[-0.018, 0.0, 0.0]], device=device).expand(n, 3)
        far_exit = far_port_pos + quat_apply(far_port_quat, rear)
        act_exit = active_port_pos + quat_apply(active_port_quat, rear)
        exit_dir = quat_apply(far_port_quat, torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(n, 3))
        arc = PluggedCableArc.solve(
            far_exit,
            act_exit,
            cable_length=self.assembly_cfg.articulation.total_length,
            num_points=self.assembly_cfg.articulation.num_segments + 1,
            exit_normal=exit_dir,
            lateral_bow=self.cfg.lateral_bow,
        )
        # Place articulation root at first segment midpoint oriented along initial arc tangent.
        seg0_pos = 0.5 * (arc.points[:, 0] + arc.points[:, 1])
        t0 = arc.tangents[:, 0]
        yaw = torch.atan2(t0[:, 1], t0[:, 0])
        pitch = -torch.asin(t0[:, 2].clamp(-0.999, 0.999))
        root = torch.zeros(n, 13, device=device)
        root[:, :3] = seg0_pos
        root[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw)
        self.wire.write_root_state_to_sim(root, env_ids=env_ids)

        joint_pos = self.wire.data.default_joint_pos[env_ids].clone()
        self.wire.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.wire.reset(env_ids)
        self._held_far[env_ids] = far_port_pos
