"""Runtime controller for IDB Board #1: 32-port poses, kinematic PhysX sync, polishing pad sensor, and neighbor cables."""

from __future__ import annotations

import math

import torch

from idb_assets.board_spec import BoardSpec, load_board_spec
from idb_tasks.utils.isaac_compat import require_isaaclab

from ...wire_base.articulation import ArticulationWire
from ...wire_base.connectors import ConnectorCfg, ConnectorGeometry
from ...wire_base.deformable import DeformableWire
from ...wire_base.math_utils import quat_apply, quat_from_euler_xyz, quat_mul
from ...wire_base.wire_cfg import WireSceneCfg


class IDBBoardRuntime:
    """Manages IDB Board #1 PhysX rigid bodies, polishing pad force sensing, and background population cables."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.board
        self.variation = env.cfg.variation
        self.spec: BoardSpec = load_board_spec(self.cfg.manifest_path)
        self.port_ids: list[str] = sorted(self.spec.ports.keys())
        self.port_index: dict[str, int] = {pid: idx for idx, pid in enumerate(self.port_ids)}
        self.neighbor_wires: list[tuple[str, str, ArticulationWire | DeformableWire]] = []
        self.polishing_contact = None
        self.frame_obj = None
        self.polishing_pad_obj = None
        self.port_objs: dict[str, object] = {}
        self.clip_objs: dict[str, object] = {}

    def spawn(self) -> None:
        require_isaaclab("IDBBoardRuntime.spawn")
        from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore[import-not-found]
        from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore[import-not-found]

        board_cfg = self.env.cfg.board_spawn
        board_quat = self._yaw_quat_tuple(self.cfg.board_yaw_rad)
        board_prim = self.env.cfg.board_prim_path
        board_cfg.func(
            board_prim,
            board_cfg,
            translation=tuple(self.cfg.board_pos),
            orientation=board_quat,
        )

        # Register RigidObject handles so per-env board pose randomization moves physical PhysX bodies + USD prims.
        self.frame_obj = RigidObject(RigidObjectCfg(prim_path=f"{board_prim}/frame", spawn=None))
        self.env.scene.rigid_objects["idb_board_frame"] = self.frame_obj

        self.polishing_pad_obj = RigidObject(RigidObjectCfg(prim_path=f"{board_prim}/polishing_pad", spawn=None))
        self.env.scene.rigid_objects["idb_board_polishing_pad"] = self.polishing_pad_obj

        for pid in self.port_ids:
            obj = RigidObject(RigidObjectCfg(prim_path=f"{board_prim}/ports/{pid}", spawn=None))
            self.port_objs[pid] = obj
            self.env.scene.rigid_objects[f"idb_board_port_{pid}"] = obj

        if self.cfg.include_cable_clips:
            for clip in self.spec.cable_clips:
                obj = RigidObject(RigidObjectCfg(prim_path=f"{board_prim}/cable_clips/{clip.name}", spawn=None))
                self.clip_objs[clip.name] = obj
                self.env.scene.rigid_objects[f"idb_board_{clip.name}"] = obj

        if self.env.cfg.sensors.enable_polishing_contact_sensor:
            self.polishing_contact = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{board_prim}/polishing_pad",
                    update_period=0.0,
                    history_length=3,
                    filter_prim_paths_expr=[f"{self.env.cfg.wire_prim_path}/end_port"],
                )
            )
            self.env.scene.sensors["polishing_pad_contact"] = self.polishing_contact

        # Spawn background population cables as siblings under /World/envs/env_.* (not nested under board Xform).
        env_root = board_prim.rsplit("/", 1)[0]
        for idx, (far_p, act_p, preset_name) in enumerate(self.env.cfg.resolved_neighbor_cables):
            w_cfg = WireSceneCfg(kind=self.env.cfg.wire.kind)
            w_cfg.apply_preset_defaults(preset_name)
            c_cfg = ConnectorCfg(
                family=self.spec.port(act_p).family,
                far_family=self.spec.port(far_p).family,
            )
            assembly = w_cfg.assembly_cfg(c_cfg)
            prim_path = f"{env_root}/neighbor_cable_{idx}"
            geom = ConnectorGeometry.from_family(c_cfg.family)
            backend_cls = ArticulationWire if w_cfg.kind == "articulation" else DeformableWire
            backend = backend_cls(
                self.env,
                prim_path=prim_path,
                assembly_cfg=assembly,
                tip_length=geom.tip_length,
                name=f"neighbor_wire_{idx}",
            )
            backend.spawn()
            backend.pin_far_end()
            self.neighbor_wires.append((far_p, act_p, backend))

    @staticmethod
    def _yaw_quat_tuple(yaw_rad: float) -> tuple[float, float, float, float]:
        return (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))

    def initialize(self) -> None:
        n = self.env.num_envs
        device = self.env.device
        origins = self.env.scene.env_origins
        base_pos = torch.tensor(self.cfg.board_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(n, 3)
        self.board_pos_w = origins + base_pos
        yaw = torch.full((n,), float(self.cfg.board_yaw_rad), device=device)
        z = torch.zeros_like(yaw)
        self.board_quat_w = quat_from_euler_xyz(z, z, yaw)

        self._local_port_pos = torch.tensor(
            [self.spec.ports[pid].mouth_pos for pid in self.port_ids], device=device, dtype=torch.float32
        )
        self._local_port_quat = torch.tensor(
            [self.spec.ports[pid].socket_quat for pid in self.port_ids], device=device, dtype=torch.float32
        )
        self._local_pad_center = torch.tensor(self.spec.polishing_pad.center, device=device, dtype=torch.float32)
        self._local_pad_normal = torch.tensor(self.spec.polishing_pad.normal, device=device, dtype=torch.float32)
        self._local_clip_poses = {
            c.name: (
                torch.tensor(c.pos, device=device, dtype=torch.float32),
                torch.tensor(c.quat, device=device, dtype=torch.float32),
            )
            for c in self.spec.cable_clips
        }

        for _, _, backend in self.neighbor_wires:
            backend.initialize()

    def _sync_board_bodies_to_sim(self, env_ids: torch.Tensor) -> None:
        """Write updated kinematic world poses to PhysX for frame, polishing pad, all 32 sockets, and clips."""
        if self.frame_obj is None:
            return
        n = len(env_ids)
        device = self.env.device
        pos_b = self.board_pos_w[env_ids]
        quat_b = self.board_quat_w[env_ids]

        def _write(obj, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(n, 13, device=device)
            st[:, :3] = pos
            st[:, 3:7] = quat
            obj.write_root_state_to_sim(st, env_ids=env_ids)
            obj.reset(env_ids)

        _write(self.frame_obj, pos_b, quat_b)
        pad_pos = pos_b + quat_apply(quat_b, self._local_pad_center.unsqueeze(0).expand(n, 3))
        _write(self.polishing_pad_obj, pad_pos, quat_b)

        for idx, pid in enumerate(self.port_ids):
            p_pos = pos_b + quat_apply(quat_b, self._local_port_pos[idx].unsqueeze(0).expand(n, 3))
            p_quat = quat_mul(quat_b, self._local_port_quat[idx].unsqueeze(0).expand(n, 4))
            _write(self.port_objs[pid], p_pos, p_quat)

        for cname, obj in self.clip_objs.items():
            lp, lq = self._local_clip_poses[cname]
            c_pos = pos_b + quat_apply(quat_b, lp.unsqueeze(0).expand(n, 3))
            c_quat = quat_mul(quat_b, lq.unsqueeze(0).expand(n, 4))
            _write(obj, c_pos, c_quat)

    def port_world_pose(self, port_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.port_index[port_id]
        n = self.board_pos_w.shape[0]
        local_p = self._local_port_pos[idx].unsqueeze(0).expand(n, 3)
        local_q = self._local_port_quat[idx].unsqueeze(0).expand(n, 4)
        pos_w = self.board_pos_w + quat_apply(self.board_quat_w, local_p)
        quat_w = quat_mul(self.board_quat_w, local_q)
        return pos_w, quat_w

    def all_ports_world_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.board_pos_w.shape[0]
        p_count = len(self.port_ids)
        b_q = self.board_quat_w.repeat_interleave(p_count, dim=0)
        p_rot = quat_apply(b_q, self._local_port_pos.repeat(n, 1)).view(n, p_count, 3)
        q_rot = quat_mul(b_q, self._local_port_quat.repeat(n, 1)).view(n, p_count, 4)
        return self.board_pos_w.unsqueeze(1) + p_rot, q_rot

    def seated_connector_pose(self, port_id: str, geometry: ConnectorGeometry) -> tuple[torch.Tensor, torch.Tensor]:
        sock_pos, sock_quat = self.port_world_pose(port_id)
        # Seat 1.0 mm past insert_seat_depth (leaving >= 2.5 mm clearance to socket backstop).
        offset_x = geometry.socket_mouth_x + geometry.insert_seat_depth + 0.001 - geometry.tip_length
        offset = torch.zeros_like(sock_pos)
        offset[:, 0] = offset_x
        return sock_pos + quat_apply(sock_quat, offset), sock_quat

    def polishing_pad_world_pose(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        n = self.board_pos_w.shape[0]
        center_w = self.board_pos_w + quat_apply(self.board_quat_w, self._local_pad_center.unsqueeze(0).expand(n, 3))
        normal_w = quat_apply(self.board_quat_w, self._local_pad_normal.unsqueeze(0).expand(n, 3))
        return center_w, normal_w, self.spec.polishing_pad.radius

    def polishing_contact_force_n(self) -> torch.Tensor:
        if self.polishing_contact is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        return torch.norm(self.polishing_contact.data.force_matrix_w, dim=-1).sum(dim=[-1, -2])

    def hold_neighbor_cables(self) -> None:
        for _, _, backend in self.neighbor_wires:
            backend.hold_far_end()

    def neighbor_active_positions(self) -> torch.Tensor | None:
        if not self.neighbor_wires:
            return None
        return torch.stack([b.state().insertion_end_pos for _, _, b in self.neighbor_wires], dim=1)

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        """Reset board pose (syncing all physical bodies to PhysX) and background neighbor cables."""
        count = len(env_ids)
        device = self.env.device
        origins = self.env.scene.env_origins[env_ids]
        base = torch.tensor(self.cfg.board_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(count, 3)
        pos = origins + base
        yaw = torch.full((count,), float(self.cfg.board_yaw_rad), device=device)

        if self.variation.randomize_board_pose:
            half_xy = torch.tensor(self.variation.board_xy_range, device=device, dtype=torch.float32)
            pos[:, :2] += (torch.rand(count, 2, device=device) * 2.0 - 1.0) * half_xy
            yaw = yaw + (torch.rand(count, device=device) * 2.0 - 1.0) * float(self.variation.board_yaw_range_rad)

        z = torch.zeros_like(yaw)
        self.board_pos_w[env_ids] = pos
        self.board_quat_w[env_ids] = quat_from_euler_xyz(z, z, yaw)
        self._sync_board_bodies_to_sim(env_ids)

        nominal_neighbors = []
        for far_p, act_p, backend in self.neighbor_wires:
            far_geom = ConnectorGeometry.from_family(self.spec.port(far_p).family)
            act_geom = ConnectorGeometry.from_family(self.spec.port(act_p).family)
            f_pos, f_quat = self.seated_connector_pose(far_p, far_geom)
            a_pos, a_quat = self.seated_connector_pose(act_p, act_geom)
            backend.reset_plugged(env_ids, f_pos[env_ids], f_quat[env_ids], a_pos[env_ids], a_quat[env_ids])
            nominal_neighbors.append(a_pos[env_ids])
        return torch.stack(nominal_neighbors, dim=1) if nominal_neighbors else None
