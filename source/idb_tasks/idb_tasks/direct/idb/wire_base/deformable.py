"""FEM deformable wire backend with rigid vertex-captured connector ports."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import require_isaaclab

from .math_utils import quat_apply, resample_polyline
from .wire_state import PluggedCableArc, WireState, tip_from_root


def map_deformable_nodes_to_arc(
    default_nodal_pos: torch.Tensor,
    arc_points: torch.Tensor,
    arc_tangents: torch.Tensor,
) -> torch.Tensor:
    """Map rest FEM cylinder nodes (N, V, 3) onto solved 3D arc points/tangents (N, K, 3).

    Subtracts the rest cylinder's Y/Z centerline offset before applying cross-section
    displacements so world spawn position never leaks into the bent nodal coordinates.
    Uses an orthogonal fallback axis whenever tangents are near-vertical.
    """
    device, dtype = default_nodal_pos.device, default_nodal_pos.dtype
    k_samples = arc_points.shape[1]

    rest0 = default_nodal_pos[0]  # (V, 3)
    rest_x = rest0[:, 0]
    rest_u = (rest_x - rest_x.min()) / (rest_x.max() - rest_x.min()).clamp_min(1e-6)
    # Remove spawn-position offset in Y and Z so local cross-section offsets stay within [-radius, +radius].
    rest_yz_local = rest0[:, 1:3] - rest0[:, 1:3].mean(dim=0, keepdim=True)

    u_idx = (rest_u * (k_samples - 1)).clamp(0.0, k_samples - 1.0001)
    lo = u_idx.long()
    frac = (u_idx - lo.float()).view(1, -1, 1)
    centers = torch.lerp(arc_points[:, lo], arc_points[:, lo + 1], frac)
    tans = torch.lerp(arc_tangents[:, lo], arc_tangents[:, lo + 1], frac)
    tans = tans / tans.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    up_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3).expand_as(tans)
    fallback_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 3).expand_as(tans)
    ref = torch.where((tans * up_z).sum(dim=-1, keepdim=True).abs() > 0.90, fallback_x, up_z)
    n1 = torch.cross(ref, tans, dim=-1)
    n1 = n1 / n1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n2 = torch.cross(tans, n1, dim=-1)
    yz = rest_yz_local.unsqueeze(0)
    return centers + yz[..., :1] * n1 + yz[..., 1:2] * n2


class DeformableWire:
    """FEM soft-body cable with independent rigid ``start_port`` and ``end_port`` connectors."""

    def __init__(self, env, prim_path: str | None = None, assembly_cfg=None, tip_length: float | None = None, name: str = "wire"):
        self.env = env
        self.cfg = env.cfg.wire
        self.prim_path = prim_path or env.cfg.wire_prim_path
        self.assembly_cfg = assembly_cfg or env.cfg.wire_spawn
        self.tip_length = env.connector.tip_length if tip_length is None else tip_length
        self.scene_key = name
        self._pinned = False

    def spawn(self) -> None:
        require_isaaclab("DeformableWire.spawn")
        from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg  # type: ignore

        assembly = self.assembly_cfg
        assembly.func(self.prim_path, assembly, translation=tuple(self.cfg.root_pos))
        self.wire = DeformableObject(DeformableObjectCfg(prim_path=f"{self.prim_path}/wire", spawn=None))
        self.start_port = RigidObject(RigidObjectCfg(prim_path=f"{self.prim_path}/start_port", spawn=None))
        self.end_port = RigidObject(RigidObjectCfg(prim_path=f"{self.prim_path}/end_port", spawn=None))
        scene = self.env.scene
        scene.deformable_objects[self.scene_key] = self.wire
        scene.rigid_objects[f"{self.scene_key}_start_port"] = self.start_port
        scene.rigid_objects[f"{self.scene_key}_end_port"] = self.end_port

    def pin_far_end(self) -> None:
        self._pinned = True

    def hold_far_end(self) -> None:
        """Keep the dynamic far port anchored at its Panel B receptacle pose."""
        if not self._pinned:
            return
        state = self.start_port.data.root_state_w.clone()
        state[:, :3] = self._held_far
        state[:, 3:7] = self._held_far_quat
        state[:, 7:13] = 0.0
        self.start_port.write_root_state_to_sim(state)

    def initialize(self) -> None:
        rest_x = self.wire.data.default_nodal_state_w[0, :, 0]
        bins = torch.linspace(rest_x.min(), rest_x.max(), self.assembly_cfg.deformable.num_segments + 1, device=rest_x.device)
        idx = torch.bucketize(rest_x, bins[1:-1])
        weights = torch.nn.functional.one_hot(idx, bins.numel() - 1).float()
        weights = weights[:, weights.sum(0) > 0]
        self._bin_weights = weights / weights.sum(0, keepdim=True)
        self._held_far = self.start_port.data.root_pos_w.clone()
        self._held_far_quat = self.start_port.data.root_quat_w.clone()

    def state(self) -> WireState:
        end, start = self.end_port.data, self.start_port.data
        bins = torch.einsum("nkd,kb->nbd", self.wire.data.nodal_pos_w, self._bin_weights)
        return WireState(
            insertion_end_pos=end.root_pos_w,
            insertion_end_quat=end.root_quat_w,
            far_end_pos=start.root_pos_w,
            far_end_quat=start.root_quat_w,
            tip_pos=tip_from_root(end.root_pos_w, end.root_quat_w, self.tip_length),
            centerline=resample_polyline(bins, self.cfg.centerline_points),
        )

    def cable_contact_paths(self) -> list[str]:
        return []

    def port_speed(self) -> torch.Tensor:
        end, start = self.end_port.data, self.start_port.data
        speed = torch.maximum(end.root_ang_vel_w.norm(dim=-1), start.root_ang_vel_w.norm(dim=-1))
        finite = torch.isfinite(end.root_pos_w).all(-1) & torch.isfinite(start.root_pos_w).all(-1)
        return torch.where(finite, speed, torch.nan)

    def reset_plugged(
        self,
        env_ids: torch.Tensor,
        far_port_pos: torch.Tensor,
        far_port_quat: torch.Tensor,
        active_port_pos: torch.Tensor,
        active_port_quat: torch.Tensor,
    ) -> None:
        """Map FEM cable nodes onto the solved 3D PluggedCableArc between both ports."""
        n = len(env_ids)
        device = self.env.device
        rear = torch.tensor([[-0.016, 0.0, 0.0]], device=device).expand(n, 3)
        far_exit = far_port_pos + quat_apply(far_port_quat, rear)
        act_exit = active_port_pos + quat_apply(active_port_quat, rear)
        exit_dir = quat_apply(far_port_quat, torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(n, 3))
        arc = PluggedCableArc.solve(
            far_exit,
            act_exit,
            cable_length=self.assembly_cfg.deformable.total_length,
            num_points=65,
            exit_normal=exit_dir,
            lateral_bow=self.cfg.lateral_bow,
        )
        nodal = self.wire.data.default_nodal_state_w[env_ids].clone()
        nodal[..., :3] = map_deformable_nodes_to_arc(nodal[..., :3], arc.points, arc.tangents)
        nodal[..., 3:] = 0.0
        self.wire.write_nodal_state_to_sim(nodal, env_ids)
        self.wire.reset(env_ids)

        for port, pos, quat in ((self.start_port, far_port_pos, far_port_quat), (self.end_port, active_port_pos, active_port_quat)):
            st = torch.zeros(n, 13, device=device)
            st[:, :3] = pos
            st[:, 3:7] = quat
            port.write_root_state_to_sim(st, env_ids=env_ids)
            port.reset(env_ids)
        self._held_far[env_ids] = far_port_pos
        self._held_far_quat[env_ids] = far_port_quat
