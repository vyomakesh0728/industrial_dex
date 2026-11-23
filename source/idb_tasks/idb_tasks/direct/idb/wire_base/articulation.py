"""Articulated capsule-chain wire backend for IDB datacenter cables."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import require_isaaclab

from .math_utils import quat_apply, quat_apply_inverse, quat_inv, quat_mul, resample_polyline
from .wire_state import PluggedCableArc, WireState, tip_from_root

HOLD_STIFFNESS = 0.5
HOLD_DAMPING = 0.5
GRAVITY = 9.81


def _quat_from_rot_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Convert batch of right-handed rotation matrices (..., 3, 3) to wxyz unit quaternions."""
    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(*rot.shape[:-2], 4, device=rot.device, dtype=rot.dtype)

    c0 = trace > 0.0
    s0 = torch.sqrt((trace[c0] + 1.0).clamp_min(1e-12)) * 2.0
    q[c0, 0] = 0.25 * s0
    q[c0, 1] = (m21[c0] - m12[c0]) / s0
    q[c0, 2] = (m02[c0] - m20[c0]) / s0
    q[c0, 3] = (m10[c0] - m01[c0]) / s0

    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s1 = torch.sqrt((1.0 + m00[c1] - m11[c1] - m22[c1]).clamp_min(1e-12)) * 2.0
    q[c1, 0] = (m21[c1] - m12[c1]) / s1
    q[c1, 1] = 0.25 * s1
    q[c1, 2] = (m01[c1] + m10[c1]) / s1
    q[c1, 3] = (m02[c1] + m20[c1]) / s1

    c2 = (~c0) & (~c1) & (m11 > m22)
    s2 = torch.sqrt((1.0 + m11[c2] - m00[c2] - m22[c2]).clamp_min(1e-12)) * 2.0
    q[c2, 0] = (m02[c2] - m20[c2]) / s2
    q[c2, 1] = (m01[c2] + m10[c2]) / s2
    q[c2, 2] = 0.25 * s2
    q[c2, 3] = (m12[c2] + m21[c2]) / s2

    c3 = (~c0) & (~c1) & (~c2)
    s3 = torch.sqrt((1.0 + m22[c3] - m00[c3] - m11[c3]).clamp_min(1e-12)) * 2.0
    q[c3, 0] = (m10[c3] - m01[c3]) / s3
    q[c3, 1] = (m02[c3] + m20[c3]) / s3
    q[c3, 2] = (m12[c3] + m21[c3]) / s3
    q[c3, 3] = 0.25 * s3
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_chain_arc_joint_state(
    arc_points: torch.Tensor,
    root_seg_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute zero-twist Bishop segment frames and relative (rotY, rotZ) joint angles along an arc.

    Args:
        arc_points: (N, num_segments + 1, 3) polyline endpoints of each capsule segment.
        root_seg_index: Index in ``[0, num_segments - 1]`` of the PhysX articulation root segment.

    Returns:
        root_pos: (N, 3) world center of ``seg_{root_seg_index}``.
        root_quat: (N, 4) wxyz world orientation of ``seg_{root_seg_index}``.
        rel_pitch_y: (N, num_segments - 1) relative Y bending angle at each inter-segment joint (rad).
        rel_yaw_z: (N, num_segments - 1) relative Z bending angle at each inter-segment joint (rad).
    """
    n, k_pts, _ = arc_points.shape
    num_segs = k_pts - 1
    device, dtype = arc_points.device, arc_points.dtype

    seg_vecs = arc_points[:, 1:] - arc_points[:, :-1]  # (N, S, 3)
    seg_centers = 0.5 * (arc_points[:, 1:] + arc_points[:, :-1])
    x_axes = seg_vecs / seg_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    # Parallel-transported Bishop frames: minimal twist from segment k-1 to segment k.
    z_ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).expand(n, 3)
    x_ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(n, 3)
    x0 = x_axes[:, 0]
    ref0 = torch.where((x0 * z_ref).sum(dim=-1, keepdim=True).abs() > 0.90, x_ref, z_ref)
    y0 = torch.cross(ref0, x0, dim=-1)
    y0 = y0 / y0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z0 = torch.cross(x0, y0, dim=-1)

    frames = [torch.stack((x0, y0, z0), dim=-1)]
    rel_pitch = []
    rel_yaw = []
    for k in range(1, num_segs):
        r_prev = frames[-1]  # (N, 3, 3)
        x_next_w = x_axes[:, k]
        # Express x_next in previous segment's local frame (where locked rotX=0).
        x_next_local = torch.einsum("nji,nj->ni", r_prev, x_next_w)
        # Intrinsic XYZ (with rx=0): R_y(pitch) @ R_z(yaw) @ [1, 0, 0] = [cp*cy, sy, -sp*cy]
        # Wait: R_y(p) @ R_z(y) @ e_x:
        #   R_z(y) @ [1,0,0] = [cos y, sin y, 0]
        #   R_y(p) @ [cos y, sin y, 0] = [cos p cos y, sin y, -sin p cos y]
        # So yaw = asin(clamp(x_local_y)), pitch = atan2(-x_local_z, x_local_x)!
        yaw = torch.asin(x_next_local[:, 1].clamp(-0.9999, 0.9999))
        pitch = torch.atan2(-x_next_local[:, 2], x_next_local[:, 0].clamp_min(1e-6))
        rel_pitch.append(pitch)
        rel_yaw.append(yaw)

        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        r_rel = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        r_rel[:, 0, 0] = cp * cy
        r_rel[:, 0, 1] = -cp * sy
        r_rel[:, 0, 2] = sp
        r_rel[:, 1, 0] = sy
        r_rel[:, 1, 1] = cy
        r_rel[:, 2, 0] = -sp * cy
        r_rel[:, 2, 1] = sp * sy
        r_rel[:, 2, 2] = cp
        frames.append(torch.bmm(r_prev, r_rel))

    root_pos = seg_centers[:, root_seg_index]
    root_quat = _quat_from_rot_matrix(frames[root_seg_index])
    return root_pos, root_quat, torch.stack(rel_pitch, dim=1), torch.stack(rel_yaw, dim=1)


def sample_physics_scales(
    count: int,
    randomize: bool,
    stiffness_range: tuple[float, float],
    friction_range: tuple[float, float],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-env cable stiffness and friction multipliers."""
    if not randomize:
        ones = torch.ones(count, device=device)
        return ones, ones.clone()
    s_lo, s_hi = stiffness_range
    f_lo, f_hi = friction_range
    s_scale = s_lo + torch.rand(count, device=device) * (s_hi - s_lo)
    f_scale = f_lo + torch.rand(count, device=device) * (f_hi - f_lo)
    return s_scale, f_scale


def resolve_bending_dofs(
    joint_names: list[str], num_segments: int
) -> tuple[dict[int, int], dict[int, int]]:
    """Map each inter-segment joint index ``k - 1`` to its PhysX ``rotY`` (pitch) and ``rotZ`` (yaw) DOF indices.

    PhysX D6 joints with ``rotX`` locked typically expose ``:1`` (rotY) and ``:2`` (rotZ),
    matching WireCraft; if a runtime compacts unlocked DOFs to two entries without ``:1``/``:2``,
    fall back to the last two matching DOFs for the joint prefix.
    """
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    pitch_dofs: dict[int, int] = {}
    yaw_dofs: dict[int, int] = {}
    for k in range(1, num_segments):
        prefix = f"joint_seg_{k}_to_parent"
        p_name = f"{prefix}:1"
        y_name = f"{prefix}:2"
        if p_name in name_to_idx and y_name in name_to_idx:
            pitch_dofs[k - 1] = name_to_idx[p_name]
            yaw_dofs[k - 1] = name_to_idx[y_name]
            continue
        matches = [idx for idx, n in enumerate(joint_names) if n == prefix or n.startswith(f"{prefix}:")]
        if len(matches) >= 2:
            pitch_dofs[k - 1] = matches[-2]
            yaw_dofs[k - 1] = matches[-1]
    return pitch_dofs, yaw_dofs


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
        self._seg_bodies = [names.index(f"seg_{i}") for i in range(count)]
        self._chain = [self._start, *self._seg_bodies, self._end]
        self._held_far = self.wire.data.body_pos_w[:, self._start].clone()
        self._far_mass = self.wire.root_physx_view.get_masses()[:, self._start : self._start + 1].to(self.env.device)

        # Resolve which segment PhysX promoted as the articulation root body.
        root_body = int((self.wire.data.body_pos_w - self.wire.data.root_pos_w.unsqueeze(1)).norm(dim=-1)[0].argmin())
        self._root_seg_idx = self._seg_bodies.index(root_body) if root_body in self._seg_bodies else 0

        # Resolve PhysX D6 bending DOFs (:1 = rotY pitch, :2 = rotZ yaw).
        self._pitch_dofs, self._yaw_dofs = resolve_bending_dofs(list(self.wire.joint_names), count)
        self._base_stiffness = float(self.assembly_cfg.articulation.stiffness)
        self._base_materials = self.wire.root_physx_view.get_material_properties().clone()

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

    def apply_physics_randomization(self, env_ids: torch.Tensor) -> None:
        """Apply per-env cable stiffness and friction randomization when enabled."""
        var = self.env.cfg.variation
        if not var.randomize_cable_physics:
            return
        s_scale, f_scale = sample_physics_scales(
            len(env_ids), True, var.stiffness_scale_range, var.friction_scale_range, device=self.env.device
        )
        all_dofs = sorted(set(self._pitch_dofs.values()) | set(self._yaw_dofs.values()))
        if all_dofs:
            stiff = (self._base_stiffness * s_scale).unsqueeze(-1).expand(-1, len(all_dofs))
            self.wire.write_joint_stiffness_to_sim(stiff, joint_ids=all_dofs, env_ids=env_ids)
        env_ids_cpu = env_ids.cpu()
        mats = self.wire.root_physx_view.get_material_properties()
        f_cpu = f_scale.cpu().view(-1, 1, 1)
        mats[env_ids_cpu, :, :2] = (self._base_materials[env_ids_cpu, :, :2] * f_cpu).clamp(0.1, 5.0)
        self.wire.root_physx_view.set_material_properties(mats, env_ids_cpu)

    def reset_plugged(
        self,
        env_ids: torch.Tensor,
        far_port_pos: torch.Tensor,
        far_port_quat: torch.Tensor,
        active_port_pos: torch.Tensor,
        active_port_quat: torch.Tensor,
    ) -> None:
        """Initialize the articulated cable along a 3D PluggedCableArc with exact joint angles."""
        n = len(env_ids)
        device = self.env.device
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
        root_pos, root_quat, rel_pitch, rel_yaw = compute_chain_arc_joint_state(
            arc.points, root_seg_index=self._root_seg_idx
        )
        root = torch.zeros(n, 13, device=device)
        root[:, :3] = root_pos
        root[:, 3:7] = root_quat
        self.wire.write_root_state_to_sim(root, env_ids=env_ids)

        joint_pos = self.wire.data.default_joint_pos[env_ids].clone()
        for k, dof in self._pitch_dofs.items():
            joint_pos[:, dof] = rel_pitch[:, k]
        for k, dof in self._yaw_dofs.items():
            joint_pos[:, dof] = rel_yaw[:, k]
        self.wire.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.wire.reset(env_ids)
        self.apply_physics_randomization(env_ids)
        self._held_far[env_ids] = far_port_pos
