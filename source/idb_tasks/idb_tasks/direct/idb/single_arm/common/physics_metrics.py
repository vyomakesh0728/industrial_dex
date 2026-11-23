"""Pure PyTorch cable physics, seating, latch, polishing, and disturbance evaluators.

Zero Isaac Sim dependency: runs identically inside Isaac Lab DirectRLEnv and in offline
unit tests. Every state tensor is vectorized over ``num_envs``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ...wire_base.connectors import ConnectorGeometry
from ...wire_base.math_utils import quat_apply, quat_inv


@dataclass
class SeatingResult:
    """Per-env geometric seating and alignment-key test for one port/socket pair."""

    is_seated: torch.Tensor
    key_aligned: torch.Tensor
    tip_depth: torch.Tensor
    axial_ok: torch.Tensor
    inside_bore: torch.Tensor
    forward_ok: torch.Tensor
    roll_cos: torch.Tensor


def evaluate_connector_seating(
    port_pos: torch.Tensor,
    port_quat: torch.Tensor,
    socket_pos: torch.Tensor,
    socket_quat: torch.Tensor,
    geometry: ConnectorGeometry,
    roll_cos_threshold: float = 0.9659,  # cos(15 deg)
    tolerance: float = 1.0e-5,
) -> SeatingResult:
    """Evaluate WireCraft leading-face-in-bore seating plus IDB alignment-key/latch roll check."""
    n = port_pos.shape[0]
    device, dtype = port_pos.device, port_pos.dtype
    face = torch.tensor([(geometry.tip_length, y, z) for y, z in geometry.tip_outline], device=device, dtype=dtype)
    hole = torch.tensor(geometry.hole, device=device, dtype=dtype)
    v_count = face.shape[0]

    inv_sock = quat_inv(socket_quat)
    face_w = quat_apply(port_quat.repeat_interleave(v_count, dim=0), face.repeat(n, 1)).view(n, v_count, 3)
    rel = face_w + (port_pos - socket_pos).unsqueeze(1)
    face_s = quat_apply(inv_sock.repeat_interleave(v_count, dim=0), rel.reshape(-1, 3)).view(n, v_count, 3)

    depth_all = face_s[..., 0] - geometry.socket_mouth_x
    seat_depth = min(geometry.insert_seat_depth, geometry.socket_depth - 0.0015, geometry.tip_length - 0.0015)
    axial_ok = (depth_all.amin(dim=-1) >= seat_depth - tolerance) & (
        depth_all.amax(dim=-1) <= geometry.socket_depth + tolerance
    )

    edge_start = hole
    edge = hole.roll(-1, dims=0) - hole
    point = face_s[..., 1:].unsqueeze(2) - edge_start
    cross = edge[:, 0] * point[..., 1] - edge[:, 1] * point[..., 0]
    inside_bore = (cross >= -tolerance * edge.norm(dim=-1)).all(dim=-1).all(dim=-1)

    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).expand(n, 3)
    port_x_in_sock = quat_apply(inv_sock, quat_apply(port_quat, x_axis))
    forward_ok = port_x_in_sock[:, 0] > 0.90

    # Alignment key / retention latch sits along +Z in both port and socket frames.
    port_z_w = quat_apply(port_quat, z_axis)
    sock_z_w = quat_apply(socket_quat, z_axis)
    roll_cos = (port_z_w * sock_z_w).sum(dim=-1)
    key_aligned = forward_ok & (roll_cos >= roll_cos_threshold)

    tip_center = torch.tensor([[geometry.tip_length, 0.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
    tip_in_sock = quat_apply(inv_sock, port_pos + quat_apply(port_quat, tip_center) - socket_pos)
    tip_depth = tip_in_sock[:, 0] - geometry.socket_mouth_x

    is_seated = axial_ok & inside_bore & key_aligned
    return SeatingResult(
        is_seated=is_seated,
        key_aligned=key_aligned,
        tip_depth=tip_depth,
        axial_ok=axial_ok,
        inside_bore=inside_bore,
        forward_ok=forward_ok,
        roll_cos=roll_cos,
    )


@dataclass
class BendMetrics:
    """Discrete Menger curvature and kink state along a cable centerline."""

    min_radius_m: torch.Tensor
    is_kinked: torch.Tensor
    max_curvature: torch.Tensor


def compute_cable_bend_radius(centerline: torch.Tensor, min_bend_radius: float) -> BendMetrics:
    """Compute minimum local bend radius along (N, K, 3) polyline via Menger curvature."""
    p0, p1, p2 = centerline[:, :-2], centerline[:, 1:-1], centerline[:, 2:]
    a = p1 - p0
    b = p2 - p1
    c = p2 - p0
    la = a.norm(dim=-1).clamp_min(1e-7)
    lb = b.norm(dim=-1).clamp_min(1e-7)
    lc = c.norm(dim=-1).clamp_min(1e-7)
    area2 = torch.cross(a, b, dim=-1).norm(dim=-1)
    curvature = (2.0 * area2) / (la * lb * lc)
    max_k = curvature.amax(dim=-1)
    min_r = 1.0 / max_k.clamp_min(1e-4)
    return BendMetrics(min_radius_m=min_r, is_kinked=min_r < min_bend_radius, max_curvature=max_k)


class RetentionLatchState:
    """Vectorized retention clip / push-pull detent state machine for IDB connectors.

    - Active-clip connectors (RJ45, LC simplex, LC duplex) require gripper latch depression
      before removal; forcing removal while latched flags ``damaged``.
    - Push-pull connectors (SC simplex) release cleanly under axial pull when grasped.
    - Re-insertion engages retention after holding a seated, key-aligned pose for
      ``engage_hold_steps`` control steps.
    """

    def __init__(
        self,
        num_envs: int,
        geometry: ConnectorGeometry,
        device: str | torch.device = "cpu",
        latch_release_force_n: float = 1.5,
        engage_hold_steps: int = 3,
    ) -> None:
        self.num_envs = num_envs
        self.geometry = geometry
        self.device = device
        self.latch_release_force_n = latch_release_force_n
        self.engage_hold_steps = engage_hold_steps

        self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.latched_port_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.released_cleanly = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.damaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._seat_hold_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._candidate_port = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_port_idx: torch.Tensor | None = None,
        initially_latched: bool = True,
    ) -> None:
        self.engaged[env_ids] = initially_latched
        if initial_port_idx is not None:
            self.latched_port_idx[env_ids] = initial_port_idx.to(self.device)
        else:
            self.latched_port_idx[env_ids] = -1
        self.released_cleanly[env_ids] = not initially_latched
        self.damaged[env_ids] = False
        self._seat_hold_count[env_ids] = 0
        self._candidate_port[env_ids] = -1

    def update(
        self,
        is_seated_on_any: torch.Tensor,
        seated_port_idx: torch.Tensor,
        key_aligned: torch.Tensor,
        is_grasped: torch.Tensor,
        gripper_latch_force_n: torch.Tensor,
        axial_pull_force_n: torch.Tensor,
    ) -> None:
        # 1. Active release check while currently latched.
        if self.geometry.requires_latch_release:
            clean_depress = self.engaged & is_grasped & (gripper_latch_force_n >= self.latch_release_force_n)
            self.engaged = torch.where(clean_depress, torch.zeros_like(self.engaged), self.engaged)
            self.released_cleanly |= clean_depress
            # Unplugging while still latched rips/damages the retention clip.
            ripped = self.engaged & (~is_seated_on_any)
            self.damaged |= ripped
            self.engaged = torch.where(ripped, torch.zeros_like(self.engaged), self.engaged)
            self.latched_port_idx = torch.where(
                clean_depress | ripped, torch.full_like(self.latched_port_idx, -1), self.latched_port_idx
            )
        else:
            # SC push-pull shroud releases cleanly when pulled out while grasped.
            pulled_out = self.engaged & (~is_seated_on_any)
            clean_pull = pulled_out & (is_grasped | (axial_pull_force_n <= self.geometry.unplug_force_n * 2.5))
            self.released_cleanly |= clean_pull
            self.damaged |= pulled_out & (~clean_pull)
            self.engaged = torch.where(pulled_out, torch.zeros_like(self.engaged), self.engaged)
            self.latched_port_idx = torch.where(
                pulled_out, torch.full_like(self.latched_port_idx, -1), self.latched_port_idx
            )

        # 2. Re-engagement check when seated & aligned in a port and currently unlatched.
        can_latch = (~self.engaged) & is_seated_on_any & key_aligned & (seated_port_idx >= 0)
        same_port = can_latch & (seated_port_idx == self._candidate_port)
        self._seat_hold_count = torch.where(
            same_port,
            self._seat_hold_count + 1,
            torch.where(can_latch, torch.ones_like(self._seat_hold_count), torch.zeros_like(self._seat_hold_count)),
        )
        self._candidate_port = torch.where(can_latch, seated_port_idx, torch.full_like(self._candidate_port, -1))

        newly_latched = can_latch & (self._seat_hold_count >= self.engage_hold_steps)
        self.engaged |= newly_latched
        self.latched_port_idx = torch.where(newly_latched, seated_port_idx, self.latched_port_idx)


class FigureEightPolishingTracker:
    """Vectorized figure-eight ferrule polishing evaluator for IDB Tasks 4 and 5.

    Tracks:
    - Ferrule placement within the circular polishing pad disc (supports 1 or 2 ferrules).
    - Contact pressure inside the IDB 1-2 N specification and excessive-force damage.
    - Continuous contact without lift-off once polishing begins.
    - True figure-eight cycles via alternating smooth CW and CCW tangent sweeps (>= 183 deg
      each with minimum lobe arc length), rejecting stationary presses, circles, and linear strokes.
    """

    def __init__(
        self,
        num_envs: int,
        required_cycles: int = 10,
        pressure_min_n: float = 1.0,
        pressure_max_n: float = 2.0,
        contact_min_n: float = 0.25,
        damage_force_n: float = 6.0,
        contact_dist_tol_m: float = 0.003,
        lobe_turn_rad: float = 3.2,
        min_lobe_arc_m: float = 0.012,
        max_step_turn_rad: float = 0.85,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.required_cycles = required_cycles
        self.pressure_min_n = pressure_min_n
        self.pressure_max_n = pressure_max_n
        self.contact_min_n = contact_min_n
        self.damage_force_n = damage_force_n
        self.contact_dist_tol_m = contact_dist_tol_m
        self.lobe_turn_rad = lobe_turn_rad
        self.min_lobe_arc_m = min_lobe_arc_m
        self.max_step_turn_rad = max_step_turn_rad
        self.device = device

        self.positioned_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.in_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.polishing_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cycles_completed = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_breaks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pressure_in_band_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ferrule_damaged = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Tangent-turn accumulator state.
        self._has_prev_pos = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._has_prev_dir = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._prev_uv = torch.zeros(num_envs, 2, device=device)
        self._prev_dir = torch.zeros(num_envs, 2, device=device)
        self._turn_sign = torch.zeros(num_envs, dtype=torch.long, device=device)  # -1, 0, +1
        self._accum_turn = torch.zeros(num_envs, device=device)
        self._accum_arc = torch.zeros(num_envs, device=device)
        self._last_completed_lobe = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.positioned_ever[env_ids] = False
        self.in_contact[env_ids] = False
        self.polishing_started[env_ids] = False
        self.cycles_completed[env_ids] = 0
        self.contact_breaks[env_ids] = 0
        self.contact_steps[env_ids] = 0
        self.pressure_in_band_steps[env_ids] = 0
        self.ferrule_damaged[env_ids] = False
        self._has_prev_pos[env_ids] = False
        self._has_prev_dir[env_ids] = False
        self._prev_uv[env_ids] = 0.0
        self._prev_dir[env_ids] = 0.0
        self._turn_sign[env_ids] = 0
        self._accum_turn[env_ids] = 0.0
        self._accum_arc[env_ids] = 0.0
        self._last_completed_lobe[env_ids] = 0

    @property
    def continuous_contact_ok(self) -> torch.Tensor:
        return self.polishing_started & (self.contact_breaks == 0)

    @property
    def pressure_ok(self) -> torch.Tensor:
        ratio = self.pressure_in_band_steps.float() / self.contact_steps.clamp_min(1).float()
        return (self.contact_steps > 0) & (ratio >= 0.80) & (~self.ferrule_damaged)

    @property
    def cycles_ok(self) -> torch.Tensor:
        return self.cycles_completed >= self.required_cycles

    def update(
        self,
        ferrule_pos_w: torch.Tensor,
        pad_center_w: torch.Tensor,
        pad_normal_w: torch.Tensor,
        pad_radius_m: float,
        contact_force_n: torch.Tensor,
    ) -> None:
        """Advance figure-eight polishing tracker by one control step."""
        # Construct orthonormal tangent basis (e_u, e_v) on the pad face.
        n_vec = pad_normal_w / pad_normal_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        up = torch.tensor([[0.0, 0.0, 1.0]], device=ferrule_pos_w.device, dtype=ferrule_pos_w.dtype).expand_as(n_vec)
        e_u = torch.cross(up, n_vec, dim=-1)
        e_u = e_u / e_u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_v = torch.cross(n_vec, e_u, dim=-1)

        rel = ferrule_pos_w - pad_center_w.unsqueeze(1)  # (N, F, 3)
        normal_dist = (rel * n_vec.unsqueeze(1)).sum(dim=-1)  # (N, F)
        u_all = (rel * e_u.unsqueeze(1)).sum(dim=-1)
        v_all = (rel * e_v.unsqueeze(1)).sum(dim=-1)
        radial_all = torch.hypot(u_all, v_all)

        # All ferrules (1 for SC, 2 for LC duplex) must lie on the pad surface.
        positioned = (normal_dist.abs() <= self.contact_dist_tol_m).all(dim=-1) & (
            radial_all <= pad_radius_m
        ).all(dim=-1)
        self.positioned_ever |= positioned

        in_contact = positioned & (contact_force_n >= self.contact_min_n)
        was_polishing = self.polishing_started & (~self.cycles_ok)
        broke_contact = was_polishing & self.in_contact & (~in_contact)
        self.contact_breaks += broke_contact.long()
        self.in_contact = in_contact

        self.ferrule_damaged |= in_contact & (contact_force_n > self.damage_force_n)
        self.contact_steps += in_contact.long()
        in_band = in_contact & (contact_force_n >= self.pressure_min_n) & (contact_force_n <= self.pressure_max_n)
        self.pressure_in_band_steps += in_band.long()

        # Track centroid UV of ferrule(s) for planar figure-8 loop detection.
        uv = torch.stack((u_all.mean(dim=-1), v_all.mean(dim=-1)), dim=-1)  # (N, 2)
        delta = uv - self._prev_uv
        step_len = delta.norm(dim=-1)
        moved = in_contact & self._has_prev_pos & (step_len >= 0.00025)

        curr_dir = delta / step_len.clamp_min(1e-9).unsqueeze(-1)
        have_turn = moved & self._has_prev_dir
        cross = self._prev_dir[:, 0] * curr_dir[:, 1] - self._prev_dir[:, 1] * curr_dir[:, 0]
        dot = (self._prev_dir * curr_dir).sum(dim=-1)
        d_theta = torch.atan2(cross, dot)
        smooth_turn = have_turn & (d_theta.abs() <= self.max_step_turn_rad) & (d_theta.abs() >= 0.005)
        step_sign = torch.where(d_theta > 0, torch.ones_like(self._turn_sign), -torch.ones_like(self._turn_sign))

        # Accumulate continuous same-sign turning sweep; reset lobe accumulator on inflection.
        same_sign = smooth_turn & ((self._turn_sign == 0) | (step_sign == self._turn_sign))
        flipped = smooth_turn & (self._turn_sign != 0) & (step_sign != self._turn_sign)

        self._accum_turn = torch.where(
            same_sign,
            self._accum_turn + d_theta.abs(),
            torch.where(flipped, d_theta.abs(), self._accum_turn),
        )
        self._accum_arc = torch.where(
            same_sign,
            self._accum_arc + step_len,
            torch.where(flipped, step_len, self._accum_arc),
        )
        self._turn_sign = torch.where(smooth_turn, step_sign, self._turn_sign)

        # Mark polishing active once meaningful pad motion occurs under contact.
        self.polishing_started |= moved & (self._accum_arc >= 0.004)

        # Detect lobe completion: >= lobe_turn_rad smooth sweep and >= min_lobe_arc_m path length.
        lobe_done = (
            smooth_turn
            & (self._accum_turn >= self.lobe_turn_rad)
            & (self._accum_arc >= self.min_lobe_arc_m)
        )
        # Opposite lobe after a prior completed lobe completes ONE figure-eight cycle!
        opposite_lobe = lobe_done & (self._last_completed_lobe != 0) & (self._turn_sign == -self._last_completed_lobe)
        self.cycles_completed += opposite_lobe.long()

        # After opposite lobe completes a full figure-8, reset last_completed_lobe to 0 so the next
        # cycle requires BOTH a fresh lobe A and fresh lobe B (preventing double-counting).
        self._last_completed_lobe = torch.where(
            opposite_lobe,
            torch.zeros_like(self._last_completed_lobe),
            torch.where(lobe_done, self._turn_sign, self._last_completed_lobe),
        )
        self._accum_turn = torch.where(lobe_done, torch.zeros_like(self._accum_turn), self._accum_turn)
        self._accum_arc = torch.where(lobe_done, torch.zeros_like(self._accum_arc), self._accum_arc)

        # Reset kinematic history when contact is lost.
        self._prev_uv = torch.where(in_contact.unsqueeze(-1), uv, self._prev_uv)
        self._has_prev_pos = in_contact
        self._prev_dir = torch.where(moved.unsqueeze(-1), curr_dir, self._prev_dir)
        self._has_prev_dir = torch.where(in_contact, self._has_prev_dir | moved, torch.zeros_like(self._has_prev_dir))


class NeighborDisturbanceMonitor:
    """Tracks neighboring cable integrity and excessive-force damage on IDB Board #1."""

    def __init__(
        self,
        num_envs: int,
        num_neighbors: int = 0,
        disturbance_pos_tol_m: float = 0.004,
        damage_force_n: float = 35.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.num_neighbors = num_neighbors
        self.disturbance_pos_tol_m = disturbance_pos_tol_m
        self.damage_force_n = damage_force_n
        self.device = device

        self.nominal_positions = torch.zeros(num_envs, max(num_neighbors, 0), 3, device=device)
        self.disturbed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.excessive_force = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.max_neighbor_disp_m = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor, nominal_positions: torch.Tensor | None = None) -> None:
        if nominal_positions is not None and self.num_neighbors > 0:
            self.nominal_positions[env_ids] = nominal_positions.to(self.device)
        self.disturbed[env_ids] = False
        self.excessive_force[env_ids] = False
        self.max_neighbor_disp_m[env_ids] = 0.0

    def update(
        self,
        current_positions: torch.Tensor | None = None,
        max_active_contact_force_n: torch.Tensor | None = None,
    ) -> None:
        if current_positions is not None and self.num_neighbors > 0 and current_positions.shape[1] > 0:
            disp = (current_positions - self.nominal_positions).norm(dim=-1).amax(dim=-1)
            self.max_neighbor_disp_m = torch.maximum(self.max_neighbor_disp_m, disp)
            self.disturbed |= disp > self.disturbance_pos_tol_m
        if max_active_contact_force_n is not None:
            self.excessive_force |= max_active_contact_force_n > self.damage_force_n

    @property
    def integrity_ok(self) -> torch.Tensor:
        return (~self.disturbed) & (~self.excessive_force)
