"""Backend-independent wire state, ferrule coordinates, and 3D plugged-cable arch solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .connectors import ConnectorGeometry
from .math_utils import quat_apply, resample_polyline


@dataclass
class WireState:
    """World-frame state shared by articulated and FEM deformable cables."""

    insertion_end_pos: torch.Tensor
    """(N, 3) world position of the active manipulation connector shoulder."""
    insertion_end_quat: torch.Tensor
    """(N, 4) wxyz world orientation of the active manipulation connector."""
    far_end_pos: torch.Tensor
    """(N, 3) world position of the far connector (anchored in Panel B)."""
    far_end_quat: torch.Tensor
    """(N, 4) wxyz world orientation of the far connector."""
    tip_pos: torch.Tensor
    """(N, 3) center of the active connector's leading face."""
    centerline: torch.Tensor
    """(N, K, 3) arc-length resampled cable centerline from far end to active end."""


def tip_from_root(root_pos: torch.Tensor, root_quat: torch.Tensor, tip_length: float) -> torch.Tensor:
    offset = torch.zeros_like(root_pos)
    offset[:, 0] = tip_length
    return root_pos + quat_apply(root_quat, offset)


def ferrule_positions(root_pos: torch.Tensor, root_quat: torch.Tensor, geometry: ConnectorGeometry) -> torch.Tensor:
    """Return world positions of all optical ferrule tips, shape (N, F, 3).

    Non-optical connectors (RJ45) return the single leading-face center with F=1.
    """
    n = root_pos.shape[0]
    if geometry.ferrule is None:
        return tip_from_root(root_pos, root_quat, geometry.tip_length).unsqueeze(1)
    f = geometry.ferrule
    local = torch.tensor(
        [(geometry.tip_length + f.protrusion, y_off, f.z_offset) for y_off in f.y_offsets],
        device=root_pos.device,
        dtype=root_pos.dtype,
    )
    k = local.shape[0]
    rot_vec = quat_apply(root_quat.repeat_interleave(k, dim=0), local.repeat(n, 1)).view(n, k, 3)
    return root_pos.unsqueeze(1) + rot_vec


def _fabrik_exact_segments(pts: torch.Tensor, seg_len: float, iterations: int = 30) -> torch.Tensor:
    """Project interior vertices so every polyline segment has exact chord length ``seg_len``.

    Keeps the first two vertices ``(0, 1)`` and last two vertices ``(-2, -1)`` fixed so
    connector-adjacent end segments stay strictly collinear with their socket axes.
    """
    p = pts.clone()
    if p.shape[1] <= 4:
        return p
    q0 = p[:, 1].clone()
    q1 = p[:, -2].clone()
    last_interior = p.shape[1] - 2
    for _ in range(iterations):
        p[:, 1] = q0
        for i in range(2, last_interior):
            d = p[:, i] - p[:, i - 1]
            p[:, i] = p[:, i - 1] + seg_len * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        p[:, last_interior] = q1
        for i in range(last_interior - 1, 1, -1):
            d = p[:, i] - p[:, i + 1]
            p[:, i] = p[:, i + 1] + seg_len * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return p


@dataclass
class PluggedCableArc:
    """Smooth 3D elastica arch connecting two ports on the vertical IDB board.

    Both cable ends exit straight along their port axes (-X_board), bow outward in -X
    (and optionally slightly downward/sideways like a real datacenter patch cord), and
    match the exact physical segment length without stretch or board penetration.
    """

    points: torch.Tensor
    """(N, K, 3) equally spaced points along the solved 3D arch from far end to active end."""
    tangents: torch.Tensor
    """(N, K, 3) unit tangent vectors along the arch."""
    bulge_m: torch.Tensor
    """(N,) solved forward bow distance in meters."""

    @classmethod
    def solve(
        cls,
        far_cable_start: torch.Tensor,
        active_cable_end: torch.Tensor,
        cable_length: float,
        num_points: int = 41,
        exit_normal: torch.Tensor | None = None,
        exit_normal_active: torch.Tensor | None = None,
        lateral_bow: tuple[float, float] = (0.0, -0.025),
        iterations: int = 24,
        enforce_exact_segments: bool = True,
    ) -> PluggedCableArc:
        """Solve a smooth C1 3D quintic Bezier arch with straight port-aligned end segments."""
        n = far_cable_start.shape[0]
        device, dtype = far_cable_start.device, far_cable_start.dtype
        if exit_normal is None:
            exit_far = torch.tensor([[-1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
        else:
            exit_far = exit_normal / exit_normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if exit_normal_active is None:
            exit_act = exit_far
        else:
            exit_act = exit_normal_active / exit_normal_active.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        num_segs = max(num_points - 1, 1)
        seg_len = cable_length / num_segs
        q0 = far_cable_start + seg_len * exit_far
        q1 = active_cable_end + seg_len * exit_act
        inner_len = max(cable_length - 2.0 * seg_len, 1e-4)
        inner_pts = max(num_points - 2, 2)

        t = torch.linspace(0.0, 1.0, max(inner_pts * 3, 61), device=device, dtype=dtype).view(1, -1, 1)
        u = 1.0 - t
        c0 = u**5
        c1 = 5.0 * (u**4) * t
        c2 = 10.0 * (u**3) * (t**2)
        c3 = 10.0 * (u**2) * (t**3)
        c4 = 5.0 * u * (t**4)
        c5 = t**5

        exit_avg = exit_far + exit_act
        exit_avg = exit_avg / exit_avg.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        span = q1 - q0
        span_perp = span - (span * exit_avg).sum(dim=-1, keepdim=True) * exit_avg
        span_norm = span_perp.norm(dim=-1, keepdim=True)
        fallback_z = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).expand(n, 3)
        span_dir = torch.where(span_norm > 1e-4, span_perp / span_norm.clamp_min(1e-6), fallback_z)
        side_dir = torch.cross(exit_avg, span_dir, dim=-1)
        side_norm = side_dir.norm(dim=-1, keepdim=True)
        fallback_y = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
        side_dir = torch.where(side_norm > 1e-4, side_dir / side_norm.clamp_min(1e-6), fallback_y)
        sag = torch.tensor([[0.0, lateral_bow[0], lateral_bow[1]]], device=device, dtype=dtype).expand(n, 3)

        proj0 = (q0 * exit_avg).sum(dim=-1, keepdim=True)
        proj1 = (q1 * exit_avg).sum(dim=-1, keepdim=True)
        front_proj = torch.maximum(proj0, proj1)
        extra0 = front_proj - proj0
        extra1 = front_proj - proj1

        def eval_inner(bulge: torch.Tensor) -> torch.Tensor:
            d0 = bulge.unsqueeze(-1) + extra0
            d1 = bulge.unsqueeze(-1) + extra1
            lead0 = 0.45 * d0
            lead1 = 0.45 * d1
            spread = (0.85 * bulge.unsqueeze(-1) - 0.35 * span_norm).clamp_min(0.0)
            side = (0.22 * bulge).unsqueeze(-1)
            p0 = q0.unsqueeze(1)
            p1 = (q0 + exit_far * lead0).unsqueeze(1)
            p2 = (q0 + exit_far * d0 - span_dir * spread + side_dir * side + sag).unsqueeze(1)
            p3 = (q1 + exit_act * d1 + span_dir * spread + side_dir * side + sag).unsqueeze(1)
            p4 = (q1 + exit_act * lead1).unsqueeze(1)
            p5 = q1.unsqueeze(1)
            return resample_polyline(c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3 + c4 * p4 + c5 * p5, inner_pts)

        lo = torch.full((n,), 0.015, device=device, dtype=dtype)
        hi = torch.full((n,), max(inner_len * 0.9, 0.15), device=device, dtype=dtype)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            pts_mid = eval_inner(mid)
            arc_len = (pts_mid[:, 1:] - pts_mid[:, :-1]).norm(dim=-1).sum(dim=-1)
            too_short = arc_len < inner_len
            lo = torch.where(too_short, mid, lo)
            hi = torch.where(too_short, hi, mid)

        bulge = 0.5 * (lo + hi)
        if num_points >= 5:
            raw = torch.cat((far_cable_start.unsqueeze(1), eval_inner(bulge), active_cable_end.unsqueeze(1)), dim=1)
            taut = (q1 - q0).norm(dim=-1) >= 0.98 * inner_len
            if enforce_exact_segments and not bool(taut.any()):
                pts = _fabrik_exact_segments(raw, seg_len, iterations=30)
            elif enforce_exact_segments:
                exact = _fabrik_exact_segments(raw, seg_len, iterations=30)
                pts = torch.where(taut.view(n, 1, 1), raw, exact)
            else:
                pts = raw
        else:
            pts = resample_polyline(torch.stack((far_cable_start, active_cable_end), dim=1), num_points)

        diffs = torch.zeros_like(pts)
        diffs[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        diffs[:, 0] = pts[:, 1] - pts[:, 0]
        diffs[:, -1] = pts[:, -1] - pts[:, -2]
        tangents = diffs / diffs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return cls(points=pts, tangents=tangents, bulge_m=bulge)
