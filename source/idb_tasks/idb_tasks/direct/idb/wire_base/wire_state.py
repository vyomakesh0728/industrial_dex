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


@dataclass
class PluggedCableArc:
    """Smooth 3D cubic Bezier arch connecting two ports on the vertical IDB board.

    Both cable ends exit perpendicular to the panel front (-X_board), bow outward in -X
    (and optionally slightly downward/sideways like a real datacenter patch cord), and
    match the physical cable length without stretch or board penetration.
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
        lateral_bow: tuple[float, float] = (0.0, -0.025),
        iterations: int = 22,
    ) -> PluggedCableArc:
        """Solve a smooth 3D cubic Bezier curve of target arc length between two plugged ports."""
        n = far_cable_start.shape[0]
        device, dtype = far_cable_start.device, far_cable_start.dtype
        if exit_normal is None:
            exit_dir = torch.tensor([[-1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
        else:
            exit_dir = exit_normal / exit_normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        t = torch.linspace(0.0, 1.0, num_points, device=device, dtype=dtype).view(1, num_points, 1)
        b0 = (1.0 - t) ** 3
        b1 = 3.0 * ((1.0 - t) ** 2) * t
        b2 = 3.0 * (1.0 - t) * (t**2)
        b3 = t**3

        # Compute unit span axis between the two ports and an orthogonal lateral axis in the panel plane.
        span = active_cable_end - far_cable_start
        span_len = span.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        span_dir = span / span_len
        side_dir = torch.cross(exit_dir, span_dir, dim=-1)
        side_norm = side_dir.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).expand(n, 3)
        side_dir = torch.where(side_norm > 1e-4, side_dir / side_norm.clamp_min(1e-6), fallback)

        # Spread control points outward into a smooth elastica-like loop matching real datacenter patch cords
        # so the apex never forms a tight hairpin kink.
        sag = torch.tensor([[0.0, lateral_bow[0], lateral_bow[1]]], device=device, dtype=dtype).expand(n, 3)

        def eval_curve(bulge: torch.Tensor) -> torch.Tensor:
            d = bulge.unsqueeze(-1)
            spread = (0.45 * bulge).unsqueeze(-1)
            side = (0.30 * bulge).unsqueeze(-1)
            p0 = far_cable_start.unsqueeze(1)
            p3 = active_cable_end.unsqueeze(1)
            p1 = (far_cable_start + exit_dir * d - span_dir * spread + side_dir * side + sag).unsqueeze(1)
            p2 = (active_cable_end + exit_dir * d + span_dir * spread + side_dir * side + sag).unsqueeze(1)
            return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

        lo = torch.full((n,), 0.02, device=device, dtype=dtype)
        hi = torch.full((n,), max(cable_length * 0.9, 0.15), device=device, dtype=dtype)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            pts = eval_curve(mid)
            arc_len = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(dim=-1)
            too_short = arc_len < cable_length
            lo = torch.where(too_short, mid, lo)
            hi = torch.where(too_short, hi, mid)

        bulge = 0.5 * (lo + hi)
        pts = resample_polyline(eval_curve(bulge), num_points)
        diffs = torch.zeros_like(pts)
        diffs[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        diffs[:, 0] = pts[:, 1] - pts[:, 0]
        diffs[:, -1] = pts[:, -1] - pts[:, -2]
        tangents = diffs / diffs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return cls(points=pts, tangents=tangents, bulge_m=bulge)
