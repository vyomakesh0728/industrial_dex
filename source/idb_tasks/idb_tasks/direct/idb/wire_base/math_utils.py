"""Vectorized wxyz quaternion and polyline utilities matching Isaac Lab math conventions."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB

if HAS_ISAACLAB:
    from isaaclab.utils.math import (  # type: ignore[import-not-found]
        quat_apply,
        quat_apply_inverse,
        quat_from_angle_axis,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
    )
else:
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack((
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ), dim=-1)

    def quat_inv(q: torch.Tensor) -> torch.Tensor:
        norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        conj = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
        return conj / norm_sq

    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        xyz = quat[..., 1:]
        t = 2.0 * torch.cross(xyz, vec, dim=-1)
        return vec + quat[..., :1] * t + torch.cross(xyz, t, dim=-1)

    def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return quat_apply(quat_inv(quat), vec)

    def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
        cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
        cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
        return torch.stack((
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ), dim=-1)

    def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        unit = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        half = 0.5 * angle.unsqueeze(-1)
        return torch.cat((torch.cos(half), unit * torch.sin(half)), dim=-1)


def resample_polyline(points: torch.Tensor, count: int) -> torch.Tensor:
    """Resample (N, K, 3) polylines to ``count`` equally spaced points by arc length."""
    segments = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
    cumulative = torch.cat([torch.zeros_like(segments[:, :1]), segments.cumsum(dim=1)], dim=1)
    total = cumulative[:, -1:].clamp_min(1e-9)
    targets = torch.linspace(0.0, 1.0, count, device=points.device, dtype=points.dtype).unsqueeze(0) * total
    right = torch.searchsorted(cumulative.contiguous(), targets.contiguous()).clamp(1, points.shape[1] - 1)
    left = right - 1
    c0, c1 = cumulative.gather(1, left), cumulative.gather(1, right)
    weight = ((targets - c0) / (c1 - c0).clamp_min(1e-9)).unsqueeze(-1)
    idx_left = left.unsqueeze(-1).expand(-1, -1, 3)
    idx_right = right.unsqueeze(-1).expand(-1, -1, 3)
    return torch.lerp(points.gather(1, idx_left), points.gather(1, idx_right), weight)
