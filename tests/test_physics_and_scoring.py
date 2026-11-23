"""Behavioral tests for connector seating, latch state, bend radius/kink, figure-8 polishing, and IDB scoring."""

from __future__ import annotations

import math

import pytest
import torch

from idb_tasks.direct.idb.single_arm.common.physics_metrics import (
    FigureEightPolishingTracker,
    NeighborDisturbanceMonitor,
    RetentionLatchState,
    compute_cable_bend_radius,
    evaluate_connector_seating,
)
from idb_tasks.direct.idb.single_arm.common.scoring import (
    score_ethernet_port_swap,
    score_lc_duplex_port_swap,
    score_lc_fiber_maintenance,
    score_sc_fiber_maintenance,
    score_sc_port_swap,
)
from idb_assets.board_spec import load_board_spec
from idb_tasks.direct.idb.wire_base.articulation import (
    _quat_from_rot_matrix,
    compute_chain_arc_joint_state,
    resolve_rotational_dofs,
    sample_physics_scales,
)
from idb_tasks.direct.idb.wire_base.connectors import ConnectorGeometry
from idb_tasks.direct.idb.wire_base.deformable import map_deformable_nodes_to_arc
from idb_tasks.direct.idb.wire_base.math_utils import quat_apply, quat_from_angle_axis
from idb_tasks.direct.idb.wire_base.wire_state import PluggedCableArc, ferrule_positions
from idb_tasks.sim.spawners.ports.connector_profiles import connector_families, get_connector_profile
from idb_tasks.sim.spawners.wires.wire_presets import get_cable_preset


@pytest.mark.parametrize("family", connector_families())
def test_seating_requires_depth_bore_alignment_and_key_roll(family: str):
    geom = ConnectorGeometry.from_family(family)
    socket_pos = torch.zeros(1, 3)
    socket_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # Exact seated pose: leading face at depth = seat_depth + 1 mm.
    depth_ok = geom.insert_seat_depth + 0.001
    port_pos_ok = torch.tensor([[geom.socket_mouth_x + depth_ok - geom.tip_length, 0.0, 0.0]])
    res_ok = evaluate_connector_seating(port_pos_ok, socket_quat, socket_pos, socket_quat, geom)
    assert bool(res_ok.is_seated[0]) and bool(res_ok.key_aligned[0])

    # Shallow depth fails seating.
    port_pos_shallow = torch.tensor([[geom.socket_mouth_x + geom.insert_seat_depth - 0.0015 - geom.tip_length, 0.0, 0.0]])
    res_shallow = evaluate_connector_seating(port_pos_shallow, socket_quat, socket_pos, socket_quat, geom)
    assert not bool(res_shallow.is_seated[0])

    # Lateral rim offset fails bore cross-section test.
    port_pos_rim = port_pos_ok.clone()
    port_pos_rim[0, 1] = 0.003
    res_rim = evaluate_connector_seating(port_pos_rim, socket_quat, socket_pos, socket_quat, geom)
    assert not bool(res_rim.is_seated[0])

    # 90-degree roll misalignment around +X fails key alignment.
    roll_90 = quat_from_angle_axis(torch.tensor([math.pi / 2.0]), torch.tensor([[1.0, 0.0, 0.0]]))
    res_rolled = evaluate_connector_seating(port_pos_ok, roll_90, socket_pos, socket_quat, geom)
    assert not bool(res_rolled.key_aligned[0])
    assert not bool(res_rolled.is_seated[0])

    # Reversed 180-degree yaw fails forward check.
    yaw_180 = quat_from_angle_axis(torch.tensor([math.pi]), torch.tensor([[0.0, 0.0, 1.0]]))
    res_rev = evaluate_connector_seating(port_pos_ok, yaw_180, socket_pos, socket_quat, geom)
    assert not bool(res_rev.is_seated[0])


def test_bend_radius_distinguishes_smooth_arc_from_kink():
    # Smooth semicircle of radius 40 mm -> R_min ~ 0.040 m (passes Cat6 23.2 mm and OS2 30 mm).
    theta = torch.linspace(0.0, math.pi, 41)
    r = 0.040
    smooth = torch.stack((r * torch.cos(theta), torch.zeros_like(theta), r * torch.sin(theta)), dim=-1).unsqueeze(0)
    res_smooth = compute_cable_bend_radius(smooth, min_bend_radius=0.030)
    assert not bool(res_smooth.is_kinked[0])
    assert math.isclose(float(res_smooth.min_radius_m[0]), r, rel_tol=0.05)

    # Sharp 90-degree kink at midpoint -> tiny bend radius (< 10 mm), must flag kinked.
    pts = torch.zeros(1, 41, 3)
    pts[0, :21, 0] = torch.linspace(0.0, 0.15, 21)
    pts[0, 20:, 0] = 0.15
    pts[0, 20:, 2] = torch.linspace(0.0, 0.15, 21)
    res_kink = compute_cable_bend_radius(pts, min_bend_radius=0.0232)
    assert bool(res_kink.is_kinked[0])
    assert float(res_kink.min_radius_m[0]) < 0.015


def test_plugged_cable_arc_matches_length_and_avoids_kink():
    far_pos = torch.tensor([[0.0, 0.024, 0.151]])
    active_pos = torch.tensor([[0.0, 0.024, 0.240]])
    arc = PluggedCableArc.solve(far_pos, active_pos, cable_length=0.30, num_points=41)
    seg_lengths = (arc.points[:, 1:] - arc.points[:, :-1]).norm(dim=-1).sum(dim=-1)
    assert math.isclose(float(seg_lengths[0]), 0.30, abs_tol=0.003)
    # Entire interior cable bows forward (-X) in front of the panel face.
    assert bool((arc.points[:, 1:-1, 0] < -0.001).all())
    # Must start un-kinked even for OS2 fiber (30 mm min bend radius).
    bend = compute_cable_bend_radius(arc.points, min_bend_radius=0.030)
    assert not bool(bend.is_kinked[0]), f"Initial plugged arc kinked: R_min={float(bend.min_radius_m[0]):.4f} m"


def test_retention_latch_state_machine_enforces_clip_depress_and_damage():
    # Active-clip connector (RJ45) starting latched in port 0.
    rj45 = ConnectorGeometry.from_family("rj45")
    latch = RetentionLatchState(num_envs=2, geometry=rj45, device="cpu")
    latch.reset(torch.tensor([0, 1]), initial_port_idx=torch.tensor([0, 0]), initially_latched=True)

    # Env 0 depresses latch cleanly while grasped, then unplugs -> no damage.
    # Env 1 rips out without depressing latch -> latch_damaged = True.
    latch.update(
        is_seated_on_any=torch.tensor([True, True]),
        seated_port_idx=torch.tensor([0, 0]),
        key_aligned=torch.tensor([True, True]),
        is_grasped=torch.tensor([True, True]),
        gripper_latch_force_n=torch.tensor([3.0, 0.0]),
        axial_pull_force_n=torch.tensor([5.0, 20.0]),
    )
    assert not bool(latch.engaged[0]) and not bool(latch.damaged[0])
    assert bool(latch.engaged[1])  # still latched while seated

    # Pull both completely out of port 0.
    latch.update(
        is_seated_on_any=torch.tensor([False, False]),
        seated_port_idx=torch.tensor([-1, -1]),
        key_aligned=torch.tensor([False, False]),
        is_grasped=torch.tensor([True, True]),
        gripper_latch_force_n=torch.tensor([3.0, 0.0]),
        axial_pull_force_n=torch.tensor([5.0, 20.0]),
    )
    assert not bool(latch.damaged[0])
    assert bool(latch.damaged[1]), "Forcing an un-released RJ45 out of its port must flag latch damage"

    # Re-insert Env 0 into target port 1 and hold for engagement steps.
    for _ in range(latch.engage_hold_steps):
        latch.update(
            is_seated_on_any=torch.tensor([True, False]),
            seated_port_idx=torch.tensor([1, -1]),
            key_aligned=torch.tensor([True, False]),
            is_grasped=torch.tensor([True, False]),
            gripper_latch_force_n=torch.tensor([0.0, 0.0]),
            axial_pull_force_n=torch.tensor([0.0, 0.0]),
        )
    assert bool(latch.engaged[0]) and int(latch.latched_port_idx[0]) == 1


def test_figure_eight_polishing_rejects_circles_and_counts_10_valid_cycles():
    pad_center = torch.tensor([[0.0, 0.0, 0.284]])
    pad_normal = torch.tensor([[-1.0, 0.0, 0.0]])
    pad_radius = 0.01689

    # 1. Pure circle (NOT a figure-8) must record 0 completed figure-8 cycles!
    tracker_circle = FigureEightPolishingTracker(num_envs=1, required_cycles=10, device="cpu")
    for k in range(400):
        t = k * (math.tau / 40.0)
        pos = pad_center + torch.tensor([[0.0, 0.008 * math.cos(t), 0.008 * math.sin(t)]])
        tracker_circle.update(pos.unsqueeze(1), pad_center, pad_normal, pad_radius, torch.tensor([1.5]))
    assert int(tracker_circle.cycles_completed[0]) == 0, "Circular motion must not count as figure-eight"

    # 2. Stationary press must record 0 cycles.
    tracker_static = FigureEightPolishingTracker(num_envs=1, required_cycles=10, device="cpu")
    for _ in range(100):
        tracker_static.update(pad_center.unsqueeze(1), pad_center, pad_normal, pad_radius, torch.tensor([1.5]))
    assert int(tracker_static.cycles_completed[0]) == 0

    # 3. Genuine Lissajous figure-8 (u = A sin(t), v = B sin(2t)) for 10 cycles at 1.5 N passes!
    tracker_ok = FigureEightPolishingTracker(num_envs=1, required_cycles=10, device="cpu")
    steps_per_cycle = 60
    for k in range(10 * steps_per_cycle + 15):
        t = k * (math.tau / steps_per_cycle)
        u = 0.010 * math.sin(t)
        v = 0.006 * math.sin(2.0 * t)
        pos = pad_center + torch.tensor([[-0.0005, u, v]])
        tracker_ok.update(pos.unsqueeze(1), pad_center, pad_normal, pad_radius, torch.tensor([1.5]))
    assert int(tracker_ok.cycles_completed[0]) >= 10
    assert bool(tracker_ok.continuous_contact_ok[0])
    assert bool(tracker_ok.pressure_ok[0])
    assert not bool(tracker_ok.ferrule_damaged[0])

    # 4. Excessive pressure (> 6 N) flags ferrule damage; low pressure (0.5 N) fails pressure band.
    tracker_bad_f = FigureEightPolishingTracker(num_envs=2, required_cycles=10, device="cpu")
    for k in range(10 * steps_per_cycle + 15):
        t = k * (math.tau / steps_per_cycle)
        u, v = 0.010 * math.sin(t), 0.006 * math.sin(2.0 * t)
        pos = pad_center.repeat(2, 1) + torch.tensor([[-0.0005, u, v], [-0.0005, u, v]])
        tracker_bad_f.update(pos.unsqueeze(1), pad_center.repeat(2, 1), pad_normal.repeat(2, 1), pad_radius, torch.tensor([0.5, 8.0]))
    assert not bool(tracker_bad_f.pressure_ok[0])
    assert bool(tracker_bad_f.ferrule_damaged[1])


def test_neighbor_disturbance_monitor_and_dual_ferrule_positions():
    lc_dx = ConnectorGeometry.from_family("lc_duplex")
    fpos = ferrule_positions(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), lc_dx)
    assert fpos.shape == (1, 2, 3)
    assert math.isclose(float(fpos[0, 1, 1] - fpos[0, 0, 1]), 0.00625, abs_tol=1e-6)

    mon = NeighborDisturbanceMonitor(num_envs=2, num_neighbors=2, device="cpu")
    nominal = torch.tensor([[[0.0, 0.05, 0.24], [0.0, 0.03, 0.24]]] * 2)
    mon.reset(torch.tensor([0, 1]), nominal_positions=nominal)
    moved = nominal.clone()
    moved[1, 0, 0] -= 0.010  # env 1 bumped neighbor 0 by 10 mm
    mon.update(current_positions=moved, max_active_contact_force_n=torch.tensor([2.0, 2.0]))
    assert not bool(mon.disturbed[0])
    assert bool(mon.disturbed[1])


def test_official_idb_rubric_point_totals_and_failure_gates():
    t = torch.tensor([True, False])
    f = torch.tensor([False, True])

    r1 = score_ethernet_port_swap(removed=t, no_kink=t, correct_port=t, inserted=t, latch_engaged=t, no_damage=t)
    assert r1.max_score == 6 and int(r1.total_score[0]) == 6 and bool(r1.passed[0])
    assert int(r1.total_score[1]) == 0 and not bool(r1.passed[1])

    r2 = score_sc_port_swap(removed=t, no_kink=t, correct_port=t, key_aligned=t, inserted=t, retention_engaged=t, no_damage=t)
    assert r2.max_score == 10 and int(r2.total_score[0]) == 10 and bool(r2.passed[0])

    r3 = score_lc_duplex_port_swap(removed=t, no_kink=t, correct_port=t, both_aligned=t, inserted=t, retention_engaged=t, no_damage=t)
    assert r3.max_score == 13 and int(r3.total_score[0]) == 13 and bool(r3.passed[0])

    r4 = score_sc_fiber_maintenance(
        removed=t, routed_to_polish_no_kink=t, positioned_on_pad=t, cycles_ok=t,
        continuous_contact=t, pressure_ok=t, routed_back_no_kink=t, correct_port=t,
        key_aligned=t, inserted=t, retention_engaged=t, no_damage=t,
    )
    assert r4.max_score == 19 and int(r4.total_score[0]) == 19 and bool(r4.passed[0])

    r5 = score_lc_fiber_maintenance(
        removed=t, routed_to_polish_no_kink=t, positioned_on_pad=t, cycles_ok=t,
        continuous_contact=t, pressure_ok=t, routed_back_no_kink=t, correct_port=t,
        both_aligned=t, inserted=t, retention_engaged=t, no_damage=t,
    )
    assert r5.max_score == 22 and int(r5.total_score[0]) == 22 and bool(r5.passed[0])

    # Partial credit: if damage occurs on Task 4, passed MUST be False even if inserted.
    r4_dmg = score_sc_fiber_maintenance(
        removed=torch.tensor([True]), routed_to_polish_no_kink=torch.tensor([True]),
        positioned_on_pad=torch.tensor([True]), cycles_ok=torch.tensor([True]),
        continuous_contact=torch.tensor([True]), pressure_ok=torch.tensor([True]),
        routed_back_no_kink=torch.tensor([True]), correct_port=torch.tensor([True]),
        key_aligned=torch.tensor([True]), inserted=torch.tensor([True]),
        retention_engaged=torch.tensor([True]), no_damage=torch.tensor([False]),
    )
    assert int(r4_dmg.total_score[0]) == 17 and not bool(r4_dmg.passed[0])


def _rot_x_y_z(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Intrinsic R_x(roll) @ R_y(pitch) @ R_z(yaw) matching PhysX 3-DOF spherical D6 kinematics."""
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    r = torch.zeros(*pitch.shape, 3, 3, dtype=pitch.dtype)
    r[..., 0, 0] = cp * cy
    r[..., 0, 1] = -cp * sy
    r[..., 0, 2] = sp
    r[..., 1, 0] = sr * sp * cy + cr * sy
    r[..., 1, 1] = -sr * sp * sy + cr * cy
    r[..., 1, 2] = -sr * cp
    r[..., 2, 0] = -cr * sp * cy + sr * sy
    r[..., 2, 1] = cr * sp * sy + sr * cy
    r[..., 2, 2] = cr * cp
    return r


def _quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    r = torch.zeros(*q.shape[:-1], 3, 3, dtype=q.dtype)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


@pytest.mark.parametrize(
    "far_pid,act_pid,preset_name",
    [
        ("B1", "A1", "cat6_ethernet"),
        ("B5", "A5", "os2_fiber_simplex"),
        ("B9", "A9", "os2_fiber_duplex"),
        ("C1", "C3", "cat6_ethernet"),
        ("C9", "C11", "os2_fiber_duplex"),
    ],
)
@pytest.mark.parametrize("root_frac", [0.0, 0.5, 1.0])
def test_articulation_chain_fk_matches_plugged_ports_and_seating(
    far_pid: str, act_pid: str, preset_name: str, root_frac: float
):
    spec = load_board_spec()
    preset = get_cable_preset(preset_name)
    f_port, a_port = spec.port(far_pid), spec.port(act_pid)
    f_prof, a_prof = get_connector_profile(f_port.family), get_connector_profile(a_port.family)
    f_geom, a_geom = ConnectorGeometry.from_family(f_port.family), ConnectorGeometry.from_family(a_port.family)

    f_sock_p, f_sock_q = torch.tensor([f_port.mouth_pos]), torch.tensor([f_port.socket_quat])
    a_sock_p, a_sock_q = torch.tensor([a_port.mouth_pos]), torch.tensor([a_port.socket_quat])
    f_seat_p = f_sock_p + quat_apply(
        f_sock_q, torch.tensor([[f_geom.socket_mouth_x + f_geom.insert_seat_depth + 0.001 - f_geom.tip_length, 0.0, 0.0]])
    )
    a_seat_p = a_sock_p + quat_apply(
        a_sock_q, torch.tensor([[a_geom.socket_mouth_x + a_geom.insert_seat_depth + 0.001 - a_geom.tip_length, 0.0, 0.0]])
    )

    f_anchor = f_prof.cable_anchor_x(min(0.012, f_prof.housing_bounds[1] - f_prof.housing_bounds[0]))
    a_anchor = a_prof.cable_anchor_x(min(0.012, a_prof.housing_bounds[1] - a_prof.housing_bounds[0]))
    f_exit = f_seat_p + quat_apply(f_sock_q, torch.tensor([[f_anchor, 0.0, 0.0]]))
    a_exit = a_seat_p + quat_apply(a_sock_q, torch.tensor([[a_anchor, 0.0, 0.0]]))
    exit_f = quat_apply(f_sock_q, torch.tensor([[-1.0, 0.0, 0.0]]))
    exit_a = quat_apply(a_sock_q, torch.tensor([[-1.0, 0.0, 0.0]]))

    num_segs = preset.num_segments
    seg_len = preset.length / num_segs
    root_idx = int(round(root_frac * (num_segs - 1)))
    arc = PluggedCableArc.solve(
        f_exit, a_exit, cable_length=preset.length, num_points=num_segs + 1, exit_normal=exit_f, exit_normal_active=exit_a
    )
    root_pos, root_quat, rel_roll, rel_pitch, rel_yaw = compute_chain_arc_joint_state(
        arc.points, far_port_quat=f_sock_q, active_port_quat=a_sock_q, root_seg_index=root_idx
    )

    # Walk exact USD FixedJoint + fixed-seg_len chain FK both forward and backward from root_idx.
    frames: list[torch.Tensor | None] = [None] * num_segs
    centers: list[torch.Tensor | None] = [None] * num_segs
    frames[root_idx] = _quat_to_rot_matrix(root_quat)
    centers[root_idx] = root_pos
    for k in range(root_idx, num_segs - 1):
        r_rel = _rot_x_y_z(rel_roll[:, k], rel_pitch[:, k], rel_yaw[:, k])
        frames[k + 1] = torch.bmm(frames[k], r_rel)
        centers[k + 1] = centers[k] + 0.5 * seg_len * frames[k][:, :, 0] + 0.5 * seg_len * frames[k + 1][:, :, 0]
    for k in range(root_idx - 1, -1, -1):
        r_rel = _rot_x_y_z(rel_roll[:, k], rel_pitch[:, k], rel_yaw[:, k])
        frames[k] = torch.bmm(frames[k + 1], r_rel.transpose(1, 2))
        centers[k] = centers[k + 1] - 0.5 * seg_len * frames[k + 1][:, :, 0] - 0.5 * seg_len * frames[k][:, :, 0]

    # Reconstruct exact start_port (FixedJoint localRot0=R_z(pi)) and end_port poses.
    rz_pi = torch.diag(torch.tensor([-1.0, -1.0, 1.0])).unsqueeze(0)
    r_start = torch.bmm(frames[0], rz_pi)
    c_start_j = centers[0] - 0.5 * seg_len * frames[0][:, :, 0]
    p_start = c_start_j - torch.bmm(r_start, torch.tensor([[[f_anchor], [0.0], [0.0]]])).squeeze(-1)

    r_end = frames[-1]
    c_end_j = centers[-1] + 0.5 * seg_len * frames[-1][:, :, 0]
    p_end = c_end_j - torch.bmm(r_end, torch.tensor([[[a_anchor], [0.0], [0.0]]])).squeeze(-1)

    assert float((p_start - f_seat_p).norm()) < 5e-4
    assert float((p_end - a_seat_p).norm()) < 5e-4

    seat_f = evaluate_connector_seating(p_start, _quat_from_rot_matrix(r_start), f_sock_p, f_sock_q, f_geom)
    seat_a = evaluate_connector_seating(p_end, _quat_from_rot_matrix(r_end), a_sock_p, a_sock_q, a_geom)
    assert bool(seat_f.is_seated[0]) and float(seat_f.roll_cos[0]) > 0.999
    assert bool(seat_a.is_seated[0]) and float(seat_a.roll_cos[0]) > 0.999

    # Verify smooth curvature (> min_bend_radius) and joint angles strictly below 0.85 rad limit.
    bend = compute_cable_bend_radius(arc.points, min_bend_radius=preset.min_bend_radius)
    assert not bool(bend.is_kinked[0]), f"{far_pid}->{act_pid} kinked: R_min={float(bend.min_radius_m[0]):.4f} m"
    max_angle = float(torch.stack((rel_roll.abs(), rel_pitch.abs(), rel_yaw.abs())).max())
    assert max_angle < 0.85, f"{far_pid}->{act_pid} exceeded joint limit: {max_angle:.3f} rad"

    # Verify 3-DOF PhysX joint resolver maps :0=roll, :1=pitch, :2=yaw.
    names_3dof = [f"joint_seg_{k}_to_parent:{d}" for k in range(1, 5) for d in (0, 1, 2)]
    r_d, p_d, y_d = resolve_rotational_dofs(names_3dof, num_segments=5)
    assert r_d[0] == 0 and p_d[0] == 1 and y_d[0] == 2


def test_deformable_nodal_arc_mapping_strips_spawn_offset_and_stays_smooth_across_vertical():
    # Spawn a rest cylinder far away at (0.35, -0.25, 1.76) with radius 2 mm.
    radius = 0.002
    rings = 31
    sides = 8
    xs = torch.linspace(0.35, 0.65, rings)
    angles = torch.linspace(0.0, math.tau * (1.0 - 1.0 / sides), sides)
    verts = []
    for x in xs:
        for a in angles:
            verts.append([float(x), -0.25 + radius * math.cos(float(a)), 1.76 + radius * math.sin(float(a))])
    default_pos = torch.tensor(verts, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

    # Construct an oblique 3D arc (with non-zero Y tangent) crossing |tans_z| == 0.90 to catch frame flips.
    t = torch.linspace(0.0, 1.0, 65)
    pts = torch.stack(
        (-0.05 * torch.sin(math.pi * t), 0.04 * torch.sin(math.pi * t), 0.15 + 0.30 * t), dim=-1
    ).unsqueeze(0).repeat(2, 1, 1)
    tans = torch.zeros_like(pts)
    tans[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    tans[:, 0] = pts[:, 1] - pts[:, 0]
    tans[:, -1] = pts[:, -1] - pts[:, -2]
    tans = tans / tans.norm(dim=-1, keepdim=True)

    mapped = map_deformable_nodes_to_arc(default_pos, pts, tans)
    assert torch.isfinite(mapped).all()
    u_idx = torch.linspace(0.0, 64.0, rings).repeat_interleave(sides)
    lo = u_idx.clamp(0.0, 63.999).long()
    frac = (u_idx - lo.float()).view(1, -1, 1)
    expected_centers = torch.lerp(pts[:, lo], pts[:, lo + 1], frac)
    radial_err = (mapped - expected_centers).norm(dim=-1)
    assert float(radial_err.max()) <= radius + 1e-4
    assert float(radial_err.min()) >= radius - 1e-4

    # Verify cross-section orientation never jumps discontinuously between adjacent rings!
    ring_v0 = mapped[0].view(rings, sides, 3)[:, 0] - expected_centers[0].view(rings, sides, 3)[:, 0]
    ring_v0 = ring_v0 / ring_v0.norm(dim=-1, keepdim=True)
    dots = (ring_v0[:-1] * ring_v0[1:]).sum(dim=-1)
    assert float(dots.min()) > 0.95, f"Discontinuous cross-section jump detected (min dot={float(dots.min()):.3f})"


def test_physics_randomization_scales_stay_within_bounds():
    s0, f0 = sample_physics_scales(16, False, (0.7, 1.3), (0.75, 1.25))
    assert torch.allclose(s0, torch.ones(16)) and torch.allclose(f0, torch.ones(16))
    s1, f1 = sample_physics_scales(64, True, (0.7, 1.3), (0.75, 1.25))
    assert float(s1.min()) >= 0.7 and float(s1.max()) <= 1.3 and float(s1.std()) > 0.05
    assert float(f1.min()) >= 0.75 and float(f1.max()) <= 1.25 and float(f1.std()) > 0.05
