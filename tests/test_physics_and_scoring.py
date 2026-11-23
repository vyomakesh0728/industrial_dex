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
from idb_tasks.direct.idb.wire_base.connectors import ConnectorGeometry
from idb_tasks.direct.idb.wire_base.math_utils import quat_from_angle_axis
from idb_tasks.direct.idb.wire_base.wire_state import PluggedCableArc, ferrule_positions
from idb_tasks.sim.spawners.ports.connector_profiles import connector_families


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
