"""Official IDB Board #1 discrete scoring rubrics (datacenter-cable-manipulation/scoring-metrics.md)
and arXiv:2607.14021 three-phase binary evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RubricBreakdown:
    """Vectorized discrete rubric output for one IDB datacenter task."""

    task_name: str
    max_score: int
    total_score: torch.Tensor
    passed: torch.Tensor
    step_scores: dict[str, torch.Tensor]
    phase_flags: dict[str, torch.Tensor]


def _pts(flag: torch.Tensor, weight: int) -> torch.Tensor:
    return flag.to(torch.long) * weight


def score_ethernet_port_swap(
    removed: torch.Tensor,
    no_kink: torch.Tensor,
    correct_port: torch.Tensor,
    inserted: torch.Tensor,
    latch_engaged: torch.Tensor,
    no_damage: torch.Tensor,
) -> RubricBreakdown:
    """Task 1: Ethernet Cable Port Swap (A1 -> A2), max 6 points."""
    steps = {
        "1_removed_from_source": _pts(removed, 1),
        "2_routed_without_kinking": _pts(removed & no_kink, 1),
        "3_correct_target_port": _pts(removed & correct_port, 1),
        "4_inserted_into_target": _pts(removed & correct_port & inserted, 1),
        "5_retention_clip_engaged": _pts(removed & correct_port & inserted & latch_engaged, 1),
        "6_no_damage": _pts(no_damage, 1),
    }
    total = sum(steps.values())
    passed = removed & no_kink & correct_port & inserted & latch_engaged & no_damage
    phases = {"unplug": removed, "insert": correct_port & inserted & latch_engaged, "integrity": no_damage & no_kink}
    return RubricBreakdown("ethernet_port_swap", 6, total, passed, steps, phases)


def score_sc_port_swap(
    removed: torch.Tensor,
    no_kink: torch.Tensor,
    correct_port: torch.Tensor,
    key_aligned: torch.Tensor,
    inserted: torch.Tensor,
    retention_engaged: torch.Tensor,
    no_damage: torch.Tensor,
) -> RubricBreakdown:
    """Task 2: SC Simplex Fiber Cable Port Swap (A5 -> A6), max 10 points."""
    steps = {
        "1_sc_removed_from_source": _pts(removed, 1),
        "2_routed_without_kinking": _pts(removed & no_kink, 1),
        "3_correct_target_port": _pts(removed & correct_port, 1),
        "4_alignment_key_oriented": _pts(removed & correct_port & key_aligned, 1),
        "5_sc_inserted_into_target": _pts(removed & correct_port & key_aligned & inserted, 2),
        "6_retention_engaged": _pts(removed & correct_port & key_aligned & inserted & retention_engaged, 2),
        "7_no_damage": _pts(no_damage, 2),
    }
    total = sum(steps.values())
    passed = removed & no_kink & correct_port & key_aligned & inserted & retention_engaged & no_damage
    phases = {"unplug": removed, "insert": correct_port & key_aligned & inserted & retention_engaged, "integrity": no_damage & no_kink}
    return RubricBreakdown("sc_port_swap", 10, total, passed, steps, phases)


def score_lc_duplex_port_swap(
    removed: torch.Tensor,
    no_kink: torch.Tensor,
    correct_port: torch.Tensor,
    both_aligned: torch.Tensor,
    inserted: torch.Tensor,
    retention_engaged: torch.Tensor,
    no_damage: torch.Tensor,
) -> RubricBreakdown:
    """Task 3: LC Duplex Fiber Cable Port Swap (A9 -> A10), max 13 points."""
    steps = {
        "1_lc_removed_from_source": _pts(removed, 1),
        "2_routed_without_kinking": _pts(removed & no_kink, 1),
        "3_correct_target_port": _pts(removed & correct_port, 1),
        "4_both_lc_aligned": _pts(removed & correct_port & both_aligned, 1),
        "5_lc_inserted_into_target": _pts(removed & correct_port & both_aligned & inserted, 3),
        "6_retention_engaged": _pts(removed & correct_port & both_aligned & inserted & retention_engaged, 3),
        "7_no_damage": _pts(no_damage, 3),
    }
    total = sum(steps.values())
    passed = removed & no_kink & correct_port & both_aligned & inserted & retention_engaged & no_damage
    phases = {"unplug": removed, "insert": correct_port & both_aligned & inserted & retention_engaged, "integrity": no_damage & no_kink}
    return RubricBreakdown("lc_duplex_port_swap", 13, total, passed, steps, phases)


def score_sc_fiber_maintenance(
    removed: torch.Tensor,
    routed_to_polish_no_kink: torch.Tensor,
    positioned_on_pad: torch.Tensor,
    cycles_ok: torch.Tensor,
    continuous_contact: torch.Tensor,
    pressure_ok: torch.Tensor,
    routed_back_no_kink: torch.Tensor,
    correct_port: torch.Tensor,
    key_aligned: torch.Tensor,
    inserted: torch.Tensor,
    retention_engaged: torch.Tensor,
    no_damage: torch.Tensor,
) -> RubricBreakdown:
    """Task 4: SC Simplex Fiber Cable Maintenance (A5 -> Polish -> A5), max 19 points."""
    cleaned = removed & positioned_on_pad & cycles_ok & continuous_contact & pressure_ok
    steps = {
        "1_sc_removed": _pts(removed, 1),
        "2_routed_to_polishing_no_kink": _pts(removed & routed_to_polish_no_kink, 1),
        "3_ferrule_positioned_on_pad": _pts(removed & positioned_on_pad, 1),
        "4_figure_eight_10_cycles": _pts(removed & positioned_on_pad & cycles_ok, 2),
        "5_continuous_contact": _pts(removed & positioned_on_pad & continuous_contact, 2),
        "6_pressure_1_to_2n": _pts(removed & positioned_on_pad & pressure_ok, 3),
        "7_routed_back_no_kink": _pts(cleaned & routed_back_no_kink, 1),
        "8_correct_target_port": _pts(cleaned & correct_port, 1),
        "9_alignment_key_oriented": _pts(cleaned & correct_port & key_aligned, 1),
        "10_sc_reinserted": _pts(cleaned & correct_port & key_aligned & inserted, 2),
        "11_retention_engaged": _pts(cleaned & correct_port & key_aligned & inserted & retention_engaged, 2),
        "12_no_damage": _pts(no_damage, 2),
    }
    total = sum(steps.values())
    passed = (
        cleaned
        & routed_to_polish_no_kink
        & routed_back_no_kink
        & correct_port
        & key_aligned
        & inserted
        & retention_engaged
        & no_damage
    )
    # arXiv:2607.14021 three-phase binary indicators (Grasp/Unplug -> Clean -> Insert).
    phases = {
        "grasp": removed,
        "clean": cleaned,
        "insert": cleaned & correct_port & key_aligned & inserted & retention_engaged,
        "integrity": no_damage & routed_to_polish_no_kink & routed_back_no_kink,
    }
    return RubricBreakdown("sc_fiber_maintenance", 19, total, passed, steps, phases)


def score_lc_fiber_maintenance(
    removed: torch.Tensor,
    routed_to_polish_no_kink: torch.Tensor,
    positioned_on_pad: torch.Tensor,
    cycles_ok: torch.Tensor,
    continuous_contact: torch.Tensor,
    pressure_ok: torch.Tensor,
    routed_back_no_kink: torch.Tensor,
    correct_port: torch.Tensor,
    both_aligned: torch.Tensor,
    inserted: torch.Tensor,
    retention_engaged: torch.Tensor,
    no_damage: torch.Tensor,
) -> RubricBreakdown:
    """Task 5: LC Duplex Fiber Cable Maintenance (A9 -> Polish -> A9), max 22 points."""
    cleaned = removed & positioned_on_pad & cycles_ok & continuous_contact & pressure_ok
    steps = {
        "1_lc_removed": _pts(removed, 1),
        "2_routed_to_polishing_no_kink": _pts(removed & routed_to_polish_no_kink, 1),
        "3_ferrules_positioned_on_pad": _pts(removed & positioned_on_pad, 1),
        "4_figure_eight_10_cycles": _pts(removed & positioned_on_pad & cycles_ok, 2),
        "5_continuous_contact": _pts(removed & positioned_on_pad & continuous_contact, 2),
        "6_pressure_1_to_2n": _pts(removed & positioned_on_pad & pressure_ok, 3),
        "7_routed_back_no_kink": _pts(cleaned & routed_back_no_kink, 1),
        "8_correct_target_port": _pts(cleaned & correct_port, 1),
        "9_both_lc_aligned": _pts(cleaned & correct_port & both_aligned, 1),
        "10_lc_reinserted": _pts(cleaned & correct_port & both_aligned & inserted, 3),
        "11_retention_engaged": _pts(cleaned & correct_port & both_aligned & inserted & retention_engaged, 3),
        "12_no_damage": _pts(no_damage, 3),
    }
    total = sum(steps.values())
    passed = (
        cleaned
        & routed_to_polish_no_kink
        & routed_back_no_kink
        & correct_port
        & both_aligned
        & inserted
        & retention_engaged
        & no_damage
    )
    phases = {
        "grasp": removed,
        "clean": cleaned,
        "insert": cleaned & correct_port & both_aligned & inserted & retention_engaged,
        "integrity": no_damage & routed_to_polish_no_kink & routed_back_no_kink,
    }
    return RubricBreakdown("lc_fiber_maintenance", 22, total, passed, steps, phases)
