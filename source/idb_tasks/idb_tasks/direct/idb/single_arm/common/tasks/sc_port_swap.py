"""Task 2: SC Simplex Fiber Cable Port Swap (A5 -> A6), LC-to-SC OS2 fiber, 10-point IDB rubric."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import configclass

from ..scoring import RubricBreakdown, score_sc_port_swap
from .base import IDBDatacenterTaskBase, IDBTaskBaseCfg


class SCPortSwapTask(IDBDatacenterTaskBase):
    """Unplug SC simplex connector from A5 (far LC end in B5), route without kinking, align key and insert into A6."""

    def evaluate(self) -> RubricBreakdown:
        correct_port = self.removed_from_source & (self.seated_port_idx == self.target_port_idx)
        key_aligned = self.target_seating.key_aligned
        inserted = correct_port & self.target_seating.is_seated
        retention = self.latch.engaged & (self.latch.latched_port_idx == self.target_port_idx)
        return score_sc_port_swap(
            removed=self.removed_from_source,
            no_kink=~self.ever_kinked,
            correct_port=correct_port | self.inserted_correct_port,
            key_aligned=key_aligned | self.inserted_correct_port,
            inserted=inserted | self.inserted_correct_port,
            retention_engaged=retention,
            no_damage=~self.damaged(),
        )


@configclass
class SCPortSwapCfg(IDBTaskBaseCfg):
    task_class: type = SCPortSwapTask
    name: str = "sc_port_swap"
    cable_preset: str = "os2_fiber_simplex"
    far_port: str = "B5"
    source_port: str = "A5"
    target_port: str = "A6"
    candidate_port_triples: list[list[str]] = [
        ["B5", "A5", "A6"],
        ["B6", "A6", "A7"],
        ["B7", "A7", "A8"],
        ["B8", "A8", "A7"],
    ]
    include_polishing: bool = False
