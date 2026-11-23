"""Task 3: LC Duplex Fiber Cable Port Swap (A9 -> A10), LC/UPC duplex OS2 fiber, 13-point IDB rubric."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import configclass

from ..scoring import RubricBreakdown, score_lc_duplex_port_swap
from .base import IDBDatacenterTaskBase, IDBTaskBaseCfg


class LCDuplexPortSwapTask(IDBDatacenterTaskBase):
    """Depress clip and unplug LC duplex connector from A9 (far end in B9), route, align both ferrules, insert into A10."""

    def evaluate(self) -> RubricBreakdown:
        correct_port = self.removed_from_source & (self.seated_port_idx == self.target_port_idx)
        both_aligned = self.target_seating.key_aligned
        inserted = correct_port & self.target_seating.is_seated
        retention = self.latch.engaged & (self.latch.latched_port_idx == self.target_port_idx)
        return score_lc_duplex_port_swap(
            removed=self.removed_from_source,
            no_kink=~self.ever_kinked,
            correct_port=correct_port | self.inserted_correct_port,
            both_aligned=both_aligned | self.inserted_correct_port,
            inserted=inserted | self.inserted_correct_port,
            retention_engaged=retention,
            no_damage=~self.damaged(),
        )


@configclass
class LCDuplexPortSwapCfg(IDBTaskBaseCfg):
    task_class: type = LCDuplexPortSwapTask
    name: str = "lc_duplex_port_swap"
    cable_preset: str = "os2_fiber_duplex"
    far_port: str = "B9"
    source_port: str = "A9"
    target_port: str = "A10"
    candidate_port_triples: list[list[str]] = [
        ["B9", "A9", "A10"],
        ["B10", "A10", "A11"],
        ["B11", "A11", "A12"],
        ["B12", "A12", "A11"],
    ]
    include_polishing: bool = False
