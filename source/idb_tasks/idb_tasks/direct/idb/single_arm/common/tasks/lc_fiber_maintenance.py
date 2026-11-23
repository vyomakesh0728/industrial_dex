"""Task 5: LC Duplex Fiber Cable Maintenance (A9 -> Polish -> A9), 22-point IDB rubric."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import configclass

from ..scoring import RubricBreakdown, score_lc_fiber_maintenance
from .base import IDBDatacenterTaskBase, IDBTaskBaseCfg


class LCFiberMaintenanceTask(IDBDatacenterTaskBase):
    """Depress clip & unplug LC duplex (A9..A12), figure-8 polish both ferrules (>=10 cycles at 1-2 N), re-insert."""

    def evaluate(self) -> RubricBreakdown:
        assert self.polishing is not None
        cleaned = self.polishing.cycles_ok
        correct_port = cleaned & (self.seated_port_idx == self.target_port_idx)
        both_aligned = self.target_seating.key_aligned
        inserted = correct_port & self.target_seating.is_seated
        retention = cleaned & self.latch.engaged & (self.latch.latched_port_idx == self.target_port_idx)
        return score_lc_fiber_maintenance(
            removed=self.removed_from_source,
            routed_to_polish_no_kink=~self.kinked_to_polish,
            positioned_on_pad=self.polishing.positioned_ever,
            cycles_ok=self.polishing.cycles_ok,
            continuous_contact=self.polishing.continuous_contact_ok,
            pressure_ok=self.polishing.pressure_ok,
            routed_back_no_kink=~self.kinked_from_polish,
            correct_port=correct_port | self.inserted_correct_port,
            both_aligned=both_aligned | self.inserted_correct_port,
            inserted=inserted | self.inserted_correct_port,
            retention_engaged=retention,
            no_damage=~self.damaged(),
        )


@configclass
class LCFiberMaintenanceCfg(IDBTaskBaseCfg):
    task_class: type = LCFiberMaintenanceTask
    name: str = "lc_fiber_maintenance"
    cable_preset: str = "os2_fiber_duplex"
    far_port: str = "B9"
    source_port: str = "A9"
    target_port: str = "A9"
    candidate_port_triples: list[list[str]] = [
        ["B9", "A9", "A9"],
        ["B10", "A10", "A10"],
        ["B11", "A11", "A11"],
        ["B12", "A12", "A12"],
    ]
    include_polishing: bool = True
    polishing_required_cycles: int = 10
    polishing_pressure_min_n: float = 1.0
    polishing_pressure_max_n: float = 2.0
