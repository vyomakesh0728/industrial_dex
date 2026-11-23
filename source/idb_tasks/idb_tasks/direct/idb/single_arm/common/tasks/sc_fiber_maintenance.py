"""Task 4: SC Simplex Fiber Cable Maintenance (A5 -> Polish -> A5), 19-point IDB rubric and arXiv:2607.14021 baseline."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import configclass

from ..scoring import RubricBreakdown, score_sc_fiber_maintenance
from .base import IDBDatacenterTaskBase, IDBTaskBaseCfg


class SCFiberMaintenanceTask(IDBDatacenterTaskBase):
    """Unplug SC connector (A5..A8), figure-8 polish ferrule (>=10 cycles at 1-2 N), re-insert into original port."""

    def evaluate(self) -> RubricBreakdown:
        assert self.polishing is not None
        cleaned = self.polishing.cycles_ok
        correct_port = cleaned & (self.seated_port_idx == self.target_port_idx)
        key_aligned = self.target_seating.key_aligned
        inserted = correct_port & self.target_seating.is_seated
        retention = cleaned & self.latch.engaged & (self.latch.latched_port_idx == self.target_port_idx)
        return score_sc_fiber_maintenance(
            removed=self.removed_from_source,
            routed_to_polish_no_kink=~self.kinked_to_polish,
            positioned_on_pad=self.polishing.positioned_ever,
            cycles_ok=self.polishing.cycles_ok,
            continuous_contact=self.polishing.continuous_contact_ok,
            pressure_ok=self.polishing.pressure_ok,
            routed_back_no_kink=~self.kinked_from_polish,
            correct_port=correct_port | self.inserted_correct_port,
            key_aligned=key_aligned | self.inserted_correct_port,
            inserted=inserted | self.inserted_correct_port,
            retention_engaged=retention,
            no_damage=~self.damaged(),
        )


@configclass
class SCFiberMaintenanceCfg(IDBTaskBaseCfg):
    task_class: type = SCFiberMaintenanceTask
    name: str = "sc_fiber_maintenance"
    cable_preset: str = "os2_fiber_simplex"
    far_port: str = "B5"
    source_port: str = "A5"
    target_port: str = "A5"
    # Matches arXiv:2607.14021 Section VIII evaluation across all four SC simplex ports A5-A8.
    candidate_port_triples: list[list[str]] = [
        ["B5", "A5", "A5"],
        ["B6", "A6", "A6"],
        ["B7", "A7", "A7"],
        ["B8", "A8", "A8"],
    ]
    include_polishing: bool = True
    polishing_required_cycles: int = 10
    polishing_pressure_min_n: float = 1.0
    polishing_pressure_max_n: float = 2.0
