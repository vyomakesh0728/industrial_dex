"""Task 1: Ethernet Cable Port Swap (A1 -> A2), RJ45 Cat6, 6-point IDB rubric."""

from __future__ import annotations

from idb_tasks.utils.isaac_compat import configclass

from ..scoring import RubricBreakdown, score_ethernet_port_swap
from .base import IDBDatacenterTaskBase, IDBTaskBaseCfg


class EthernetPortSwapTask(IDBDatacenterTaskBase):
    """Unplug Cat6 RJ45 connector from A1, route without kinking, insert & latch into A2."""

    def evaluate(self) -> RubricBreakdown:
        correct_port = self.removed_from_source & (self.seated_port_idx == self.target_port_idx)
        inserted = correct_port & self.target_seating.is_seated
        latch_engaged = self.latch.engaged & (self.latch.latched_port_idx == self.target_port_idx)
        return score_ethernet_port_swap(
            removed=self.removed_from_source,
            no_kink=~self.ever_kinked,
            correct_port=correct_port | self.inserted_correct_port,
            inserted=inserted | self.inserted_correct_port,
            latch_engaged=latch_engaged,
            no_damage=~self.damaged(),
        )


@configclass
class EthernetPortSwapCfg(IDBTaskBaseCfg):
    task_class: type = EthernetPortSwapTask
    name: str = "ethernet_port_swap"
    cable_preset: str = "cat6_ethernet"
    far_port: str = "B1"
    source_port: str = "A1"
    target_port: str = "A2"
    candidate_port_triples: list[list[str]] = [
        ["B1", "A1", "A2"],
        ["B2", "A2", "A3"],
        ["B3", "A3", "A4"],
        ["B4", "A4", "A3"],
    ]
    include_polishing: bool = False
