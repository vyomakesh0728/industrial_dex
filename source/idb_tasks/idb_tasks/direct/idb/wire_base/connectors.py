"""Connector configuration and derived geometry shared by wire backends and task logic."""

from __future__ import annotations

from dataclasses import dataclass

from idb_tasks.sim.spawners.ports import (
    FerruleSpec,
    IDBPortCfg,
    IDBSocketCfg,
    default_visual_usd,
    get_connector_profile,
    normalize_family,
    offset_polygon,
)
from idb_tasks.utils.isaac_compat import configclass


@configclass
class ConnectorCfg:
    """User-facing connector configuration for active and far cable ends."""

    family: str = "sc_simplex"
    far_family: str | None = None
    """Connector family on the cable's far end (start_port); None matches ``family``."""
    socket_clearance: float | None = None
    port_mass: float = 0.01
    port_static_friction: float = 2.0
    port_dynamic_friction: float = 2.0

    @property
    def resolved_far_family(self) -> str:
        return normalize_family(self.family if self.far_family is None else self.far_family)

    def port_cfg(self, family: str | None = None) -> IDBPortCfg:
        fam = normalize_family(self.family if family is None else family)
        return IDBPortCfg(
            family=fam,
            mass=self.port_mass,
            static_friction=self.port_static_friction,
            dynamic_friction=self.port_dynamic_friction,
            visual_usd_path=default_visual_usd(fam, "port"),
        )

    def socket_cfg(self, family: str | None = None) -> IDBSocketCfg:
        fam = normalize_family(self.family if family is None else family)
        return IDBSocketCfg(
            family=fam,
            clearance=self.socket_clearance,
            visual_usd_path=default_visual_usd(fam, "socket"),
        )


@dataclass(frozen=True)
class ConnectorGeometry:
    """Immutable scalar and polygon geometry read by seating, latch, and polishing checks."""

    family: str
    standard: str
    tip_length: float
    grasp_x: float
    socket_depth: float
    socket_mouth_x: float
    insert_seat_depth: float
    requires_latch_release: bool
    unplug_force_n: float
    tip_outline: tuple[tuple[float, float], ...]
    hole: tuple[tuple[float, float], ...]
    ferrule: FerruleSpec | None

    @classmethod
    def from_family(cls, name: str, clearance: float | None = None) -> ConnectorGeometry:
        fam = normalize_family(name)
        prof = get_connector_profile(fam)
        clr = prof.clearance if clearance is None else float(clearance)
        return cls(
            family=fam,
            standard=prof.standard,
            tip_length=prof.tip_length,
            grasp_x=prof.grasp_x,
            socket_depth=prof.socket_depth,
            socket_mouth_x=0.0,
            insert_seat_depth=prof.insert_seat_depth,
            requires_latch_release=prof.requires_latch_release,
            unplug_force_n=prof.unplug_force_n,
            tip_outline=tuple(prof.tip_outline),
            hole=tuple(offset_polygon(list(prof.tip_outline), clr)),
            ferrule=prof.ferrule,
        )
