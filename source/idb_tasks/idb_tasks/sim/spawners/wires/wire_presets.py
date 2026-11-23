"""Physical cable presets calibrated to IDB Board #1 bill of materials (bom.csv)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CablePreset:
    """Physical and geometric parameters for one datacenter patch cable type."""

    name: str
    bom_part: str
    length: float
    radius: float
    outer_diameter: float
    min_bend_radius: float
    """Minimum permissible bend radius in meters (kink threshold)."""
    num_segments: int
    stiffness: float
    """Articulation bending drive stiffness in Nm/rad."""
    damping: float
    density: float
    fem_num_segments: int
    fem_youngs_modulus: float
    fem_dynamic_friction: float
    color: tuple[float, float, float]
    start_family: str
    end_family: str


_PRESETS: dict[str, CablePreset] = {
    "cat6_ethernet": CablePreset(
        name="cat6_ethernet",
        bom_part="Cat6 UTP Patch Cable 12in (C6UTPSGPVC-70717)",
        length=0.30,
        radius=0.0029,
        outer_diameter=0.0058,
        # IDB scoring-metrics.md Task 1: bend radius >= 4x cable diameter (4 * 5.8 mm = 23.2 mm).
        min_bend_radius=0.0232,
        num_segments=14,
        stiffness=0.35,
        damping=0.006,
        density=1150.0,
        fem_num_segments=36,
        fem_youngs_modulus=5.5e7,
        fem_dynamic_friction=4.0,
        color=(0.06, 0.36, 0.85),
        start_family="rj45",
        end_family="rj45",
    ),
    "os2_fiber_simplex": CablePreset(
        name="os2_fiber_simplex",
        bom_part="LC/UPC to SC/UPC Simplex OS2 Patch Cable 12in (SMXXSX-12285)",
        length=0.30,
        radius=0.0018,
        outer_diameter=0.0030,
        # IEC 60794 / TIA-568 OS2 single-mode minimum bend radius during handling (30 mm).
        min_bend_radius=0.030,
        num_segments=16,
        stiffness=0.14,
        damping=0.004,
        density=900.0,
        fem_num_segments=40,
        fem_youngs_modulus=2.8e7,
        fem_dynamic_friction=3.5,
        color=(0.95, 0.82, 0.08),
        start_family="lc_simplex",
        end_family="sc_simplex",
    ),
    "os2_fiber_duplex": CablePreset(
        name="os2_fiber_duplex",
        bom_part="LC/UPC to LC/UPC Duplex OS2 Patch Cable 12in (SMLCDX-282145)",
        length=0.30,
        radius=0.0022,
        outer_diameter=0.0040,
        min_bend_radius=0.030,
        num_segments=16,
        stiffness=0.20,
        damping=0.005,
        density=950.0,
        fem_num_segments=40,
        fem_youngs_modulus=3.6e7,
        fem_dynamic_friction=3.5,
        color=(0.95, 0.82, 0.08),
        start_family="lc_duplex",
        end_family="lc_duplex",
    ),
}


def cable_presets() -> tuple[str, ...]:
    return tuple(_PRESETS)


def get_cable_preset(name: str) -> CablePreset:
    try:
        return _PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown cable preset {name!r}; choose from {cable_presets()}.") from None
