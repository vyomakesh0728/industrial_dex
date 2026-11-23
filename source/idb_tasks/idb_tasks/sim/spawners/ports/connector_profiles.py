"""IDB datacenter connector profiles shared by spawners, geometry, and CAD export.

Frames match WireCraft (all lengths in meters, right-handed Z-up):
- Port root: tip rear shoulder. Local +X points toward the leading face / ferrule tip.
  Local +Z points toward the retention latch or alignment key.
- Socket root: mouth_x behind the mouth plane. Local +X points inward into the cavity.
  Local +Z points toward the latch/key relief slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def rectangle(y_min: float, y_max: float, z_min: float, z_max: float) -> tuple[tuple[float, float], ...]:
    """Return a counterclockwise rectangle in the local YZ plane."""
    return ((y_min, z_min), (y_max, z_min), (y_max, z_max), (y_min, z_max))


@dataclass(frozen=True)
class FerruleSpec:
    """Optical ferrule tip coordinates in the port frame (meters)."""

    y_offsets: tuple[float, ...]
    z_offset: float
    radius: float
    protrusion: float


@dataclass(frozen=True)
class ConnectorProfile:
    """Dimension and collision specification for one IDB datacenter connector family."""

    family: str
    standard: str
    category: str
    tip_length: float
    tip_outline: tuple[tuple[float, float], ...]
    housing_bounds: tuple[float, float, float, float, float, float]
    socket_outer_bounds: tuple[float, float, float, float]
    socket_depth: float
    clearance: float
    grasp_x: float
    latch_bounds: tuple[float, float, float, float, float, float] | None = None
    housing_sections: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    latch_collision_height_scale: float = 1.0
    requires_latch_release: bool = False
    unplug_force_n: float = 12.0
    insert_seat_depth: float = 0.009
    ferrule: FerruleSpec | None = None
    port_color: tuple[float, float, float] = (0.12, 0.38, 0.82)
    boot_color: tuple[float, float, float] = (0.10, 0.30, 0.72)
    socket_color: tuple[float, float, float] = (0.10, 0.42, 0.78)

    @property
    def collision_latch_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """Effective fixed-key bounds shared by the port collider and socket relief."""
        if self.latch_bounds is None:
            return None
        scale = self.latch_collision_height_scale
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("Latch collision height scale must be finite and in (0, 1].")
        *bounds, z0, z1 = self.latch_bounds
        return (*bounds, z0, z0 + (z1 - z0) * scale)

    def cable_anchor_x(self, embed_depth: float = 0.008) -> float:
        """Cable endpoint inside the housing, measured along local X in the port frame."""
        housing_len = self.housing_bounds[1] - self.housing_bounds[0]
        if not 0 < embed_depth <= housing_len:
            raise ValueError(f"Cable embed depth {embed_depth} must fit in housing length {housing_len}.")
        return self.housing_bounds[0] + embed_depth


_PROFILES: dict[str, ConnectorProfile] = {
    "rj45": ConnectorProfile(
        family="rj45",
        standard="IEC 60603-7 (8P8C Cat6)",
        category="datacenter",
        tip_length=0.013,
        tip_outline=rectangle(-0.00584, 0.00584, -0.002225, 0.004975),
        housing_bounds=(-0.034, 0.0, -0.00584, 0.00584, -0.00553, 0.00553),
        # STEP rj45-port width is 17.00 mm on 17.50 mm pitch (0.50 mm inter-port gap).
        socket_outer_bounds=(-0.0084, 0.0084, -0.0072, 0.0102),
        socket_depth=0.016,
        clearance=0.0003,
        grasp_x=-0.019,
        latch_bounds=(-0.0045, 0.0125, -0.00305, 0.00305, 0.004975, 0.0095),
        latch_collision_height_scale=0.5,
        housing_sections=(
            (-0.034, -0.009, -0.00553, 0.00553, -0.00553, 0.00553),
            (-0.0095, 0.0, -0.00584, 0.00584, -0.003325, 0.004975),
        ),
        requires_latch_release=True,
        unplug_force_n=15.0,
        insert_seat_depth=0.010,
        ferrule=None,
        port_color=(0.78, 0.84, 0.90),
        boot_color=(0.08, 0.34, 0.78),
        socket_color=(0.08, 0.36, 0.74),
    ),
    "sc_simplex": ConnectorProfile(
        family="sc_simplex",
        standard="IEC 61754-4 (SC/UPC Simplex)",
        category="datacenter",
        tip_length=0.0125,
        tip_outline=rectangle(-0.0035, 0.0035, -0.0030, 0.0030),
        housing_bounds=(-0.040, 0.0, -0.0044, 0.0044, -0.0037, 0.0037),
        # STEP sc-port envelope: 9.41 mm (along latch Z) x 21.80 mm (along Y), 17.00 mm pitch.
        socket_outer_bounds=(-0.0075, 0.0075, -0.00465, 0.00465),
        socket_depth=0.0152,
        clearance=0.00025,
        grasp_x=-0.016,
        latch_bounds=(-0.002, 0.0115, -0.0011, 0.0011, 0.0030, 0.0039),
        latch_collision_height_scale=0.85,
        housing_sections=(
            (-0.040, -0.022, -0.0024, 0.0024, -0.0024, 0.0024),
            (-0.022, 0.0, -0.0044, 0.0044, -0.0037, 0.0037),
        ),
        requires_latch_release=False,
        unplug_force_n=10.0,
        insert_seat_depth=0.0095,
        ferrule=FerruleSpec(y_offsets=(0.0,), z_offset=0.0, radius=0.00125, protrusion=0.0015),
        port_color=(0.06, 0.42, 0.86),
        boot_color=(0.05, 0.34, 0.74),
        socket_color=(0.06, 0.40, 0.84),
    ),
    "lc_simplex": ConnectorProfile(
        family="lc_simplex",
        standard="IEC 61754-20 (LC/UPC Simplex)",
        category="datacenter",
        tip_length=0.0105,
        tip_outline=rectangle(-0.0023, 0.0023, -0.0023, 0.0023),
        housing_bounds=(-0.032, 0.0, -0.0026, 0.0026, -0.0026, 0.0026),
        # STEP lc-simplex-port envelope: 7.00 mm (W) x 13.00 mm (H), 17.00 mm pitch.
        socket_outer_bounds=(-0.0035, 0.0035, -0.0045, 0.0062),
        socket_depth=0.0135,
        clearance=0.0002,
        grasp_x=-0.014,
        latch_bounds=(-0.003, 0.0095, -0.0011, 0.0011, 0.0023, 0.0041),
        latch_collision_height_scale=0.65,
        housing_sections=(
            (-0.032, -0.016, -0.0018, 0.0018, -0.0018, 0.0018),
            (-0.016, 0.0, -0.0026, 0.0026, -0.0026, 0.0026),
        ),
        requires_latch_release=True,
        unplug_force_n=8.0,
        insert_seat_depth=0.0080,
        ferrule=FerruleSpec(y_offsets=(0.0,), z_offset=0.0, radius=0.000625, protrusion=0.0012),
        port_color=(0.24, 0.58, 0.90),
        boot_color=(0.94, 0.94, 0.96),
        socket_color=(0.24, 0.56, 0.88),
    ),
    "lc_duplex": ConnectorProfile(
        family="lc_duplex",
        standard="IEC 61754-20 (LC/UPC Duplex)",
        category="datacenter",
        tip_length=0.0110,
        # Twin LC noses spaced 6.25 mm along local Y (outer span 10.85 mm).
        tip_outline=rectangle(-0.005425, 0.005425, -0.0023, 0.0023),
        housing_bounds=(-0.036, 0.0, -0.0058, 0.0058, -0.0028, 0.0028),
        # STEP lc-duplex-port envelope: 9.30 mm (along latch Z) x 21.80 mm (along Y), 17.00 mm pitch.
        socket_outer_bounds=(-0.0092, 0.0092, -0.0046, 0.0046),
        socket_depth=0.0155,
        clearance=0.00025,
        grasp_x=-0.016,
        latch_bounds=(-0.004, 0.010, -0.0042, 0.0042, 0.0023, 0.0040),
        latch_collision_height_scale=0.65,
        housing_sections=(
            (-0.036, -0.018, -0.0052, 0.0052, -0.0020, 0.0020),
            (-0.018, 0.0, -0.0058, 0.0058, -0.0028, 0.0028),
        ),
        requires_latch_release=True,
        unplug_force_n=11.0,
        insert_seat_depth=0.0085,
        ferrule=FerruleSpec(y_offsets=(-0.003125, 0.003125), z_offset=0.0, radius=0.000625, protrusion=0.0012),
        port_color=(0.24, 0.58, 0.90),
        boot_color=(0.94, 0.94, 0.96),
        socket_color=(0.24, 0.58, 0.88),
    ),
}

_ALIASES = {
    "ethernet": "rj45",
    "sc": "sc_simplex",
    "lc": "lc_duplex",
    "lc_sx": "lc_simplex",
    "lc_dx": "lc_duplex",
}


def connector_families() -> tuple[str, ...]:
    return tuple(_PROFILES)


def normalize_family(name: str) -> str:
    family = _ALIASES.get(name.lower(), name.lower())
    if family not in _PROFILES:
        raise ValueError(f"Unknown IDB connector family {name!r}; choose from {connector_families()}.")
    return family


def get_connector_profile(family: str) -> ConnectorProfile:
    return _PROFILES[normalize_family(family)]
