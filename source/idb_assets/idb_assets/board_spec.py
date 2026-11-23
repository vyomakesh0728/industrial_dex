"""Pure-Python geometry manifest loader for IDB Board #1 (Datacenter Cable Manipulation).

All positions and orientations are expressed in the right-handed Z-up Board Local Frame
(meters, wxyz quaternions):
- X = 0.0 at the front face of Panels A/B/C; +X points INTO the board (insertion axis).
- Y = 0.0 at horizontal board center; +Y points LEFT when facing the board (Column 1 -> +Y, Column 12 -> -Y).
- Z = 0.0 at the bottom of the aluminum extrusion base (table contact plane); +Z points UP.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from idb_assets import IDB_ASSET_DIR


@dataclass(frozen=True)
class PortSpec:
    """Single receptacle port on IDB Board #1."""

    port_id: str
    panel: str
    column: int
    family: str
    mouth_pos: tuple[float, float, float]
    """Front mouth center in Board Local Frame (meters)."""
    socket_quat: tuple[float, float, float, float]
    """wxyz orientation of socket frame (+X into panel, +Z along latch/key direction)."""
    latch_dir: tuple[float, float, float]
    """Unit direction of the retention latch or alignment key in Board Local Frame."""
    roll_rad: float
    """Roll about +X from upright (+Z latch) to this port's orientation."""
    protrusion_m: float
    """Distance the coupler mouth protrudes forward (-X) of the panel front face."""
    cad_envelope_m: tuple[float, float, float]
    """(depth_x, width_y, height_z) bounding size from official STEP CAD."""


@dataclass(frozen=True)
class PolishingPadSpec:
    """Fiber polishing station disc mounted atop Panel A."""

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    thickness: float
    bracket_origin: tuple[float, float, float]


@dataclass(frozen=True)
class ExtrusionBarSpec:
    """One 20x20x250 mm aluminum extrusion member of the board frame."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class PanelSpec:
    """One 3D-printed 1U patch panel (Panel A, B, or C)."""

    panel: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    usd_relpath: str


@dataclass(frozen=True)
class CableClipSpec:
    """One side-mounted cable management clip (optional scene attachment)."""

    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]


@dataclass(frozen=True)
class BoardSpec:
    """Complete CAD-derived specification for IDB Board #1."""

    version: str
    source_step: str
    footprint_m: tuple[float, float, float]
    ports: dict[str, PortSpec]
    panels: dict[str, PanelSpec]
    polishing_pad: PolishingPadSpec
    extrusions: tuple[ExtrusionBarSpec, ...]
    cable_clips: tuple[CableClipSpec, ...]

    def port(self, port_id: str) -> PortSpec:
        try:
            return self.ports[port_id]
        except KeyError:
            raise KeyError(f"Unknown IDB Board #1 port {port_id!r}; valid ports: {sorted(self.ports)}.") from None

    def ports_by_panel(self, panel: str) -> list[PortSpec]:
        return sorted((p for p in self.ports.values() if p.panel == panel), key=lambda p: p.column)

    def ports_by_family(self, family: str) -> list[PortSpec]:
        return sorted((p for p in self.ports.values() if p.family == family), key=lambda p: (p.panel, p.column))


def load_board_spec(manifest_path: str | Path | None = None) -> BoardSpec:
    """Load the CAD-extracted board geometry manifest into typed domain objects."""
    path = Path(manifest_path) if manifest_path is not None else Path(IDB_ASSET_DIR) / "board" / "board_geometry.json"
    if not path.is_file():
        raise FileNotFoundError(f"Board geometry manifest not found at {path}; run scripts/idb/export_assets.py.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    ports = {
        pid: PortSpec(
            port_id=pid,
            panel=p["panel"],
            column=int(p["column"]),
            family=p["family"],
            mouth_pos=tuple(float(v) for v in p["mouth_pos"]),
            socket_quat=tuple(float(v) for v in p["socket_quat"]),
            latch_dir=tuple(float(v) for v in p["latch_dir"]),
            roll_rad=float(p["roll_rad"]),
            protrusion_m=float(p["protrusion_m"]),
            cad_envelope_m=tuple(float(v) for v in p["cad_envelope_m"]),
        )
        for pid, p in raw["ports"].items()
    }
    panels = {
        name: PanelSpec(
            panel=name,
            center=tuple(float(v) for v in p["center"]),
            size=tuple(float(v) for v in p["size"]),
            usd_relpath=p["usd_relpath"],
        )
        for name, p in raw["panels"].items()
    }
    pad_raw = raw["polishing_pad"]
    pad = PolishingPadSpec(
        center=tuple(float(v) for v in pad_raw["center"]),
        normal=tuple(float(v) for v in pad_raw["normal"]),
        radius=float(pad_raw["radius"]),
        thickness=float(pad_raw["thickness"]),
        bracket_origin=tuple(float(v) for v in pad_raw["bracket_origin"]),
    )
    extrusions = tuple(
        ExtrusionBarSpec(
            name=e["name"],
            center=tuple(float(v) for v in e["center"]),
            size=tuple(float(v) for v in e["size"]),
        )
        for e in raw["extrusions"]
    )
    clips = tuple(
        CableClipSpec(
            name=c["name"],
            pos=tuple(float(v) for v in c["pos"]),
            quat=tuple(float(v) for v in c["quat"]),
        )
        for c in raw["cable_clips"]
    )
    return BoardSpec(
        version=raw["version"],
        source_step=raw["source_step"],
        footprint_m=tuple(float(v) for v in raw["footprint_m"]),
        ports=ports,
        panels=panels,
        polishing_pad=pad,
        extrusions=extrusions,
        cable_clips=clips,
    )
