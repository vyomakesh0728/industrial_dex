"""Behavioral tests for CAD-extracted IDB Board #1 geometry and OpenUSD assets."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pxr import Usd, UsdGeom

from idb_assets import IDB_ASSET_DIR

ASSET_ROOT = Path(IDB_ASSET_DIR)
MANIFEST_PATH = ASSET_ROOT / "board" / "board_geometry.json"

EXPECTED_PORTS = (
    [f"A{i}" for i in range(1, 13)]
    + [f"B{i}" for i in range(1, 13)]
    + [f"C{i}" for i in (1, 2, 3, 4, 9, 10, 11, 12)]
)

EXPECTED_USD_ASSETS = (
    "board/panel_a.usda",
    "board/panel_b.usda",
    "board/panel_c.usda",
    "board/fiber_polishing_station.usda",
    "board/cable_management_clip.usda",
    "board/idb_board_assembly.usda",
    "connectors/rj45/port.usda",
    "connectors/rj45/socket.usda",
    "connectors/sc_simplex/port.usda",
    "connectors/sc_simplex/socket.usda",
    "connectors/lc_simplex/port.usda",
    "connectors/lc_simplex/socket.usda",
    "connectors/lc_duplex/port.usda",
    "connectors/lc_duplex/socket.usda",
)


def test_board_manifest_exists_and_matches_official_cad():
    assert MANIFEST_PATH.is_file(), f"Missing board geometry manifest at {MANIFEST_PATH}"
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ports = data["ports"]
    assert sorted(ports.keys()) == sorted(EXPECTED_PORTS), f"Port set mismatch: {sorted(ports.keys())}"

    # Verify exact 1U panel spacing (44.45 mm) and 2U A-to-B spacing (88.90 mm) from official STEP.
    z_a = ports["A1"]["mouth_pos"][2]
    z_b = ports["B1"]["mouth_pos"][2]
    z_c = ports["C1"]["mouth_pos"][2]
    assert math.isclose(z_b - z_c, 0.04445, abs_tol=0.0005), f"B-C vertical delta {z_b - z_c}"
    assert math.isclose(z_a - z_b, 0.08890, abs_tol=0.0005), f"A-B vertical delta {z_a - z_b}"

    # Verify port horizontal pitches from STEP: RJ45 = 17.5 mm, SC/LC = 17.0 mm, A1 is LEFT (+Y) of A12 (-Y).
    assert ports["A1"]["mouth_pos"][1] > ports["A12"]["mouth_pos"][1]
    assert math.isclose(ports["A1"]["mouth_pos"][1] - ports["A2"]["mouth_pos"][1], 0.0175, abs_tol=0.0002)
    assert math.isclose(ports["A5"]["mouth_pos"][1] - ports["A6"]["mouth_pos"][1], 0.0170, abs_tol=0.0002)
    assert math.isclose(ports["A9"]["mouth_pos"][1] - ports["A10"]["mouth_pos"][1], 0.0170, abs_tol=0.0002)

    # Verify latch/key orientation per instructions.md.
    assert ports["A1"]["latch_dir"] == [0.0, 0.0, 1.0]   # top
    assert ports["B1"]["latch_dir"] == [0.0, 0.0, -1.0]  # bottom
    assert ports["C1"]["latch_dir"] == [0.0, 0.0, 1.0]   # top
    assert ports["A5"]["latch_dir"] == [0.0, 1.0, 0.0]   # left
    assert ports["B5"]["latch_dir"] == [0.0, 0.0, 1.0]   # up
    assert ports["A9"]["latch_dir"] == [0.0, 1.0, 0.0]   # left

    # Verify fiber polishing station sits centered horizontally above Panel A.
    pad = data["polishing_pad"]
    assert math.isclose(pad["center"][1], 0.0, abs_tol=0.001)
    assert pad["center"][2] > z_a + 0.025
    assert 0.010 <= pad["radius"] <= 0.025
    assert pad["normal"] == [-1.0, 0.0, 0.0]  # faces robot (-X)


def test_all_usd_assets_load_cleanly_in_openusd():
    for rel in EXPECTED_USD_ASSETS:
        usd_path = ASSET_ROOT / rel
        assert usd_path.is_file(), f"Missing USD asset: {rel}"
        stage = Usd.Stage.Open(str(usd_path))
        assert stage is not None, f"Failed to open USD stage: {rel}"
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, f"{rel} must be Z-up"
        assert math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, abs_tol=1e-6), f"{rel} must be meters"
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid(), f"{rel} missing valid defaultPrim"

        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert meshes, f"{rel} contains no UsdGeom.Mesh prims"
        for prim in meshes:
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            assert pts and len(pts) >= 4, f"{rel}:{prim.GetPath()} has insufficient vertices"
            assert counts and len(counts) >= 4, f"{rel}:{prim.GetPath()} has insufficient faces"
            for pt in pts:
                assert all(math.isfinite(c) for c in pt), f"Non-finite vertex in {rel}:{prim.GetPath()}"
