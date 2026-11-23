#!/usr/bin/env python3
"""Deterministic CAD-to-OpenUSD asset pipeline for IDB Board #1 (Datacenter Cable Manipulation).

Reads official Analog Devices IDB CAD files (datacenter-benchmark-assembly-v3.5.step and the
five fabrication .3MF meshes), extracts exact port/panel/frame/polishing-pad geometry into
board_geometry.json, converts the 3MF meshes to Z-up meter OpenUSD (.usda) assets, builds the
assembled board visual USD, and builds matched visual connector port/socket USD assets.

Usage:
    uv run python scripts/idb/export_assets.py --source-dir <path-to-idb-cad>
    uv run python scripts/idb/export_assets.py --fetch
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from idb_assets import IDB_ASSET_DIR
from idb_tasks.sim.spawners.ports.connector_profiles import (
    ConnectorProfile,
    connector_families,
    get_connector_profile,
    rectangle,
)

IDB_RAW_BASE = (
    "https://media.githubusercontent.com/media/analogdevicesinc/"
    "ag-industrial-dexterity-benchmarks/main/datacenter-cable-manipulation"
)

CAD_FILES = {
    "fabrication/panel-a-v3.3MF": "fabrication/panel-a-v3.3MF",
    "fabrication/panel-b-v3.3MF": "fabrication/panel-b-v3.3MF",
    "fabrication/panel-c-v3.3MF": "fabrication/panel-c-v3.3MF",
    "fabrication/fiber-polishing-v3.3MF": "fabrication/fiber-polishing-v3.3MF",
    "fabrication/cable-management.3MF": "fabrication/cable-management.3MF",
    "source/datacenter-benchmark-assembly-v3.5.step": "source/datacenter-benchmark-assembly-v3.5.step",
}

# Exact STEP -> Board Local Frame mapping (mm -> meters, right-handed det = +1):
#   X_b = (145.0 - Z_s) / 1000.0   (0 at panel front face, +X into board)
#   Y_b = (115.0 - X_s) / 1000.0   (0 at horizontal center, +Y left toward col 1)
#   Z_b = (Y_s + 10.0) / 1000.0    (0 at base bottom on table, +Z up)
STEP_PANEL_FRONT_Z_MM = 145.0
STEP_CENTER_X_MM = 115.0
STEP_BASE_BOTTOM_Y_MM = -10.0


def step_mm_to_board_m(pt_step_mm: np.ndarray) -> np.ndarray:
    """Convert STEP assembly coordinates (mm, Y-up, +Z front) to Board Local Frame (m, Z-up, +X into board)."""
    arr = np.asarray(pt_step_mm, dtype=np.float64)
    out = np.empty_like(arr)
    out[..., 0] = (STEP_PANEL_FRONT_Z_MM - arr[..., 2]) / 1000.0
    out[..., 1] = (STEP_CENTER_X_MM - arr[..., 0]) / 1000.0
    out[..., 2] = (arr[..., 1] - STEP_BASE_BOTTOM_Y_MM) / 1000.0
    return out


def quat_from_roll_x(roll_rad: float) -> tuple[float, float, float, float]:
    """Return wxyz unit quaternion for rotation about local +X."""
    half = 0.5 * roll_rad
    return (round(math.cos(half), 8), round(math.sin(half), 8), 0.0, 0.0)


def fetch_cad_files(dest_dir: Path) -> None:
    """Download official IDB CAD source files from GitHub LFS media."""
    for rel_remote, rel_local in CAD_FILES.items():
        out_path = dest_dir / rel_local
        if out_path.is_file() and out_path.stat().st_size > 1024:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{IDB_RAW_BASE}/{rel_remote}"
        print(f"[export_assets] downloading {rel_remote}...")
        urllib.request.urlretrieve(url, out_path)


def parse_step_assembly(step_path: Path) -> list[dict]:
    """Parse STEP AP214 assembly hierarchy and return world-space part bounding boxes in STEP mm."""
    text = step_path.read_text(encoding="utf-8", errors="ignore")
    data = text.split("DATA;")[1].split("ENDSEC;")[0]
    entities: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", data, re.S):
        entities[int(m.group(1))] = (m.group(2), m.group(3).replace("\r", "").replace("\n", " "))

    prod_name = {
        eid: re.match(r"'([^']*)'", args.strip()).group(1)
        for eid, (et, args) in entities.items()
        if et == "PRODUCT"
    }
    pdf_to_prod: dict[int, str] = {}
    for eid, (et, args) in entities.items():
        if "PRODUCT_DEFINITION_FORMATION" in et:
            for ref in (int(x) for x in re.findall(r"#(\d+)", args)):
                if ref in prod_name:
                    pdf_to_prod[eid] = prod_name[ref]
    pd_to_prod: dict[int, str] = {}
    for eid, (et, args) in entities.items():
        if et == "PRODUCT_DEFINITION":
            for ref in (int(x) for x in re.findall(r"#(\d+)", args)):
                if ref in pdf_to_prod:
                    pd_to_prod[eid] = pdf_to_prod[ref]

    pds_to_pd: dict[int, int] = {}
    for eid, (et, args) in entities.items():
        if et == "PRODUCT_DEFINITION_SHAPE":
            refs = [int(x) for x in re.findall(r"#(\d+)", args)]
            if refs and refs[0] in pd_to_prod:
                pds_to_pd[eid] = refs[0]
    sr_to_pd: dict[int, int] = {}
    for eid, (et, args) in entities.items():
        if et == "SHAPE_DEFINITION_REPRESENTATION":
            refs = [int(x) for x in re.findall(r"#(\d+)", args)]
            if refs[0] in pds_to_pd:
                sr_to_pd[refs[1]] = pds_to_pd[refs[0]]
    for eid, (et, args) in entities.items():
        if et == "SHAPE_REPRESENTATION_RELATIONSHIP":
            refs = [int(x) for x in re.findall(r"#(\d+)", args)]
            if refs[0] in sr_to_pd:
                sr_to_pd[refs[1]] = sr_to_pd[refs[0]]
            elif refs[1] in sr_to_pd:
                sr_to_pd[refs[0]] = sr_to_pd[refs[1]]

    def get_vec(eid: int) -> np.ndarray:
        _, args = entities[eid]
        m = re.search(r"\(\s*([^)]+)\s*\)", args)
        return np.array([float(x) for x in m.group(1).split(",")], dtype=np.float64)

    def get_axis2(eid: int) -> np.ndarray:
        _, args = entities[eid]
        refs = [int(x) for x in re.findall(r"#(\d+)", args)]
        origin = get_vec(refs[0])
        z = get_vec(refs[1])
        z /= np.linalg.norm(z)
        x = get_vec(refs[2])
        x = x - np.dot(x, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4)
        mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = x, y, z, origin
        return mat

    nauos = {
        eid: tuple(int(x) for x in re.findall(r"#(\d+)", args)[:2])
        for eid, (et, args) in entities.items()
        if et == "NEXT_ASSEMBLY_USAGE_OCCURRENCE"
    }
    pds_to_nauo: dict[int, int] = {}
    for eid, (et, args) in entities.items():
        if et == "PRODUCT_DEFINITION_SHAPE":
            refs = [int(x) for x in re.findall(r"#(\d+)", args)]
            if refs and refs[0] in nauos:
                pds_to_nauo[eid] = refs[0]

    idt_axes = {
        eid: (
            get_axis2(int(re.findall(r"#(\d+)", args)[0])),
            get_axis2(int(re.findall(r"#(\d+)", args)[1])),
        )
        for eid, (et, args) in entities.items()
        if et == "ITEM_DEFINED_TRANSFORMATION"
    }
    rrwt_to_idt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m in re.finditer(r"#(\d+)\s*=\s*\((.*?)\)\s*;", data, re.S):
        eid, body = int(m.group(1)), m.group(2)
        if "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" in body:
            for ref in (int(x) for x in re.findall(r"#(\d+)", body)):
                if ref in idt_axes:
                    rrwt_to_idt[eid] = idt_axes[ref]

    children: dict[int, list[tuple[int, np.ndarray]]] = {}
    for eid, (et, args) in entities.items():
        if et == "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION":
            refs = [int(x) for x in re.findall(r"#(\d+)", args)]
            if refs[0] in rrwt_to_idt and refs[1] in pds_to_nauo:
                t1, t2 = rrwt_to_idt[refs[0]]
                parent_pd, child_pd = nauos[pds_to_nauo[refs[1]]]
                children.setdefault(parent_pd, []).append((child_pd, t1 @ np.linalg.inv(t2)))

    def get_points(sr_id: int) -> np.ndarray | None:
        visited: set[int] = set()
        stack = [sr_id]
        pts: list[list[float]] = []
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in entities:
                continue
            visited.add(cur)
            et, args = entities[cur]
            if et == "VERTEX_POINT":
                for ref in (int(x) for x in re.findall(r"#(\d+)", args)):
                    if ref in entities and entities[ref][0] == "CARTESIAN_POINT":
                        m = re.search(r"\(\s*([^)]+)\s*\)", entities[ref][1])
                        pts.append([float(x) for x in m.group(1).split(",")])
            elif et not in ("AXIS2_PLACEMENT_3D", "DIRECTION", "VECTOR", "CARTESIAN_POINT", "PRODUCT_DEFINITION"):
                for ref in (int(x) for x in re.findall(r"#(\d+)", args)):
                    if ref not in visited:
                        stack.append(ref)
        return np.array(pts, dtype=np.float64) if pts else None

    pd_pts = {
        pd: get_points(sr)
        for sr, pd in sr_to_pd.items()
        if entities[sr][0] == "ADVANCED_BREP_SHAPE_REPRESENTATION"
    }

    root_pd = next(pd for pd, name in pd_to_prod.items() if name == "datacenter-benchmark-assembly-v3.5")
    records: list[dict] = []

    def walk(pd: int, mat: np.ndarray, parent_name: str = "") -> None:
        name = pd_to_prod[pd]
        pts_local = pd_pts.get(pd)
        if pts_local is not None and len(pts_local) > 0:
            pts_w = (pts_local @ mat[:3, :3].T) + mat[:3, 3]
            records.append({
                "parent": parent_name,
                "name": name,
                "origin_step_mm": mat[:3, 3].copy(),
                "min_step_mm": pts_w.min(axis=0),
                "max_step_mm": pts_w.max(axis=0),
            })
        for child_pd, t_rel in children.get(pd, []):
            walk(child_pd, mat @ t_rel, name)

    walk(root_pd, np.eye(4))
    return records


def load_3mf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices (mm) and triangle indices from a 3MF archive."""
    with zipfile.ZipFile(path) as archive:
        model_name = next(n for n in archive.namelist() if n.endswith(".model"))
        root = ET.fromstring(archive.read(model_name))
    ns = {"m": root.tag.split("}")[0].strip("{")}
    verts = np.array(
        [[float(v.attrib["x"]), float(v.attrib["y"]), float(v.attrib["z"])] for v in root.findall(".//m:vertex", ns)],
        dtype=np.float64,
    )
    tris = np.array(
        [[int(t.attrib["v1"]), int(t.attrib["v2"]), int(t.attrib["v3"])] for t in root.findall(".//m:triangle", ns)],
        dtype=np.int32,
    )
    return verts, tris


def _create_stage(path: Path, default_prim: str = "Visual") -> tuple[Usd.Stage, UsdGeom.Xform]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _define_trimesh(
    stage: Usd.Stage,
    prim_path: str,
    verts: np.ndarray,
    tris: np.ndarray,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateSubdivisionSchemeAttr("none")
    pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.reshape(-1).tolist()))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*mn.tolist()), Gf.Vec3f(*mx.tolist())]))
    return mesh


def _define_box_mesh(
    stage: Usd.Stage,
    prim_path: str,
    bounds: tuple[float, float, float, float, float, float],
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    x0, x1, y0, y1, z0, z1 = bounds
    verts = np.array([
        [x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1],
        [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1],
    ], dtype=np.float64)
    tris = np.array([
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    return _define_trimesh(stage, prim_path, verts, tris, color)


def _define_cylinder_x_mesh(
    stage: Usd.Stage,
    prim_path: str,
    x0: float,
    x1: float,
    y_center: float,
    z_center: float,
    radius: float,
    color: tuple[float, float, float],
    sides: int = 16,
) -> UsdGeom.Mesh:
    ring0 = [[x0, y_center + radius * math.cos(i * math.tau / sides), z_center + radius * math.sin(i * math.tau / sides)] for i in range(sides)]
    ring1 = [[x1, y_center + radius * math.cos(i * math.tau / sides), z_center + radius * math.sin(i * math.tau / sides)] for i in range(sides)]
    c0_idx = 2 * sides
    c1_idx = 2 * sides + 1
    verts = np.array(ring0 + ring1 + [[x0, y_center, z_center], [x1, y_center, z_center]], dtype=np.float64)
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris.append([i, j, j + sides])
        tris.append([i, j + sides, i + sides])
        tris.append([j, i, c0_idx])
        tris.append([i + sides, j + sides, c1_idx])
    return _define_trimesh(stage, prim_path, verts, np.array(tris, dtype=np.int32), color)


def export_3mf_parts(cad_dir: Path, out_board_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert the 5 official IDB fabrication 3MF meshes into Z-up meter OpenUSD assets."""
    exported: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Panels A, B, C: local 3MF frame has front face at z=10 mm, width along x, height along y.
    # Right-handed rotation to board frame: X_b = (10 - z)/1000, Y_b = -x/1000, Z_b = y/1000 (det = +1).
    panel_color = (0.93, 0.94, 0.96)
    for label, src_name, dst_name in (
        ("A", "panel-a-v3.3MF", "panel_a.usda"),
        ("B", "panel-b-v3.3MF", "panel_b.usda"),
        ("C", "panel-c-v3.3MF", "panel_c.usda"),
    ):
        v_mm, tris = load_3mf_mesh(cad_dir / "fabrication" / src_name)
        v_m = np.column_stack(((10.0 - v_mm[:, 2]) / 1000.0, -v_mm[:, 0] / 1000.0, v_mm[:, 1] / 1000.0))
        stage, _ = _create_stage(out_board_dir / dst_name)
        _define_trimesh(stage, f"/Visual/panel_{label.lower()}", v_m, tris, panel_color)
        stage.GetRootLayer().Save()
        exported[dst_name] = (v_m, tris)

    # Fiber polishing station: local 3MF frame has x in [-30, 30], y in [0, 12] (pad front at y=12),
    # z in [-44.45, 0] (0 at top, -44.45 at bottom touching Panel A).
    # In STEP assembly, it is rotated so X_s = x + 115, Y_s = 252 - z, Z_s = 135 + y.
    # Mapping to local board-aligned frame centered at bracket bottom-front-center (y=10 -> X_b=0, z=0 -> Z_b=0.04445):
    # Using exact STEP->board formula minus bracket base origin (0, 0, 0.262):
    #   X_b = (10.0 - y) / 1000.0   (front disc at X_b = -0.002 m, rear face at X_b = +0.010 m)
    #   Y_b = -x / 1000.0
    #   Z_b = -z / 1000.0           (0 at Panel A top edge, +0.04445 at station top)
    # Check det: X=-y, Y=-x, Z=-z -> det = -1? Check: R = [[0,-1,0],[-1,0,0],[0,0,-1]]; det = -(-1)*(-1) = -1!
    # Wait: in STEP, fiber-polishing-v3 had T_step with X=(1,0,0), Y=(0,0,1), Z=(0,-1,0) -> det = +1!
    # So STEP_pos = (115 + x, 252 - z, 135 + y).
    # Applying right-handed step_mm_to_board_m gives:
    #   X_b = (145 - (135 + y))/1000 = (10 - y)/1000
    #   Y_b = (115 - (115 + x))/1000 = -x/1000
    #   Z_b = (252 - z + 10)/1000 = (262 - z)/1000 -> relative to Z_base=0.262: Z_rel = -z/1000!
    # Wait: why did R = [[0,-1,0],[-1,0,0],[0,0,-1]] have det = (-1)*(-1)*(-1)? No!
    # Compute det([[0,-1,0],[-1,0,0],[0,0,-1]]) = -(-1)*( (-1)*(-1) - 0 ) = +1 * (+1) = +1!! Right-handed!!
    v_pol_mm, tris_pol = load_3mf_mesh(cad_dir / "fabrication" / "fiber-polishing-v3.3MF")
    v_pol_m = np.column_stack(((10.0 - v_pol_mm[:, 1]) / 1000.0, -v_pol_mm[:, 0] / 1000.0, -v_pol_mm[:, 2] / 1000.0))
    stage, _ = _create_stage(out_board_dir / "fiber_polishing_station.usda")
    _define_trimesh(stage, "/Visual/fiber_polishing_station", v_pol_m, tris_pol, (0.94, 0.94, 0.96))
    stage.GetRootLayer().Save()
    exported["fiber_polishing_station.usda"] = (v_pol_m, tris_pol)

    # Cable management clip: local 3MF frame x in [-10, 10], y in [-6, 6], z in [0, 12] (mm -> m).
    v_clip_mm, tris_clip = load_3mf_mesh(cad_dir / "fabrication" / "cable-management.3MF")
    v_clip_m = v_clip_mm / 1000.0
    stage, _ = _create_stage(out_board_dir / "cable_management_clip.usda")
    _define_trimesh(stage, "/Visual/cable_management_clip", v_clip_m, tris_clip, (0.20, 0.22, 0.25))
    stage.GetRootLayer().Save()
    exported["cable_management_clip.usda"] = (v_clip_m, tris_clip)

    return exported


def build_connector_visual_usd(profile: ConnectorProfile, out_dir: Path) -> None:
    """Create multi-part visual port.usda and socket.usda matching the ConnectorProfile frame."""
    fam_dir = out_dir / profile.family
    fam_dir.mkdir(parents=True, exist_ok=True)

    # --- Male connector port visual (shoulder at X=0, tip along +X, housing/boot along -X) ---
    stage, _ = _create_stage(fam_dir / "port.usda")
    y_vals = [p[0] for p in profile.tip_outline]
    z_vals = [p[1] for p in profile.tip_outline]
    _define_box_mesh(
        stage,
        "/Visual/tip",
        (0.0, profile.tip_length, min(y_vals), max(y_vals), min(z_vals), max(z_vals)),
        profile.port_color,
    )
    sections = profile.housing_sections or (profile.housing_bounds,)
    for idx, bounds in enumerate(sections):
        color = profile.boot_color if (idx == 0 and len(sections) > 1) else profile.port_color
        _define_box_mesh(stage, f"/Visual/housing_{idx}", bounds, color)
    if profile.latch_bounds is not None:
        _define_box_mesh(stage, "/Visual/latch", profile.latch_bounds, (0.92, 0.93, 0.96))
    if profile.ferrule is not None:
        for idx, y_off in enumerate(profile.ferrule.y_offsets):
            _define_cylinder_x_mesh(
                stage,
                f"/Visual/ferrule_{idx}",
                profile.tip_length - 0.0005,
                profile.tip_length + profile.ferrule.protrusion,
                y_off,
                profile.ferrule.z_offset,
                profile.ferrule.radius,
                (0.96, 0.96, 0.98),
            )
    stage.GetRootLayer().Save()

    # --- Female receptacle socket visual (mouth at X=0, bore along +X) ---
    stage_s, _ = _create_stage(fam_dir / "socket.usda")
    oy0, oy1, oz0, oz1 = profile.socket_outer_bounds
    c = profile.clearance
    hy0, hy1 = min(y_vals) - c, max(y_vals) + c
    hz0, hz1 = min(z_vals) - c, max(z_vals) + c
    latch = profile.collision_latch_bounds
    latch_top = max(hz1, latch[5] + c) if latch is not None else hz1
    d = profile.socket_depth
    walls = {
        "wall_left": (0.0, d, oy0, hy0, oz0, oz1),
        "wall_right": (0.0, d, hy1, oy1, oz0, oz1),
        "wall_bottom": (0.0, d, hy0, hy1, oz0, hz0),
        "wall_top": (0.0, d, hy0, hy1, latch_top, oz1),
        "backstop": (d, d + 0.002, oy0, oy1, oz0, oz1),
    }
    if latch is not None:
        ly0, ly1 = latch[2] - c, latch[3] + c
        walls["shoulder_left"] = (0.0, d, hy0, ly0, hz1, latch_top)
        walls["shoulder_right"] = (0.0, d, ly1, hy1, hz1, latch_top)
    for name, bounds in walls.items():
        if min(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) > 1e-6:
            _define_box_mesh(stage_s, f"/Visual/{name}", bounds, profile.socket_color)
    stage_s.GetRootLayer().Save()


def build_board_manifest_and_assembly(
    step_records: list[dict],
    cad_dir: Path,
    exported_3mf: dict[str, tuple[np.ndarray, np.ndarray]],
    out_board_dir: Path,
) -> dict:
    """Build board_geometry.json and assembled idb_board_assembly.usda from exact STEP data."""
    # Identify polishing pad disc center and radius from fiber-polishing-v3.3MF.
    v_pol_mm, _ = load_3mf_mesh(cad_dir / "fabrication" / "fiber-polishing-v3.3MF")
    front = v_pol_mm[v_pol_mm[:, 1] > 11.9]
    pad_cx_mm = float(front[:, 0].mean())
    pad_cz_mm = float(front[:, 2].mean())
    pad_radius_m = float(np.hypot(front[:, 0] - pad_cx_mm, front[:, 2] - pad_cz_mm).max() / 1000.0)
    # Polishing station sits at STEP (115, 252, 135) with pad face at Z_s = 147.0 mm -> X_b = -0.002 m.
    pad_center_step = np.array([115.0 + pad_cx_mm, 252.0 - pad_cz_mm, 135.0 + 12.0])
    pad_center_board = step_mm_to_board_m(pad_center_step)
    pad_origin_board = step_mm_to_board_m(np.array([115.0, 252.0, 145.0]))

    # Extract all 7 aluminum 2020 extrusions.
    extrusions = []
    ext_records = [r for r in step_records if r["name"] == "extrusion-2020-250"]
    for idx, r in enumerate(ext_records):
        corners = np.vstack([r["min_step_mm"], r["max_step_mm"]])
        b_corners = step_mm_to_board_m(corners)
        b_min, b_max = b_corners.min(axis=0), b_corners.max(axis=0)
        extrusions.append({
            "name": f"extrusion_{idx}",
            "center": [round(float(v), 6) for v in 0.5 * (b_min + b_max)],
            "size": [round(float(v), 6) for v in (b_max - b_min)],
        })

    # Extract panels A, B, C.
    panels = {}
    panel_map = {
        "panel-A-v3": ("A", "board/panel_a.usda"),
        "panel-B-v3": ("B", "board/panel_b.usda"),
        "panel-C-v3": ("C", "board/panel_c.usda"),
    }
    for r in step_records:
        if r["name"] in panel_map:
            key, rel = panel_map[r["name"]]
            b_corners = step_mm_to_board_m(np.vstack([r["min_step_mm"], r["max_step_mm"]]))
            b_min, b_max = b_corners.min(axis=0), b_corners.max(axis=0)
            panels[key] = {
                "center": [round(float(v), 6) for v in 0.5 * (b_min + b_max)],
                "size": [round(float(v), 6) for v in (b_max - b_min)],
                "usd_relpath": rel,
            }

    # Extract all 32 ports grouped by panel and sorted left-to-right (ascending STEP X -> descending Board Y).
    step_to_family = {
        "rj45-port": "rj45",
        "sc-port": "sc_simplex",
        "lc-simplex-port": "lc_simplex",
        "lc-duplex-port": "lc_duplex",
    }
    # Official latch/key orientations from datacenter-cable-manipulation/instructions.md:
    # Roll about +X from upright (+Z latch):
    #   top/up (+Z_board):   roll = 0.0
    #   bottom (-Z_board):   roll = pi
    #   left   (+Y_board):   roll = -pi/2  (R_x(-pi/2) @ [0,0,1] = [0,+1,0])
    latch_rules = {
        ("A", "rj45"): ([0.0, 0.0, 1.0], 0.0),
        ("A", "sc_simplex"): ([0.0, 1.0, 0.0], -0.5 * math.pi),
        ("A", "lc_duplex"): ([0.0, 1.0, 0.0], -0.5 * math.pi),
        ("B", "rj45"): ([0.0, 0.0, -1.0], math.pi),
        ("B", "lc_simplex"): ([0.0, 0.0, 1.0], 0.0),
        ("B", "lc_duplex"): ([0.0, 1.0, 0.0], -0.5 * math.pi),
        ("C", "rj45"): ([0.0, 0.0, 1.0], 0.0),
        ("C", "lc_duplex"): ([0.0, 1.0, 0.0], -0.5 * math.pi),
    }

    ports: dict[str, dict] = {}
    for sub_name, panel_letter in (
        ("panel-A-subassembly", "A"),
        ("panel-B-subassembly", "B"),
        ("panel-C-subassembly", "C"),
    ):
        panel_ports = [r for r in step_records if r["parent"] == sub_name and r["name"] in step_to_family]
        panel_ports.sort(key=lambda r: float(0.5 * (r["min_step_mm"][0] + r["max_step_mm"][0])))
        # Column numbering: A and B have cols 1..12; C has cols 1..4 and 9..12 (5..8 blank).
        columns = list(range(1, 13)) if panel_letter in ("A", "B") else [1, 2, 3, 4, 9, 10, 11, 12]
        assert len(panel_ports) == len(columns), f"Panel {panel_letter}: expected {len(columns)} ports, got {len(panel_ports)}"
        for col, r in zip(columns, panel_ports):
            family = step_to_family[r["name"]]
            ctr_step = 0.5 * (r["min_step_mm"] + r["max_step_mm"])
            # Front mouth plane in STEP is max Z_step of the coupler body.
            mouth_step = np.array([ctr_step[0], ctr_step[1], r["max_step_mm"][2]])
            mouth_board = step_mm_to_board_m(mouth_step)
            b_corners = step_mm_to_board_m(np.vstack([r["min_step_mm"], r["max_step_mm"]]))
            b_size = b_corners.max(axis=0) - b_corners.min(axis=0)
            latch_dir, roll_rad = latch_rules[(panel_letter, family)]
            port_id = f"{panel_letter}{col}"
            ports[port_id] = {
                "panel": panel_letter,
                "column": col,
                "family": family,
                "mouth_pos": [round(float(v), 6) for v in mouth_board],
                "socket_quat": list(quat_from_roll_x(roll_rad)),
                "latch_dir": latch_dir,
                "roll_rad": round(roll_rad, 8),
                "protrusion_m": round(float(-mouth_board[0]), 6),
                "cad_envelope_m": [round(float(v), 6) for v in b_size],
            }

    # 4 optional cable-management clips mounted along left/right aluminum uprights between Panels A and B.
    cable_clips = [
        {"name": "clip_left_upper", "pos": [0.005, 0.125, 0.200], "quat": [1.0, 0.0, 0.0, 0.0]},
        {"name": "clip_left_lower", "pos": [0.005, 0.125, 0.175], "quat": [1.0, 0.0, 0.0, 0.0]},
        {"name": "clip_right_upper", "pos": [0.005, -0.125, 0.200], "quat": [0.0, 0.0, 0.0, 1.0]},
        {"name": "clip_right_lower", "pos": [0.005, -0.125, 0.175], "quat": [0.0, 0.0, 0.0, 1.0]},
    ]

    all_b = np.vstack([
        step_mm_to_board_m(np.vstack([r["min_step_mm"], r["max_step_mm"]]))
        for r in step_records
    ])
    footprint = (all_b.max(axis=0) - all_b.min(axis=0)).tolist()

    manifest = {
        "version": "idb-board-1-v3.5",
        "source_step": "datacenter-cable-manipulation/source/datacenter-benchmark-assembly-v3.5.step",
        "coordinate_frame": "Z-up meters; X=0 at panel front face (+X into board), Y=0 horizontal center (+Y left), Z=0 base bottom",
        "footprint_m": [round(float(v), 6) for v in footprint],
        "panels": panels,
        "polishing_pad": {
            "center": [round(float(v), 6) for v in pad_center_board],
            "normal": [-1.0, 0.0, 0.0],
            "radius": round(pad_radius_m, 6),
            "thickness": 0.002,
            "bracket_origin": [round(float(v), 6) for v in pad_origin_board],
        },
        "extrusions": extrusions,
        "ports": ports,
        "cable_clips": cable_clips,
    }
    manifest_path = out_board_dir / "board_geometry.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Compose visual-only assembled IDB Board #1 USD (for inspection / rendering reference).
    stage, _ = _create_stage(out_board_dir / "idb_board_assembly.usda", default_prim="IDBBoard")
    frame_xf = UsdGeom.Xform.Define(stage, "/IDBBoard/Frame")
    for ext in extrusions:
        cx, cy, cz = ext["center"]
        sx, sy, sz = ext["size"]
        _define_box_mesh(
            stage,
            f"/IDBBoard/Frame/{ext['name']}",
            (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2),
            (0.12, 0.12, 0.13),
        )
    for letter, pinfo in panels.items():
        v_m, tris = exported_3mf[f"panel_{letter.lower()}.usda"]
        px, py, pz = pinfo["center"]
        # Panel local X already spans [0, 0.0115], Y spans [-0.125, 0.125], Z spans [-0.022225, 0.022225].
        shifted = v_m + np.array([0.0, py, pz])
        _define_trimesh(stage, f"/IDBBoard/Panel_{letter}", shifted, tris, (0.93, 0.94, 0.96))

    v_pol, tris_pol = exported_3mf["fiber_polishing_station.usda"]
    pol_shifted = v_pol + np.array([0.0, 0.0, pad_origin_board[2]])
    _define_trimesh(stage, "/IDBBoard/FiberPolishingStation", pol_shifted, tris_pol, (0.95, 0.95, 0.97))

    # Add visual port housings for all 32 ports.
    UsdGeom.Xform.Define(stage, "/IDBBoard/Ports")
    for pid, pinfo in ports.items():
        prof = get_connector_profile(pinfo["family"])
        xf = UsdGeom.Xform.Define(stage, f"/IDBBoard/Ports/{pid}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*pinfo["mouth_pos"]))
        w, x, y, z = pinfo["socket_quat"]
        xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        xf.GetPrim().GetReferences().AddReference(f"../connectors/{pinfo['family']}/socket.usda")
    stage.GetRootLayer().Save()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=None, help="Directory holding IDB CAD fabrication/ and source/ folders.")
    parser.add_argument("--fetch", action="store_true", help="Download official IDB CAD files if missing.")
    parser.add_argument("--output-dir", type=Path, default=Path(IDB_ASSET_DIR), help="Target directory for USD assets and manifest.")
    args = parser.parse_args()

    cad_dir = args.source_dir or (args.output_dir / "_cad_cache")
    if args.fetch:
        fetch_cad_files(cad_dir)

    step_path = cad_dir / "source" / "datacenter-benchmark-assembly-v3.5.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing STEP assembly at {step_path}. Pass --source-dir or --fetch.")

    out_board = args.output_dir / "board"
    out_conn = args.output_dir / "connectors"
    out_board.mkdir(parents=True, exist_ok=True)
    out_conn.mkdir(parents=True, exist_ok=True)

    for family in connector_families():
        build_connector_visual_usd(get_connector_profile(family), out_conn)
    exported_3mf = export_3mf_parts(cad_dir, out_board)
    step_records = parse_step_assembly(step_path)
    manifest = build_board_manifest_and_assembly(step_records, cad_dir, exported_3mf, out_board)
    print(
        f"[export_assets] exported {len(manifest['ports'])} ports, {len(manifest['panels'])} panels, "
        f"{len(manifest['extrusions'])} extrusions, and {len(connector_families())} connector families to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
