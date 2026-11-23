"""Behavioral USD spawner tests using real OpenUSD (pxr) stages."""

from __future__ import annotations

import math

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from idb_assets.board_spec import load_board_spec
from idb_tasks.sim.spawners.board.board_spawner import build_idb_board_usd
from idb_tasks.sim.spawners.ports.connector_profiles import connector_families, get_connector_profile
from idb_tasks.sim.spawners.ports.connector_spawner import (
    build_connector_port_usd,
    build_connector_socket_usd,
    offset_polygon,
)
from idb_tasks.sim.spawners.wires.wire_presets import cable_presets, get_cable_preset
from idb_tasks.sim.spawners.wires.wire_spawner import build_articulated_wire_usd, build_deformable_wire_mesh_usd


@pytest.mark.parametrize("family", connector_families())
def test_socket_collider_walls_never_bridge_bore(family: str):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    profile = get_connector_profile(family)
    root = build_connector_socket_usd(stage, f"/World/socket_{family}", profile)
    assert root.IsValid()

    # Check that each wall mesh uses convexHull and NO wall's AABB covers the bore center (y=0, z=0) inside [0, depth).
    wall_meshes = [c for c in root.GetChildren() if c.IsA(UsdGeom.Mesh)]
    wall_names = {c.GetName() for c in wall_meshes}
    assert {"left", "right", "bottom", "top", "backstop"}.issubset(wall_names)

    for prim in wall_meshes:
        assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} missing CollisionAPI"
        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
        assert mesh_col.GetApproximationAttr().Get() == "convexHull"
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        if prim.GetName() != "backstop":
            # Bore centerline (y=0, z=0) at mid-depth must lie strictly outside every side wall box.
            mid_x = 0.5 * profile.socket_depth
            covers_center = (min(xs) < mid_x < max(xs)) and (min(ys) < 0.0 < max(ys)) and (min(zs) < 0.0 < max(zs))
            assert not covers_center, f"{family} wall {prim.GetName()} bridges socket bore!"


@pytest.mark.parametrize("family", connector_families())
def test_port_spawner_creates_rigid_body_and_collider_parts(family: str):
    stage = Usd.Stage.CreateInMemory()
    profile = get_connector_profile(family)
    root = build_connector_port_usd(stage, f"/World/port_{family}", profile, mass=0.012)
    assert root.IsValid()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)
    assert math.isclose(UsdPhysics.MassAPI(root).GetMassAttr().Get(), 0.012, rel_tol=1e-5)

    child_names = {c.GetName() for c in root.GetChildren()}
    assert "tip" in child_names
    assert "housing" in child_names
    if profile.collision_latch_bounds is not None:
        assert "latch" in child_names


def test_board_spawner_builds_all_32_ports_and_polishing_pad():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    spec = load_board_spec()
    root = build_idb_board_usd(stage, "/World/IDBBoard", spec, include_cable_clips=True)
    assert root.IsValid()

    ports_root = stage.GetPrimAtPath("/World/IDBBoard/ports")
    assert ports_root.IsValid()
    spawned_ports = {c.GetName() for c in ports_root.GetChildren()}
    assert spawned_ports == set(spec.ports.keys())
    assert len(spawned_ports) == 32

    pad_prim = stage.GetPrimAtPath("/World/IDBBoard/polishing_pad")
    assert pad_prim.IsValid()
    assert pad_prim.HasAPI(UsdPhysics.RigidBodyAPI)

    clips_root = stage.GetPrimAtPath("/World/IDBBoard/cable_clips")
    assert clips_root.IsValid()
    assert len(list(clips_root.GetChildren())) == 4

    # Panels must sit under /frame so kinematic board pose moves panel visuals rigidly.
    for letter in ("a", "b", "c"):
        assert stage.GetPrimAtPath(f"/World/IDBBoard/frame/panels/panel_{letter}").IsValid()


def test_heterogeneous_wire_spawner_supports_lc_to_sc_and_deformable():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Articulated SC-simplex to LC-simplex patch cable (Tasks 2 & 4).
    preset = get_cable_preset("os2_fiber_simplex")
    wire_root = build_articulated_wire_usd(
        stage,
        "/World/wire_sc_lc",
        length=preset.length,
        num_segments=preset.num_segments,
        radius=preset.radius,
        start_family="lc_simplex",
        end_family="sc_simplex",
        color=preset.color,
    )
    assert wire_root.IsValid()
    assert wire_root.HasAPI(UsdPhysics.ArticulationRootAPI)
    names = {c.GetName() for c in wire_root.GetChildren()}
    assert "start_port" in names and "end_port" in names
    assert f"seg_{preset.num_segments - 1}" in names
    assert "joint_start_port" in names and "joint_end_port" in names

    # Verify start_port and end_port have distinct connector tip lengths matching LC simplex vs SC simplex.
    start_tip = UsdGeom.Mesh(stage.GetPrimAtPath("/World/wire_sc_lc/start_port/tip")).GetPointsAttr().Get()
    end_tip = UsdGeom.Mesh(stage.GetPrimAtPath("/World/wire_sc_lc/end_port/tip")).GetPointsAttr().Get()
    assert math.isclose(max(p[0] for p in start_tip), get_connector_profile("lc_simplex").tip_length, abs_tol=1e-5)
    assert math.isclose(max(p[0] for p in end_tip), get_connector_profile("sc_simplex").tip_length, abs_tol=1e-5)

    # Inter-segment bending joints must lock axial torsion (rotX) and drive bending (rotY, rotZ).
    j1 = stage.GetPrimAtPath("/World/wire_sc_lc/joint_seg_1_to_parent")
    assert j1.IsValid()
    rot_x_lim = UsdPhysics.LimitAPI(j1, "rotX")
    assert rot_x_lim.GetLowAttr().Get() > rot_x_lim.GetHighAttr().Get()
    assert not UsdPhysics.DriveAPI(j1, "rotX").GetStiffnessAttr().HasAuthoredValue()
    assert UsdPhysics.DriveAPI(j1, "rotY").GetStiffnessAttr().Get() > 0.0
    assert UsdPhysics.DriveAPI(j1, "rotZ").GetStiffnessAttr().Get() > 0.0

    # Deformable FEM mesh wire.
    fem_prim = build_deformable_wire_mesh_usd(
        stage,
        "/World/wire_fem",
        length=0.30,
        num_segments=30,
        radius=0.002,
        color=(0.95, 0.82, 0.08),
    )
    assert fem_prim.IsValid() and fem_prim.IsA(UsdGeom.Mesh)
    pts = UsdGeom.Mesh(fem_prim).GetPointsAttr().Get()
    xs = [p[0] for p in pts]
    assert math.isclose(max(xs) - min(xs), 0.30, abs_tol=1e-4)
    assert set(cable_presets()) == {"cat6_ethernet", "os2_fiber_simplex", "os2_fiber_duplex"}


def test_offset_polygon_grows_outward_by_exact_clearance():
    base = [(-0.005, -0.003), (0.005, -0.003), (0.005, 0.003), (-0.005, 0.003)]
    grown = offset_polygon(base, 0.0005)
    ys = [p[0] for p in grown]
    zs = [p[1] for p in grown]
    assert math.isclose(min(ys), -0.0055, abs_tol=1e-8)
    assert math.isclose(max(ys), 0.0055, abs_tol=1e-8)
    assert math.isclose(min(zs), -0.0035, abs_tol=1e-8)
    assert math.isclose(max(zs), 0.0035, abs_tol=1e-8)
