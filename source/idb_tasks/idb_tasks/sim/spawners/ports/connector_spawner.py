"""Procedural USD connector port and keyed socket spawners matching WireCraft.

Socket colliders are built from separate convex wall pieces around the bore and latch/key
relief slot. A single convex hull is never applied across a socket, which would seal the
opening and block insertion.
"""

from __future__ import annotations

import math
import os
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from idb_assets import IDB_ASSET_DIR
from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, configclass, require_isaaclab

from .connector_profiles import ConnectorProfile, get_connector_profile, normalize_family, rectangle


def polygon_area(points: list[tuple[float, float]]) -> float:
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1])) / 2.0


def offset_polygon(points: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    """Offset convex counterclockwise YZ edges outward by per-side clearance distance."""
    lines = []
    for a, b in zip(points, points[1:] + points[:1]):
        dy, dz = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dy, dz)
        ny, nz = dz / norm, -dy / norm
        lines.append((ny, nz, ny * a[0] + nz * a[1] + distance))
    result = []
    for prev_line, curr_line in zip(lines[-1:] + lines[:-1], lines):
        a, b, c = prev_line
        d, e, f = curr_line
        det = a * e - b * d
        result.append(((c * e - b * f) / det, (a * f - c * d) / det))
    return result


def _outside_edge(
    points: list[tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
) -> list[tuple[float, float]]:
    def side(p: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    result = []
    for p, q in zip(points, points[1:] + points[:1]):
        sp, sq = side(p), side(q)
        if sp <= 0:
            result.append(p)
        if (sp < 0 < sq) or (sq < 0 < sp):
            t = sp / (sp - sq)
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return result


def _define_convex_prism(
    stage: Usd.Stage,
    prim_path: str,
    x_min: float,
    x_max: float,
    outline: list[tuple[float, float]],
    color: tuple[float, float, float],
    purpose: str | None = None,
) -> UsdGeom.Mesh | None:
    if len(outline) < 3 or abs(polygon_area(outline)) < 1e-14 or x_max - x_min <= 1e-10:
        return None
    points = [Gf.Vec3f(x, y, z) for x in (x_min, x_max) for y, z in outline]
    n = len(outline)
    faces = [list(reversed(range(n))), list(range(n, 2 * n))]
    faces.extend([i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateExtentAttr([
        Gf.Vec3f(x_min, min(p[0] for p in outline), min(p[1] for p in outline)),
        Gf.Vec3f(x_max, max(p[0] for p in outline), max(p[1] for p in outline)),
    ])
    if purpose is not None:
        UsdGeom.Imageable(mesh.GetPrim()).CreatePurposeAttr(purpose)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def _define_convex_box(
    stage: Usd.Stage,
    prim_path: str,
    bounds: tuple[float, float, float, float, float, float],
    color: tuple[float, float, float],
    purpose: str | None = None,
) -> UsdGeom.Mesh | None:
    x0, x1, y0, y1, z0, z1 = bounds
    if min(x1 - x0, y1 - y0, z1 - z0) <= 1e-10:
        return None
    return _define_convex_prism(stage, prim_path, x0, x1, list(rectangle(y0, y1, z0, z1)), color, purpose=purpose)


def default_visual_usd(family: str, role: str) -> str:
    """Resolve the package-bundled visual USDA path for a connector port or socket."""
    return os.path.join(IDB_ASSET_DIR, "connectors", normalize_family(family), f"{role}.usda")


def _attach_visual(stage: Usd.Stage, prim_path: str, visual_usd_path: str | None, offset_x: float = 0.0) -> None:
    if not visual_usd_path:
        return
    if not os.path.isfile(visual_usd_path):
        raise FileNotFoundError(f"Connector visual USD not found: {visual_usd_path}")
    visual = UsdGeom.Xform.Define(stage, f"{prim_path}/visual")
    visual.GetPrim().GetReferences().AddReference(visual_usd_path)
    if offset_x != 0.0:
        visual.AddTranslateOp().Set(Gf.Vec3d(offset_x, 0.0, 0.0))
    for child in stage.GetPrimAtPath(prim_path).GetChildren():
        if child.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(child).CreatePurposeAttr("guide")


def build_connector_port_usd(
    stage: Usd.Stage,
    prim_path: str,
    profile: ConnectorProfile,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    mass: float = 0.01,
    visual_usd_path: str | None = None,
    apply_rigid_body: bool = True,
) -> Usd.Prim:
    """Author a rigid connector plug with convex collider parts and optional CAD visual."""
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w, x, y, z = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    if apply_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass_api.CreateMassAttr().Set(float(mass))

    _define_convex_prism(stage, f"{prim_path}/tip", 0.0, profile.tip_length, list(profile.tip_outline), profile.port_color)
    for idx, bounds in enumerate(profile.housing_sections or (profile.housing_bounds,)):
        name = "housing" if idx == 0 else f"housing_{idx}"
        _define_convex_box(stage, f"{prim_path}/{name}", bounds, profile.boot_color if idx == 0 else profile.port_color)
    latch = profile.collision_latch_bounds
    if latch is not None:
        _define_convex_box(stage, f"{prim_path}/latch", latch, (0.92, 0.93, 0.96))

    _attach_visual(stage, prim_path, visual_usd_path)
    return root.GetPrim()


def build_connector_socket_usd(
    stage: Usd.Stage,
    prim_path: str,
    profile: ConnectorProfile,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clearance: float | None = None,
    depth: float | None = None,
    mouth_x: float = 0.0,
    backstop_thickness: float = 0.002,
    visual_usd_path: str | None = None,
    apply_rigid_body: bool = False,
) -> Usd.Prim:
    """Author a keyed receptacle socket from disjoint convex wall pieces around an open bore."""
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    clr = profile.clearance if clearance is None else float(clearance)
    dep = profile.socket_depth if depth is None else float(depth)
    outer = profile.socket_outer_bounds
    if not all(math.isfinite(v) for v in (clr, dep, mouth_x, backstop_thickness, *outer)):
        raise ValueError("Socket dimensions must be finite.")
    if clr < 0 or dep <= 0 or backstop_thickness < 0:
        raise ValueError("Socket clearance/backstop must be >= 0 and depth > 0.")

    hole = offset_polygon(list(profile.tip_outline), clr)
    y0, y1 = min(p[0] for p in hole), max(p[0] for p in hole)
    z0, z1 = min(p[1] for p in hole), max(p[1] for p in hole)
    oy0, oy1, oz0, oz1 = outer
    latch = profile.collision_latch_bounds
    latch_top = max(z1, latch[5] + clr) if latch is not None else z1
    if not (oy0 < y0 < y1 < oy1 and oz0 < z0 < latch_top < oz1):
        raise ValueError(f"Socket outer bounds {outer} too small for bore ({y0},{y1},{z0},{latch_top}).")

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w, x, y, z = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    if apply_rigid_body:
        rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(True)

    x0, x1 = mouth_x, mouth_x + dep
    walls = {
        "left": (x0, x1, oy0, y0, oz0, oz1),
        "right": (x0, x1, y1, oy1, oz0, oz1),
        "bottom": (x0, x1, y0, y1, oz0, z0),
        "top": (x0, x1, y0, y1, latch_top, oz1),
    }
    if latch is not None:
        ly0, ly1 = latch[2] - clr, latch[3] + clr
        walls["shoulder_left"] = (x0, x1, y0, ly0, z1, latch_top)
        walls["shoulder_right"] = (x0, x1, ly1, y1, z1, latch_top)
    for name, bounds in walls.items():
        _define_convex_box(stage, f"{prim_path}/{name}", bounds, profile.socket_color)

    inner_rect = list(rectangle(y0, y1, z0, z1))
    for idx, (a, b) in enumerate(zip(hole, hole[1:] + hole[:1])):
        poly = _outside_edge(inner_rect, a, b)
        _define_convex_prism(stage, f"{prim_path}/key_{idx}", x0, x1, poly, profile.socket_color)

    if backstop_thickness > 0:
        _define_convex_box(stage, f"{prim_path}/backstop", (x1, x1 + backstop_thickness, *outer), profile.socket_color)

    _attach_visual(stage, prim_path, visual_usd_path, offset_x=mouth_x)
    return root.GetPrim()


def _bind_isaac_physics(stage: Usd.Stage, prim_path: str, cfg: Any) -> None:
    """Apply Isaac Lab rigid/collision/material schemas when running inside Isaac Sim."""
    from isaaclab.sim import schemas  # type: ignore[import-not-found]
    from isaaclab.sim.utils import bind_physics_material  # type: ignore[import-not-found]

    mat_path = f"{prim_path}/physics_material"
    if cfg.physics_material is not None:
        cfg.physics_material.func(mat_path, cfg.physics_material)
    if getattr(cfg, "rigid_props", None) is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage)
    if getattr(cfg, "mass_props", None) is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage)
    col_props = getattr(cfg, "collision_props", None)
    for child in stage.GetPrimAtPath(prim_path).GetChildren():
        if child.IsA(UsdGeom.Mesh):
            if col_props is not None:
                schemas.define_collision_properties(str(child.GetPath()), col_props, stage)
            if cfg.physics_material is not None:
                bind_physics_material(str(child.GetPath()), mat_path)


def _spawn_port_impl(prim_path: str, cfg: IDBPortCfg, translation=None, orientation=None) -> Usd.Prim:
    require_isaaclab("spawn_idb_port")
    from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]

    stage = get_current_stage()
    profile = get_connector_profile(cfg.family)
    vis = cfg.visual_usd_path if cfg.visual_usd_path is not None else default_visual_usd(profile.family, "port")
    prim = build_connector_port_usd(
        stage,
        prim_path,
        profile,
        translation=translation,
        orientation=orientation,
        mass=cfg.mass,
        visual_usd_path=vis,
        apply_rigid_body=True,
    )
    _bind_isaac_physics(stage, prim_path, cfg)
    return prim


def _spawn_socket_impl(prim_path: str, cfg: IDBSocketCfg, translation=None, orientation=None) -> Usd.Prim:
    require_isaaclab("spawn_idb_socket")
    from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]

    stage = get_current_stage()
    profile = get_connector_profile(cfg.family)
    vis = cfg.visual_usd_path if cfg.visual_usd_path is not None else default_visual_usd(profile.family, "socket")
    prim = build_connector_socket_usd(
        stage,
        prim_path,
        profile,
        translation=translation,
        orientation=orientation,
        clearance=cfg.clearance,
        depth=cfg.depth,
        mouth_x=cfg.mouth_x,
        backstop_thickness=cfg.backstop_thickness,
        visual_usd_path=vis,
        apply_rigid_body=True,
    )
    _bind_isaac_physics(stage, prim_path, cfg)
    return prim


if HAS_ISAACLAB:
    from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg  # type: ignore
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg  # type: ignore
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg  # type: ignore
    from isaaclab.sim.utils import clone  # type: ignore

    spawn_idb_port = clone(_spawn_port_impl)
    spawn_idb_socket = clone(_spawn_socket_impl)
else:
    RigidObjectSpawnerCfg = object  # type: ignore[assignment,misc]
    spawn_idb_port = _spawn_port_impl
    spawn_idb_socket = _spawn_socket_impl


@configclass
class IDBPortCfg(RigidObjectSpawnerCfg):  # type: ignore[misc]
    """Isaac Lab spawner config for an IDB datacenter connector plug."""

    func: Any = spawn_idb_port
    family: str = "sc_simplex"
    mass: float = 0.01
    static_friction: float = 2.0
    dynamic_friction: float = 2.0
    visual_usd_path: str | None = None
    rigid_props: Any = (
        RigidBodyPropertiesCfg(max_depenetration_velocity=5.0, max_linear_velocity=5.0, max_angular_velocity=60.0)
        if HAS_ISAACLAB
        else None
    )
    mass_props: Any = MassPropertiesCfg(mass=0.01) if HAS_ISAACLAB else None
    collision_props: Any = CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0) if HAS_ISAACLAB else None
    physics_material: Any = (
        RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, friction_combine_mode="max")
        if HAS_ISAACLAB
        else None
    )


@configclass
class IDBSocketCfg(RigidObjectSpawnerCfg):  # type: ignore[misc]
    """Isaac Lab spawner config for an IDB datacenter receptacle socket."""

    func: Any = spawn_idb_socket
    family: str = "sc_simplex"
    clearance: float | None = None
    depth: float | None = None
    mouth_x: float = 0.0
    backstop_thickness: float = 0.002
    visual_usd_path: str | None = None
    rigid_props: Any = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True) if HAS_ISAACLAB else None
    collision_props: Any = CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0) if HAS_ISAACLAB else None
    physics_material: Any = (
        RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5, friction_combine_mode="multiply")
        if HAS_ISAACLAB
        else None
    )
