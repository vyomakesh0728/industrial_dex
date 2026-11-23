"""USD spawner for the CAD-accurate IDB Board #1 (Datacenter Cable Manipulation).

Spawns:
- Kinematic extrusion frame (7 aluminum 2020 bars) with convex box colliders.
- CAD visual meshes for Panels A, B, and C (from official .3MF files) with solid-margin
  colliders that leave all 32 port openings unobstructed.
- All 32 keyed convex-wall receptacle sockets (RJ45, SC simplex, LC simplex, LC duplex)
  placed at their exact STEP assembly coordinates and latch/key orientations.
- Dedicated kinematic rigid body ``polishing_pad`` at the Fiber Polishing Station disc so
  an Isaac Lab ContactSensor can measure exact 1-2 N ferrule polishing contact forces.
- Optional 4 side-mounted cable-management clips (from official cable-management.3MF).
"""

from __future__ import annotations

import math
import os
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from idb_assets import IDB_ASSET_DIR
from idb_assets.board_spec import BoardSpec, load_board_spec
from idb_tasks.sim.spawners.ports.connector_profiles import get_connector_profile
from idb_tasks.sim.spawners.ports.connector_spawner import (
    _define_convex_box,
    build_connector_socket_usd,
    default_visual_usd,
)
from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, configclass, require_isaaclab


def _define_disc_collider(
    stage: Usd.Stage,
    prim_path: str,
    x0: float,
    x1: float,
    radius: float,
    color: tuple[float, float, float],
    sides: int = 24,
) -> UsdGeom.Mesh:
    ring0 = [Gf.Vec3f(x0, radius * math.cos(i * math.tau / sides), radius * math.sin(i * math.tau / sides)) for i in range(sides)]
    ring1 = [Gf.Vec3f(x1, radius * math.cos(i * math.tau / sides), radius * math.sin(i * math.tau / sides)) for i in range(sides)]
    points = ring0 + ring1
    faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    faces.extend([i, (i + 1) % sides, (i + 1) % sides + sides, i + sides] for i in range(sides))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateExtentAttr([Gf.Vec3f(x0, -radius, -radius), Gf.Vec3f(x1, radius, radius)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def build_idb_board_usd(
    stage: Usd.Stage,
    prim_path: str,
    spec: BoardSpec | None = None,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    socket_clearance_override: float | None = None,
    include_cable_clips: bool = False,
) -> Usd.Prim:
    """Author the full IDB Board #1 hierarchy on ``stage`` at ``prim_path``."""
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    board = spec or load_board_spec()

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w, x, y, z = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))

    # Static/kinematic board structural body (extrusions, panel borders, 32 keyed sockets).
    frame_body = UsdGeom.Xform.Define(stage, f"{prim_path}/frame")
    rb_frame = UsdPhysics.RigidBodyAPI.Apply(frame_body.GetPrim())
    rb_frame.CreateKinematicEnabledAttr().Set(True)

    for ext in board.extrusions:
        cx, cy, cz = ext.center
        sx, sy, sz = ext.size
        _define_convex_box(
            stage,
            f"{prim_path}/frame/{ext.name}",
            (cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0, cz - sz / 2.0, cz + sz / 2.0),
            (0.12, 0.12, 0.14),
        )

    # Panels A, B, C: official 3MF visual meshes + top/bottom/side frame border colliders leaving port strip open.
    panels_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/panels")
    for letter, panel in board.panels.items():
        p_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/panels/panel_{letter.lower()}")
        p_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, panel.center[1], panel.center[2]))
        usd_file = os.path.join(IDB_ASSET_DIR, panel.usd_relpath)
        if os.path.isfile(usd_file):
            vis = UsdGeom.Xform.Define(stage, f"{prim_path}/panels/panel_{letter.lower()}/visual")
            vis.GetPrim().GetReferences().AddReference(usd_file)
        # Upper/lower horizontal strips (outside +/-11.5 mm port band) and left/right mounting ears.
        pz_c = panel.center[2]
        _define_convex_box(stage, f"{prim_path}/frame/panel_{letter.lower()}_top_rail", (0.0, 0.011, -0.125, 0.125, pz_c + 0.012, pz_c + 0.0222), (0.92, 0.93, 0.95), purpose="guide")
        _define_convex_box(stage, f"{prim_path}/frame/panel_{letter.lower()}_bot_rail", (0.0, 0.011, -0.125, 0.125, pz_c - 0.0222, pz_c - 0.012), (0.92, 0.93, 0.95), purpose="guide")
        _define_convex_box(stage, f"{prim_path}/frame/panel_{letter.lower()}_left_ear", (0.0, 0.011, 0.105, 0.125, pz_c - 0.012, pz_c + 0.012), (0.92, 0.93, 0.95), purpose="guide")
        _define_convex_box(stage, f"{prim_path}/frame/panel_{letter.lower()}_right_ear", (0.0, 0.011, -0.125, -0.105, pz_c - 0.012, pz_c + 0.012), (0.92, 0.93, 0.95), purpose="guide")

    # All 32 keyed convex-wall receptacle sockets (under frame body hierarchy for shared kinematic motion,
    # also exposed under /ports/{port_id} for clear inspection).
    UsdGeom.Xform.Define(stage, f"{prim_path}/ports")
    for port_id, port in board.ports.items():
        profile = get_connector_profile(port.family)
        vis_path = default_visual_usd(port.family, "socket")
        build_connector_socket_usd(
            stage,
            f"{prim_path}/ports/{port_id}",
            profile,
            translation=port.mouth_pos,
            orientation=port.socket_quat,
            clearance=socket_clearance_override,
            mouth_x=0.0,
            backstop_thickness=0.002,
            visual_usd_path=vis_path if os.path.isfile(vis_path) else None,
            apply_rigid_body=True,
        )

    # Separate kinematic rigid body for the Fiber Polishing Station so ContactSensor reads ferrule-pad force directly.
    pad = board.polishing_pad
    pad_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/polishing_pad")
    pad_xf.AddTranslateOp().Set(Gf.Vec3d(*pad.center))
    rb_pad = UsdPhysics.RigidBodyAPI.Apply(pad_xf.GetPrim())
    rb_pad.CreateKinematicEnabledAttr().Set(True)
    # Disc collider lies on X in [0.0, 0.012] relative to pad front face pad.center.
    _define_disc_collider(stage, f"{prim_path}/polishing_pad/pad_disc", 0.0, 0.012, pad.radius, (0.95, 0.95, 0.97))
    pol_vis_file = os.path.join(IDB_ASSET_DIR, "board", "fiber_polishing_station.usda")
    if os.path.isfile(pol_vis_file):
        pol_vis = UsdGeom.Xform.Define(stage, f"{prim_path}/polishing_pad/visual")
        # fiber_polishing_station.usda origin is at bracket bottom-front-center (bracket_origin).
        offset = tuple(pad.bracket_origin[i] - pad.center[i] for i in range(3))
        pol_vis.AddTranslateOp().Set(Gf.Vec3d(*offset))
        pol_vis.GetPrim().GetReferences().AddReference(pol_vis_file)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/polishing_pad/pad_disc")).CreatePurposeAttr("guide")

    if include_cable_clips:
        UsdGeom.Xform.Define(stage, f"{prim_path}/cable_clips")
        clip_vis_file = os.path.join(IDB_ASSET_DIR, "board", "cable_management_clip.usda")
        for clip in board.cable_clips:
            c_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/cable_clips/{clip.name}")
            c_xf.AddTranslateOp().Set(Gf.Vec3d(*clip.pos))
            cw, cx, cy, cz = clip.quat
            c_xf.AddOrientOp().Set(Gf.Quatf(cw, cx, cy, cz))
            rb_c = UsdPhysics.RigidBodyAPI.Apply(c_xf.GetPrim())
            rb_c.CreateKinematicEnabledAttr().Set(True)
            _define_convex_box(stage, f"{prim_path}/cable_clips/{clip.name}/collider", (-0.010, 0.010, -0.006, 0.006, 0.0, 0.012), (0.20, 0.22, 0.25), purpose="guide")
            if os.path.isfile(clip_vis_file):
                c_vis = UsdGeom.Xform.Define(stage, f"{prim_path}/cable_clips/{clip.name}/visual")
                c_vis.GetPrim().GetReferences().AddReference(clip_vis_file)

    return root.GetPrim()


def _spawn_board_impl(prim_path: str, cfg: IDBBoardCfg, translation=None, orientation=None) -> Usd.Prim:
    require_isaaclab("spawn_idb_board")
    from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]
    from isaaclab.sim import schemas  # type: ignore[import-not-found]

    stage = get_current_stage()
    prim = build_idb_board_usd(
        stage,
        prim_path,
        spec=load_board_spec(cfg.manifest_path),
        translation=translation,
        orientation=orientation,
        socket_clearance_override=cfg.socket_clearance_override,
        include_cable_clips=cfg.include_cable_clips,
    )
    if cfg.activate_polishing_contact_sensor:
        from pxr import PhysxSchema  # type: ignore[import-not-found]

        pad_prim = stage.GetPrimAtPath(f"{prim_path}/polishing_pad")
        PhysxSchema.PhysxContactReportAPI.Apply(pad_prim).CreateThresholdAttr().Set(0.0)
    if cfg.collision_props is not None:
        for desc in Usd.PrimRange(prim):
            if desc.IsA(UsdGeom.Mesh) and desc.HasAPI(UsdPhysics.CollisionAPI):
                schemas.define_collision_properties(str(desc.GetPath()), cfg.collision_props, stage)
    return prim


if HAS_ISAACLAB:
    from isaaclab.sim.schemas import CollisionPropertiesCfg  # type: ignore
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg  # type: ignore
    from isaaclab.sim.utils import clone  # type: ignore

    spawn_idb_board = clone(_spawn_board_impl)
else:
    SpawnerCfg = object  # type: ignore[assignment,misc]
    spawn_idb_board = _spawn_board_impl


@configclass
class IDBBoardCfg(SpawnerCfg):  # type: ignore[misc]
    """Isaac Lab spawner configuration for IDB Board #1."""

    func: Any = spawn_idb_board
    manifest_path: str | None = None
    socket_clearance_override: float | None = None
    include_cable_clips: bool = False
    activate_polishing_contact_sensor: bool = True
    collision_props: Any = CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0) if HAS_ISAACLAB else None
