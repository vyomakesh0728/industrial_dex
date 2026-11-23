"""Heterogeneous-endpoint articulated and FEM deformable wire USD spawners.

Follows WireCraft's WireAssemblyCfg convention:
- Cable spans local X in [0, length].
- ``start_port`` (far end, e.g. B1/B5/B9) sits at X=0 pointing outward along -X.
- ``end_port`` (active manipulation end, e.g. A1/A5/A9) sits at X=length pointing outward along +X.
- Supports distinct connector families on start_port and end_port (required for IDB's
  LC-simplex to SC-simplex fiber patch cables in Tasks 2 and 4).
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
except ModuleNotFoundError:  # Inside Isaac Sim python.sh prior to SimulationApp startup
    Gf = Sdf = Usd = UsdGeom = UsdPhysics = Vt = None  # type: ignore[assignment]

from idb_tasks.sim.spawners.ports.connector_profiles import get_connector_profile
from idb_tasks.sim.spawners.ports.connector_spawner import build_connector_port_usd, default_visual_usd
from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, HAS_PHYSX_SCHEMA, configclass, require_isaaclab


def _apply_d6_bending_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    pos0: tuple[float, float, float],
    pos1: tuple[float, float, float],
    stiffness: float,
    damping: float,
    joint_limit_rad: float | None,
) -> UsdPhysics.Joint:
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # Lock all 3 translational DOFs; drive bending (rotY, rotZ) and compliant torsion (rotX)
    # so 30 cm datacenter patch cords can bridge ports with distinct latch/key roll orientations.
    for axis in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        lim.CreateLowAttr().Set(1.0)
        lim.CreateHighAttr().Set(-1.0)

    for axis, stiff_scale in (("rotX", 0.5), ("rotY", 1.0), ("rotZ", 1.0)):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateStiffnessAttr().Set(float(stiffness) * stiff_scale)
        drive.CreateDampingAttr().Set(float(damping))
        if joint_limit_rad is not None:
            lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            deg = math.degrees(joint_limit_rad)
            lim.CreateLowAttr().Set(-deg)
            lim.CreateHighAttr().Set(deg)
    return joint


def build_articulated_wire_usd(
    stage: Usd.Stage,
    prim_path: str,
    length: float,
    num_segments: int,
    radius: float,
    start_family: str,
    end_family: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    stiffness: float = 0.25,
    damping: float = 0.005,
    joint_limit: float | None = 0.85,
    density: float = 950.0,
    capsule_overlap: float | None = 0.004,
    port_mass: float = 0.01,
    port_embed_depth: float = 0.012,
    color: tuple[float, float, float] = (0.06, 0.36, 0.85),
    self_collision: bool = True,
) -> Usd.Prim:
    """Author a WireCraft-compatible articulated capsule chain with heterogeneous start/end ports."""
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    if num_segments < 2 or length <= 0 or radius <= 0:
        raise ValueError("Wire requires num_segments >= 2, length > 0, and radius > 0.")

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w, x, y, z = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    seg_len = length / num_segments
    overlap = (2.0 * radius) if capsule_overlap is None else float(capsule_overlap)
    capsule_height = max(seg_len + overlap - 2.0 * radius, 0.001)
    seg_mass = max(density * math.pi * radius * radius * seg_len, 1.0e-4)

    seg_paths: list[str] = []
    for i in range(num_segments):
        seg_path = f"{prim_path}/seg_{i}"
        seg_paths.append(seg_path)
        cx = (i + 0.5) * seg_len
        cap = UsdGeom.Capsule.Define(stage, seg_path)
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(float(radius))
        cap.CreateHeightAttr(float(capsule_height))
        cap.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cap.AddTranslateOp().Set(Gf.Vec3d(cx, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        UsdPhysics.MassAPI.Apply(cap.GetPrim()).CreateMassAttr().Set(float(seg_mass))
        if i > 0:
            _apply_d6_bending_joint(
                stage,
                f"{prim_path}/joint_seg_{i}_to_parent",
                seg_paths[i - 1],
                seg_path,
                (0.5 * seg_len, 0.0, 0.0),
                (-0.5 * seg_len, 0.0, 0.0),
                stiffness=stiffness,
                damping=damping,
                joint_limit_rad=joint_limit,
            )
            # Filter collision between adjacent overlapping capsules so they never fight each other.
            UsdPhysics.FilteredPairsAPI.Apply(cap.GetPrim()).CreateFilteredPairsRel().AddTarget(Sdf.Path(seg_paths[i - 1]))

    # Start port at X=0 pointing along -X (rotated 180 deg about Z -> quat (0, 0, 0, 1)).
    start_prof = get_connector_profile(start_family)
    start_embed = min(port_embed_depth, start_prof.housing_bounds[1] - start_prof.housing_bounds[0])
    start_anchor = start_prof.cable_anchor_x(start_embed)
    start_pos = (start_anchor, 0.0, 0.0)  # rotated 180 deg around Z: port +X -> cable -X, so anchor at x=0 -> root at +start_anchor?
    # Check: p_world = root + R_z(pi) @ [anchor_x, 0, 0] = root - anchor_x. Setting p_world = 0 -> root = anchor_x (negative).
    # Wait: anchor_x < 0 (behind shoulder), so -anchor_x > 0 -> root_x = anchor_x (negative, sticking out toward -X)!
    start_root_x = start_anchor
    start_vis = default_visual_usd(start_prof.family, "port")
    start_prim = build_connector_port_usd(
        stage,
        f"{prim_path}/start_port",
        start_prof,
        translation=(start_root_x, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        mass=port_mass,
        visual_usd_path=start_vis if os.path.isfile(start_vis) else None,
        apply_rigid_body=True,
    )
    j_start = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/joint_start_port")
    j_start.CreateBody0Rel().SetTargets([Sdf.Path(seg_paths[0])])
    j_start.CreateBody1Rel().SetTargets([start_prim.GetPath()])
    j_start.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.5 * seg_len, 0.0, 0.0))
    j_start.CreateLocalRot0Attr().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
    j_start.CreateLocalPos1Attr().Set(Gf.Vec3f(start_anchor, 0.0, 0.0))
    j_start.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    UsdPhysics.FilteredPairsAPI.Apply(start_prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(seg_paths[0])])

    # End port at X=length pointing along +X (identity orientation).
    # p_world = root + anchor_x = length -> root = length - anchor_x (> length since anchor_x < 0).
    end_prof = get_connector_profile(end_family)
    end_embed = min(port_embed_depth, end_prof.housing_bounds[1] - end_prof.housing_bounds[0])
    end_anchor = end_prof.cable_anchor_x(end_embed)
    end_root_x = length - end_anchor
    end_vis = default_visual_usd(end_prof.family, "port")
    end_prim = build_connector_port_usd(
        stage,
        f"{prim_path}/end_port",
        end_prof,
        translation=(end_root_x, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        mass=port_mass,
        visual_usd_path=end_vis if os.path.isfile(end_vis) else None,
        apply_rigid_body=True,
    )
    j_end = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/joint_end_port")
    j_end.CreateBody0Rel().SetTargets([Sdf.Path(seg_paths[-1])])
    j_end.CreateBody1Rel().SetTargets([end_prim.GetPath()])
    j_end.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5 * seg_len, 0.0, 0.0))
    j_end.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j_end.CreateLocalPos1Attr().Set(Gf.Vec3f(end_anchor, 0.0, 0.0))
    j_end.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    UsdPhysics.FilteredPairsAPI.Apply(end_prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(seg_paths[-1])])

    if HAS_PHYSX_SCHEMA:
        from pxr import PhysxSchema  # type: ignore[import-not-found]

        art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        art.CreateEnabledSelfCollisionsAttr().Set(bool(self_collision))
        art.CreateSolverPositionIterationCountAttr().Set(64)
        art.CreateSolverVelocityIterationCountAttr().Set(4)

    return root.GetPrim()


def build_deformable_wire_mesh_usd(
    stage: Usd.Stage,
    prim_path: str,
    length: float,
    num_segments: int,
    radius: float,
    color: tuple[float, float, float] = (0.95, 0.82, 0.08),
    radial_subdivisions: int = 10,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Author a closed cylindrical triangle mesh for PhysX FEM deformable wire simulation."""
    if num_segments < 2 or radius <= 0 or length <= 0 or radial_subdivisions < 3:
        raise ValueError("Invalid deformable wire dimensions.")
    verts = []
    for i in range(num_segments + 1):
        x = length * i / num_segments - 0.5 * length
        for j in range(radial_subdivisions):
            theta = j * math.tau / radial_subdivisions
            verts.append([x, radius * math.cos(theta), radius * math.sin(theta)])
    bot_idx = len(verts)
    verts.append([-0.5 * length, 0.0, 0.0])
    top_idx = len(verts)
    verts.append([0.5 * length, 0.0, 0.0])

    counts: list[int] = []
    indices: list[int] = []
    for i in range(num_segments):
        for j in range(radial_subdivisions):
            v1 = i * radial_subdivisions + j
            v2 = i * radial_subdivisions + (j + 1) % radial_subdivisions
            v3 = (i + 1) * radial_subdivisions + (j + 1) % radial_subdivisions
            v4 = (i + 1) * radial_subdivisions + j
            counts.extend([3, 3])
            indices.extend([v1, v2, v3, v1, v3, v4])
    for j in range(radial_subdivisions):
        v1, v2 = j, (j + 1) % radial_subdivisions
        counts.append(3)
        indices.extend([v2, v1, bot_idx])
    for j in range(radial_subdivisions):
        v1 = num_segments * radial_subdivisions + j
        v2 = num_segments * radial_subdivisions + (j + 1) % radial_subdivisions
        counts.append(3)
        indices.extend([v1, v2, top_idx])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.array(verts, dtype=np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.array(counts, dtype=np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.array(indices, dtype=np.int32)))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-0.5 * length, -radius, -radius),
        Gf.Vec3f(0.5 * length, radius, radius),
    ]))
    if translation is not None:
        mesh.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        mesh.AddOrientOp().Set(Gf.Quatf(*orientation))
    return mesh.GetPrim()


def _spawn_wire_assembly_impl(prim_path: str, cfg: IDBWireAssemblyCfg, translation=None, orientation=None) -> Usd.Prim:
    require_isaaclab("spawn_idb_wire_assembly")
    from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import-not-found]
    from pxr import PhysxSchema  # type: ignore[import-not-found]

    stage = get_current_stage()
    if cfg.kind == "articulation":
        w = cfg.articulation
        prim = build_articulated_wire_usd(
            stage,
            prim_path,
            length=w.total_length,
            num_segments=w.num_segments,
            radius=w.radius,
            start_family=cfg.start_family,
            end_family=cfg.end_family,
            translation=translation,
            orientation=orientation,
            stiffness=w.stiffness,
            damping=w.damping,
            joint_limit=w.joint_limit,
            density=w.density,
            capsule_overlap=w.capsule_overlap,
            port_mass=cfg.port_mass,
            port_embed_depth=w.port_cable_embed_depth,
            color=w.color,
            self_collision=w.self_collision,
        )
        # Apply PhysX velocity / damping limits to all segments AND connector ports matching WireCraft stability settings.
        body_paths = [f"{prim_path}/seg_{i}" for i in range(w.num_segments)] + [
            f"{prim_path}/start_port",
            f"{prim_path}/end_port",
        ]
        for bpath in body_paths:
            body_prim = stage.GetPrimAtPath(bpath)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            rb.CreateLinearDampingAttr().Set(float(w.segment_linear_damping))
            rb.CreateAngularDampingAttr().Set(float(w.segment_angular_damping))
            rb.CreateMaxLinearVelocityAttr().Set(float(w.max_linear_velocity))
            rb.CreateMaxAngularVelocityAttr().Set(float(w.max_angular_velocity))
            rb.CreateMaxDepenetrationVelocityAttr().Set(float(w.max_depenetration_velocity))
        return prim

    # FEM Deformable backend with vertex-captured rigid start_port and end_port.
    from isaaclab.sim.schemas import DeformableBodyPropertiesCfg, define_deformable_body_properties  # type: ignore

    d = cfg.deformable
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w_q, x_q, y_q, z_q = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(w_q, x_q, y_q, z_q))
    mesh_path = f"{prim_path}/wire"
    mesh_prim = build_deformable_wire_mesh_usd(
        stage,
        mesh_path,
        length=d.total_length,
        num_segments=d.num_segments,
        radius=d.radius,
        color=d.color,
        radial_subdivisions=d.radial_subdivisions,
        translation=(0.5 * d.total_length, 0.0, 0.0),
    )
    mat_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(mesh_prim)
    mat_api.CreateYoungsModulusAttr().Set(float(d.youngs_modulus))
    mat_api.CreatePoissonsRatioAttr().Set(float(d.poissons_ratio))
    mat_api.CreateDynamicFrictionAttr().Set(float(d.dynamic_friction))
    mat_api.CreateDensityAttr().Set(float(d.density))
    define_deformable_body_properties(
        mesh_path,
        DeformableBodyPropertiesCfg(
            rest_offset=0.0,
            contact_offset=0.001,
            simulation_hexahedral_resolution=d.num_segments,
            collision_simplification=True,
            solver_position_iteration_count=32,
            vertex_velocity_damping=3.0,
            sleep_threshold=0.01,
        ),
        stage,
    )
    for side, family, orient, sign in (
        ("start", cfg.start_family, (0.0, 0.0, 0.0, 1.0), 0.0),
        ("end", cfg.end_family, (1.0, 0.0, 0.0, 0.0), 1.0),
    ):
        prof = get_connector_profile(family)
        embed = min(0.008, prof.housing_bounds[1] - prof.housing_bounds[0])
        anchor = prof.cable_anchor_x(embed)
        px = anchor if side == "start" else (d.total_length - anchor)
        vis = default_visual_usd(prof.family, "port")
        port_path = f"{prim_path}/{side}_port"
        build_connector_port_usd(
            stage,
            port_path,
            prof,
            translation=(px, 0.0, 0.0),
            orientation=orient,
            mass=cfg.port_mass,
            visual_usd_path=vis if os.path.isfile(vis) else None,
            apply_rigid_body=True,
        )
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, f"{prim_path}/attachment_{side}")
        attachment.GetActor0Rel().SetTargets([Sdf.Path(port_path)])
        attachment.GetActor1Rel().SetTargets([Sdf.Path(mesh_path)])
        auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
        auto.CreateEnableDeformableVertexAttachmentsAttr(True)
        auto.CreateDeformableVertexOverlapOffsetAttr(float(cfg.attachment_offset))
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(port_path)).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(mesh_path)]
        )
    return root.GetPrim()


if HAS_ISAACLAB:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg  # type: ignore
    from isaaclab.sim.utils import clone  # type: ignore

    spawn_idb_wire_assembly = clone(_spawn_wire_assembly_impl)
else:
    SpawnerCfg = object  # type: ignore[assignment,misc]
    spawn_idb_wire_assembly = _spawn_wire_assembly_impl


@configclass
class IDBArticulationWireCfg:
    """Parameters for an articulated capsule-chain datacenter cable."""

    num_segments: int = 16
    total_length: float = 0.30
    radius: float = 0.002
    stiffness: float = 0.20
    damping: float = 0.005
    joint_limit: float = 0.85
    density: float = 950.0
    self_collision: bool = True
    color: tuple[float, float, float] = (0.95, 0.82, 0.08)
    capsule_overlap: float | None = 0.004
    solver_position_iteration_count: int = 64
    solver_velocity_iteration_count: int = 4
    max_linear_velocity: float = 5.0
    max_angular_velocity: float = 60.0
    max_depenetration_velocity: float = 0.1
    segment_linear_damping: float = 5.0
    segment_angular_damping: float = 5.0
    port_cable_embed_depth: float = 0.012


@configclass
class IDBDeformableWireCfg:
    """Parameters for a PhysX FEM deformable datacenter cable."""

    num_segments: int = 40
    total_length: float = 0.30
    radius: float = 0.002
    radial_subdivisions: int = 10
    color: tuple[float, float, float] = (0.95, 0.82, 0.08)
    youngs_modulus: float = 3.5e7
    poissons_ratio: float = 0.49
    dynamic_friction: float = 3.5
    density: float = 950.0


@configclass
class IDBWireAssemblyCfg(SpawnerCfg):  # type: ignore[misc]
    """Unified wire assembly spawner supporting heterogeneous start and end connectors."""

    func: Any = spawn_idb_wire_assembly
    kind: str = "articulation"
    start_family: str = "lc_simplex"
    end_family: str = "sc_simplex"
    port_mass: float = 0.01
    attachment_offset: float = 0.012
    articulation: IDBArticulationWireCfg = IDBArticulationWireCfg()
    deformable: IDBDeformableWireCfg = IDBDeformableWireCfg()
