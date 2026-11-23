"""Configuration inputs and single-pass resolver for single-arm IDB environments.

Follows WireCraft's exact contract: every user-editable setting is a plain input on these
configclasses, and every derived spawn config or space dimension is constructed once inside
:func:`resolve_env_cfg` after all override layers merge.
"""

from __future__ import annotations

from dataclasses import MISSING
import importlib
from typing import Any

from idb_assets.board_spec import load_board_spec
from idb_tasks.sim.spawners.board import IDBBoardCfg
from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, configclass

from ...wire_base.connectors import ConnectorCfg
from ...wire_base.wire_cfg import WireSceneCfg

PRIM_PATH = "/World/envs/env_.*"
MODES = ("demo", "random", "eval")
POPULATION_DENSITIES = ("sparse", "medium", "dense", "custom")

# arXiv:2607.14021 Table I camera & sensor configurations.
PAPER_SENSOR_CONFIGS: dict[int, dict[str, Any]] = {
    1: {"cameras": ["wrist_d405_rgb", "scene_d435i_rgb"], "enable_wrist_ft": True},
    2: {"cameras": ["wrist_d405_rgb", "scene_adtf3175_tof"], "enable_wrist_ft": True},
    3: {"cameras": ["wrist_d405_rgb", "wrist_d405_depth"], "enable_wrist_ft": True},
    4: {"cameras": ["wrist_d405_rgb", "scene_d435i_depth"], "enable_wrist_ft": True},
    5: {"cameras": ["wrist_d405_depth"], "enable_wrist_ft": True},
    6: {"cameras": ["wrist_d405_rgb"], "enable_wrist_ft": False},
}

# Standard background cable topologies for IDB Board #1 difficulty modulation (Section III-A).
# Each entry is (far_port_id, active_port_id, cable_preset).
POPULATION_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("B2", "A2", "cat6_ethernet"),
    ("B3", "A3", "cat6_ethernet"),
    ("B4", "A4", "cat6_ethernet"),
    ("C1", "B1", "cat6_ethernet"),
    ("C2", "B2", "cat6_ethernet"),
    ("C3", "B3", "cat6_ethernet"),
    ("C4", "B4", "cat6_ethernet"),
    ("B6", "A6", "os2_fiber_simplex"),
    ("B7", "A7", "os2_fiber_simplex"),
    ("B8", "A8", "os2_fiber_simplex"),
    ("B10", "A10", "os2_fiber_duplex"),
    ("B11", "A11", "os2_fiber_duplex"),
    ("B12", "A12", "os2_fiber_duplex"),
    ("C9", "B9", "os2_fiber_duplex"),
    ("C10", "B10", "os2_fiber_duplex"),
    ("C11", "B11", "os2_fiber_duplex"),
    ("C12", "B12", "os2_fiber_duplex"),
)


if HAS_ISAACLAB:
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg  # type: ignore[import-not-found]
    from isaaclab.envs import DirectRLEnvCfg  # type: ignore[import-not-found]
    from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import-not-found]
    from isaaclab.sensors import TiledCameraCfg  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationCfg  # type: ignore[import-not-found]
    from isaaclab.terrains import TerrainImporterCfg  # type: ignore[import-not-found]
else:
    @configclass
    class SimulationCfg:
        dt: float = 1.0 / 120.0
        render_interval: int = 2
        device: str = "cuda:0"

    @configclass
    class InteractiveSceneCfg:
        num_envs: int = 1
        env_spacing: float = 2.5
        replicate_physics: bool = False
        clone_in_fabric: bool = False

    @configclass
    class DirectRLEnvCfg:
        decimation: int = 2
        episode_length_s: float = 30.0
        action_space: int = 7
        observation_space: int = 0
        state_space: int = 0
        sim: SimulationCfg = SimulationCfg()
        scene: InteractiveSceneCfg = InteractiveSceneCfg()


@configclass
class CameraIntrinsicsCfg:
    """Pinhole camera intrinsics for workstation and IDB Table I sensors."""

    focal_length: float = 24.0
    focus_distance: float = 400.0
    horizontal_aperture: float = 20.955
    clipping_range: tuple[float, float] = (0.05, 2.5)


@configclass
class SensorSuiteCfg:
    """Multimodal perception and wrist wrench sensor suite matching IDB paper Table I."""

    paper_config_id: int | None = None
    """Optional 1-6 selector from arXiv:2607.14021 Table I; overrides camera list when set."""
    enable_cameras: bool = False
    cameras: list[str] = []
    """Allowed names: side, side2, scene_d435i_rgb, scene_d435i_depth, scene_adtf3175_tof, wrist_d405_rgb, wrist_d405_depth."""
    camera_resolution: int = 224
    render_stride: int = 1
    intrinsics: CameraIntrinsicsCfg = CameraIntrinsicsCfg()
    enable_wrist_ft: bool = True
    """Expose 6-DOF wrist force/torque wrench in observations['ft'] and task evaluators."""
    enable_polishing_contact_sensor: bool = True
    """Attach ContactSensor on Board #1's fiber polishing pad for 1-2 N force feedback."""
    side_camera_pose: tuple[float, ...] = (0.25, 0.55, 1.32, 0.56, 0.14, 0.28, -0.76)
    side2_camera_pose: tuple[float, ...] = (0.05, -0.50, 1.25, 0.88, -0.18, 0.18, 0.40)
    scene_d435i_pose: tuple[float, ...] = (0.12, 0.28, 1.28, 0.85, 0.12, 0.28, -0.43)
    scene_adtf3175_pose: tuple[float, ...] = (0.12, -0.28, 1.28, 0.85, -0.12, 0.28, 0.43)


@configclass
class BoardSceneCfg:
    """Placement and structural options for IDB Board #1 on the workstation table."""

    board_pos: tuple[float, float, float] = (0.56, 0.0, 0.7825)
    """Board origin in env-local world coordinates (front panel X, board center Y, table top Z)."""
    board_yaw_rad: float = 0.0
    include_cable_clips: bool = False
    socket_clearance_override: float | None = None
    manifest_path: str | None = None


@configclass
class VariationCfg:
    """Configurable scene variations: board density, pose jitter, physics, lighting, port selection."""

    population_density: str = "sparse"
    """Board cable population: ``sparse`` (0 neighbors, paper baseline), ``medium`` (up to 6), ``dense`` (up to 14), or ``custom``."""
    neighbor_cables: list[list[str]] = []
    """Explicit neighbor cables when population_density == 'custom': [[far_port, active_port, preset], ...]."""
    randomize_board_pose: bool = False
    board_xy_range: tuple[float, float] = (0.02, 0.04)
    board_yaw_range_rad: float = 0.087
    randomize_cable_physics: bool = False
    stiffness_scale_range: tuple[float, float] = (0.75, 1.30)
    friction_scale_range: tuple[float, float] = (0.80, 1.25)
    randomize_lighting: bool = False
    light_intensity_range: tuple[float, float] = (900.0, 2200.0)
    randomize_target_port: bool = False
    """Sample source/target ports from the task's valid candidate ports on reset (e.g., A5..A8 for SC maintenance)."""


@configclass
class RobotCfg:
    """Robot-specific parameters consumed by RobotAdapter."""

    name: str = MISSING
    adapter: str = MISSING
    usd_path: str = ""
    articulation: Any = None
    arm_joints: list[str] = MISSING
    gripper_joint: str = MISSING
    ee_body: str = MISSING
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_parent: str = ""
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_open_pos: float = MISSING
    gripper_closed_pos: float = MISSING
    gripper_max_step: float | None = None
    gripper_release_step: float | None = None
    grasp_opening_range: tuple[float, float] = MISSING
    release_opening: float = MISSING
    fully_closed_opening: float = MISSING
    finger_bodies: tuple[str, str] = MISSING
    pad_static_friction: float = 5.0
    pad_dynamic_friction: float = 5.0
    pad_friction_combine_mode: str = "max"
    grasp_rest_quat: tuple[float, float, float, float] = MISSING
    ft_sensor_body: str | None = None
    ik_joint_reference: str = "control"
    max_joint_step: float | None = 0.05
    joint_lower: list[float] | None = None
    joint_upper: list[float] | None = None


@configclass
class SingleArmIDBEnvCfg(DirectRLEnvCfg):
    """Top-level configuration for every single-arm IDB Board #1 environment."""

    episode_length_s: float = 30.0
    decimation: int = 2
    action_space: int = 7
    observation_space: int = 0
    state_space: int = 0
    seed: int = 42

    mode: str = "demo"
    variant: str = "sparse"

    robot: RobotCfg = MISSING
    task: Any = MISSING
    wire: WireSceneCfg = WireSceneCfg()
    connector: ConnectorCfg = ConnectorCfg()
    board: BoardSceneCfg = BoardSceneCfg()
    variation: VariationCfg = VariationCfg()
    sensors: SensorSuiteCfg = SensorSuiteCfg()

    control_mode: str = "cartesian"
    fix_ee_orient: bool = False
    action_scale: float = 0.01

    enable_terminations: bool = True
    terminate_on_time_out: bool = True
    terminate_on_success: bool = False
    terminate_on_damage: bool = False
    ee_safe_zone: list[float] = [0.05, 0.85, -0.45, 0.45, 0.76, 1.38]
    port_drop_height: float = 0.74
    fault_port_speed: float = 1000.0
    fault_joint_speed: float = 20.0

    table_pos: tuple[float, float, float] = (0.35, 0.0, 0.39125)
    table_size: tuple[float, float, float] = (1.10, 1.20, 0.7825)
    table_top_z: float = 0.7825

    sim: Any = (
        SimulationCfg(
            dt=1.0 / 120.0,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            physx=sim_utils.PhysxCfg(
                gpu_max_rigid_contact_count=1048576,
                gpu_max_rigid_patch_count=1048576,
                gpu_found_lost_pairs_capacity=1048576,
                gpu_found_lost_aggregate_pairs_capacity=1048576,
                gpu_total_aggregate_pairs_capacity=1048576,
                gpu_max_soft_body_contacts=1048576,
                gpu_max_particle_contacts=1048576,
                gpu_heap_capacity=134217728,
                gpu_temp_buffer_capacity=67108864,
                gpu_max_num_partitions=8,
                max_velocity_iteration_count=1,
            ),
        )
        if HAS_ISAACLAB
        else SimulationCfg()
    )
    scene: Any = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False, clone_in_fabric=False
    )

    # Derived by resolve_env_cfg — never set via presets or CLI.
    resolved: bool = False
    wire_prim_path: str = f"{PRIM_PATH}/active_wire"
    board_prim_path: str = f"{PRIM_PATH}/idb_board"
    wire_spawn: Any = None
    board_spawn: Any = None
    resolved_neighbor_cables: list[tuple[str, str, str]] = []
    table_cfg: Any = None
    camera_cfgs: dict[str, Any] = {}
    composition: dict[str, Any] = {}


def _import_symbol(path: str):
    module_name, symbol = path.split(":")
    return getattr(importlib.import_module(module_name), symbol)


def _select_neighbor_cables(cfg: SingleArmIDBEnvCfg, reserved_ports: set[str]) -> list[tuple[str, str, str]]:
    density = cfg.variation.population_density
    if density not in POPULATION_DENSITIES:
        raise ValueError(f"variation.population_density must be one of {POPULATION_DENSITIES}, got {density!r}.")
    if density == "sparse":
        return []
    if density == "custom":
        chosen = [(str(a), str(b), str(p)) for a, b, p in cfg.variation.neighbor_cables]
    else:
        limit = 6 if density == "medium" else 14
        chosen = []
        used = set(reserved_ports)
        for far_p, act_p, preset in POPULATION_CATALOG:
            if far_p in used or act_p in used:
                continue
            chosen.append((far_p, act_p, preset))
            used.add(far_p)
            used.add(act_p)
            if len(chosen) >= limit:
                break
    # Validate all chosen neighbor ports exist on Board #1 and do not collide with active task ports.
    board = load_board_spec(cfg.board.manifest_path)
    seen = set(reserved_ports)
    for far_p, act_p, _ in chosen:
        board.port(far_p)
        board.port(act_p)
        if far_p in seen or act_p in seen:
            raise ValueError(f"Neighbor cable ({far_p}, {act_p}) conflicts with reserved or duplicate port.")
        seen.update((far_p, act_p))
    return chosen


def _build_fixed_camera_cfg(cfg: SingleArmIDBEnvCfg, name: str, pose: tuple[float, ...], data_type: str) -> Any:
    if not HAS_ISAACLAB:
        return {"name": name, "pose": tuple(pose), "data_type": data_type, "resolution": cfg.sensors.camera_resolution}
    return TiledCameraCfg(
        prim_path=f"{PRIM_PATH}/Camera_{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]), convention="world"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cfg.sensors.intrinsics.focal_length,
            focus_distance=cfg.sensors.intrinsics.focus_distance,
            horizontal_aperture=cfg.sensors.intrinsics.horizontal_aperture,
            clipping_range=tuple(cfg.sensors.intrinsics.clipping_range),
        ),
        width=cfg.sensors.camera_resolution,
        height=cfg.sensors.camera_resolution,
    )


def resolve_env_cfg(cfg: SingleArmIDBEnvCfg) -> SingleArmIDBEnvCfg:
    """Validate merged inputs and derive all spawn configs, sensor configs, and spaces once."""
    if cfg.resolved:
        return cfg
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}.")
    if cfg.control_mode not in ("cartesian", "joint"):
        raise ValueError(f"control_mode must be 'cartesian' or 'joint', got {cfg.control_mode!r}.")
    if cfg.fix_ee_orient and cfg.control_mode != "cartesian":
        raise ValueError("fix_ee_orient requires control_mode='cartesian'.")
    if cfg.decimation < 1 or cfg.episode_length_s <= 0 or cfg.action_scale <= 0:
        raise ValueError("decimation, episode_length_s, and action_scale must be positive.")
    if len(cfg.ee_safe_zone) != 6:
        raise ValueError("ee_safe_zone must have 6 floats [x_min, x_max, y_min, y_max, z_min, z_max].")
    if cfg.scene.num_envs < 1:
        raise ValueError("scene.num_envs must be positive.")
    if cfg.wire.kind == "deformable" and "cuda" not in str(cfg.sim.device):
        raise ValueError("Deformable FEM wires require a CUDA device.")

    # Apply paper Table I sensor configuration shorthand when requested.
    if cfg.sensors.paper_config_id is not None:
        if cfg.sensors.paper_config_id not in PAPER_SENSOR_CONFIGS:
            raise ValueError(f"sensors.paper_config_id must be in 1..6, got {cfg.sensors.paper_config_id!r}.")
        preset = PAPER_SENSOR_CONFIGS[cfg.sensors.paper_config_id]
        cfg.sensors.enable_cameras = True
        cfg.sensors.cameras = list(preset["cameras"])
        cfg.sensors.enable_wrist_ft = bool(preset["enable_wrist_ft"])

    cfg.sim.render_interval = cfg.decimation * cfg.sensors.render_stride
    cfg.scene.replicate_physics = False

    arm_dof = len(cfg.robot.arm_joints)
    cfg.action_space = (arm_dof + 1) if cfg.control_mode == "joint" else (4 if cfg.fix_ee_orient else 7)

    # Resolve task-specific contract (ports, connector families, observation dim).
    cfg.task.resolve(cfg)
    cfg.wire_spawn = cfg.wire.assembly_cfg(cfg.connector)
    cfg.board_spawn = IDBBoardCfg(
        manifest_path=cfg.board.manifest_path,
        socket_clearance_override=cfg.board.socket_clearance_override,
        include_cable_clips=cfg.board.include_cable_clips,
        activate_polishing_contact_sensor=cfg.sensors.enable_polishing_contact_sensor,
    )
    reserved_ports = cfg.task.reserved_ports()
    cfg.resolved_neighbor_cables = _select_neighbor_cables(cfg, reserved_ports)
    cfg.observation_space = cfg.task.observation_dim(arm_dof + 1)

    if HAS_ISAACLAB:
        adapter = _import_symbol(cfg.robot.adapter)
        cfg.robot.articulation.prim_path = f"{PRIM_PATH}/{adapter.prim_name}"
        if cfg.robot.usd_path:
            cfg.robot.articulation.spawn.usd_path = cfg.robot.usd_path
        cfg.table_cfg = RigidObjectCfg(
            prim_path=f"{PRIM_PATH}/table",
            spawn=sim_utils.CuboidCfg(
                size=tuple(cfg.table_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.26, 0.29), metallic=0.2, roughness=0.6),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cfg.table_pos), rot=(1.0, 0.0, 0.0, 0.0)),
        )

    # Resolve camera configs (fixed workstation + paper Table I scene/wrist sensors).
    cfg.camera_cfgs = {}
    if cfg.sensors.enable_cameras:
        adapter = _import_symbol(cfg.robot.adapter)
        wrist_cams = adapter.camera_cfgs(cfg)
        fixed_map = {
            "side": (cfg.sensors.side_camera_pose, "rgb"),
            "side2": (cfg.sensors.side2_camera_pose, "rgb"),
            "scene_d435i_rgb": (cfg.sensors.scene_d435i_pose, "rgb"),
            "scene_d435i_depth": (cfg.sensors.scene_d435i_pose, "distance_to_camera"),
            "scene_adtf3175_tof": (cfg.sensors.scene_adtf3175_pose, "distance_to_camera"),
        }
        valid_names = sorted(list(fixed_map) + list(wrist_cams))
        for cam_name in cfg.sensors.cameras:
            if cam_name in fixed_map:
                pose, dtype = fixed_map[cam_name]
                cfg.camera_cfgs[cam_name] = _build_fixed_camera_cfg(cfg, cam_name, pose, dtype)
            elif cam_name in wrist_cams:
                cfg.camera_cfgs[cam_name] = wrist_cams[cam_name]
            else:
                raise ValueError(f"Unknown camera {cam_name!r}; choose from {valid_names}.")

    cfg.resolved = True
    return cfg
