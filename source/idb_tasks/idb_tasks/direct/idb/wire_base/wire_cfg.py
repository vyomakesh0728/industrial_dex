"""Wire scene configuration inputs shared by articulated and FEM deformable cables."""

from __future__ import annotations

from idb_tasks.sim.spawners.wires import (
    IDBArticulationWireCfg,
    IDBDeformableWireCfg,
    IDBWireAssemblyCfg,
    get_cable_preset,
)
from idb_tasks.utils.isaac_compat import configclass

from .connectors import ConnectorCfg


@configclass
class WireSceneCfg:
    """User-facing wire physics and reset inputs; spawn config is derived during resolve."""

    kind: str = "articulation"
    """``articulation`` (capsule chain) or ``deformable`` (FEM)."""
    preset: str = "os2_fiber_simplex"
    """Cable BOM preset: ``cat6_ethernet``, ``os2_fiber_simplex``, or ``os2_fiber_duplex``."""
    root_pos: tuple[float, float, float] = (0.35, 0.0, 0.98)
    length: float = 0.30
    radius: float = 0.0018
    min_bend_radius: float = 0.030
    color: tuple[float, float, float] = (0.95, 0.82, 0.08)
    centerline_points: int = 41
    lateral_bow: tuple[float, float] = (0.0, -0.025)

    # Articulation cable settings (WireCraft stability parameters)
    num_segments: int = 16
    stiffness: float = 0.14
    damping: float = 0.004
    joint_limit: float = 0.65
    density: float = 900.0
    solver_position_iterations: int = 64
    solver_velocity_iterations: int = 4
    capsule_overlap: float | None = 0.003
    max_linear_velocity: float = 5.0
    max_angular_velocity: float = 60.0
    max_depenetration_velocity: float = 0.1
    segment_linear_damping: float = 5.0
    segment_angular_damping: float = 5.0
    port_cable_embed_depth: float = 0.012

    # FEM deformable cable settings
    fem_num_segments: int = 40
    fem_radius: float = 0.0020
    fem_youngs_modulus: float = 2.8e7
    fem_density: float = 900.0
    fem_dynamic_friction: float = 3.5
    attachment_offset: float = 0.012

    def apply_preset_defaults(self, preset_name: str | None = None) -> None:
        """Populate physical parameters from the named IDB BOM cable preset."""
        p = get_cable_preset(self.preset if preset_name is None else preset_name)
        self.preset = p.name
        self.length = p.length
        self.radius = p.radius
        self.min_bend_radius = p.min_bend_radius
        self.num_segments = p.num_segments
        self.stiffness = p.stiffness
        self.damping = p.damping
        self.density = p.density
        self.fem_num_segments = p.fem_num_segments
        self.fem_radius = max(p.radius, 0.0018)
        self.fem_youngs_modulus = p.fem_youngs_modulus
        self.fem_density = p.density
        self.fem_dynamic_friction = p.fem_dynamic_friction
        self.color = p.color

    def validate(self) -> None:
        if self.kind not in ("articulation", "deformable"):
            raise ValueError(f"wire.kind must be 'articulation' or 'deformable', got {self.kind!r}.")
        if self.length <= 0 or self.radius <= 0 or self.min_bend_radius <= 0:
            raise ValueError("wire.length, wire.radius, and wire.min_bend_radius must be positive.")
        if self.centerline_points < 5:
            raise ValueError("wire.centerline_points must be at least 5 for curvature evaluation.")

    def assembly_cfg(self, connector: ConnectorCfg) -> IDBWireAssemblyCfg:
        """Build the unified heterogeneous wire assembly spawner config."""
        self.validate()
        return IDBWireAssemblyCfg(
            kind=self.kind,
            start_family=connector.resolved_far_family,
            end_family=connector.family,
            port_mass=connector.port_mass,
            attachment_offset=self.attachment_offset,
            articulation=IDBArticulationWireCfg(
                num_segments=self.num_segments,
                total_length=self.length,
                radius=self.radius,
                stiffness=self.stiffness,
                damping=self.damping,
                joint_limit=self.joint_limit,
                density=self.density,
                self_collision=True,
                color=tuple(self.color),
                capsule_overlap=self.capsule_overlap,
                solver_position_iteration_count=self.solver_position_iterations,
                solver_velocity_iteration_count=self.solver_velocity_iterations,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_depenetration_velocity=self.max_depenetration_velocity,
                segment_linear_damping=self.segment_linear_damping,
                segment_angular_damping=self.segment_angular_damping,
                port_cable_embed_depth=self.port_cable_embed_depth,
            ),
            deformable=IDBDeformableWireCfg(
                num_segments=self.fem_num_segments,
                total_length=self.length,
                radius=self.fem_radius,
                color=tuple(self.color),
                youngs_modulus=self.fem_youngs_modulus,
                density=self.fem_density,
                dynamic_friction=self.fem_dynamic_friction,
            ),
        )
