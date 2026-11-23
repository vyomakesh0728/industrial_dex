# Architecture and physics design

This document explains the structural choices behind the IDB Board #1 simulation suite, how the package maps to WireCraft's architecture, and how CAD geometry, cable physics, connector collisions, retention latches, and polishing contact are modeled.

## WireCraft alignment and architectural choices

Two one-way-door design decisions shaped the codebase before implementation began.

### Standalone Isaac Lab extension instead of a full Isaac Lab fork

WireCraft ships inside a full fork of Isaac Lab v2.2.1 (over 1,600 files including `isaaclab_rl`, `isaaclab_mimic`, and `rsl_rl` training scripts). Copying upstream Isaac Lab into this repository would pull out-of-scope RL and imitation-learning code into a simulation-only deliverable and obscure the IDB implementation.

Instead, this repository packages two installable Isaac Lab extensions (`source/idb_assets` and `source/idb_tasks`) while preserving WireCraft's exact internal module layout and dependency direction:

```text
single_arm/{franka,ur5} -> single_arm/common -> wire_base -> sim/spawners -> idb_assets
```

Core domain logic (CAD board manifest, connector profiles, OpenUSD procedural builders, PyTorch seating tests, retention latch state machine, Menger bend-radius curvature, figure-eight polishing tracker, neighbor integrity monitor, and IDB rubric scorers) sits inside pure Python, OpenUSD (`pxr`), and PyTorch modules with no `isaacsim` import at module load time. The Isaac Lab `DirectRLEnv` shell (`SingleArmIDBEnv`) and `@clone` spawners call those modules directly inside Isaac Sim. Outside Isaac Sim, `idb_tasks/utils/isaac_compat.py` permits config resolution, registry inspection, OpenUSD stage validation, and PyTorch tests while raising an explicit `RuntimeError` if live physics execution is attempted without Isaac Lab.

### One shared `DirectRLEnv` and modular task contracts

Rather than duplicating five monolithic `DirectRLEnv` subclasses, one `SingleArmIDBEnv` coordinates four collaborators:

1. `RobotAdapter` (`single_arm/common/robot.py`, subclassed by `FrankaAdapter` and `UR5Adapter`) owns articulation spawning, virtual `grasp_center` tool attachment, differential IK, finger contact sensors, 6-DOF wrist wrench sensing, and wrist cameras.
2. `IDBBoardRuntime` (`single_arm/common/board.py`) owns IDB Board #1 transforms, per-env board pose randomization, the 32-port world-pose table, the polishing pad `ContactSensor`, and optional neighboring population cables.
3. `ArticulationWire` or `DeformableWire` (`wire_base/{articulation,deformable}.py`) owns the active patch cord and resets both ends cleanly into their starting receptacles.
4. `IDBDatacenterTaskBase` and its five subclasses (`single_arm/common/tasks/*.py`) own task-specific port pairs, phase state machines, observations, and official IDB scoring rubrics.

## Official CAD extraction and coordinate frame

The script `scripts/idb/export_assets.py` reads Analog Devices' `datacenter-benchmark-assembly-v3.5.step` and five fabrication `.3MF` files (`panel-a-v3.3MF`, `panel-b-v3.3MF`, `panel-c-v3.3MF`, `fiber-polishing-v3.3MF`, and `cable-management.3MF`).

SolidWorks exports the STEP assembly in millimeters with `+Y` up and panel faces pointing toward `+Z`. The pipeline maps every part into a right-handed (`det = +1`) Z-up meter **Board Local Frame** that matches WireCraft's `+X` insertion convention:

- `X_b = (145.0 - Z_step) / 1000.0`: `X_b = 0` lies on the front face of Panels A, B, and C, and `+X_b` points into the ports.
- `Y_b = (115.0 - X_step) / 1000.0`: `Y_b = 0` sits at horizontal board center, `+Y_b` points left toward Column 1, and `-Y_b` points right toward Column 12.
- `Z_b = (Y_step + 10.0) / 1000.0`: `Z_b = 0` rests directly on the table top (`table_top_z = 0.7825 m`).

Extracted CAD measurements verified against `datacenter-benchmark-assembly-v3.5.step`:

- Board footprint: `250.0 mm` wide, `290.0 mm` base depth, `306.5 mm` tall including the polishing station.
- Panel vertical centers above base bottom: Panel C at `106.43 mm`, Panel B at `150.88 mm` (`+44.45 mm`, 1U flush), Panel A at `239.78 mm` (`+88.90 mm`, 2U above Panel B).
- Port horizontal pitches: `17.50 mm` for RJ45 (`A1` through `A4`, `B1` through `B4`, `C1` through `C4`) and `17.00 mm` for SC simplex (`A5` through `A8`), LC simplex (`B5` through `B8`), and LC duplex (`A9` through `A12`, `B9` through `B12`, `C9` through `C12`).
- Port mouth protrusions forward of the panel front face (`-X_b`): RJ45 flush (`0.0 mm`), LC simplex `4.10 mm`, SC simplex `15.20 mm`, LC duplex `16.05 mm`.
- Fiber polishing pad: centered horizontally at `Y_b = 0.0 mm`, `Z_b = 284.15 mm`, front disc at `X_b = -2.0 mm` (`2.0 mm` proud of Panel A), disc radius `16.89 mm`, outward surface normal `(-1, 0, 0)`.

## Connector geometry and collision handling

`sim/spawners/ports/connector_profiles.py` defines four industrial connector profiles (`rj45`, `sc_simplex`, `lc_simplex`, `lc_duplex`) sized strictly inside their STEP coupler envelopes so adjacent sockets on Panels A, B, and C maintain positive clearance (`> 0.1 mm` inter-socket gap on RJ45, `> 7 mm` on fiber couplers).

Every receptacle socket is built by `build_connector_socket_usd` from disjoint convex-hull wall prisms (`left`, `right`, `bottom`, `top`, `shoulder_left`, `shoulder_right`, `key_*`, and `backstop`). No single convex hull ever spans a socket mouth. Each socket's local `+Z` aligns with the port's retention latch or alignment key (`0 rad` for latch top, `pi rad` for latch bottom, `-pi/2 rad` for latch/key left per `instructions.md`).

For optical connectors, `ConnectorProfile.ferrule` stores exact ceramic ferrule tip positions (`2.5 mm` single ferrule on SC simplex, `1.25 mm` single ferrule on LC simplex, twin `1.25 mm` ferrules spaced `6.25 mm` apart on LC duplex).

## Cable physics and plugged initialization

Two cable backends mirror WireCraft:

1. **Articulated capsule chain (`ArticulationWire`)**: 14 to 16 overlapping rigid capsules linked by PhysX D6 angular drives (bending `rotY`/`rotZ` at preset stiffness plus compliant axial torsion `rotX` at `0.5x` stiffness so 30 cm patch cords naturally bridge ports with distinct latch/key roll orientations, `solver_position_iteration_count = 64`, `max_depenetration_velocity = 0.1 m/s`, segment linear and angular damping `5.0`, adjacent-capsule collision filtering). Rigid `start_port` and `end_port` bodies attach via `UsdPhysics.FixedJoint` and support heterogeneous families (such as `lc_simplex` at `B5` and `sc_simplex` at `A5`).
2. **FEM deformable cable (`DeformableWire`)**: closed cylindrical tetrahedral mesh with `PhysxDeformableBodyMaterialAPI` (Young's modulus, Poisson's ratio, dynamic friction, density) coupled to rigid `start_port` and `end_port` bodies via `PhysxPhysicsAttachment` and `PhysxAutoAttachmentAPI` (requires CUDA).

Three BOM-calibrated presets live in `sim/spawners/wires/wire_presets.py`:

| Preset | Cable length | Outer diameter | Minimum bend radius | Articulation stiffness |
| --- | --- | --- | --- | --- |
| `cat6_ethernet` | `0.30 m` | `5.8 mm` | `23.2 mm` (4x diameter) | `0.35 Nm/rad` |
| `os2_fiber_simplex` | `0.30 m` | `3.0 mm` | `30.0 mm` | `0.14 Nm/rad` |
| `os2_fiber_duplex` | `0.30 m` | `4.0 mm` | `30.0 mm` | `0.20 Nm/rad` |

At reset, `PluggedCableArc.solve` solves a C1-continuous 3D quintic Bezier elastica loop with port-aligned straight entry/exit segments and FABRIK fixed-segment projection so every capsule segment matches its exact USD length (`< 0.1 mm` endpoint error) while keeping initial curvature above `30 mm` (`R_min > 30 mm`) without board penetration. `ArticulationWire` seeds `seg_0` at `R(far_port) @ R_z(pi)` (matching `joint_start_port`), parallel-transports Bishop frames, distributes the exact residual axial twist uniformly across joints so both `start_port` and `end_port` begin seated with `roll_cos = 1.0`, and extracts intrinsic XYZ `(rotX, rotY, rotZ)` joint angles anchored at PhysX's chosen root segment. `DeformableWire` strips rest-mesh world offsets and maps every FEM vertex onto continuous parallel-transported Bishop frames with exact attachment anchors. During stepping, the far connector in Panel B is held by a deadbeat mass-scaled spring (`HOLD_STIFFNESS = 0.5`, `HOLD_DAMPING = 0.5`) so pulling the active end never destabilizes the PhysX articulation solver. When `variation.randomize_board_pose` is enabled, `IDBBoardRuntime` synchronizes kinematic PhysX root states for `/frame` (including nested `/frame/panels`), `/polishing_pad`, all 32 `/ports/*` sockets, and cable clips.

## Interaction mechanics and completion evaluators

All evaluators live in `single_arm/common/physics_metrics.py` and `single_arm/common/scoring.py`:

- **Seating and key alignment**: WireCraft's leading-face-in-bore convex polygon check combined with a latch/key roll alignment gate (`dot(port_z, socket_z) >= cos(15 deg)`).
- **Retention latch state machine**: Active-clip connectors (`rj45`, `lc_simplex`, `lc_duplex`) require gripper pinch depression (`>= 1.5 N`) before removal; pulling out while latched flags `latch.damaged`. Push-pull SC connectors release cleanly under axial pull when grasped. Re-insertion latches after holding a seated, aligned pose for `3` consecutive steps.
- **Bend-radius kink detection**: Discrete Menger curvature along the 41-point resampled centerline flags a kink whenever `R_min < wire.min_bend_radius`.
- **Figure-eight polishing tracker**: Projects all ferrule tips onto the polishing pad disc (`R = 16.89 mm`), checks continuous contact (`>= 0.25 N`), checks pressure inside `1.0` to `2.0 N` (flagging ferrule damage above `6.0 N`), and counts complete figure-eight cycles by requiring alternating smooth CW and CCW tangent sweeps (`>= 3.2 rad` and `>= 12 mm` arc length per lobe, rejecting stationary presses, circles, and linear strokes).
- **Neighbor disturbance and integrity**: Tracks displacement of every background cable on the board (`> 4 mm` flags disturbance) and peak contact force (`> 35 N` flags damage).

## Multimodal sensor suite

`SensorSuiteCfg` supports both WireCraft's fixed workstation cameras (`side`, `side2`) and all six sensor configurations from Table I of arXiv:2607.14021:

- Wrist Intel RealSense D405 RGB and depth (`wrist_d405_rgb`, `wrist_d405_depth`)
- Scene Intel RealSense D435i RGB and depth (`scene_d435i_rgb`, `scene_d435i_depth`)
- Scene ADI EVAL-ADTF3175 indirect ToF depth (`scene_adtf3175_tof`)
- 6-DOF wrist force/torque wrench via `body_incoming_joint_wrench_b` (`panda_hand` on Franka, `gripper_base_link` on UR5)
- Dedicated `ContactSensor` on `/idb_board/polishing_pad` filtered to `/active_wire/end_port`
