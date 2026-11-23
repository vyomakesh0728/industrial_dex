# Environment reference

Reference for all 20 registered IDB Board #1 Gymnasium environments, action and observation spaces, layered configuration presets, scene variations, and Table I sensor configurations.

Generate the live registry JSON at any time with:

```bash
python scripts/idb/run.py --list out/ids.json
```

## Gymnasium ID naming

```text
IDB-<Robot>-<Wire>-<Task>-v0
```

| Field | Values |
| --- | --- |
| `Robot` | `Franka` (7-DOF arm + parallel gripper), `UR5` (6-DOF arm + Robotiq 2F-85) |
| `Wire` | `Articulation` (capsule chain), `Deformable` (PhysX FEM soft body, CUDA only) |
| `Task` | `EthernetPortSwap`, `SCPortSwap`, `LCDuplexPortSwap`, `SCFiberMaintenance`, `LCFiberMaintenance` |

## Registered environments

| Gym ID | Source to target | Far port | Active and far connector | Action dim | Obs dim | Rubric max |
| --- | --- | --- | --- | --- | --- | --- |
| `IDB-Franka-Articulation-EthernetPortSwap-v0` | `A1` to `A2` | `B1` | `rj45` and `rj45` | 7 | 65 | 6 |
| `IDB-Franka-Deformable-EthernetPortSwap-v0` | `A1` to `A2` | `B1` | `rj45` and `rj45` | 7 | 65 | 6 |
| `IDB-UR5-Articulation-EthernetPortSwap-v0` | `A1` to `A2` | `B1` | `rj45` and `rj45` | 7 | 63 | 6 |
| `IDB-UR5-Deformable-EthernetPortSwap-v0` | `A1` to `A2` | `B1` | `rj45` and `rj45` | 7 | 63 | 6 |
| `IDB-Franka-Articulation-SCPortSwap-v0` | `A5` to `A6` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 65 | 10 |
| `IDB-Franka-Deformable-SCPortSwap-v0` | `A5` to `A6` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 65 | 10 |
| `IDB-UR5-Articulation-SCPortSwap-v0` | `A5` to `A6` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 63 | 10 |
| `IDB-UR5-Deformable-SCPortSwap-v0` | `A5` to `A6` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 63 | 10 |
| `IDB-Franka-Articulation-LCDuplexPortSwap-v0` | `A9` to `A10` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 65 | 13 |
| `IDB-Franka-Deformable-LCDuplexPortSwap-v0` | `A9` to `A10` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 65 | 13 |
| `IDB-UR5-Articulation-LCDuplexPortSwap-v0` | `A9` to `A10` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 63 | 13 |
| `IDB-UR5-Deformable-LCDuplexPortSwap-v0` | `A9` to `A10` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 63 | 13 |
| `IDB-Franka-Articulation-SCFiberMaintenance-v0` | `A5` to pad to `A5` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 76 | 19 |
| `IDB-Franka-Deformable-SCFiberMaintenance-v0` | `A5` to pad to `A5` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 76 | 19 |
| `IDB-UR5-Articulation-SCFiberMaintenance-v0` | `A5` to pad to `A5` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 74 | 19 |
| `IDB-UR5-Deformable-SCFiberMaintenance-v0` | `A5` to pad to `A5` | `B5` | `sc_simplex` and `lc_simplex` | 7 | 74 | 19 |
| `IDB-Franka-Articulation-LCFiberMaintenance-v0` | `A9` to pad to `A9` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 76 | 22 |
| `IDB-Franka-Deformable-LCFiberMaintenance-v0` | `A9` to pad to `A9` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 76 | 22 |
| `IDB-UR5-Articulation-LCFiberMaintenance-v0` | `A9` to pad to `A9` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 74 | 22 |
| `IDB-UR5-Deformable-LCFiberMaintenance-v0` | `A9` to pad to `A9` | `B9` | `lc_duplex` and `lc_duplex` | 7 | 74 | 22 |

## Action space

All environments default to `control_mode: cartesian` with `action_scale: 0.01`, `dt = 1/120 s`, and `decimation = 2` (60 Hz control rate).

| Control mode | Dimension | Layout |
| --- | --- | --- |
| `cartesian` (`fix_ee_orient=false`, default) | 7 | `[dx, dy, dz, drx, dry, drz, gripper]` (`gripper < 0` opens, `>= 0` closes) |
| `cartesian` (`fix_ee_orient=true`) | 4 | `[dx, dy, dz, gripper]` |
| `joint` | `J` (`8` Franka, `7` UR5) | `[dq_1 .. dq_arm, gripper]` |

## Observation dictionary

`_get_observations()` returns a dictionary with finite tensors on `env.device`:

- `policy`: flat state tensor of shape `(num_envs, observation_space)` described below.
- `ft`: `(num_envs, 6)` wrist force/torque wrench `[fx, fy, fz, tx, ty, tz]` (included when `sensors.enable_wrist_ft=true`).
- `polishing_force_n`: `(num_envs, 1)` ferrule-to-pad contact normal force in Newtons (included when `sensors.enable_polishing_contact_sensor=true`).
- `<camera_name>`: `(num_envs, H, W, C)` normalized RGB (`[0, 1]`) or metric depth image for each camera enabled in `sensors.cameras`.

### Flat `policy` observation vector layout

Let `J` be the tracked joint count (`8` for Franka, `7` for UR5).

| Slice | Width | Description |
| --- | --- | --- |
| Joint positions | `J` | Arm joints + primary gripper joint (rad or m) |
| Joint velocities | `J` | Arm joint velocities + gripper velocity |
| End-effector pose | 7 | Env-local tool position `(x, y, z)` + `wxyz` quaternion |
| Gripper telemetry | 3 | Normalized opening `[0, 1]`, left pad force `/ 10 N`, right pad force `/ 10 N` |
| Wrist wrench | 6 | 6-DOF wrench (zeros when `sensors.enable_wrist_ft=false`) |
| Active connector state | 10 | Connector shoulder local pos (3), `wxyz` quat (4), leading-face tip local pos (3) |
| Source port geometry | 6 | Source port mouth local pos (3), vector `source_mouth - connector_shoulder` (3) |
| Target port geometry | 10 | Target port mouth local pos (3), vector `target_mouth - connector_shoulder` (3), target socket `wxyz` quat (4) |
| Task status scalars | 7 | `removed_from_source`, `latch_engaged`, `target_tip_depth`, `target_roll_cos`, `min_bend_radius_m`, `ever_kinked`, `integrity_ok` |
| Polishing features (Tasks 4 and 5 only) | 11 | Pad center local pos (3), `pad_center - ferrule_mean` (3), `pad_force_n` (1), `in_contact` (1), `cycle_fraction` (1), `continuous_contact_ok` (1), `pressure_ok` (1) |

## Layered configuration priority

`single_arm/common/compose.py` merges layers from lowest to highest priority, then calls `resolve_env_cfg` once:

1. `SingleArmIDBEnvCfg` defaults
2. Robot config (`single_arm/{franka,ur5}/robot_cfg.py`)
3. Task preset (`presets/task/<task>.yaml`) and cable BOM defaults
4. Wire preset (`presets/wire/{articulation,deformable}.yaml`)
5. Connector preset (`presets/connector/{rj45,sc_simplex,lc_duplex}.yaml`)
6. Variation preset (`presets/variation/{sparse,medium,dense,randomized}.yaml`)
7. Mode preset (`presets/mode/{demo,random,eval}.yaml`)
8. `--experiment file.yaml` files in command-line order (`env:` top-level mapping only)
9. Explicit `env.*=value` CLI arguments and typed flags (`--num_envs`, `--seed`, `--device`)

Every run writes the final resolved config to `<output>/params/env.yaml` and layer provenance to `<output>/params/selection.yaml`.

## Scene variation presets

| Preset | Neighbor cables | Cable clips | Board pose jitter | Cable physics jitter | Lighting jitter | Random target port |
| --- | --- | --- | --- | --- | --- | --- |
| `sparse` | 0 (paper baseline) | Off | Off | Off | Off | Off |
| `medium` | 6 disjoint cables | On | Off | Off | Off | Off |
| `dense` | 13 to 14 disjoint cables | On | On (`±20 mm X, ±40 mm Y, ±5° yaw`) | On | On | Off |
| `randomized` | 6 disjoint cables | On | On | On | On | On |

Custom neighbor layouts are supported by setting `env.variation.population_density=custom` and providing `env.variation.neighbor_cables=[[far,active,preset],...]`.

## Paper Table I sensor configurations

Pass `--sensor-config <1..6>` or `env.sensors.paper_config_id=<1..6>` to replicate the six modality configurations from arXiv:2607.14021 Table I:

| Config ID | Enabled cameras | Wrist F/T |
| --- | --- | --- |
| 1 | `wrist_d405_rgb`, `scene_d435i_rgb` | Enabled |
| 2 | `wrist_d405_rgb`, `scene_adtf3175_tof` | Enabled |
| 3 | `wrist_d405_rgb`, `wrist_d405_depth` | Enabled |
| 4 | `wrist_d405_rgb`, `scene_d435i_depth` | Enabled |
| 5 | `wrist_d405_depth` | Enabled |
| 6 | `wrist_d405_rgb` | Disabled |
