# Industrial Dexterity Benchmark (IDB) datacenter simulation suite

Executable Isaac Sim and Isaac Lab simulation environments for IDB Board #1 (datacenter cable management), structured around WireCraft's extension layout and `DirectRLEnv` interface.

This repository ships simulation environments, CAD-derived OpenUSD assets, physics evaluators, and task documentation only. It contains no RL or IL training code, no policy models, no agent configurations, and no dataset collection scripts.

## Benchmark coverage

The suite registers 20 Gymnasium environments covering two robots, two cable physics models, and all five official IDB Board #1 tasks from [ag-industrial-dexterity-benchmarks](https://github.com/analogdevicesinc/ag-industrial-dexterity-benchmarks) and [arXiv:2607.14021](https://arxiv.org/pdf/2607.14021).

| Task | Source to target | Cable type | Rubric max | Documentation |
| --- | --- | --- | --- | --- |
| Task 1: Ethernet port swap | `A1` to `A2` (far `B1`) | Cat6 RJ45 patch cord | 6 | [Task 1](docs/tasks/task1_ethernet_port_swap.md) |
| Task 2: SC simplex port swap | `A5` to `A6` (far `B5`) | LC-to-SC simplex OS2 fiber | 10 | [Task 2](docs/tasks/task2_sc_port_swap.md) |
| Task 3: LC duplex port swap | `A9` to `A10` (far `B9`) | LC duplex OS2 fiber | 13 | [Task 3](docs/tasks/task3_lc_duplex_port_swap.md) |
| Task 4: SC simplex fiber maintenance | `A5` to polishing pad to `A5` (far `B5`) | LC-to-SC simplex OS2 fiber | 19 | [Task 4](docs/tasks/task4_sc_fiber_maintenance.md) |
| Task 5: LC duplex fiber maintenance | `A9` to polishing pad to `A9` (far `B9`) | LC duplex OS2 fiber | 22 | [Task 5](docs/tasks/task5_lc_duplex_fiber_maintenance.md) |

## Repository layout

| Component | Path |
| --- | --- |
| CAD-derived OpenUSD meshes and 32-port board manifest | `source/idb_assets/idb_assets/` |
| Connector, wire, and IDB Board #1 USD spawners | `source/idb_tasks/idb_tasks/sim/spawners/{ports,wires,board}/` |
| Wire backends, connector geometry, and plugged-cable arch solver | `source/idb_tasks/idb_tasks/direct/idb/wire_base/` |
| Shared `DirectRLEnv`, config composer, board runtime, metrics, and rubric scorers | `source/idb_tasks/idb_tasks/direct/idb/single_arm/common/` |
| Franka Panda/FR3 and UR5 + Robotiq 2F-85 robot adapters | `source/idb_tasks/idb_tasks/direct/idb/single_arm/{franka,ur5}/` |
| Simulation runner, matrix verifier, CAD exporter, and offline check | `scripts/idb/` |
| Architecture, environment reference, and task specifications | `docs/` |
| Offline OpenUSD, PyTorch, config, CLI, and scope-gate tests | `tests/` |

## Installation

### Inside an Isaac Lab v2.2.1 workstation (NVIDIA GPU, Linux)

Install Isaac Sim 4.5 or 5.1 and Isaac Lab v2.2.1 first. Then install both local extensions into the Isaac Lab Python environment.

```bash
pip install -e source/idb_assets
pip install -e source/idb_tasks
```

Set `WIRECRAFT_ROBOT_ASSET_DIR` to the folder holding `ur5_with_gripper_camera_v2.usd` when running UR5 environments. Franka environments use Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`.

### Local development and offline verification (no GPU required)

Sync the workspace with `uv` on Python 3.11 and run the offline check script.

```bash
uv sync --python 3.11
uv run python scripts/idb/check_offline.py
```

## Quickstart commands

Export the full 20-environment registry to JSON (runs with or without Isaac Sim):

```bash
python scripts/idb/run.py --list out/ids.json
```

Validate a task, variation, and Table I sensor configuration without launching Isaac Sim:

```bash
python scripts/idb/run.py \
	--task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
	--variation medium \
	--sensor-config 1 \
	--dry-run \
	--output out/preview_sc_maint
```

Launch a headless physics simulation with side-camera video recording on an Isaac Sim host:

```bash
python scripts/idb/run.py --headless \
	--task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
	--variation sparse \
	--driver random \
	--num_envs 4 \
	--video \
	--output out/sc_maint_random
```

Verify all 20 registered environments in Isaac Sim:

```bash
python scripts/idb/verify_matrix.py --driver random --num_envs 2 --steps 120 --output out/idb_matrix
```

Regenerate all OpenUSD assets and `board_geometry.json` directly from Analog Devices' official STEP and 3MF CAD files:

```bash
uv run python scripts/idb/export_assets.py --fetch
```

## Documentation index

- [Architecture and physics design](docs/ARCHITECTURE.md)
- [Environment registry, spaces, sensors, and variations](docs/ENVIRONMENTS.md)
- [Task 1: Ethernet cable port swap](docs/tasks/task1_ethernet_port_swap.md)
- [Task 2: SC simplex fiber cable port swap](docs/tasks/task2_sc_port_swap.md)
- [Task 3: LC duplex fiber cable port swap](docs/tasks/task3_lc_duplex_port_swap.md)
- [Task 4: SC simplex fiber cable maintenance](docs/tasks/task4_sc_fiber_maintenance.md)
- [Task 5: LC duplex fiber cable maintenance](docs/tasks/task5_lc_duplex_fiber_maintenance.md)
- [Decision trail](decisions.tsv)

## References

- DORLR WireCraft: [github.com/DORLR/WireCraft](https://github.com/DORLR/WireCraft)
- Analog Devices IDB CAD and rubrics: [github.com/analogdevicesinc/ag-industrial-dexterity-benchmarks](https://github.com/analogdevicesinc/ag-industrial-dexterity-benchmarks)
- Industrial Dexterity Benchmark paper: [arXiv:2607.14021](https://arxiv.org/pdf/2607.14021)
