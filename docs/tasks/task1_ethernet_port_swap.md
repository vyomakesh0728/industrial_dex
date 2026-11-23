# Task 1: Ethernet cable port swap

Unplug a Cat6 RJ45 patch cable from port `A1`, route it without kinking, and insert and latch it into adjacent port `A2` without damaging the connector, port, or neighboring cables.

## Objective

- Source port: `A1` (Panel A, Column 1, RJ45 keystone jack, retention latch at top `+Z`).
- Target port: `A2` (Panel A, Column 2, `17.50 mm` to the right along `-Y`, retention latch at top `+Z`).
- Far anchor port: `B1` (Panel B, Column 1, RJ45 keystone jack, retention latch at bottom `-Z`).
- Candidate randomized triples when `variation.randomize_target_port=true`: `(B1, A1, A2)`, `(B2, A2, A3)`, `(B3, A3, A4)`, `(B4, A4, A3)`.

## Scene setup

- **Board**: IDB Board #1 mounted at `board_pos = (0.56, 0.0, 0.7825) m` facing `-X` toward the robot.
- **Cable**: `cat6_ethernet` preset (`0.30 m` length, `5.8 mm` outer diameter, `23.2 mm` minimum bend radius = `4x` cable diameter, blue jacket, `0.35 Nm/rad` articulation bending stiffness).
- **Connectors**: `rj45` (`IEC 60603-7 8P8C`) on both ends, tip length `13.0 mm`, socket depth `16.0 mm`, `0.30 mm` per-side bore clearance, active top retention clip requiring gripper release before extraction.
- **Initial state**: Far connector seated and spring-anchored in `B1`; active connector seated and latched in `A1`; cable initialized along a smooth `PluggedCableArc` bowing forward in `-X` with `R_min > 40 mm`.

## Interactions

1. Approach `A1` and grasp the RJ45 housing (`grasp_x = -19 mm`).
2. Depress the top retention clip (`gripper_pinch_force >= 1.5 N` while grasped) to disengage `RetentionLatchState`. Pulling the plug out while still latched flags `latch.damaged`.
3. Extract the connector axially along `-X` until the leading face clears the `A1` mouth by more than `5 mm` (`removed_from_source = true`).
4. Translate `17.50 mm` right (`-Y`) to `A2` while maintaining cable bend radius `>= 23.2 mm` everywhere along the 41-point centerline.
5. Align the RJ45 plug with the `A2` bore and top latch slot (`roll_cos >= cos(15 deg)`).
6. Push along `+X` until the entire leading face crosses `insert_seat_depth = 10.0 mm` inside the `A2` bore and hold for `3` control steps to engage the retention clip.

## Observations and spaces

- Registered Gym IDs: `IDB-{Franka,UR5}-{Articulation,Deformable}-EthernetPortSwap-v0`
- Action space: `7` (Cartesian delta pose + gripper)
- Observation space: `65` (Franka) or `63` (UR5) in `obs["policy"]`, plus optional `obs["ft"]` (`6`) and configured camera tensors.

## Success, failure, and scoring rubric

Source rubric: `datacenter-cable-manipulation/scoring-metrics.md` (Task 1, max 6 points).

| Step | Criterion | Max points | Evaluated by |
| --- | --- | --- | --- |
| 1 | Connector removed from `A1` | 1 | `removed_from_source` (`tip_depth < -5 mm`) |
| 2 | Cable routed without kinking (`R >= 4x D`) | 1 | `~ever_kinked` (`R_min >= 23.2 mm`) |
| 3 | Correct port (`A2`) | 1 | `seated_port_idx == target_port_idx` |
| 4 | Connector inserted into `A2` | 1 | `target_seating.is_seated` (whole leading face `>= 10 mm` inside bore) |
| 5 | Retention clip engaged | 1 | `latch.engaged` on `A2` (`3` seated steps) |
| 6 | No damage to connector, port, or neighbors | 1 | `~latch.damaged & neighbors.integrity_ok` |
| **Total** | | **6** | Binary pass requires all 6 points and zero simulation faults |

Failure and termination triggers (in `eval` mode): simulation fault (`joint_speed >= 20 rad/s` or `port_speed >= 1000 rad/s`), out-of-bounds end-effector, joint limit violation, connector drop below `0.74 m`, clip rip-out or neighbor disturbance damage (`terminate_on_damage=true`), or episode timeout (`60 s`).

## Running Task 1

```bash
# Dry-run config verification (no Isaac Sim required)
python scripts/idb/run.py --task IDB-Franka-Articulation-EthernetPortSwap-v0 --variation medium --dry-run

# Live headless simulation on Isaac Sim workstation
python scripts/idb/run.py --headless --task IDB-Franka-Articulation-EthernetPortSwap-v0 \
	--variation sparse --driver random --num_envs 4 --video --output out/task1_eth_swap
```
