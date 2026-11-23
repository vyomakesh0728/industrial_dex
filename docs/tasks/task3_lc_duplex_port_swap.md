# Task 3: LC duplex fiber cable port swap

Depress the retention clip on an LC/UPC duplex fiber connector at port `A9`, extract both mated ferrules cleanly, route the zipcord without kinking, align both LC housings, and insert and latch the connector into adjacent port `A10`.

## Objective

- Source port: `A9` (Panel A, Column 9, LC duplex coupler protruding `16.05 mm`, retention latch facing left `+Y`).
- Target port: `A10` (Panel A, Column 10, `17.00 mm` to the right along `-Y`, retention latch facing left `+Y`).
- Far anchor port: `B9` (Panel B, Column 9, LC duplex coupler, retention latch facing left `+Y`).
- Candidate randomized triples when `variation.randomize_target_port=true`: `(B9, A9, A10)`, `(B10, A10, A11)`, `(B11, A11, A12)`, `(B12, A12, A11)`.

## Scene setup

- **Board**: IDB Board #1 at `(0.56, 0.0, 0.7825) m`.
- **Cable**: `os2_fiber_duplex` preset (`0.30 m` yellow duplex zipcord, `4.0 mm` effective diameter, `30.0 mm` minimum bend radius, `0.20 Nm/rad` articulation bending stiffness).
- **Connectors**: `lc_duplex` (`IEC 61754-20`, twin `1.25 mm` ceramic ferrules spaced `6.25 mm` vertically along local `Y`, tip span `10.85 x 4.60 mm`, tip length `11.0 mm`, socket depth `15.5 mm`, `0.25 mm` per-side bore clearance, active shared duplex retention clip).
- **Initial state**: Far LC duplex connector seated and spring-anchored in `B9`; active LC duplex connector seated and latched in `A9`; cable initialized along a smooth `PluggedCableArc`.

## Interactions

1. Approach `A9` and grasp the LC duplex clip housing (`grasp_x = -16 mm`).
2. Depress the left-facing retention latch (`gripper_pinch_force >= 1.5 N` while grasped) before pulling back. Extracting the connector without depressing the clip triggers `latch.damaged`.
3. Pull axially along `-X` until both LC noses clear the `A9` mouth by more than `5 mm`.
4. Translate `17.00 mm` right (`-Y`) to `A10` without violating the `30.0 mm` OS2 bend radius limit.
5. Align both vertical LC plugs and the left-facing retention latch with `A10` (`roll_cos >= cos(15 deg)`).
6. Push along `+X` until the full dual-nose leading face crosses `insert_seat_depth = 8.5 mm` inside `A10` and holds for `3` steps to engage retention.

## Observations and spaces

- Registered Gym IDs: `IDB-{Franka,UR5}-{Articulation,Deformable}-LCDuplexPortSwap-v0`
- Action space: `7` (Cartesian delta pose + gripper)
- Observation space: `65` (Franka) or `63` (UR5) in `obs["policy"]`, plus optional `obs["ft"]` (`6`) and configured camera tensors.

## Success, failure, and scoring rubric

Source rubric: `datacenter-cable-manipulation/scoring-metrics.md` (Task 3, max 13 points).

| Step | Criterion | Max points | Evaluated by |
| --- | --- | --- | --- |
| 1 | LC duplex connector removed from `A9` | 1 | `removed_from_source` |
| 2 | Cable routed without kinking | 1 | `~ever_kinked` (`R_min >= 30.0 mm`) |
| 3 | Correct port (`A10`) | 1 | `seated_port_idx == target_port_idx` |
| 4 | Both LC connectors properly aligned | 1 | `target_seating.key_aligned` |
| 5 | LC duplex connector inserted into `A10` | 3 | `target_seating.is_seated` (`depth >= 8.5 mm`) |
| 6 | Retention mechanism engaged | 3 | `latch.engaged` on `A10` |
| 7 | No damage to ferrules, boots, clip, or port | 3 | `~latch.damaged & neighbors.integrity_ok` |
| **Total** | | **13** | Binary pass requires all 13 points and zero simulation faults |

## Running Task 3

```bash
# Dry-run config verification
python scripts/idb/run.py --task IDB-Franka-Articulation-LCDuplexPortSwap-v0 --variation medium --dry-run

# Live headless simulation on Isaac Sim workstation
python scripts/idb/run.py --headless --task IDB-Franka-Articulation-LCDuplexPortSwap-v0 \
	--variation sparse --driver random --num_envs 4 --video --output out/task3_lc_swap
```
