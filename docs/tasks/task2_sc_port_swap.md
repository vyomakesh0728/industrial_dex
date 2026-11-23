# Task 2: SC simplex fiber cable port swap

Unplug an SC/UPC simplex fiber connector from port `A5`, route the single-mode patch cord without kinking, align the SC key, and insert the connector into adjacent port `A6` without damaging the ferrule, boot, port, or neighboring cables.

## Objective

- Source port: `A5` (Panel A, Column 5, SC simplex coupler protruding `15.20 mm`, alignment key facing left `+Y`).
- Target port: `A6` (Panel A, Column 6, `17.00 mm` to the right along `-Y`, alignment key facing left `+Y`).
- Far anchor port: `B5` (Panel B, Column 5, LC simplex coupler protruding `4.10 mm`, latch facing up `+Z`).
- Candidate randomized triples when `variation.randomize_target_port=true`: `(B5, A5, A6)`, `(B6, A6, A7)`, `(B7, A7, A8)`, `(B8, A8, A7)`.

## Scene setup

- **Board**: IDB Board #1 at `(0.56, 0.0, 0.7825) m`.
- **Cable**: `os2_fiber_simplex` preset (`0.30 m` yellow OS2 single-mode patch cord, `3.0 mm` jacket outer diameter, `30.0 mm` minimum bend radius, `0.14 Nm/rad` articulation bending stiffness).
- **Heterogeneous connectors**: `lc_simplex` (`IEC 61754-20`) at far end `B5`; `sc_simplex` (`IEC 61754-4`, `2.5 mm` ceramic ferrule, tip length `12.5 mm`, socket depth `15.2 mm`, `0.25 mm` per-side clearance) at active end `A5`.
- **Initial state**: Far LC simplex end seated and held in `B5`; active SC simplex end seated and latched in `A5`; cable initialized along a smooth `PluggedCableArc` with `R_min > 40 mm`.

## Interactions

1. Approach `A5` and grasp the outer push-pull SC shroud (`grasp_x = -16 mm`).
2. Pull axially along `-X` to release the SC push-pull detent cleanly (`requires_latch_release=false`) and clear the `A5` coupler mouth by more than `5 mm`.
3. Translate `17.00 mm` right (`-Y`) to `A6` while keeping minimum bend radius `>= 30.0 mm` along the cable centerline.
4. Orient the SC alignment key ridge along `+Y` (matching `A6`'s left-facing keyway within `15 deg`).
5. Push along `+X` until the leading face passes `insert_seat_depth = 9.5 mm` inside the `A6` bore and holds for `3` control steps to engage the retention detent.

## Observations and spaces

- Registered Gym IDs: `IDB-{Franka,UR5}-{Articulation,Deformable}-SCPortSwap-v0`
- Action space: `7` (Cartesian delta pose + gripper)
- Observation space: `65` (Franka) or `63` (UR5) in `obs["policy"]`, plus optional `obs["ft"]` (`6`) and configured camera tensors.

## Success, failure, and scoring rubric

Source rubric: `datacenter-cable-manipulation/scoring-metrics.md` (Task 2, max 10 points).

| Step | Criterion | Max points | Evaluated by |
| --- | --- | --- | --- |
| 1 | SC connector removed from `A5` | 1 | `removed_from_source` |
| 2 | Cable routed without kinking | 1 | `~ever_kinked` (`R_min >= 30.0 mm`) |
| 3 | Correct port (`A6`) | 1 | `seated_port_idx == target_port_idx` |
| 4 | Alignment key properly oriented | 1 | `target_seating.key_aligned` (`roll_cos >= cos(15 deg)`) |
| 5 | SC connector inserted into `A6` | 2 | `target_seating.is_seated` (`depth >= 9.5 mm`) |
| 6 | Retention mechanism engaged | 2 | `latch.engaged` on `A6` |
| 7 | No damage to ferrule, boot, or port | 2 | `~latch.damaged & neighbors.integrity_ok` |
| **Total** | | **10** | Binary pass requires all 10 points and zero simulation faults |

## Running Task 2

```bash
# Dry-run config verification
python scripts/idb/run.py --task IDB-Franka-Articulation-SCPortSwap-v0 --variation medium --dry-run

# Live headless simulation on Isaac Sim workstation
python scripts/idb/run.py --headless --task IDB-Franka-Articulation-SCPortSwap-v0 \
	--variation sparse --driver random --num_envs 4 --video --output out/task2_sc_swap
```
