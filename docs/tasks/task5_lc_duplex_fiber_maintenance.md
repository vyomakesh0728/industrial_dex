# Task 5: LC duplex fiber cable maintenance

Depress the retention clip on an LC/UPC duplex fiber connector at port `A9` (or `A9` through `A12`), extract both ferrules cleanly, polish both `1.25 mm` ceramic ferrules on the fiber polishing station using at least 10 continuous figure-eight motions at `1.0` to `2.0 N`, and re-insert and latch the connector back into its original port.

## Objective

- Source and re-insertion port: `A9` by default (LC duplex coupler protruding `16.05 mm`, retention latch facing left `+Y`).
- Far anchor port: `B9` (LC duplex coupler, retention latch facing left `+Y`).
- Polishing station: circular raised pad atop Panel A at board-local `(-0.002, 0.0, 0.28415) m`, radius `16.89 mm`, facing `-X`.
- Candidate randomized triples when `variation.randomize_target_port=true`: `(B9, A9, A9)`, `(B10, A10, A10)`, `(B11, A11, A11)`, `(B12, A12, A12)`.

## Scene setup

- **Board**: IDB Board #1 at `(0.56, 0.0, 0.7825) m` with instrumented `polishing_pad` rigid body.
- **Cable**: `os2_fiber_duplex` preset (`0.30 m` yellow OS2 duplex zipcord, `30.0 mm` minimum bend radius, `0.20 Nm/rad` articulation bending stiffness).
- **Connectors**: `lc_duplex` (`IEC 61754-20`, twin `1.25 mm` ceramic ferrules spaced `6.25 mm` vertically along local `Y`, active shared retention latch requiring gripper depression before unplugging).
- **Dual-ferrule polishing validation**: `ferrule_positions()` returns both ferrule tips `(N, 2, 3)`. `FigureEightPolishingTracker` verifies that **both** ferrules lie simultaneously on the polishing disc (`normal_dist <= 3 mm` and radial offset `<= 16.89 mm`) during every polishing step.

## Interactions

1. **Removal phase**: Grasp the LC duplex housing at `A9` (`grasp_x = -16 mm`), depress the left-facing retention latch (`>= 1.5 N` pinch force), and pull back along `-X` until both noses clear `A9` by more than `5 mm`. Route to the polishing pad without kinking (`R_min >= 30.0 mm`).
2. **Polishing phase**: Place both LC ferrules flat on the circular polishing pad disc, maintain continuous contact (`>= 0.25 N`) inside `1.0` to `2.0 N` (`> 6.0 N` flags ferrule damage), and complete at least `10` figure-eight cycles without lift-off.
3. **Re-installation phase**: Route back to `A9` without kinking, align both LC housings and the left-facing latch (`roll_cos >= cos(15 deg)`), push past `8.5 mm` seat depth, and hold `3` steps to engage retention.

## Observations and spaces

- Registered Gym IDs: `IDB-{Franka,UR5}-{Articulation,Deformable}-LCFiberMaintenance-v0`
- Action space: `7` (Cartesian delta pose + gripper)
- Observation space: `76` (Franka) or `74` (UR5) in `obs["policy"]`, `obs["ft"]` (`6`), `obs["polishing_force_n"]` (`1`), plus enabled cameras.

## Success, failure, and scoring rubric

Source rubric: `datacenter-cable-manipulation/scoring-metrics.md` (Task 5, max 22 points).

| Step | Phase | Criterion | Max points | Evaluated by |
| --- | --- | --- | --- | --- |
| 1 | Removal | LC duplex connector removed from `A9` | 1 | `removed_from_source` |
| 2 | Removal | Cable routed to polishing station without kinking | 1 | `~kinked_to_polish` (`R_min >= 30.0 mm`) |
| 3 | Polishing | Both LC ferrules positioned on polishing surface | 1 | `polishing.positioned_ever` (all 2 ferrules) |
| 4 | Polishing | Minimum 10 complete figure-eight motions | 2 | `polishing.cycles_ok` (`cycles >= 10`) |
| 5 | Polishing | Continuous contact maintained during polishing | 2 | `polishing.continuous_contact_ok` |
| 6 | Polishing | Appropriate pressure applied (`1` to `2 N`) | 3 | `polishing.pressure_ok` (`>= 80%` in band, no damage) |
| 7 | Re-installation | Cable routed back to port without kinking | 1 | `~kinked_from_polish` |
| 8 | Re-installation | Correct port (`A9`) | 1 | `seated_port_idx == target_port_idx` |
| 9 | Re-installation | Both LC connectors properly aligned | 1 | `target_seating.key_aligned` |
| 10 | Re-installation | LC duplex connector inserted into `A9` | 3 | `target_seating.is_seated` (`depth >= 8.5 mm`) |
| 11 | Re-installation | Retention mechanism engaged | 3 | `latch.engaged` on `A9` |
| 12 | Integrity | No damage to ferrules, boots, clip, port, or neighbors | 3 | `~damaged()` |
| **Total** | | | **22** | Binary pass requires all 22 points and zero simulation faults |

## Running Task 5

```bash
# Dry-run config verification
python scripts/idb/run.py --task IDB-Franka-Articulation-LCFiberMaintenance-v0 --variation medium --dry-run

# Live headless simulation on Isaac Sim workstation
python scripts/idb/run.py --headless --task IDB-Franka-Articulation-LCFiberMaintenance-v0 \
	--variation sparse --sensor-config 1 --driver random --num_envs 4 --video --output out/task5_lc_maint
```
