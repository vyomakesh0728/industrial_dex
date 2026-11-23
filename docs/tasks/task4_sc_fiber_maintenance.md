# Task 4: SC simplex fiber cable maintenance

Unplug an SC/UPC simplex fiber connector from port `A5` (or any SC port `A5` through `A8`), clean the `2.5 mm` ceramic ferrule on the top-mounted fiber polishing station using at least 10 continuous figure-eight motions at `1.0` to `2.0 N` contact pressure, and re-insert the connector back into its original port.

This task is the baseline benchmark evaluated in Section III-D and Section VIII of [arXiv:2607.14021](https://arxiv.org/pdf/2607.14021).

## Objective

- Source and re-insertion port: `A5` by default (SC simplex coupler, alignment key facing left `+Y`).
- Far anchor port: `B5` (LC simplex coupler, latch facing up `+Z`).
- Polishing station: circular raised pad mounted centered atop Panel A at board-local `(-0.002, 0.0, 0.28415) m`, radius `16.89 mm`, facing `-X` toward the robot.
- Candidate randomized triples when `variation.randomize_target_port=true` (reproducing the paper's 4-port evaluation protocol): `(B5, A5, A5)`, `(B6, A6, A6)`, `(B7, A7, A7)`, `(B8, A8, A8)`.

## Scene setup

- **Board**: IDB Board #1 at `(0.56, 0.0, 0.7825) m` with dedicated kinematic rigid body `polishing_pad` instrumented with an Isaac Lab `ContactSensor` filtered to `/active_wire/end_port`.
- **Cable**: `os2_fiber_simplex` preset (`0.30 m` yellow OS2 fiber, `30.0 mm` minimum bend radius, `0.14 Nm/rad` articulation stiffness).
- **Heterogeneous connectors**: `lc_simplex` at far end `B5` and `sc_simplex` (`2.5 mm` ceramic ferrule protruding `1.5 mm` beyond the tip face) at active end `A5`.
- **Sensors**: Wrist 6-DOF F/T wrench (`enable_wrist_ft=true`) and polishing pad contact sensor (`enable_polishing_contact_sensor=true`) are enabled by default. Use `--sensor-config <1..6>` to replicate any Table I sensor ablation from arXiv:2607.14021.
- **Difficulty modulation**: `sparse` matches the paper's lowest-difficulty protocol (single active cable on an otherwise unpopulated board); `medium` and `dense` add 6 to 14 surrounding cables.

## Interactions

1. **Removal phase**: Grasp the SC shroud at `A5` (`grasp_x = -16 mm`) and pull straight back along `-X` until clear of the coupler mouth by more than `5 mm`. Route up to the polishing station without kinking (`R_min >= 30.0 mm`).
2. **Polishing phase**: Bring the SC ferrule onto the polishing pad disc (`normal_dist <= 3 mm`, radial distance `<= 16.89 mm`). Establish continuous contact (`>= 0.25 N`) inside the `1.0` to `2.0 N` normal force band (`> 6.0 N` flags ferrule damage). Execute at least `10` complete figure-eight cycles (each cycle requires alternating smooth CW and CCW tangent sweeps `>= 3.2 rad` and `>= 12 mm` arc length per lobe) without breaking contact.
3. **Re-installation phase**: Route back to `A5` without kinking (`R_min >= 30.0 mm`), align the SC key facing left (`+Y`, `roll_cos >= cos(15 deg)`), insert past `9.5 mm` seat depth, and hold `3` steps to engage retention.

## Observations and spaces

- Registered Gym IDs: `IDB-{Franka,UR5}-{Articulation,Deformable}-SCFiberMaintenance-v0`
- Action space: `7` (Cartesian delta pose + gripper)
- Observation space: `76` (Franka) or `74` (UR5) in `obs["policy"]` (includes the 11 polishing pad and cycle-progress features), `obs["ft"]` (`6`), `obs["polishing_force_n"]` (`1`), plus enabled Table I cameras.

## Success, failure, and scoring rubric

Source rubric: `datacenter-cable-manipulation/scoring-metrics.md` (Task 4, max 19 points).

| Step | Phase | Criterion | Max points | Evaluated by |
| --- | --- | --- | --- | --- |
| 1 | Removal | SC connector removed from `A5` | 1 | `removed_from_source` |
| 2 | Removal | Cable routed to polishing station without kinking | 1 | `~kinked_to_polish` (`R_min >= 30.0 mm`) |
| 3 | Polishing | SC ferrule positioned on polishing surface | 1 | `polishing.positioned_ever` |
| 4 | Polishing | Minimum 10 complete figure-eight motions | 2 | `polishing.cycles_ok` (`cycles >= 10`) |
| 5 | Polishing | Continuous contact maintained during polishing | 2 | `polishing.continuous_contact_ok` (`contact_breaks == 0`) |
| 6 | Polishing | Appropriate pressure applied (`1` to `2 N`) | 3 | `polishing.pressure_ok` (`>= 80%` in band, no damage) |
| 7 | Re-installation | Cable routed back to port without kinking | 1 | `~kinked_from_polish` (`R_min >= 30.0 mm`) |
| 8 | Re-installation | Correct port (`A5`) | 1 | `seated_port_idx == target_port_idx` |
| 9 | Re-installation | Alignment key properly oriented | 1 | `target_seating.key_aligned` |
| 10 | Re-installation | SC connector inserted into `A5` | 2 | `target_seating.is_seated` (`depth >= 9.5 mm`) |
| 11 | Re-installation | Retention mechanism engaged | 2 | `latch.engaged` on `A5` |
| 12 | Integrity | No damage to connector, ferrule, port, or neighbors | 2 | `~damaged()` (`~latch.damaged & ~ferrule_damaged & neighbors.integrity_ok`) |
| **Total** | | | **19** | Binary pass requires all 19 points and zero simulation faults |

In addition to the 19-point rubric, `env.extras["phase_flags"]` exposes the arXiv:2607.14021 three-phase binary indicators (`grasp`, `clean`, `insert`, `integrity`) so downstream evaluators can track per-phase completion across all four SC ports (`A5` through `A8`).

## Running Task 4

```bash
# Dry-run matching paper Table I Config 1 (D405 wrist RGB + D435i scene RGB + wrist F/T)
python scripts/idb/run.py --task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
	--variation sparse --sensor-config 1 --dry-run --output out/task4_cfg1

# Live headless simulation across all four SC ports with board randomization
python scripts/idb/run.py --headless --task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
	--variation randomized --sensor-config 2 --driver random --num_envs 4 --video --output out/task4_sc_maint
```
