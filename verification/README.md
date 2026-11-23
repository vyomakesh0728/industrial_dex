# Live Isaac Sim and Isaac Lab GPU verification

Executed on an NVIDIA GeForce RTX 3090 (24 GB VRAM, driver `580.142`, Vulkan headless) running NVIDIA's `nvcr.io/nvidia/isaac-lab:2.2.0` container (Isaac Sim 5.0 + Isaac Lab 2.2.0) with `WIRECRAFT_ROBOT_ASSET_DIR` pointing to `ur5_with_gripper_camera_v2.usd`.

## Summary of live checks

- **20-environment verification matrix** ([matrix_report.md](matrix_report.md), [matrix_report.json](matrix_report.json)): **20/20 passed**, `0` physics faults, `0` errors, `0` non-finite tensors across both robots (`Franka`, `UR5`), both wire backends (`Articulation`, `Deformable` CUDA FEM), and all 5 IDB Board #1 tasks (`num_envs=2`, `steps=60`, `driver=random`, partial and full resets exercised on every environment).
- **Kinematic board pose sync and polishing contact sensor** ([physx_sensors_check.json](physx_sensors_check.json)): randomized per-environment board poses synchronized `/frame` (`0.0 mm` error), `/polishing_pad` (`0.0 mm` error), and `/ports/A5` (`0.0 mm` error) directly in PhysX; pressing the active connector against `/idb_board/polishing_pad` triggered the dedicated `ContactSensor` (`0.3927 N` peak normal force).
- **Table I multimodal sensor configs 1 through 6 and all 4 scene variations**: all 6 rollout captures below passed (`exit 0`, zero errors, zero non-finite tensors) with `TiledCamera` RGB/depth/ToF rendering and 512x512 side-camera `rollout.mp4` export enabled simultaneously.

## Rollout gallery

The first two GIFs below are embedded in the top-level [README.md](../README.md). All six GIFs, source MP4 recordings (`videos/`), and per-run result JSON logs (`results/`) are archived in this directory.

| Demo | Task ID | Variation | Table I Sensor Config | Wire Backend | Faults | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| Franka SC fiber maintenance | `IDB-Franka-Articulation-SCFiberMaintenance-v0` | `medium` (6 neighbors + clips) | Config 1 (`wrist_d405_rgb`, `scene_d435i_rgb`, F/T) | Articulation | 0 | [GIF](gifs/franka_art_sc_maintenance_medium.gif) · [MP4](videos/franka_art_sc_maintenance_medium.mp4) · [JSON](results/franka_art_sc_maintenance_medium.json) |
| UR5 SC port swap (CUDA FEM) | `IDB-UR5-Deformable-SCPortSwap-v0` | `medium` (6 neighbors + clips) | Config 4 (`wrist_d405_rgb`, `scene_d435i_depth`, F/T) | Deformable FEM | 0 | [GIF](gifs/ur5_fem_sc_swap_medium.gif) · [MP4](videos/ur5_fem_sc_swap_medium.mp4) · [JSON](results/ur5_fem_sc_swap_medium.json) |
| Franka LC maintenance (CUDA FEM) | `IDB-Franka-Deformable-LCFiberMaintenance-v0` | `sparse` | Config 3 (`wrist_d405_rgb`, `wrist_d405_depth`, F/T) | Deformable FEM | 0 | [GIF](gifs/franka_fem_lc_maintenance_sparse.gif) · [MP4](videos/franka_fem_lc_maintenance_sparse.mp4) · [JSON](results/franka_fem_lc_maintenance_sparse.json) |
| Franka LC duplex swap (randomized) | `IDB-Franka-Articulation-LCDuplexPortSwap-v0` | `randomized` (pose, physics, lighting, port) | Config 5 (`wrist_d405_depth`, F/T) | Articulation | 1 | [GIF](gifs/franka_art_lc_duplex_randomized.gif) · [MP4](videos/franka_art_lc_duplex_randomized.mp4) · [JSON](results/franka_art_lc_duplex_randomized.json) |
| UR5 Ethernet port swap (dense) | `IDB-UR5-Articulation-EthernetPortSwap-v0` | `dense` (13 neighbors + clips + jitter) | Config 2 (`wrist_d405_rgb`, `scene_adtf3175_tof`, F/T) | Articulation | 7 | [GIF](gifs/ur5_art_ethernet_swap_dense.gif) · [MP4](videos/ur5_art_ethernet_swap_dense.mp4) · [JSON](results/ur5_art_ethernet_swap_dense.json) |
| UR5 LC fiber maintenance | `IDB-UR5-Articulation-LCFiberMaintenance-v0` | `sparse` | Config 6 (`wrist_d405_rgb`, no F/T) | Articulation | 1 | [GIF](gifs/ur5_art_lc_maintenance_sparse.gif) · [MP4](videos/ur5_art_lc_maintenance_sparse.mp4) · [JSON](results/ur5_art_lc_maintenance_sparse.json) |

### Additional rollout previews

| Franka Deformable LC Maintenance (`sparse`, Config 3) | Franka Articulation LC Duplex Swap (`randomized`, Config 5) |
| --- | --- |
| ![Franka Deformable LC Maintenance](gifs/franka_fem_lc_maintenance_sparse.gif) | ![Franka Articulation LC Duplex Swap](gifs/franka_art_lc_duplex_randomized.gif) |

| UR5 Articulation Ethernet Swap (`dense`, Config 2) | UR5 Articulation LC Maintenance (`sparse`, Config 6) |
| --- | --- |
| ![UR5 Articulation Ethernet Swap](gifs/ur5_art_ethernet_swap_dense.gif) | ![UR5 Articulation LC Maintenance](gifs/ur5_art_lc_maintenance_sparse.gif) |
