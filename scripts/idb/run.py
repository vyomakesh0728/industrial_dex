#!/usr/bin/env python3
"""Run any IDB Board #1 datacenter simulation environment in Isaac Sim + Isaac Lab.

Follows WireCraft's scripts/wirecraft/run.py interface and launch ordering (AppLauncher
is started before importing Omniverse/Isaac Lab environment modules for live runs;
simulation-only: zero, random, or idle viewer drivers; no RL/IL training, policies,
or dataset collection).

Examples:
    # List all 20 registered Gym IDs (works with or without Isaac Sim):
    python scripts/idb/run.py --list ids.json

    # Preview and validate resolved configuration without launching Isaac Sim:
    python scripts/idb/run.py --task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
        --variation medium --sensor-config 1 --dry-run --output out/preview

    # Run live physics simulation in Isaac Sim + Isaac Lab:
    python scripts/idb/run.py --headless --task IDB-Franka-Articulation-SCFiberMaintenance-v0 \
        --variation sparse --driver random --num_envs 4 --video --output out/sc_maint_demo
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
import traceback

ORIGINAL_ARGV = list(sys.argv[1:])


def build_parser(with_app_launcher: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="Registered Gym ID; overrides selector flags below.")
    parser.add_argument("--robot", default="franka", choices=("franka", "ur5"))
    parser.add_argument(
        "--job",
        default="sc_fiber_maintenance",
        choices=(
            "ethernet_port_swap",
            "sc_port_swap",
            "lc_duplex_port_swap",
            "sc_fiber_maintenance",
            "lc_fiber_maintenance",
        ),
    )
    parser.add_argument("--wire", default="articulation", choices=("articulation", "deformable"))
    parser.add_argument("--variation", default="sparse", choices=("sparse", "medium", "dense", "randomized"))
    parser.add_argument("--mode", default="demo", choices=("demo", "random", "eval"))
    parser.add_argument("--driver", default="random", choices=("zero", "random", "idle"))
    parser.add_argument(
        "--sensor-config",
        type=int,
        default=None,
        choices=(1, 2, 3, 4, 5, 6),
        help="arXiv:2607.14021 Table I sensor config.",
    )
    parser.add_argument("--experiment", action="append", default=[], help="Experiment YAML file (env: section only).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=0, help="Control steps; 0 derives from episode length.")
    parser.add_argument("--output", default=None, help="Output directory for params/, result.json, and optional video.")
    parser.add_argument("--video", action="store_true", help="Record env 0 from the side camera.")
    parser.add_argument("--list", default=None, help="Write all registered IDB Gym IDs and resolved specs to JSON and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose and validate config, save params/, and exit without launching Isaac Sim.",
    )
    if with_app_launcher:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default=None)
        parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless without GUI.")
        parser.add_argument("--enable_cameras", action="store_true", help="Enable Omniverse camera rendering.")
    return parser


def _import_idb_runtime():
    """Import Gymnasium, PyTorch, and idb_tasks AFTER AppLauncher (for live runs)."""
    import gymnasium as gym
    import torch

    import idb_tasks  # noqa: F401
    from idb_tasks.direct.idb.single_arm.common.compose import (
        apply_cli,
        compose_env_cfg,
        legacy_overrides,
        parse_cli_overrides,
        save_resolved,
    )
    from idb_tasks.direct.idb.single_arm.common.env_cfg import resolve_env_cfg
    from idb_tasks.direct.idb.single_arm.common.registry import SPECS

    return {
        "gym": gym,
        "torch": torch,
        "apply_cli": apply_cli,
        "compose_env_cfg": compose_env_cfg,
        "legacy_overrides": legacy_overrides,
        "parse_cli_overrides": parse_cli_overrides,
        "save_resolved": save_resolved,
        "resolve_env_cfg": resolve_env_cfg,
        "SPECS": SPECS,
    }


def select_task(args_cli: argparse.Namespace, specs: dict) -> str:
    if args_cli.task:
        if args_cli.task not in specs:
            raise SystemExit(f"Unknown task {args_cli.task!r}. Run with --list to inspect all {len(specs)} registered IDs.")
        return args_cli.task
    wanted = (args_cli.robot, args_cli.job, args_cli.wire)
    for task_id, spec in specs.items():
        if (spec.robot, spec.task, spec.wire) == wanted:
            return task_id
    raise SystemExit(
        f"No registered IDB environment matches robot={args_cli.robot}, job={args_cli.job}, wire={args_cli.wire}."
    )


def export_registry_json(out_path: str, rt: dict) -> None:
    payload = {}
    for task_id, spec in sorted(rt["SPECS"].items()):
        resolved = rt["resolve_env_cfg"](rt["compose_env_cfg"](spec, mode="demo"))
        entry = dataclasses.asdict(spec)
        entry["variant_overrides"] = dict(entry["variant_overrides"])
        entry["action_space"] = resolved.action_space
        entry["observation_space"] = resolved.observation_space
        entry["source_port"] = resolved.task.source_port
        entry["target_port"] = resolved.task.target_port
        entry["far_port"] = resolved.task.far_port
        entry["cable_preset"] = resolved.task.cable_preset
        entry["include_polishing"] = resolved.task.include_polishing
        payload[task_id] = entry
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[idb] wrote {len(payload)} registered environments to {out_path}")


def _compose_from_cli(task_id: str, args_cli: argparse.Namespace, extra_tokens: list[str], rt: dict):
    spec = rt["SPECS"][task_id]
    cfg = rt["compose_env_cfg"](
        spec,
        mode=args_cli.mode,
        variation=args_cli.variation,
        experiments=args_cli.experiment,
    )
    explicit = rt["parse_cli_overrides"](extra_tokens)
    if args_cli.sensor_config is not None:
        explicit["env.sensors.paper_config_id"] = args_cli.sensor_config
    if args_cli.video:
        explicit.setdefault("env.sensors.enable_cameras", True)
        existing_cams = list(explicit.get("env.sensors.cameras", []))
        if "side" not in existing_cams:
            existing_cams.append("side")
        explicit["env.sensors.cameras"] = existing_cams
        explicit.setdefault("env.sensors.camera_resolution", 512)
        explicit.setdefault("env.sensors.render_stride", 4)
    rt["apply_cli"](cfg, explicit, rt["legacy_overrides"](args_cli, ORIGINAL_ARGV))
    return rt["resolve_env_cfg"](cfg)


def _check_tensors(env, obs: dict, reward, action, report: dict, torch_mod) -> None:
    for name, tensor in (("obs", obs["policy"]), ("reward", reward), ("action", action)):
        if not torch_mod.isfinite(tensor).all():
            report["nonfinite"].append(name)
        if tensor.device.type != torch_mod.device(env.device).type:
            report["errors"].append(f"{name} on {tensor.device}, expected {env.device}")
    if obs["policy"].shape != (env.num_envs, env.cfg.observation_space):
        report["errors"].append(f"obs shape {tuple(obs['policy'].shape)} != {(env.num_envs, env.cfg.observation_space)}")
    if not torch_mod.isfinite(env.wire_state.centerline).all():
        report["nonfinite"].append("wire")


def _run_simulation(env, driver: str, steps_override: int, report: dict, torch_mod, video_writer=None) -> None:
    steps = steps_override or int(env.max_episode_length)
    obs, _ = env.reset()
    motion = 0.0
    last_q = env.arm.joint_pos.clone()
    for step in range(steps):
        if driver == "random":
            action = 2.0 * torch_mod.rand(env.num_envs, env.cfg.action_space, device=env.device) - 1.0
        else:
            action = torch_mod.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        obs, reward, terminated, truncated, extras = env.step(action)
        _check_tensors(env, obs, reward, action, report, torch_mod)
        done = terminated | truncated
        report["episodes"] += int(done.sum())
        report["partial_resets"] += int(done.any() and not done.all())
        motion += float((env.arm.joint_pos - last_q).abs().sum())
        last_q = env.arm.joint_pos.clone()
        report["steps"] = step + 1
        if "rubric_score" in extras:
            report["last_rubric_score"] = extras["rubric_score"].tolist()
            report["rubric_max"] = int(extras["rubric_max"])
        if video_writer is not None and step % env.cfg.sensors.render_stride == 0:
            frame = env.tiled_cameras["side"].data.output["rgb"][0, ..., :3].to(torch_mod.uint8).cpu().numpy()
            video_writer.append_data(frame)
        if step == steps // 3 and env.num_envs > 1:
            before = env.episode_length_buf[1:].clone()
            env._reset_idx(torch_mod.tensor([0], device=env.device))
            env.scene.write_data_to_sim()
            env.sim.forward()
            report["partial_resets"] += 1
            if not torch_mod.equal(before, env.episode_length_buf[1:]):
                report["errors"].append("partial reset modified another env")
        if step == (2 * steps) // 3:
            obs, _ = env.reset()
            report["full_resets"] += 1
    report["joint_motion"] = round(motion, 5)
    motion_ok = (motion > 1e-3) if driver == "random" else True
    report["passed"] = not report["nonfinite"] and not report["errors"] and motion_ok


def main() -> int:
    has_isaaclab_pkg = importlib.util.find_spec("isaaclab") is not None
    parser = build_parser(with_app_launcher=has_isaaclab_pkg)
    args_cli, extra_tokens = parser.parse_known_args()

    if args_cli.video or args_cli.sensor_config is not None or any(
        tok.lower() == "env.sensors.enable_cameras=true" for tok in extra_tokens
    ):
        args_cli.enable_cameras = True

    # Offline branches (--list / --dry-run) run without booting Omniverse Kit.
    if args_cli.list:
        rt = _import_idb_runtime()
        export_registry_json(args_cli.list, rt)
        return 0

    if args_cli.dry_run:
        rt = _import_idb_runtime()
        task_id = select_task(args_cli, rt["SPECS"])
        env_cfg = _compose_from_cli(task_id, args_cli, extra_tokens, rt)
        output_dir = os.path.abspath(args_cli.output or os.path.join("logs", "idb", task_id, args_cli.variation))
        os.makedirs(output_dir, exist_ok=True)
        rt["save_resolved"](output_dir, env_cfg)
        summary = {
            "task": task_id,
            "mode": env_cfg.mode,
            "variation": env_cfg.variant,
            "action_space": env_cfg.action_space,
            "observation_space": env_cfg.observation_space,
            "neighbor_cables": len(env_cfg.resolved_neighbor_cables),
            "cameras": sorted(env_cfg.camera_cfgs.keys()),
            "wrist_ft": env_cfg.sensors.enable_wrist_ft,
            "dry_run": True,
            "passed": True,
        }
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[idb] dry-run resolved {task_id} -> {output_dir}")
        return 0

    # Live Isaac Sim + Isaac Lab execution: boot AppLauncher BEFORE importing idb_tasks/isaaclab.envs.
    if not has_isaaclab_pkg:
        raise RuntimeError(
            "scripts/idb/run.py live simulation requires NVIDIA Isaac Sim (4.5/5.1) and Isaac Lab (v2.2.1). "
            "Run this command on an Isaac Lab workstation or use --dry-run / scripts/idb/check_offline.py."
        )
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    # Strip env.* override tokens before SimulationApp reads sys.argv (matching WireCraft run.py).
    sys.argv = [sys.argv[0]]
    simulation_app = AppLauncher(args_cli).app

    rt = _import_idb_runtime()
    task_id = select_task(args_cli, rt["SPECS"])
    env_cfg = _compose_from_cli(task_id, args_cli, extra_tokens, rt)
    output_dir = os.path.abspath(args_cli.output or os.path.join("logs", "idb", task_id, args_cli.variation))
    os.makedirs(output_dir, exist_ok=True)
    rt["save_resolved"](output_dir, env_cfg)

    report = {
        "task": task_id,
        "mode": env_cfg.mode,
        "variation": env_cfg.variant,
        "driver": args_cli.driver,
        "seed": env_cfg.seed,
        "num_envs": env_cfg.scene.num_envs,
        "action_space": env_cfg.action_space,
        "observation_space": env_cfg.observation_space,
        "steps": 0,
        "episodes": 0,
        "partial_resets": 0,
        "full_resets": 0,
        "nonfinite": [],
        "errors": [],
        "passed": False,
        "composition": env_cfg.composition,
        "argv": ORIGINAL_ARGV,
    }
    start_t = time.time()
    writer = None
    try:
        env = rt["gym"].make(task_id, cfg=env_cfg).unwrapped
        if args_cli.video:
            import imageio.v2 as imageio  # type: ignore[import-not-found]

            fps = 1.0 / (env.step_dt * env.cfg.sensors.render_stride)
            writer = imageio.get_writer(os.path.join(output_dir, "rollout.mp4"), fps=fps, macro_block_size=1)
        _run_simulation(env, args_cli.driver, args_cli.steps, report, rt["torch"], video_writer=writer)
        report["faults"] = env.fault_count
        report["fault_log"] = env.fault_log[:20]
        env.close()
    except Exception:
        report["errors"].append(traceback.format_exc()[-2000:])
    finally:
        if writer is not None:
            writer.close()
        report["wall_time_s"] = round(time.time() - start_t, 2)
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    print(f"[idb] {task_id} ({args_cli.driver}): passed={report['passed']} errors={report['errors'][:1]}")
    exit_code = 0 if report["passed"] else 1
    if exit_code:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
