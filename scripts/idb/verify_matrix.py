#!/usr/bin/env python3
"""Headless verification matrix runner across all 20 registered IDB Gym environments.

Runs scripts/idb/run.py in isolated subprocesses on an Isaac Sim + Isaac Lab workstation
and writes matrix_report.json and matrix_report.md. Use --dry-run on non-GPU hosts to
verify config resolution across all 20 environments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import idb_tasks  # noqa: F401
from idb_tasks.direct.idb.single_arm.common.registry import SPECS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/idb_matrix"))
    parser.add_argument("--driver", default="random", choices=("zero", "random"))
    parser.add_argument("--variation", default="sparse", choices=("sparse", "medium", "dense", "randomized"))
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--robot", default=None, choices=("franka", "ur5"))
    parser.add_argument("--wire", default=None, choices=("articulation", "deformable"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config resolution without launching Isaac Sim.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    run_script = Path(__file__).resolve().parent / "run.py"

    selected = [
        task_id
        for task_id, spec in sorted(SPECS.items())
        if (args.robot is None or spec.robot == args.robot) and (args.wire is None or spec.wire == args.wire)
    ]

    results = []
    for task_id in selected:
        out_dir = args.output / task_id
        cmd = [
            sys.executable,
            str(run_script),
            "--task",
            task_id,
            "--variation",
            args.variation,
            "--driver",
            args.driver,
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--output",
            str(out_dir),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        else:
            cmd.append("--headless")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        res_file = out_dir / "result.json"
        if res_file.is_file():
            entry = json.loads(res_file.read_text(encoding="utf-8"))
        else:
            entry = {"task": task_id, "passed": False, "errors": [proc.stderr[-800:] or f"exit {proc.returncode}"]}
        entry["exit_code"] = proc.returncode
        results.append(entry)
        status = "PASS" if entry.get("passed") and proc.returncode == 0 else "FAIL"
        print(f"[verify_matrix] {status}  {task_id}")

    passed = sum(1 for r in results if r.get("passed") and r.get("exit_code") == 0)
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "dry_run": args.dry_run,
        "results": results,
    }
    (args.output / "matrix_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# IDB verification matrix report",
        "",
        f"Mode: `{'dry-run' if args.dry_run else args.driver}` | Variation: `{args.variation}` | Passed: `{passed}/{len(results)}`",
        "",
        "| Task ID | Status | Action Space | Obs Space | Errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        st = "PASS" if r.get("passed") and r.get("exit_code") == 0 else "FAIL"
        errs = "; ".join(r.get("errors", []))[:80]
        lines.append(
            f"| `{r['task']}` | {st} | {r.get('action_space', '-')} | {r.get('observation_space', '-')} | {errs or '-'} |"
        )
    (args.output / "matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[verify_matrix] summary: {passed}/{len(results)} passed -> {args.output / 'matrix_report.md'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
