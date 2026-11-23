#!/usr/bin/env python3
"""Deterministic verification runner for offline checks (CAD, OpenUSD, PyTorch, config, scope).

Runs all checks executable without NVIDIA Isaac Sim and reports clearly what is VERIFIED
locally versus what requires an Isaac Sim + Isaac Lab GPU host.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

from idb_tasks.utils.isaac_compat import HAS_ISAACLAB, HAS_ISAACSIM

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n=== {label} ===")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    ok = proc.returncode == 0
    print(f"[{label}] {'VERIFIED' if ok else 'FAILED'} (exit={proc.returncode})")
    return ok


def main() -> int:
    ok = True
    ok &= _run("pytest suite (CAD, OpenUSD, spawners, physics, scoring, config, tasks, scope)", [sys.executable, "-m", "pytest", "-v"])

    with tempfile.TemporaryDirectory(prefix="idb_offline_check_") as tmp:
        tmp_path = Path(tmp)
        ok &= _run(
            "CLI registry export (run.py --list)",
            [sys.executable, "scripts/idb/run.py", "--list", str(tmp_path / "ids.json")],
        )
        ok &= _run(
            "20-env matrix config resolution (verify_matrix.py --dry-run)",
            [sys.executable, "scripts/idb/verify_matrix.py", "--dry-run", "--output", str(tmp_path / "matrix")],
        )

    print("\n=== Summary ===")
    print(f"Offline checks (CAD/OpenUSD/PyTorch/config/rubrics/scope): {'VERIFIED' if ok else 'NOT VERIFIED'}")
    if HAS_ISAACLAB and HAS_ISAACSIM:
        print("Isaac Sim + Isaac Lab detected: run scripts/idb/verify_matrix.py (without --dry-run) for live GPU physics verification.")
    else:
        print("Live Isaac Sim / PhysX stepping: NOT VERIFIED on this host (isaacsim/isaaclab not installed; requires x86_64 Linux + NVIDIA GPU).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
