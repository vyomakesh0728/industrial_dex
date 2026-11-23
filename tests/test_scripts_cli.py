"""Behavioral CLI tests for scripts/idb/run.py and scripts/idb/verify_matrix.py."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_py_list_exports_all_20_environments(tmp_path):
    out_json = tmp_path / "ids.json"
    proc = subprocess.run(
        [sys.executable, "scripts/idb/run.py", "--list", str(out_json)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 20
    assert "IDB-Franka-Articulation-SCFiberMaintenance-v0" in data
    assert data["IDB-Franka-Articulation-SCFiberMaintenance-v0"]["observation_space"] == 76
    assert data["IDB-UR5-Articulation-EthernetPortSwap-v0"]["observation_space"] == 63


def test_run_py_dry_run_saves_resolved_config_and_provenance(tmp_path):
    out_dir = tmp_path / "demo_run"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/idb/run.py",
            "--task",
            "IDB-Franka-Articulation-SCFiberMaintenance-v0",
            "--variation",
            "medium",
            "--sensor-config",
            "2",
            "--dry-run",
            "--output",
            str(out_dir),
            "env.seed=123",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "params" / "env.yaml").is_file()
    assert (out_dir / "params" / "selection.yaml").is_file()
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert res["passed"] is True
    assert res["neighbor_cables"] == 6
    assert set(res["cameras"]) == {"wrist_d405_rgb", "scene_adtf3175_tof"}


def test_verify_matrix_dry_run_resolves_all_20_envs(tmp_path):
    out_dir = tmp_path / "matrix"
    proc = subprocess.run(
        [sys.executable, "scripts/idb/verify_matrix.py", "--dry-run", "--output", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out_dir / "matrix_report.json").read_text(encoding="utf-8"))
    assert report["total"] == 20 and report["passed"] == 20 and report["failed"] == 0
    assert (out_dir / "matrix_report.md").is_file()
