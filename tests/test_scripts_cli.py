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


def test_run_py_video_preserves_side_camera_with_sensor_config_and_defers_runtime_imports(tmp_path):
    import ast

    # 1. Static check: scripts/idb/run.py must NOT import idb_tasks/gymnasium/torch/isaaclab.envs at module top level
    # so AppLauncher runs before Omniverse/Isaac Lab module imports during live execution.
    tree = ast.parse((REPO_ROOT / "scripts/idb/run.py").read_text(encoding="utf-8"))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    for forbidden in ("idb_tasks", "gymnasium", "torch", "isaaclab", "isaacsim"):
        assert forbidden not in top_imports, f"scripts/idb/run.py imports {forbidden} before AppLauncher!"

    # 2. Behavioral check: --video combined with --sensor-config 1 must keep the 'side' camera alongside Table I cams.
    out_dir = tmp_path / "video_sensor"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/idb/run.py",
            "--task",
            "IDB-Franka-Articulation-SCFiberMaintenance-v0",
            "--sensor-config",
            "1",
            "--video",
            "--dry-run",
            "--output",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert set(res["cameras"]) == {"side", "wrist_d405_rgb", "scene_d435i_rgb"}

    # 3. Verify --sensor-config also sets enable_cameras=True before AppLauncher boots.
    run_src = (REPO_ROOT / "scripts/idb/run.py").read_text(encoding="utf-8")
    assert "args_cli.sensor_config is not None" in run_src
    assert "sys.argv = [sys.argv[0]]" in run_src
