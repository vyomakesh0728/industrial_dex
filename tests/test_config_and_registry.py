"""Config composition, layer priority, scene variations, sensor configs, and 20-ID Gym registry tests."""

from __future__ import annotations

import argparse

import gymnasium as gym
import pytest

import idb_tasks  # noqa: F401
from idb_tasks.direct.idb.single_arm.common import compose
from idb_tasks.direct.idb.single_arm.common.compose import (
    apply_cli,
    apply_env_layer,
    apply_overrides,
    compose_env_cfg,
    legacy_overrides,
    parse_cli_overrides,
)
from idb_tasks.direct.idb.single_arm.common.env_cfg import resolve_env_cfg
from idb_tasks.direct.idb.single_arm.common.registry import SPECS

FRANKA_SC_MAINT = "IDB-Franka-Articulation-SCFiberMaintenance-v0"
UR5_ETH_SWAP = "IDB-UR5-Articulation-EthernetPortSwap-v0"
FRANKA_LC_FEM = "IDB-Franka-Deformable-LCFiberMaintenance-v0"


def test_all_20_gym_ids_registered_and_resolvable():
    assert len(SPECS) == 20
    for task_id, spec in SPECS.items():
        assert task_id in gym.registry, f"{task_id} missing from gymnasium.registry"
        cfg = compose_env_cfg(spec, mode="demo")
        resolved = resolve_env_cfg(cfg)
        assert resolved.resolved is True
        assert resolved.action_space == 7
        expected_base = 65 if spec.robot == "franka" else 63
        expected_obs = expected_base + (11 if "Maintenance" in task_id else 0)
        assert resolved.observation_space == expected_obs, f"{task_id}: obs {resolved.observation_space} != {expected_obs}"


def test_heterogeneous_sc_cable_resolves_lc_far_and_sc_active():
    spec = SPECS[FRANKA_SC_MAINT]
    cfg = resolve_env_cfg(compose_env_cfg(spec, mode="eval"))
    assert cfg.wire_spawn.start_family == "lc_simplex"
    assert cfg.wire_spawn.end_family == "sc_simplex"
    assert cfg.wire.min_bend_radius == pytest.approx(0.030)
    assert cfg.task.source_port == "A5" and cfg.task.target_port == "A5" and cfg.task.far_port == "B5"


def test_variation_presets_modulate_board_density_and_clips():
    spec = SPECS[UR5_ETH_SWAP]
    sparse = resolve_env_cfg(compose_env_cfg(spec, mode="demo", variation="sparse"))
    medium = resolve_env_cfg(compose_env_cfg(spec, mode="demo", variation="medium"))
    dense = resolve_env_cfg(compose_env_cfg(spec, mode="demo", variation="dense"))

    assert len(sparse.resolved_neighbor_cables) == 0
    assert len(medium.resolved_neighbor_cables) == 6
    assert len(dense.resolved_neighbor_cables) >= 12
    assert not sparse.board.include_cable_clips and dense.board.include_cable_clips
    assert dense.variation.randomize_board_pose and dense.variation.randomize_cable_physics

    # Confirm no neighbor cable collides with active task reserved ports (B1, A1, A2).
    reserved = {"B1", "A1", "A2"}
    for far_p, act_p, _ in dense.resolved_neighbor_cables:
        assert far_p not in reserved and act_p not in reserved


def test_paper_table1_sensor_configs_resolve_cameras_and_ft():
    spec = SPECS[FRANKA_SC_MAINT]
    for cfg_id in range(1, 7):
        cfg = compose_env_cfg(spec, mode="eval")
        apply_overrides(cfg, {"sensors": {"paper_config_id": cfg_id}}, derived=compose.DERIVED_ENV_FIELDS)
        resolved = resolve_env_cfg(cfg)
        assert resolved.sensors.enable_cameras is True
        assert len(resolved.camera_cfgs) >= 1
        if cfg_id == 6:
            assert resolved.sensors.enable_wrist_ft is False
            assert list(resolved.camera_cfgs) == ["wrist_d405_rgb"]
        elif cfg_id == 1:
            assert set(resolved.camera_cfgs) == {"wrist_d405_rgb", "scene_d435i_rgb"}
        elif cfg_id == 2:
            assert set(resolved.camera_cfgs) == {"wrist_d405_rgb", "scene_adtf3175_tof"}


def test_layer_priority_experiment_files_and_cli(tmp_path):
    exp1 = tmp_path / "a.yaml"
    exp2 = tmp_path / "b.yaml"
    exp1.write_text("env:\n  action_scale: 0.02\n  wire:\n    stiffness: 0.5\n", encoding="utf-8")
    exp2.write_text("env:\n  action_scale: 0.03\n", encoding="utf-8")

    cfg = compose_env_cfg(SPECS[UR5_ETH_SWAP], mode="demo", experiments=[str(exp1), str(exp2)])
    assert cfg.action_scale == pytest.approx(0.03)
    assert cfg.wire.stiffness == pytest.approx(0.5)

    explicit = parse_cli_overrides(["env.action_scale=0.04", "env.seed=99"])
    args = argparse.Namespace(num_envs=4, seed=99)
    legacy = legacy_overrides(args, ["--num_envs", "4", "--seed", "99"])
    apply_cli(cfg, explicit, legacy)
    assert cfg.action_scale == pytest.approx(0.04)
    assert cfg.seed == 99 and cfg.scene.num_envs == 4


def test_merge_rules_reject_bad_types_unknown_keys_derived_fields_and_agent_section(tmp_path):
    cfg = compose_env_cfg(SPECS[UR5_ETH_SWAP], mode="demo")
    with pytest.raises(KeyError, match="wire.nonexistent"):
        apply_overrides(cfg, {"wire": {"nonexistent": 1}})
    for bad in ({"decimation": 1.5}, {"enable_terminations": 1}, {"action_scale": True}, {"wire": {"lateral_bow": [1.0]}}):
        with pytest.raises(ValueError):
            apply_overrides(cfg, bad)
    with pytest.raises(ValueError, match="derived"):
        apply_overrides(cfg, {"wire_spawn": 1}, derived=compose.DERIVED_ENV_FIELDS)
    with pytest.raises(ValueError, match="connector preset"):
        apply_env_layer(cfg, "connector", {"wire": {"radius": 0.01}})

    # Experiment YAML must reject any 'agent:' section (simulation-only scope).
    bad_exp = tmp_path / "bad.yaml"
    bad_exp.write_text("env:\n  seed: 7\nagent:\n  max_iterations: 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simulation-only"):
        compose_env_cfg(SPECS[UR5_ETH_SWAP], mode="demo", experiments=[str(bad_exp)])


def test_instances_do_not_share_mutable_state():
    a = compose_env_cfg(SPECS[FRANKA_SC_MAINT], mode="demo")
    b = compose_env_cfg(SPECS[FRANKA_SC_MAINT], mode="demo")
    a.wire.stiffness = 99.0
    a.robot.arm_joints.append("extra")
    a.ee_safe_zone[0] = -99.0
    assert b.wire.stiffness != 99.0
    assert "extra" not in b.robot.arm_joints
    assert b.ee_safe_zone[0] != -99.0


def test_deformable_wire_rejects_cpu_device():
    cfg = compose_env_cfg(SPECS[FRANKA_LC_FEM], mode="demo")
    cfg.sim.device = "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        resolve_env_cfg(cfg)


def test_ur5_depth_camera_defines_valid_spawn_and_clipping_range():
    import inspect
    from idb_tasks.direct.idb.single_arm.ur5.robot import UR5Adapter

    cfg = resolve_env_cfg(compose_env_cfg(SPECS[UR5_ETH_SWAP], mode="eval"))
    cams = UR5Adapter.camera_cfgs(cfg)
    assert cams["wrist_d405_depth"]["prim_path"].endswith("wrist_3_link/wrist_d405_depth")
    assert cams["wrist_d405_depth"]["clipping_range"] == (0.04, 1.5)
    # Also verify live Isaac Lab branch spawns a PinholeCameraCfg with clipping_range for distance_to_camera.
    src = inspect.getsource(UR5Adapter.camera_cfgs)
    depth_block = src.split('"wrist_d405_depth": TiledCameraCfg(', 1)[1]
    assert "PinholeCameraCfg" in depth_block and "clipping_range=" in depth_block
    assert "spawn=None" not in depth_block


def test_population_catalog_partitions_all_32_ports_and_compat_does_not_probe_omni_spec():
    import ast
    from pathlib import Path
    from idb_tasks.direct.idb.single_arm.common.env_cfg import POPULATION_CATALOG

    seen: list[str] = []
    for far_p, act_p, _ in POPULATION_CATALOG:
        seen.extend([far_p, act_p])
    assert len(seen) == 32 and len(set(seen)) == 32

    compat_src = (
        Path(__file__).resolve().parents[1]
        / "source/idb_tasks/idb_tasks/utils/isaac_compat.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(compat_src)
    find_spec_args = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "find_spec"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert "omni" not in find_spec_args
