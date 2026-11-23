"""End-to-end vectorized PyTorch lifecycle tests for all 5 modular IDB datacenter tasks."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import idb_tasks  # noqa: F401
from idb_tasks.direct.idb.single_arm.common.board import IDBBoardRuntime
from idb_tasks.direct.idb.single_arm.common.compose import compose_env_cfg
from idb_tasks.direct.idb.single_arm.common.env_cfg import resolve_env_cfg
from idb_tasks.direct.idb.single_arm.common.registry import SPECS
from idb_tasks.direct.idb.wire_base.connectors import ConnectorGeometry
from idb_tasks.direct.idb.wire_base.wire_state import PluggedCableArc, WireState, tip_from_root


def _build_task_harness(task_id: str, num_envs: int = 4, variation: str = "sparse"):
    """Construct a real task instance wired to real BoardSpec/IDBBoardRuntime and PyTorch state tensors."""
    spec = SPECS[task_id]
    cfg = resolve_env_cfg(compose_env_cfg(spec, mode="eval", variation=variation))
    n_joints = len(cfg.robot.arm_joints) + 1
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=num_envs,
        device="cpu",
        connector=ConnectorGeometry.from_family(cfg.connector.family, cfg.connector.socket_clearance),
        far_connector=ConnectorGeometry.from_family(cfg.connector.resolved_far_family, cfg.connector.socket_clearance),
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
        arm=SimpleNamespace(
            joint_pos=torch.zeros(num_envs, n_joints),
            joint_vel=torch.zeros(num_envs, n_joints),
            ee_pos=torch.tensor([[0.35, 0.0, 1.0]]).repeat(num_envs, 1),
            ee_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            opening=torch.ones(num_envs),
            force_left=torch.zeros(num_envs),
            force_right=torch.zeros(num_envs),
            gripper_pinch_force=torch.zeros(num_envs),
            is_grasping=torch.zeros(num_envs, dtype=torch.bool),
            wrench=torch.zeros(num_envs, 6),
        ),
        ee_local_pos=torch.tensor([[0.35, 0.0, 1.0]]).repeat(num_envs, 1),
        ee_local_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
        port_local_pos=torch.zeros(num_envs, 3),
        tip_local_pos=torch.zeros(num_envs, 3),
        wire_state=None,
        board=None,
    )
    board = IDBBoardRuntime(env)
    board.initialize()
    board._test_polishing_force = torch.zeros(num_envs)
    board.polishing_contact_force_n = lambda: board._test_polishing_force
    env.board = board

    task = cfg.task.task_class(env)
    task.initialize()
    all_ids = torch.arange(num_envs)
    far_pos, far_quat, act_pos, act_quat = task.reset(all_ids, nominal_neighbors=None)

    def set_connector(pos: torch.Tensor, quat: torch.Tensor, cable_pts: torch.Tensor | None = None):
        if cable_pts is None:
            arc = PluggedCableArc.solve(far_pos, pos, cable_length=cfg.wire.length, num_points=cfg.wire.centerline_points)
            cable_pts = arc.points
        env.wire_state = WireState(
            insertion_end_pos=pos.clone(),
            insertion_end_quat=quat.clone(),
            far_end_pos=far_pos.clone(),
            far_end_quat=far_quat.clone(),
            tip_pos=tip_from_root(pos, quat, env.connector.tip_length),
            centerline=cable_pts.clone(),
        )
        env.port_local_pos = pos.clone()
        env.tip_local_pos = env.wire_state.tip_pos.clone()

    set_connector(act_pos, act_quat)
    task.update()
    return env, task, set_connector, (far_pos, far_quat, act_pos, act_quat)


@pytest.mark.parametrize(
    "task_id,max_score",
    [
        ("IDB-Franka-Articulation-EthernetPortSwap-v0", 6),
        ("IDB-Franka-Articulation-SCPortSwap-v0", 10),
        ("IDB-Franka-Articulation-LCDuplexPortSwap-v0", 13),
        ("IDB-Franka-Articulation-SCFiberMaintenance-v0", 19),
        ("IDB-Franka-Articulation-LCFiberMaintenance-v0", 22),
    ],
)
def test_all_five_tasks_observation_shape_and_partial_reset_isolation(task_id: str, max_score: int):
    env, task, set_connector, (_, _, act_pos, act_quat) = _build_task_harness(task_id, num_envs=4)
    obs = task.observations()
    assert obs.shape == (4, env.cfg.observation_space)
    assert torch.isfinite(obs).all()

    rubric0 = task.evaluate()
    assert rubric0.max_score == max_score
    assert not bool(rubric0.passed.any())

    # Unplug all 4 envs cleanly (depress latch, pull out -X by 6 cm).
    env.arm.is_grasping[:] = True
    env.arm.gripper_pinch_force[:] = 3.0
    task.update()
    pulled = act_pos.clone()
    pulled[:, 0] -= 0.06
    set_connector(pulled, act_quat)
    task.update()
    assert bool(task.removed_from_source.all())

    # Partial reset ONLY env 1: envs 0, 2, 3 must remain removed while env 1 resets cleanly!
    task.reset(torch.tensor([1]), nominal_neighbors=None)
    assert not bool(task.removed_from_source[1])
    assert bool(task.removed_from_source[0]) and bool(task.removed_from_source[2]) and bool(task.removed_from_source[3])


@pytest.mark.parametrize(
    "task_id,expected_max",
    [
        ("IDB-Franka-Articulation-EthernetPortSwap-v0", 6),
        ("IDB-UR5-Articulation-SCPortSwap-v0", 10),
        ("IDB-Franka-Articulation-LCDuplexPortSwap-v0", 13),
    ],
)
def test_port_swap_tasks_full_success_and_wrong_port_rejection(task_id: str, expected_max: int):
    env, task, set_connector, (_, _, act_pos, act_quat) = _build_task_harness(task_id, num_envs=2)
    target_pos, target_quat = env.board.seated_connector_pose(env.cfg.task.target_port, env.connector)

    # Step 1: Depress clip and unplug both envs cleanly from source port.
    env.arm.is_grasping[:] = True
    env.arm.gripper_pinch_force[:] = 3.0
    task.update()
    pulled = act_pos.clone()
    pulled[:, 0] -= 0.06
    set_connector(pulled, act_quat)
    task.update()
    assert bool(task.removed_from_source.all())

    # Step 2: Env 0 inserts into TARGET port; Env 1 re-inserts into SOURCE (wrong) port!
    env.arm.gripper_pinch_force[:] = 0.0
    dest_pos = torch.stack((target_pos[0], act_pos[1]), dim=0)
    dest_quat = torch.stack((target_quat[0], act_quat[1]), dim=0)
    set_connector(dest_pos, dest_quat)
    for _ in range(4):
        task.update()

    rubric = task.evaluate()
    assert bool(rubric.passed[0]) and int(rubric.total_score[0]) == expected_max
    assert not bool(rubric.passed[1]), "Inserting back into source port must NOT pass a port swap task"


@pytest.mark.parametrize(
    "task_id,expected_max",
    [
        ("IDB-Franka-Articulation-SCFiberMaintenance-v0", 19),
        ("IDB-UR5-Articulation-LCFiberMaintenance-v0", 22),
    ],
)
def test_fiber_maintenance_tasks_require_polishing_before_reinsertion(task_id: str, expected_max: int):
    env, task, set_connector, (_, _, act_pos, act_quat) = _build_task_harness(task_id, num_envs=2)
    pad_center, _, _ = env.board.polishing_pad_world_pose()

    # Step 1: Clean unplug from source port.
    env.arm.is_grasping[:] = True
    env.arm.gripper_pinch_force[:] = 3.0
    task.update()
    pulled = act_pos.clone()
    pulled[:, 0] -= 0.06
    set_connector(pulled, act_quat)
    task.update()

    # Step 2: Env 0 executes 10 figure-8 cycles on polishing pad at 1.5 N; Env 1 skips polishing!
    ferrule_x = env.connector.tip_length + env.connector.ferrule.protrusion
    steps_per_cycle = 50
    for k in range(10 * steps_per_cycle + 12):
        t = k * (math.tau / steps_per_cycle)
        u = 0.009 * math.sin(t)
        v = 0.0055 * math.sin(2.0 * t)
        p0 = pad_center[0] + torch.tensor([-(ferrule_x + 0.0005), u, v])
        p1 = pulled[1]
        set_connector(torch.stack((p0, p1), dim=0), act_quat)
        env.board._test_polishing_force = torch.tensor([1.5, 0.0])
        task.update()

    assert bool(task.polishing.cycles_ok[0]) and bool(task.polishing.pressure_ok[0])
    assert not bool(task.polishing.cycles_ok[1])

    # Step 3: Both envs re-insert cleanly into original source port.
    env.arm.gripper_pinch_force[:] = 0.0
    env.board._test_polishing_force = torch.tensor([0.0, 0.0])
    set_connector(act_pos, act_quat)
    for _ in range(4):
        task.update()

    rubric = task.evaluate()
    assert bool(rubric.passed[0]) and int(rubric.total_score[0]) == expected_max
    assert bool(rubric.phase_flags["grasp"][0]) and bool(rubric.phase_flags["clean"][0]) and bool(rubric.phase_flags["insert"][0])
    assert not bool(rubric.passed[1]), "Skipping figure-eight polishing must NOT pass maintenance task"


def test_refresh_after_reset_prevents_step0_latch_rip_damage():
    env, task, set_connector, (_, _, act_pos, act_quat) = _build_task_harness(
        "IDB-Franka-Articulation-EthernetPortSwap-v0", num_envs=2
    )
    # Simulate a transient wrench spike on reset before the first physics step.
    env.arm.wrench[:, 0] = 25.0
    all_ids = torch.tensor([0, 1])
    task.reset(all_ids, nominal_neighbors=None)
    set_connector(act_pos, act_quat)
    task.refresh_after_reset(all_ids)
    assert not bool(task.latch.damaged.any())
    assert not bool(task.removed_from_source.any())
    assert task.target_seating is not None
    obs = task.observations()
    assert obs.shape == (2, env.cfg.observation_space) and torch.isfinite(obs).all()


def test_board_reset_syncs_kinematic_physx_bodies_when_randomized():
    env, _, _, _ = _build_task_harness("IDB-Franka-Articulation-EthernetPortSwap-v0", num_envs=2, variation="randomized")
    board = env.board

    class _Recorder:
        def __init__(self):
            self.last_state = None
            self.last_ids = None
            self.reset_calls = 0

        def write_root_state_to_sim(self, state, env_ids=None):
            self.last_state = state.clone()
            self.last_ids = env_ids.clone() if env_ids is not None else None

        def reset(self, env_ids=None):
            self.reset_calls += 1

    board.frame_obj = _Recorder()
    board.polishing_pad_obj = _Recorder()
    board.port_objs = {pid: _Recorder() for pid in board.port_ids}
    board.clip_objs = {c.name: _Recorder() for c in board.spec.cable_clips}

    ids = torch.tensor([0, 1])
    board.reset(ids)

    assert board.frame_obj.reset_calls == 1
    assert torch.allclose(board.frame_obj.last_state[:, :3], board.board_pos_w, atol=1e-6)
    pad_c, _, _ = board.polishing_pad_world_pose()
    assert torch.allclose(board.polishing_pad_obj.last_state[:, :3], pad_c, atol=1e-6)
    for pid in board.port_ids:
        p_pos, p_quat = board.port_world_pose(pid)
        assert torch.allclose(board.port_objs[pid].last_state[:, :3], p_pos, atol=1e-6)
        assert torch.allclose(board.port_objs[pid].last_state[:, 3:7], p_quat, atol=1e-6)
