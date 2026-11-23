"""Gymnasium registration for all 20 IDB Board #1 datacenter simulation environments.

ID pattern mirrors WireCraft:
    IDB-{Franka,UR5}-{Articulation,Deformable}-{TaskName}-v0
"""

from __future__ import annotations

import gymnasium as gym

from .compose import TaskSpec, env_entry_point

ENV_ENTRY = "idb_tasks.direct.idb.single_arm.common.env:SingleArmIDBEnv"

ROBOTS: dict[str, tuple[str, str]] = {
    "franka": ("Franka", "idb_tasks.direct.idb.single_arm.franka.robot_cfg:build_robot_cfg"),
    "ur5": ("UR5", "idb_tasks.direct.idb.single_arm.ur5.robot_cfg:build_robot_cfg"),
}

WIRE_NAMES: dict[str, str] = {
    "articulation": "Articulation",
    "deformable": "Deformable",
}

# (task_key, gym_suffix, active_connector, far_connector)
TASKS: tuple[tuple[str, str, str, str], ...] = (
    ("ethernet_port_swap", "EthernetPortSwap", "rj45", "rj45"),
    ("sc_port_swap", "SCPortSwap", "sc_simplex", "lc_simplex"),
    ("lc_duplex_port_swap", "LCDuplexPortSwap", "lc_duplex", "lc_duplex"),
    ("sc_fiber_maintenance", "SCFiberMaintenance", "sc_simplex", "lc_simplex"),
    ("lc_fiber_maintenance", "LCFiberMaintenance", "lc_duplex", "lc_duplex"),
)

SPECS: dict[str, TaskSpec] = {}
_REGISTERED = False


def register_all() -> None:
    """Register all 20 IDB datacenter environments idempotently with Gymnasium."""
    global _REGISTERED
    if _REGISTERED:
        return
    for robot_key, (robot_label, robot_cfg_fn) in ROBOTS.items():
        for wire_key, wire_label in WIRE_NAMES.items():
            for task_key, task_suffix, active_conn, far_conn in TASKS:
                task_id = f"IDB-{robot_label}-{wire_label}-{task_suffix}-v0"
                spec = TaskSpec(
                    robot=robot_key,
                    robot_cfg=robot_cfg_fn,
                    task=task_key,
                    wire=wire_key,
                    connector=active_conn,
                    far_connector=far_conn,
                )
                SPECS[task_id] = spec
                if task_id not in gym.registry:
                    gym.register(
                        id=task_id,
                        entry_point=ENV_ENTRY,
                        disable_env_checker=True,
                        kwargs={"env_cfg_entry_point": env_entry_point(spec)},
                    )
    _REGISTERED = True
