"""Layered environment configuration composition mirroring WireCraft's compose.py.

Priority from lowest to highest:
1. SingleArmIDBEnvCfg defaults
2. robot (<robot>/robot_cfg.py)
3. task (presets/task/<task>.yaml)
4. wire (presets/wire/<wire>.yaml)
5. connector (presets/connector/<family>.yaml, optional)
6. variation (presets/variation/<variation>.yaml, optional)
7. mode (presets/mode/<mode>.yaml)
8. --experiment YAML files in the order given (must have top-level 'env:' mapping only)
9. explicit env.* CLI overrides

Merge rules (identical to WireCraft):
- Mappings merge recursively; lists and tuples replace whole (tuples keep length).
- Unknown keys raise KeyError; wrong types raise ValueError (int accepted for float,
  never float for int, never bool for number).
- Presets may only set top-level keys permitted by ENV_LAYER_KEYS.
- Derived fields produced by resolve_env_cfg cannot be overridden.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import importlib
import os
import random
import sys
from typing import Any

import yaml

from .env_cfg import MODES, SingleArmIDBEnvCfg, resolve_env_cfg

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")

_SCENE_KEYS = ("episode_length_s", "decimation", "control_mode", "fix_ee_orient", "action_scale")
ENV_LAYER_KEYS: dict[str, tuple[str, ...]] = {
    "robot": ("robot",),
    "task": (
        "task", "wire", "connector", "board", "sensors", "enable_terminations",
        "terminate_on_time_out", "terminate_on_success", "terminate_on_damage",
        "ee_safe_zone", "fault_joint_speed", "fault_port_speed", *_SCENE_KEYS,
    ),
    "wire": ("wire", "task", "connector", "episode_length_s", "fix_ee_orient", "action_scale"),
    "connector": ("connector",),
    "variation": ("variation", "board", "sensors", "wire", "task"),
    "mode": (
        "mode", "wire", "task", "scene", "sensors", "variation",
        "enable_terminations", "terminate_on_time_out", "terminate_on_success",
        "terminate_on_damage", *_SCENE_KEYS,
    ),
}

DERIVED_ENV_FIELDS = (
    "resolved",
    "wire_prim_path",
    "board_prim_path",
    "wire_spawn",
    "board_spawn",
    "resolved_neighbor_cables",
    "table_cfg",
    "camera_cfgs",
    "action_space",
    "observation_space",
    "state_space",
    "composition",
    "task.task_class",
)


@dataclass(frozen=True)
class TaskSpec:
    """One registered IDB simulation combination mapped 1:1 to a Gym ID."""

    robot: str
    robot_cfg: str
    task: str
    wire: str
    connector: str
    far_connector: str = ""
    variation: str = "sparse"
    variant_overrides: Mapping[str, Any] = field(default_factory=dict)
    leaf_dir: str = ""


def _coerce(current: Any, value: Any, where: str) -> Any:
    if value is None or current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected bool, got {value!r}.")
    elif isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected float, got {value!r}.")
        return float(value)
    elif isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected int, got {value!r}.")
    elif isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected str, got {value!r}.")
    elif isinstance(current, (list, tuple)):
        if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
            raise ValueError(f"{where}: expected a sequence, got {value!r}.")
        if isinstance(current, tuple):
            if len(value) != len(current):
                raise ValueError(f"{where}: expected {len(current)} items, got {len(value)}.")
            return tuple(_coerce(c, v, f"{where}[{i}]") for i, (c, v) in enumerate(zip(current, value)))
        return list(value)
    elif isinstance(current, dict) or hasattr(current, "__dataclass_fields__"):
        raise ValueError(f"{where}: expected a mapping, got {value!r}.")
    return value


def apply_overrides(target: Any, data: Mapping[str, Any], path: str = "", derived: Sequence[str] = ()) -> None:
    """Recursively merge ``data`` into a configclass or dict under WireCraft merge rules."""
    for key, value in data.items():
        where = f"{path}.{key}" if path else key
        if where in derived:
            raise ValueError(f"{where} is derived during resolve and cannot be overridden.")
        if isinstance(target, dict):
            current = target.get(key)
        elif hasattr(target, key):
            current = getattr(target, key)
        else:
            raise KeyError(f"Unknown config field {where!r}.")
        if isinstance(value, Mapping) and current is not None:
            if isinstance(current, (bool, int, float, str, list, tuple)):
                raise ValueError(f"{where}: expected scalar or list, got mapping.")
            apply_overrides(current, value, where, derived)
            continue
        coerced = _coerce(current, value, where)
        if isinstance(target, dict):
            target[key] = coerced
        else:
            setattr(target, key, coerced)


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a YAML mapping.")
    return data


def apply_env_layer(cfg: SingleArmIDBEnvCfg, layer: str, data: Mapping[str, Any]) -> None:
    allowed = ENV_LAYER_KEYS[layer]
    stray = sorted(set(data) - set(allowed))
    if stray:
        raise ValueError(f"The {layer} preset cannot set {stray}; allowed keys are {list(allowed)}.")
    apply_overrides(cfg, data, derived=DERIVED_ENV_FIELDS)


def _layer_files(spec: TaskSpec, mode: str, variation: str, layer: str) -> list[str]:
    name_map = {
        "task": spec.task,
        "wire": spec.wire,
        "connector": spec.connector,
        "variation": variation,
        "mode": mode,
    }
    name = name_map[layer]
    if not name:
        return []
    paths = [os.path.join(PRESET_DIR, layer, f"{name}.yaml")]
    if spec.leaf_dir:
        paths.append(os.path.join(spec.leaf_dir, "task_cfg", f"{layer}_{name}.yaml"))
    return [p for p in paths if os.path.exists(p)]


def _experiment_sections(paths: Sequence[str]) -> list[tuple[str, dict[str, Any]]]:
    sections = []
    for path in paths:
        data = read_yaml(path)
        stray = sorted(set(data) - {"env"})
        if stray:
            raise ValueError(
                f"{path}: simulation-only experiment YAML must only contain top-level 'env:', found {stray}."
            )
        sections.append((path, data))
    return sections


def _load_callable(path: str) -> Callable:
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _task_cfg(name: str):
    from .tasks import (
        EthernetPortSwapCfg,
        LCDuplexPortSwapCfg,
        LCFiberMaintenanceCfg,
        SCFiberMaintenanceCfg,
        SCPortSwapCfg,
    )

    mapping = {
        "ethernet_port_swap": EthernetPortSwapCfg,
        "sc_port_swap": SCPortSwapCfg,
        "lc_duplex_port_swap": LCDuplexPortSwapCfg,
        "sc_fiber_maintenance": SCFiberMaintenanceCfg,
        "lc_fiber_maintenance": LCFiberMaintenanceCfg,
    }
    try:
        return mapping[name]()
    except KeyError:
        raise ValueError(f"Unknown IDB task {name!r}; choose from {sorted(mapping)}.") from None


def compose_env_cfg(
    spec: TaskSpec,
    mode: str = "demo",
    variation: str | None = None,
    experiments: Sequence[str] = (),
    script_defaults: Mapping[str, Any] | None = None,
) -> SingleArmIDBEnvCfg:
    """Compose unresolved environment config for one TaskSpec. Each call returns independent objects."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    selected_variation = spec.variation if variation is None else variation
    cfg = SingleArmIDBEnvCfg()
    record = [f"defaults:{type(cfg).__name__}"]
    cfg.robot = _load_callable(spec.robot_cfg)()
    record.append(f"robot:{spec.robot_cfg}")
    cfg.task = _task_cfg(spec.task)
    cfg.wire.kind = spec.wire
    cfg.connector.family = spec.connector
    cfg.connector.far_family = spec.far_connector or spec.connector
    cfg.variant = selected_variation

    # Apply cable BOM physical defaults before layer YAML overrides.
    if hasattr(cfg.task, "cable_preset"):
        cfg.wire.apply_preset_defaults(cfg.task.cable_preset)
        record.append(f"wire_preset:{cfg.task.cable_preset}")

    for layer in ("task", "wire", "connector", "variation", "mode"):
        for path in _layer_files(spec, mode, selected_variation, layer):
            apply_env_layer(cfg, layer, read_yaml(path))
            record.append(f"{layer}:{os.path.relpath(path, os.path.dirname(PRESET_DIR))}")
        if layer == "task" and spec.variant_overrides:
            apply_env_layer(cfg, "task", copy.deepcopy(dict(spec.variant_overrides)))
            record.append(f"task:variant={selected_variation}")

    cfg.mode = mode
    if script_defaults:
        apply_overrides(cfg, copy.deepcopy(dict(script_defaults)), derived=DERIVED_ENV_FIELDS)
        record.append(f"script_defaults:{dict(script_defaults)}")
    for path, data in _experiment_sections(experiments):
        if "env" in data:
            apply_overrides(cfg, data["env"], derived=DERIVED_ENV_FIELDS)
            record.append(f"experiment:{path}")

    cfg.composition = {
        "spec": {
            "robot": spec.robot,
            "task": spec.task,
            "wire": spec.wire,
            "connector": spec.connector,
            "far_connector": spec.far_connector or spec.connector,
            "variation": selected_variation,
        },
        "mode": mode,
        "env_layers": record,
        "cli": [],
    }
    return cfg


_context: dict[str, Any] = {"mode": "demo", "variation": None, "experiments": ()}


def configure(mode: str = "demo", variation: str | None = None, experiments: Sequence[str] = ()) -> None:
    """Set mode, variation preset, and experiment YAML files for Gym entry points."""
    _context.update(mode=mode, variation=variation, experiments=tuple(experiments))


def env_entry_point(spec: TaskSpec) -> Callable[[], SingleArmIDBEnvCfg]:
    def build_env_cfg() -> SingleArmIDBEnvCfg:
        return compose_env_cfg(spec, _context["mode"], _context["variation"], _context["experiments"])

    return build_env_cfg


CLI_FLAGS: dict[str, tuple[str, ...]] = {
    "num_envs": ("env.scene.num_envs",),
    "seed": ("env.seed",),
    "action_scale": ("env.action_scale",),
    "device": ("env.sim.device",),
}


def legacy_overrides(args_cli: Any, argv: Sequence[str]) -> dict[str, Any]:
    """Map explicitly typed CLI flags into dotted env.* paths."""
    typed = {a.split("=")[0].lstrip("-") for a in argv if a.startswith("--")}
    result: dict[str, Any] = {}
    for flag, paths in CLI_FLAGS.items():
        value = getattr(args_cli, flag, None)
        if flag not in typed or value is None:
            continue
        if flag == "seed" and value == -1:
            value = random.randint(0, 10000)
        for p in paths:
            result[p] = value
    return result


def parse_cli_overrides(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse ``env.a.b=value`` CLI tokens as YAML scalars/lists."""
    result: dict[str, Any] = {}
    for token in tokens:
        path, sep, text = token.partition("=")
        if not sep or not path.startswith("env."):
            raise ValueError(f"Unrecognized override {token!r}; expected env.<field>=<value>.")
        result[path] = yaml.safe_load(text)
    return result


def _nested(path: str, value: Any) -> dict[str, Any]:
    out = value
    for key in reversed(path.split(".")):
        out = {key: out}
    return out


def apply_cli(env_cfg: SingleArmIDBEnvCfg, explicit: Mapping[str, Any], legacy: Mapping[str, Any]) -> None:
    """Apply explicit CLI overrides and reject conflicts with typed flags."""
    for path, value in legacy.items():
        if path in explicit and explicit[path] != value:
            raise ValueError(f"Conflicting values for {path}: {explicit[path]!r} vs {value!r}.")
    for path, value in {**legacy, **explicit}.items():
        side, _, rest = path.partition(".")
        if side != "env":
            raise ValueError(f"Only 'env.*' overrides are supported in simulation-only mode, got {path!r}.")
        apply_overrides(env_cfg, _nested(rest, value), derived=DERIVED_ENV_FIELDS)
        env_cfg.composition["cli"].append(f"{path}={value!r}")


def save_resolved(log_dir: str, env_cfg: SingleArmIDBEnvCfg) -> None:
    """Save resolved environment config and composition provenance to params/*.yaml."""
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, "env.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(env_cfg.to_dict(), f, sort_keys=False)
    selection = {"env": env_cfg.composition, "argv": list(sys.argv)}
    with open(os.path.join(params_dir, "selection.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(selection, f, sort_keys=False)
