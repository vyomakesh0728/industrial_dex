"""Compatibility boundary for running pure config/USD/PyTorch code outside Isaac Sim.

When Isaac Lab is installed (production Linux/NVIDIA environment), every symbol resolves
directly to the real ``isaaclab`` / ``isaacsim`` implementation. Outside Isaac Sim
(offline verification on macOS/CPU), lightweight config stubs allow config composition,
Gym registry inspection, real ``pxr`` OpenUSD building, and real PyTorch task/scoring tests
without ever faking live physics execution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
import importlib.util
import sys
from typing import Any

HAS_ISAACLAB_PKG: bool = importlib.util.find_spec("isaaclab") is not None
# Kit is only active once SimulationApp has initialized Omniverse modules in sys.modules
# (find_spec("omni") can return True before SimulationApp boots on pip-installed Isaac Sim).
_KIT_ACTIVE: bool = any(
    mod in sys.modules for mod in ("omni.kit.app", "omni.isaac.core", "isaacsim.core.api", "carb")
)
HAS_ISAACLAB: bool = HAS_ISAACLAB_PKG and _KIT_ACTIVE
HAS_ISAACSIM: bool = importlib.util.find_spec("isaacsim") is not None and _KIT_ACTIVE
HAS_PHYSX_SCHEMA: bool = importlib.util.find_spec("pxr.PhysxSchema") is not None


def _to_dict_recursive(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict_recursive(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_dict_recursive(v) for v in obj)
    return obj


if HAS_ISAACLAB:
    from isaaclab.utils import configclass  # type: ignore[import-not-found]
else:
    def configclass(cls=None, **kwargs):  # type: ignore[misc]
        """Minimal configclass-compatible dataclass decorator for offline config testing."""
        def wrap(target):
            # Mirror Isaac Lab's configclass: wrap mutable defaults AND MISSING via default_factory
            # so MISSING fields can sit after defaulted fields and subclasses inherit cleanly.
            from dataclasses import MISSING, field as dc_field

            annotations = getattr(target, "__annotations__", {})
            for name in list(annotations):
                if hasattr(target, name):
                    val = getattr(target, name)
                    if val is MISSING:
                        setattr(target, name, dc_field(default_factory=lambda: MISSING))
                    elif isinstance(val, (list, dict)) or (is_dataclass(val) and not isinstance(val, type)):
                        setattr(target, name, dc_field(default_factory=lambda v=val: copy.deepcopy(v)))
            dc = dataclass(target, **kwargs)

            def to_dict(self) -> dict[str, Any]:
                return _to_dict_recursive(self)

            def copy_fn(self):
                return copy.deepcopy(self)

            def replace_fn(self, **changes):
                cloned = copy.deepcopy(self)
                for k, v in changes.items():
                    setattr(cloned, k, v)
                return cloned

            if not hasattr(dc, "to_dict"):
                dc.to_dict = to_dict
            if not hasattr(dc, "copy"):
                dc.copy = copy_fn
            if not hasattr(dc, "replace"):
                dc.replace = replace_fn
            return dc

        return wrap(cls) if cls is not None else wrap


def require_isaaclab(feature: str) -> None:
    """Fail explicitly when a caller tries to run live Isaac Sim simulation without Isaac Lab."""
    if not HAS_ISAACLAB:
        raise RuntimeError(
            f"{feature} requires NVIDIA Isaac Sim (4.5/5.1) and Isaac Lab (v2.2.1). "
            "Run this command on an Isaac Lab workstation or run offline checks via scripts/idb/check_offline.py."
        )
