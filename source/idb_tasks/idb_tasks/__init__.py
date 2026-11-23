"""Industrial Dexterity Benchmark (IDB) Isaac Lab simulation environments."""

from __future__ import annotations

from .direct.idb.single_arm.common.registry import SPECS, register_all

__version__ = "0.1.0"

register_all()

__all__ = ["SPECS", "__version__", "register_all"]
