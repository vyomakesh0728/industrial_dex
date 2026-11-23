"""Wire backend abstractions, connector geometry, and 3D cable state."""

from .connectors import ConnectorCfg, ConnectorGeometry
from .wire_state import PluggedCableArc, WireState, ferrule_positions, tip_from_root

__all__ = [
    "ConnectorCfg",
    "ConnectorGeometry",
    "PluggedCableArc",
    "WireState",
    "ferrule_positions",
    "tip_from_root",
]
