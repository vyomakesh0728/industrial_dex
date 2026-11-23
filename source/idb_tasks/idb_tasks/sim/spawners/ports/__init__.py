"""Connector profiles and procedural port/socket USD spawners."""

from .connector_profiles import (
    ConnectorProfile,
    FerruleSpec,
    connector_families,
    get_connector_profile,
    normalize_family,
    rectangle,
)
from .connector_spawner import (
    IDBPortCfg,
    IDBSocketCfg,
    build_connector_port_usd,
    build_connector_socket_usd,
    default_visual_usd,
    offset_polygon,
    spawn_idb_port,
    spawn_idb_socket,
)

__all__ = [
    "ConnectorProfile",
    "FerruleSpec",
    "IDBPortCfg",
    "IDBSocketCfg",
    "build_connector_port_usd",
    "build_connector_socket_usd",
    "connector_families",
    "default_visual_usd",
    "get_connector_profile",
    "normalize_family",
    "offset_polygon",
    "rectangle",
    "spawn_idb_port",
    "spawn_idb_socket",
]
