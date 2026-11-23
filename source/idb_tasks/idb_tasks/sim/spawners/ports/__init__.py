"""Connector profiles and procedural port/socket USD spawners."""

from .connector_profiles import (
    ConnectorProfile,
    FerruleSpec,
    connector_families,
    get_connector_profile,
    normalize_family,
    rectangle,
)

__all__ = [
    "ConnectorProfile",
    "FerruleSpec",
    "connector_families",
    "get_connector_profile",
    "normalize_family",
    "rectangle",
]
