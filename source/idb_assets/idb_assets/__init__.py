"""CAD-derived OpenUSD assets and geometry manifests for IDB datacenter tasks."""

from __future__ import annotations

import os

IDB_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "idb"))

__all__ = ["IDB_ASSET_DIR"]
