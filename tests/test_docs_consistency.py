"""Verify documentation completeness, relative links, and alignment with the live Gym registry."""

from __future__ import annotations

from pathlib import Path
import re

import idb_tasks  # noqa: F401
from idb_tasks.direct.idb.single_arm.common.registry import SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/ENVIRONMENTS.md",
    "docs/tasks/task1_ethernet_port_swap.md",
    "docs/tasks/task2_sc_port_swap.md",
    "docs/tasks/task3_lc_duplex_port_swap.md",
    "docs/tasks/task4_sc_fiber_maintenance.md",
    "docs/tasks/task5_lc_duplex_fiber_maintenance.md",
)


def test_all_required_docs_exist_and_links_resolve():
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for rel in REQUIRED_DOCS:
        doc = REPO_ROOT / rel
        assert doc.is_file(), f"Missing documentation file: {rel}"
        text = doc.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            target_path = (doc.parent / target.split("#")[0]).resolve()
            assert target_path.exists(), f"Broken relative link in {rel}: {target}"


def test_environments_doc_lists_every_registered_gym_id():
    env_doc = (REPO_ROOT / "docs/ENVIRONMENTS.md").read_text(encoding="utf-8")
    for task_id in SPECS:
        assert f"`{task_id}`" in env_doc, f"Gym ID {task_id} missing from docs/ENVIRONMENTS.md"
