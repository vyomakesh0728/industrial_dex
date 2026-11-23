"""Structural scope gate: enforce simulation environments and documentation only.

Prohibits RL/IL training pipelines, policies, agents, and dataset collection scripts
inside source/ and scripts/.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATH_TOKENS = (
    "rsl_rl",
    "ppo",
    "diffusion",
    "lerobot",
    "collect_demos",
    "eval_checkpoint",
    "train.py",
    "play.py",
    "agents",
    "scripted",
    "policy",
    "policies",
    "dataset",
)

FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"^\s*(?:from|import)\s+rsl_rl\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+isaaclab_rl\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+isaaclab_mimic\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+lerobot\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+stable_baselines3\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+skrl\b", re.M),
    re.compile(r"^\s*(?:from|import)\s+rl_games\b", re.M),
)


def _iter_code_files():
    for root_name in ("source", "scripts"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path


def test_no_out_of_scope_files():
    violations = []
    for path in _iter_code_files():
        rel = path.relative_to(REPO_ROOT).as_posix().lower()
        parts = set(path.relative_to(REPO_ROOT).parts)
        for token in FORBIDDEN_PATH_TOKENS:
            if token in parts or path.name.lower() == token or token in rel:
                violations.append(f"{rel} (matched forbidden token {token!r})")
                break
    assert not violations, "Out-of-scope RL/IL/policy/dataset files found:\n" + "\n".join(violations)


def test_no_out_of_scope_imports():
    violations = []
    for path in _iter_code_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_IMPORT_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
    assert not violations, "Out-of-scope RL/IL imports found:\n" + "\n".join(violations)
