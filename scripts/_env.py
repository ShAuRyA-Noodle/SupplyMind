"""Shared .env loader for SupplyMind scripts (stdlib only, no external deps).

Replaces the divergent hand-rolled .env parsers scattered across the scripts/
directory with one implementation. Reads KEY=VALUE lines from the repository
.env into os.environ.

Security: this module NEVER logs, prints, or returns secret values. load_env()
returns only the list of KEY names it set; the smoke test below prints names.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: str | os.PathLike | None = None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE pairs from a .env file into ``os.environ``.

    Args:
        path: .env file to read; defaults to ``<repo root>/.env``.
        override: if False (default), existing environment variables are left
            untouched; if True, values from the file take precedence.

    Returns:
        Sorted list of KEY names that were present in the file (names only —
        values are never returned or logged). A missing file returns ``[]``.
    """
    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    names: set[str] = set()
    if not env_path.exists():
        return []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
        names.add(key)
    return sorted(names)


if __name__ == "__main__":
    loaded = load_env()
    print(f"[_env] loaded {len(loaded)} key name(s) from .env: {', '.join(loaded)}")
