"""Repository-root paths for Stage 1 corpus scripts.

Paths are resolved from this file's location, so commands work regardless of the
current working directory (e.g. ``python corpus/fetch_datatracker.py`` from repo root).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
