"""Small, dependency-light helpers shared by the final Transformer pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "process_id": os.getpid(),
    }


def hash_directory(path: Path) -> tuple[str, dict[str, str]]:
    """Hash all files under a local model directory without loading any model."""
    if not path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    files = {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    if not files:
        raise RuntimeError(f"Local model directory is empty: {path}")
    return sha256_text(stable_json(files)), files


def source_file_hashes() -> dict[str, str]:
    source_root = ROOT / "models" / "transformer_final"
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(source_root.glob("*.py"))
        if path.is_file()
    }


@contextmanager
def immutable_directory(destination: Path) -> Iterator[Path]:
    """Build off to the side, publish atomically, and preserve failures for diagnosis."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    building = destination.parent / f".{destination.name}.building-{nonce}"
    failed = destination.parent / f".{destination.name}.failed-{nonce}"
    building.mkdir()
    try:
        yield building
        if destination.exists():
            raise FileExistsError(
                f"Immutable output appeared during build: {destination}"
            )
        building.replace(destination)
    except Exception:
        if building.exists():
            building.replace(failed)
        raise
