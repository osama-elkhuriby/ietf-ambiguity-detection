"""Shared text normalization and identifier helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

WHITESPACE_RE = re.compile(r"\s+")
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_prose(text: str) -> str:
    """Collapse rendering whitespace without changing punctuation or wording."""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_multiline(text: str) -> str:
    """Preserve line structure for artwork and source code."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    nonempty = [line for line in lines if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in nonempty), default=0)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines)


def join_wrapped_lines(lines: list[str]) -> str:
    """Join hard-wrapped RFC prose while repairing line-break hyphenation."""
    result = ""
    for line in lines:
        part = line.strip()
        if not part:
            continue
        if not result:
            result = part
        elif result.endswith("-") and part[:1].islower():
            result += part
        else:
            result += " " + part
    return normalize_prose(result)


def safe_identifier(value: str) -> str:
    normalized = SAFE_ID_RE.sub("-", value.strip()).strip("-")
    return normalized or "unknown"


def block_identifier(document_id: str, section_id: str, index: int) -> str:
    return f"{safe_identifier(document_id)}:{safe_identifier(section_id)}:b{index:04d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

