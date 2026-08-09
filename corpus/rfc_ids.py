"""RFC number parsing shared by corpus fetch scripts (no network dependencies)."""
from __future__ import annotations

import re


def parse_rfc_number(value: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return None
    return int(digits)


def normalize_rfc_number(value: str) -> int:
    num = parse_rfc_number(value)
    if num is None:
        raise SystemExit(f"Not a valid RFC number: '{value}'")
    return num


def normalize_rfc_id(value: str) -> str:
    """Accepts '9460', 'rfc9460', or 'RFC9460' -> returns 'rfc9460'."""
    num = normalize_rfc_number(value)
    return f"rfc{num}"
