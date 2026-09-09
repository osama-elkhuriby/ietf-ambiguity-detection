"""Alignment normalization never removes protocol tokens or folds case."""

import re
import unicodedata

FORMAL_TYPES = {"artwork", "sourcecode", "table", "formal_syntax"}


def normalize(text: str, block_type: str) -> tuple[str, list[str]]:
    value = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    steps = ["NFC", "line_endings"]
    if block_type in FORMAL_TYPES:
        return "\n".join(x.rstrip() for x in value.splitlines()).strip("\n"), steps + [
            "trailing_space"
        ]
    # Only unambiguous bullet glyphs; numeric/alphabetic markers may be identifiers.
    if block_type == "list_item":
        value, count = re.subn(r"^\s*[-*•]\s+", "", value, count=1)
        if count:
            steps.append("structural_bullet")
    return re.sub(r"\s+", " ", value).strip(), steps + ["prose_whitespace"]
