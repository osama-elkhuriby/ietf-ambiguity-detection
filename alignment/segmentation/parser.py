"""Format- and profile-detecting entry point for document segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .models import Document
from .text_parser import parse_text_document
from .xml_parser import parse_xml_document


def parse_document(
    path: str | Path,
    document_type: Literal["auto", "internet-draft", "rfc"] = "auto",
) -> Document:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix == ".xml":
        if document_type == "rfc":
            raise ValueError("The RFC profile currently supports rendered .txt input only")
        return parse_xml_document(source)
    if suffix == ".txt":
        return parse_text_document(source, document_type)
    raise ValueError(f"Unsupported source format: {source.suffix or '(none)'}")
