"""CLI for structural parsing followed by clause classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alignment.segmentation import parse_document

from .segmenter import segment_document


def _write_json(path: Path, payload: dict[str, object], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and classify clauses from an Internet-Draft or RFC source."
    )
    parser.add_argument("source", type=Path, help="Path to a source .txt or .xml file")
    parser.add_argument(
        "--document-type",
        choices=("auto", "internet-draft", "rfc"),
        default="auto",
        help="Text profile override (default: infer from filename/header).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="JSON destination; default is stdout"
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        document = parse_document(args.source, document_type=args.document_type)
        result = segment_document(document)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.output:
        _write_json(args.output, payload, pretty=not args.compact)
        destination = str(args.output)
    else:
        print(
            json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2)
        )
        destination = "stdout"

    summary = result.summary()
    print(
        f"Classified {result.document_id}: {summary['clauses']} clauses, "
        f"{summary['normative']} normative, {summary['contextual']} contextual, "
        f"{summary['excluded']} excluded, {summary['manual_review']} manual review "
        f"-> {destination}",
        file=sys.stderr,
    )
    return 0
