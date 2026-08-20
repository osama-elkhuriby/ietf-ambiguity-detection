"""Command-line interface for parsing one Internet-Draft or RFC document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_document


def _write_json(path: Path, payload: dict[str, object], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one Internet-Draft TXT/XML or RFC TXT source into a canonical "
            "section/block tree."
        )
    )
    parser.add_argument("source", type=Path, help="Path to a source .txt or .xml file")
    parser.add_argument(
        "--document-type",
        choices=("auto", "internet-draft", "rfc"),
        default="auto",
        help="Text profile override (default: infer from filename and document header).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write JSON to this path. If omitted, JSON is written to stdout.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of the default indented representation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        document = parse_document(args.source, document_type=args.document_type)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = document.to_dict()
    if args.output:
        _write_json(args.output, payload, pretty=not args.compact)
        destination = str(args.output)
    else:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=None if args.compact else 2,
                sort_keys=True,
            )
        )
        destination = "stdout"

    summary = document.summary()
    print(
        f"Parsed {document.document_id}: "
        f"{document.document_type}, "
        f"{summary['sections']} sections, {summary['blocks']} blocks, "
        f"{summary['references']} references, {summary['warnings']} warnings -> {destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
