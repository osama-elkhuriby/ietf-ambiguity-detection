"""CLI for the versioned analysis-only repair; no training commands exist."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .analysis import analyze, validate_analysis, verify_parent_artifacts


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Verify frozen parent and all reused artifacts")
    analyze_parser = sub.add_parser("analyze", help="Run repaired CPU-only analysis")
    analyze_parser.add_argument("--bootstrap-replicates", type=int)
    sub.add_parser("validate-analysis", help="Validate repaired outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "preflight":
        _print(verify_parent_artifacts())
    elif args.command == "analyze":
        _print(analyze(bootstrap_replicates=args.bootstrap_replicates))
    elif args.command == "validate-analysis":
        _print(validate_analysis())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
