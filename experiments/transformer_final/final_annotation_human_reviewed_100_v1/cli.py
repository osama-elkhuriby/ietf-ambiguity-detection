"""Command-line interface for the frozen final-annotation experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from . import analysis
from .core import (
    EXPERIMENT_ROOT,
    FAILURES_DIR,
    LOGS_DIR,
    ROOT,
    _write_json,
    iter_specs,
    preflight,
    prepare,
    status,
    utc_now,
    validate_cell,
    validate_cells,
    verify_frozen,
)

LOCK_PATH = EXPERIMENT_ROOT / "TRAINING_ACTIVE.lock"


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _select_specs(args: argparse.Namespace, kind: str) -> list[tuple[Path, dict[str, Any]]]:
    selected = []
    for path, spec in iter_specs(kind):
        if args.all:
            selected.append((path, spec))
            continue
        if args.method and spec["method"] != args.method:
            continue
        if kind == "outer" and args.fold and int(spec["outer_fold"]) != args.fold:
            continue
        if kind == "inner":
            if args.outer_fold and int(spec["outer_fold"]) != args.outer_fold:
                continue
            if args.inner_fold and int(spec["inner_fold"]) != args.inner_fold:
                continue
        selected.append((path, spec))
    if not selected:
        raise RuntimeError("No cell matches the explicit selection")
    if not args.all:
        if kind == "outer" and (args.method is None or args.fold is None):
            raise RuntimeError("Select --all or both --method and --fold")
        if kind == "inner" and (
            args.method is None or args.outer_fold is None or args.inner_fold is None
        ):
            raise RuntimeError(
                "Select --all or --method, --outer-fold, and --inner-fold"
            )
    return selected


def _acquire_lock() -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        detail = LOCK_PATH.read_text(encoding="utf-8") if LOCK_PATH.is_file() else ""
        raise RuntimeError(
            f"Another training command may be active. Inspect {LOCK_PATH}: {detail}"
        ) from exc
    os.write(descriptor, f"pid={os.getpid()} started={utc_now()}\n".encode())
    return descriptor


def _run_selected(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    verify_frozen()
    selected = _select_specs(args, kind)
    descriptor = _acquire_lock()
    completed = []
    skipped = []
    try:
        for spec_path, spec in selected:
            if Path(spec["completion_path"]).is_file():
                validate_cell(spec)
                skipped.append(spec["cell_id"])
                print(f"SKIP validated complete cell: {spec['cell_id']}", flush=True)
                continue
            failure_path = FAILURES_DIR / f"{spec['cell_id']}.json"
            if failure_path.exists():
                raise RuntimeError(
                    f"Prior failure evidence exists for {spec['cell_id']}; no automatic retry. "
                    f"Inspect {failure_path} and version the experiment if code/protocol changes are needed."
                )
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            stdout_path = LOGS_DIR / f"{spec['cell_id']}.stdout.log"
            stderr_path = LOGS_DIR / f"{spec['cell_id']}.stderr.log"
            if stdout_path.exists() or stderr_path.exists():
                raise FileExistsError(f"Attempt logs already exist for {spec['cell_id']}")
            print(f"START {spec['cell_id']}", flush=True)
            command = [
                sys.executable,
                "-m",
                "experiments.transformer_final.final_annotation_human_reviewed_100_v1.runner",
                "--spec",
                str(spec_path),
            ]
            try:
                with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
                    "x", encoding="utf-8"
                ) as stderr:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        env=os.environ.copy(),
                    )
                if process.returncode != 0:
                    raise RuntimeError(f"Cell exited with code {process.returncode}")
                validated = validate_cell(spec)
                completed.append(validated)
                print(f"COMPLETE {spec['cell_id']}", flush=True)
            except Exception as exc:
                _write_json(
                    failure_path,
                    {
                        "schema_version": 1,
                        "failed_at": utc_now(),
                        "cell_id": spec["cell_id"],
                        "command": command,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                        "automatic_retry": False,
                    },
                    exclusive=True,
                )
                raise RuntimeError(
                    f"Stopped immediately after {spec['cell_id']} failed. "
                    f"See {failure_path}. No retry was attempted."
                ) from exc
    finally:
        os.close(descriptor)
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    return {"kind": kind, "completed_now": completed, "skipped_valid_complete": skipped}


def _add_training_selection(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.add_argument("--all", action="store_true", help="Run every cell sequentially")
    parser.add_argument("--method", choices=("focal", "partial4"))
    if kind == "outer":
        parser.add_argument("--fold", type=int, choices=range(1, 6))
    else:
        parser.add_argument("--outer-fold", type=int, choices=range(1, 6))
        parser.add_argument("--inner-fold", type=int, choices=range(1, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Read-only dataset/protocol/model identity checks")
    sub.add_parser("prepare", help="Build and freeze preparation artifacts; never trains")
    sub.add_parser("status", help="Read-only cell status")
    outer = sub.add_parser("train-outer", help="Explicitly run outer CUDA cells")
    _add_training_selection(outer, "outer")
    validate_outer = sub.add_parser("validate-outer", help="Validate outer artifacts")
    validate_outer.add_argument("--require-all", action="store_true")
    inner = sub.add_parser("train-inner", help="Explicitly run inner CUDA cells")
    _add_training_selection(inner, "inner")
    validate_inner = sub.add_parser("validate-inner", help="Validate inner artifacts")
    validate_inner.add_argument("--require-all", action="store_true")
    analyze_parser = sub.add_parser("analyze", help="CPU-only final nested analysis")
    analyze_parser.add_argument("--bootstrap-replicates", type=int)
    sub.add_parser("validate-analysis", help="Validate final reports and Top-20 integrity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        _print(preflight())
    elif args.command == "prepare":
        _print(prepare())
    elif args.command == "status":
        _print(status())
    elif args.command == "train-outer":
        _print(_run_selected(args, "outer"))
    elif args.command == "validate-outer":
        _print(validate_cells("outer", require_all=args.require_all))
    elif args.command == "train-inner":
        _print(_run_selected(args, "inner"))
    elif args.command == "validate-inner":
        _print(validate_cells("inner", require_all=args.require_all))
    elif args.command == "analyze":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _print(analysis.analyze(bootstrap_replicates=args.bootstrap_replicates))
    elif args.command == "validate-analysis":
        _print(analysis.validate_analysis())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

