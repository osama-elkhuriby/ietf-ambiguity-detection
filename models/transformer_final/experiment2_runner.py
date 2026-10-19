"""Sequential, fail-closed runner for the five-cell annotation-ready pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_conda_library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
os.environ["PATH"] = str(_conda_library_bin) + os.pathsep + os.environ.get("PATH", "")

import torch
import pandas as pd

from .common import (
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    source_file_hashes,
    write_json,
)
from .config import load_config
from .dataset import load_effective_dataset, verify_dataset
from .experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PROTOCOL,
    EXPERIMENT_ID,
    load_annotation_ready_protocol,
)
from .experiment2_analysis import create_experiment2_analysis
from .experiment_runner import _assert_no_training_processes
from .run_validation import validate_completed_run


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale inventory temporary file exists: {temporary}")
    write_json(temporary, value)
    temporary.replace(path)


def _inventory_path(config: dict[str, Any]) -> Path:
    return (
        resolve_path(config["outputs"]["audit_dir"])
        / "run_management"
        / "completion_inventory.json"
    )


def _check_preflight(
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    protocol_path: Path,
    model_path: Path,
) -> tuple[str, dict[str, str]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Experiment 2 requires exactly one visible CUDA GPU")
    environment = protocol["environment"]
    observed = {
        "interpreter": str(Path(sys.executable).resolve()),
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
    }
    if any(observed[key] != environment[key] for key in observed):
        raise RuntimeError(
            f"Experiment 2 runtime differs from frozen protocol: {observed}"
        )
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    if not driver or len(set(driver)) != 1 or driver[0].strip() != environment["driver"]:
        raise RuntimeError(f"NVIDIA driver differs from frozen protocol: {driver}")
    dataset_dir = resolve_path(config["outputs"]["dataset_dir"])
    dataset_manifest = verify_dataset(dataset_dir)
    if (
        sha256_file(protocol_path) != protocol.get("_loaded_protocol_sha256")
        or sha256_file(dataset_dir / "manifest.json")
        != protocol["dataset"]["dataset_manifest_sha256"]
        or dataset_manifest["effective_dataset_sha256"]
        != protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(resolve_path(config["data"]["source_csv"]))
        != protocol["dataset"]["source_sha256"]
        or source_file_hashes() != protocol["source_snapshot"]["files"]
    ):
        raise RuntimeError("Dataset, protocol, or source snapshot hash mismatch")
    source_audit = (
        resolve_path(config["outputs"]["audit_dir"]) / "dataset_source_audit.json"
    )
    if sha256_file(source_audit) != protocol["dataset"]["source_audit_sha256"]:
        raise RuntimeError("Dataset source-audit hash differs from frozen protocol")
    for fold, expected_hash in protocol["dataset"]["fold_manifest_sha256"].items():
        if sha256_file(dataset_dir / "folds" / f"fold_{fold}.json") != expected_hash:
            raise RuntimeError(f"Fold {fold} hash differs from frozen protocol")
    model_hash, model_files = hash_directory(model_path.resolve())
    if (
        model_hash != protocol["model"]["source_sha256"]
        or model_files != protocol["model"]["model_file_hashes"]
        or str(model_path.resolve()) != protocol["model"]["local_path"]
    ):
        raise RuntimeError("Local pretrained model/tokenizer hash mismatch")
    free_memory, _ = torch.cuda.mem_get_info(0)
    if free_memory < 8 * 1024**3:
        raise RuntimeError(f"Insufficient GPU memory: {free_memory} bytes free")
    _assert_no_training_processes()
    return model_hash, model_files


def _load_or_create_inventory(
    path: Path, protocol_sha256: str, frozen_source_hashes: dict[str, str]
) -> dict[str, Any]:
    if path.exists():
        inventory = read_json(path)
        if (
            inventory.get("protocol_sha256") != protocol_sha256
            or inventory.get("source_file_hashes") != frozen_source_hashes
        ):
            raise RuntimeError("Existing pilot inventory belongs to another protocol")
        active = inventory.get("active_cell")
        if active is not None:
            pid = int(active.get("pid") or 0)
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError as exc:
                    raise RuntimeError(
                        f"Previous pilot cell was interrupted: {active['cell']}; "
                        "inspect logs and artifacts before any continuation"
                    ) from exc
                raise RuntimeError(
                    f"Pilot cell {active['cell']} is still active under PID {pid}"
                )
            raise RuntimeError(
                f"Pilot inventory has unresolved active cell {active['cell']}"
            )
        return inventory
    inventory = {
        "schema_version": 1,
        "protocol_id": EXPERIMENT_ID,
        "protocol_sha256": protocol_sha256,
        "source_file_hashes": frozen_source_hashes,
        "active_cell": None,
        "cells": {},
        "analysis": None,
    }
    _atomic_json(path, inventory)
    return inventory


def _validate_cell(
    *,
    run: Path,
    fold: int,
    seed: int,
    config: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    effective: Any,
    dataset_manifest: dict[str, Any],
    dataset_dir: Path,
    model_hash: str,
    model_files: dict[str, str],
) -> pd.DataFrame:
    return validate_completed_run(
        run=run,
        condition="A2",
        fold=fold,
        seed=seed,
        config=config,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        effective=effective,
        dataset_manifest=dataset_manifest,
        dataset_dir=dataset_dir,
        model_source_sha256=model_hash,
        model_file_hashes=model_files,
    )


def _launch_cell(
    *,
    fold: int,
    seed: int,
    run: Path,
    config_path: Path,
    protocol_path: Path,
    model_path: Path,
    inventory: dict[str, Any],
    inventory_path: Path,
    log_dir: Path,
) -> float:
    key = f"fold_{fold}/seed_{seed}"
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {run}")
    stdout_path = log_dir / f"fold_{fold}_seed_{seed}.stdout.log"
    stderr_path = log_dir / f"fold_{fold}_seed_{seed}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"Pilot logs already exist for {key}")
    command = [
        sys.executable,
        "-m",
        "models.transformer_final.train",
        "--config",
        str(config_path),
        "--condition",
        "A2",
        "--fold",
        str(fold),
        "--seed",
        str(seed),
        "--model-path",
        str(model_path),
        "--precision",
        "fp32",
        "--protocol-manifest",
        str(protocol_path),
        "--diagnostic-numerics",
        "--output",
        str(run),
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PATH"] = (
        str(Path(sys.executable).resolve().parent / "Library" / "bin")
        + os.pathsep
        + env.get("PATH", "")
    )
    inventory["active_cell"] = {
        "cell": key,
        "status": "starting",
        "pid": None,
        "run_path": str(run),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _atomic_json(inventory_path, inventory)
    started = time.time()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        inventory["active_cell"]["status"] = "running"
        inventory["active_cell"]["pid"] = process.pid
        _atomic_json(inventory_path, inventory)
        print(f"Started {key}; PID={process.pid}; output={run}", flush=True)
        return_code = process.wait()
    elapsed = time.time() - started
    if return_code != 0:
        inventory["cells"][key] = {
            "status": "failed",
            "exit_code": return_code,
            "run_path": str(run),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "elapsed_seconds": elapsed,
        }
        inventory["active_cell"] = None
        _atomic_json(inventory_path, inventory)
        raise RuntimeError(
            f"Pilot training failed for {key} with exit code {return_code}; "
            f"inspect {stdout_path} and {stderr_path}"
        )
    inventory["active_cell"]["status"] = "validating"
    _atomic_json(inventory_path, inventory)
    return elapsed


def run_annotation_ready_pilot(
    *,
    config_path: Path = DEFAULT_EXPERIMENT_CONFIG,
    protocol_path: Path = DEFAULT_PROTOCOL,
    model_path: Path = Path(r"D:\RFC\models\deberta-v3-base"),
) -> Path:
    config = load_config(config_path)
    protocol, protocol_sha256 = load_annotation_ready_protocol(
        protocol_path, config
    )
    protocol["_loaded_protocol_sha256"] = protocol_sha256
    effective, dataset_manifest, dataset_dir = load_effective_dataset(config)
    model_hash, model_files = _check_preflight(
        config=config,
        protocol=protocol,
        protocol_path=protocol_path.resolve(),
        model_path=model_path,
    )
    inventory_path = _inventory_path(config)
    inventory = _load_or_create_inventory(
        inventory_path,
        protocol_sha256,
        protocol["source_snapshot"]["files"],
    )
    log_dir = inventory_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runs_root = resolve_path(config["outputs"]["runs_dir"])
    seed = int(protocol["training"]["seed"])

    for fold in protocol["training"]["folds"]:
        key = f"fold_{fold}/seed_{seed}"
        entry = inventory["cells"].get(key)
        run = runs_root / f"fold_{fold}" / f"seed_{seed}"
        if entry and entry.get("status") == "complete":
            _validate_cell(
                run=run,
                fold=fold,
                seed=seed,
                config=config,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                effective=effective,
                dataset_manifest=dataset_manifest,
                dataset_dir=dataset_dir,
                model_hash=model_hash,
                model_files=model_files,
            )
            continue
        if entry:
            raise RuntimeError(
                f"Pilot cell {key} has prior status {entry.get('status')}; refusing retry"
            )
        _assert_no_training_processes()
        if source_file_hashes() != protocol["source_snapshot"]["files"]:
            raise RuntimeError("Source files changed before the next pilot cell")
        elapsed = _launch_cell(
            fold=fold,
            seed=seed,
            run=run,
            config_path=config_path.resolve(),
            protocol_path=protocol_path.resolve(),
            model_path=model_path.resolve(),
            inventory=inventory,
            inventory_path=inventory_path,
            log_dir=log_dir,
        )
        try:
            predictions = _validate_cell(
                run=run,
                fold=fold,
                seed=seed,
                config=config,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                effective=effective,
                dataset_manifest=dataset_manifest,
                dataset_dir=dataset_dir,
                model_hash=model_hash,
                model_files=model_files,
            )
            if source_file_hashes() != protocol["source_snapshot"]["files"]:
                raise RuntimeError("Source files changed during the pilot cell")
        except Exception:
            inventory["cells"][key] = {
                "status": "failed_validation",
                "run_path": str(run),
                "elapsed_seconds": elapsed,
            }
            inventory["active_cell"] = None
            _atomic_json(inventory_path, inventory)
            raise
        inventory["cells"][key] = {
            "status": "complete",
            "run_path": str(run),
            "rows": len(predictions),
            "elapsed_seconds": elapsed,
        }
        inventory["active_cell"] = None
        _atomic_json(inventory_path, inventory)
        print(f"Validated complete {key}", flush=True)

    expected_cells = {
        f"fold_{fold}/seed_{seed}" for fold in protocol["training"]["folds"]
    }
    if set(inventory["cells"]) != expected_cells or any(
        item.get("status") != "complete" for item in inventory["cells"].values()
    ):
        raise RuntimeError("Pilot inventory does not contain exactly five valid cells")
    if source_file_hashes() != protocol["source_snapshot"]["files"]:
        raise RuntimeError("Source snapshot changed before final analysis")
    analysis_path = resolve_path(config["outputs"]["aggregate_dir"])
    result = create_experiment2_analysis(
        config_path=config_path,
        protocol_path=protocol_path,
        inventory_path=inventory_path,
        analysis_path=analysis_path,
    )
    inventory["analysis"] = str(result)
    _atomic_json(inventory_path, inventory)
    print(f"Experiment 2 analysis complete: {result}", flush=True)
    return inventory_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--model-path", type=Path, default=Path(r"D:\RFC\models\deberta-v3-base")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = run_annotation_ready_pilot(
        config_path=args.config,
        protocol_path=args.protocol,
        model_path=args.model_path,
    )
    print(f"Experiment 2 completion inventory: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
