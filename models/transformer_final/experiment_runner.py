"""Sequential, resumable runner for the frozen FP32 Experiment 1 cells."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# CUDA DLL lookup on Windows must include the active Conda environment before torch imports.
_conda_library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
os.environ["PATH"] = str(_conda_library_bin) + os.pathsep + os.environ.get("PATH", "")

import torch

from .aggregate import (
    DEFAULT_PROTOCOL,
    DEFAULT_RUN_OVERRIDES,
    aggregate_condition,
)
from .common import (
    ROOT,
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    source_file_hashes,
    write_json,
)
from .compare import compare
from .config import DEFAULT_CONFIG, load_config
from .dataset import load_effective_dataset
from .protocol import load_fp32_protocol
from .research_report import write_research_report
from .run_validation import validate_completed_run
from .train import (
    _assert_finite_model_parameters,
    _check_forward_numerics,
    _clip_gradients,
    _prepare_model_for_training,
)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale inventory temporary file exists: {temporary}")
    write_json(temporary, value)
    temporary.replace(path)


def _assert_no_training_processes() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdecimal():
            raise RuntimeError(
                f"Unrecognized GPU compute-process listing {value!r}; refusing training"
            )
        if int(value) == os.getpid():
            continue
        process = subprocess.run(
            ["tasklist", "/FI", f"PID eq {value}", "/FO", "CSV", "/NH"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        try:
            process_row = next(csv.reader([process.stdout.strip()]))
        except (StopIteration, IndexError, csv.Error) as exc:
            raise RuntimeError(
                f"Cannot identify GPU compute process PID {value}; refusing training"
            ) from exc
        image_name = process_row[0].casefold()
        if image_name.startswith("info: no tasks"):
            raise RuntimeError(
                f"GPU process PID {value} disappeared during inspection"
            )
        if image_name in {"python.exe", "pythonw.exe"}:
            raise RuntimeError(
                f"Another Python process occupies the GPU (PID {value}); refusing training"
            )


def _verify_environment(
    *,
    config_path: Path,
    model_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    if platform.python_version() != protocol["environment"]["python"]:
        raise RuntimeError(
            f"Wrong Python version: {platform.python_version()}"
        )
    if Path(sys.executable).resolve() != Path(
        protocol["environment"]["interpreter"]
    ).resolve():
        raise RuntimeError(f"Wrong selected Conda interpreter: {sys.executable}")
    expected_environment = protocol["environment"]
    import importlib.metadata

    for package in (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tokenizers",
        "sentencepiece",
        "safetensors",
    ):
        version = importlib.metadata.version(package)
        if version != expected_environment[package]:
            raise RuntimeError(f"{package} version mismatch: {version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable; CPU fallback is prohibited")
    if torch.cuda.get_device_name(0) != expected_environment["gpu"]:
        raise RuntimeError(f"Unexpected GPU: {torch.cuda.get_device_name(0)}")
    if torch.version.cuda != expected_environment["cuda_runtime"]:
        raise RuntimeError(f"CUDA runtime mismatch: {torch.version.cuda}")
    if torch.backends.cudnn.version() != expected_environment["cudnn"]:
        raise RuntimeError(
            f"cuDNN version mismatch: {torch.backends.cudnn.version()}"
        )
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    if not driver or len(set(driver)) != 1 or driver[0].strip() != expected_environment["driver"]:
        raise RuntimeError(f"CUDA driver mismatch: {driver}")
    if str(model_path.resolve()) != protocol["model"]["local_path"]:
        raise RuntimeError("Model path differs from frozen FP32 protocol")
    model_hash, model_files = hash_directory(model_path)
    if model_hash != protocol["model"]["source_sha256"]:
        raise RuntimeError("Local pretrained model hash mismatch")
    if sha256_file(resolve_path(config["data"]["source_csv"])) != protocol["dataset"]["source_sha256"]:
        raise RuntimeError("Dataset A source hash mismatch")
    if (
        dataset_manifest["effective_dataset_sha256"]
        != protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(resolve_path(config["outputs"]["dataset_dir"]) / "manifest.json")
        != protocol["dataset"]["dataset_manifest_sha256"]
        or sha256_file(config_path.resolve()) != config["_config_sha256"]
    ):
        raise RuntimeError("Dataset or configuration hash mismatch")
    dataset_dir = resolve_path(config["outputs"]["dataset_dir"])
    for fold in protocol["training"]["folds"]:
        fold_hash = sha256_file(dataset_dir / "folds" / f"fold_{fold}.json")
        if fold_hash != protocol["dataset"]["fold_manifest_sha256"][str(fold)]:
            raise RuntimeError(f"Fold {fold} manifest hash mismatch")
    if not (Path(sys.executable).resolve().parent / "Library" / "bin" / "nvrtc-builtins64_133.dll").is_file():
        raise RuntimeError("The selected environment is missing the required NVRTC DLL")
    free_memory, _ = torch.cuda.mem_get_info(0)
    if free_memory < 8 * 1024**3:
        raise RuntimeError(f"Insufficient free GPU memory before training: {free_memory}")
    _assert_no_training_processes()
    required_guards = (
        _assert_finite_model_parameters,
        _check_forward_numerics,
        _clip_gradients,
        _prepare_model_for_training,
    )
    if not all(callable(guard) for guard in required_guards):
        raise RuntimeError("Required numerical safeguards are unavailable")
    return model_hash, model_files


def _inventory_path(protocol_path: Path) -> Path:
    return (
        protocol_path.parent.parent
        / "run_management"
        / "fp32_protocol_v1"
        / "completion_inventory.json"
    )


def _load_or_create_inventory(
    *,
    path: Path,
    protocol_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if path.exists():
        inventory = read_json(path)
        if (
            inventory.get("protocol_sha256") != protocol_sha256
            or inventory.get("source_file_hashes") != source_hashes
        ):
            raise RuntimeError("Existing completion inventory belongs to other code/protocol")
        if inventory.get("active_cell") is not None:
            active = inventory["active_cell"]
            pid = int(active.get("pid") or 0)
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError:
                    active["status"] = "interrupted"
                    _atomic_write_json(path, inventory)
                    raise RuntimeError(
                        f"Previous cell was interrupted: {active['cell']}; "
                        "inspect its logs and output before authorizing continuation"
                    )
                raise RuntimeError(
                    f"Experiment process PID {pid} is still active for {active['cell']}"
                )
            raise RuntimeError(
                f"Inventory has an unresolved active cell: {active['cell']}"
            )
        return inventory
    inventory = {
        "schema_version": 1,
        "protocol_id": "original_annotation_experiment_1_fp32_v1",
        "protocol_sha256": protocol_sha256,
        "source_file_hashes": source_hashes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active_cell": None,
        "cells": {},
        "aggregations": {},
        "comparison": None,
    }
    _atomic_write_json(path, inventory)
    return inventory


def _cell_key(condition: str, fold: int, seed: int) -> str:
    return f"{condition}/fold_{fold}/seed_{seed}"


def _cell_directory(
    runs_root: Path,
    condition: str,
    fold: int,
    seed: int,
    overrides: dict[str, Any],
) -> Path:
    key = _cell_key(condition, fold, seed)
    override = overrides.get(key)
    if override is not None:
        return (runs_root / override["path"]).resolve()
    return (runs_root / condition / f"fold_{fold}" / f"seed_{seed}").resolve()


def _ensure_aggregate(
    *,
    condition: str,
    inventory: dict[str, Any],
    inventory_path: Path,
    config_path: Path,
    protocol_path: Path,
    overrides_path: Path,
    config: dict[str, Any],
) -> None:
    if condition in inventory["aggregations"]:
        return
    aggregate_path = resolve_path(config["outputs"]["aggregate_dir"]) / condition
    if aggregate_path.exists():
        raise FileExistsError(
            f"Aggregate exists without a verified inventory entry: {aggregate_path}"
        )
    result = aggregate_condition(
        config_path,
        condition,
        protocol_path=protocol_path,
        run_overrides_path=overrides_path,
    )
    inventory["aggregations"][condition] = str(result)
    _atomic_write_json(inventory_path, inventory)
    print(f"{condition} aggregation verified: {result}", flush=True)


def _validate_cell(
    *,
    run: Path,
    condition: str,
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
    legacy: bool,
) -> None:
    validate_completed_run(
        run=run,
        condition=condition,
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
        legacy_protocol_compatible=legacy,
    )


def _launch_cell(
    *,
    condition: str,
    fold: int,
    seed: int,
    run: Path,
    log_dir: Path,
    inventory: dict[str, Any],
    inventory_path: Path,
    config_path: Path,
    protocol_path: Path,
    model_path: Path,
    diagnostic_numerics: bool,
) -> float:
    key = _cell_key(condition, fold, seed)
    if run.exists():
        if (run / "run_manifest.json").is_file():
            raise RuntimeError(
                f"Unlisted completed run exists and requires verification: {run}"
            )
        raise FileExistsError(f"Refusing to overwrite existing output: {run}")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{condition}_fold_{fold}_seed_{seed}.stdout.log"
    stderr_path = log_dir / f"{condition}_fold_{fold}_seed_{seed}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"Cell logs already exist for {key}")
    command = [
        sys.executable,
        "-m",
        "models.transformer_final.train",
        "--config",
        str(config_path),
        "--condition",
        condition,
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
        "--output",
        str(run),
    ]
    if diagnostic_numerics:
        command.append("--diagnostic-numerics")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PATH"] = str(Path(sys.executable).resolve().parent / "Library" / "bin") + os.pathsep + env["PATH"]
    inventory["active_cell"] = {
        "cell": key,
        "status": "starting",
        "pid": None,
        "run_path": str(run),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _atomic_write_json(inventory_path, inventory)
    print(f"Starting {key}; output={run}", flush=True)
    started = time.time()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        inventory["active_cell"]["status"] = "running"
        inventory["active_cell"]["pid"] = process.pid
        _atomic_write_json(inventory_path, inventory)
        print(f"{key} child PID={process.pid}", flush=True)
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
        _atomic_write_json(inventory_path, inventory)
        raise RuntimeError(
            f"Training failed for {key} with exit code {return_code}; "
            f"logs: {stdout_path}, {stderr_path}"
        )
    inventory["active_cell"]["status"] = "validating"
    _atomic_write_json(inventory_path, inventory)
    return elapsed


def run_experiment(
    *,
    config_path: Path,
    protocol_path: Path,
    overrides_path: Path,
    model_path: Path,
) -> Path:
    config = load_config(config_path)
    protocol, protocol_sha256 = load_fp32_protocol(protocol_path, config)
    override_file = read_json(overrides_path)
    overrides = override_file.get("overrides", {})
    if (
        override_file.get("schema_version") != 1
        or override_file.get("protocol_id") != protocol["protocol_id"]
        or overrides
        != {
            "A1/fold_1/seed_1729": {
                "path": "A1/fold_1/seed_1729_diagnostic_numerics_v2",
                "legacy_protocol_compatible": True,
            }
        }
    ):
        raise RuntimeError("Run-path override mapping does not match frozen protocol")
    seen_override_paths: set[str] = set()
    for key, value in overrides.items():
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("path"), str)
            or Path(value["path"]).is_absolute()
            or value["path"] in seen_override_paths
        ):
            raise RuntimeError(f"Invalid or duplicate explicit run override: {key}")
        seen_override_paths.add(value["path"])
    effective, dataset_manifest, dataset_dir = load_effective_dataset(config)
    model_hash, model_files = _verify_environment(
        config_path=config_path,
        model_path=model_path,
        config=config,
        protocol=protocol,
        dataset_manifest=dataset_manifest,
    )
    source_hashes = source_file_hashes()
    inventory_path = _inventory_path(protocol_path)
    inventory = _load_or_create_inventory(
        path=inventory_path,
        protocol_sha256=protocol_sha256,
        source_hashes=source_hashes,
    )
    log_dir = inventory_path.parent / "logs"
    runs_root = resolve_path(config["outputs"]["runs_dir"])
    required_cells = [
        (condition, fold, seed)
        for condition in ("A1", "A2")
        for seed in config["training"]["seeds"]
        for fold in range(1, int(config["folds"]["count"]) + 1)
    ]
    required_keys = {_cell_key(*cell) for cell in required_cells}
    unknown_inventory_cells = set(inventory["cells"]) - required_keys
    if unknown_inventory_cells:
        raise RuntimeError(f"Unexpected cells in completion inventory: {unknown_inventory_cells}")

    for condition, fold, seed in required_cells:
        if condition == "A2":
            _ensure_aggregate(
                condition="A1",
                inventory=inventory,
                inventory_path=inventory_path,
                config_path=config_path,
                protocol_path=protocol_path,
                overrides_path=overrides_path,
                config=config,
            )
        key = _cell_key(condition, fold, seed)
        run = _cell_directory(runs_root, condition, fold, seed, overrides)
        entry = inventory["cells"].get(key)
        is_legacy = bool(
            overrides.get(key, {}).get("legacy_protocol_compatible", False)
        )
        if entry and entry.get("status") == "complete":
            _validate_cell(
                run=run,
                condition=condition,
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
                legacy=is_legacy,
            )
            continue
        if entry and entry.get("status") in {
            "failed",
            "failed_validation",
            "interrupted",
        }:
            raise RuntimeError(
                f"Cell {key} has prior status {entry['status']}; refusing automatic retry"
            )
        if key == "A1/fold_1/seed_1729":
            _validate_cell(
                run=run,
                condition=condition,
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
                legacy=True,
            )
            inventory["cells"][key] = {
                "status": "complete",
                "run_path": str(run),
                "reuse": "verified pre-protocol FP32 diagnostic run; no retroactive source hash",
            }
            _atomic_write_json(inventory_path, inventory)
            print(f"Verified and reused {key}", flush=True)
            continue
        if (run / "run_manifest.json").is_file():
            _validate_cell(
                run=run,
                condition=condition,
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
                legacy=is_legacy,
            )
            inventory["cells"][key] = {
                "status": "complete",
                "run_path": str(run),
                "reuse": "found completed and fully verified immutable run",
            }
            _atomic_write_json(inventory_path, inventory)
            print(f"Verified existing {key}", flush=True)
            continue
        _assert_no_training_processes()
        elapsed = _launch_cell(
            condition=condition,
            fold=fold,
            seed=seed,
            run=run,
            log_dir=log_dir,
            inventory=inventory,
            inventory_path=inventory_path,
            config_path=config_path,
            protocol_path=protocol_path,
            model_path=model_path,
            diagnostic_numerics=True,
        )
        _validate_cell(
            run=run,
            condition=condition,
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
            legacy=False,
        )
        inventory["cells"][key] = {
            "status": "complete",
            "run_path": str(run),
            "elapsed_seconds": elapsed,
        }
        inventory["active_cell"] = None
        _atomic_write_json(inventory_path, inventory)
        print(f"Verified complete {key}", flush=True)

    _ensure_aggregate(
        condition="A1",
        inventory=inventory,
        inventory_path=inventory_path,
        config_path=config_path,
        protocol_path=protocol_path,
        overrides_path=overrides_path,
        config=config,
    )
    _ensure_aggregate(
        condition="A2",
        inventory=inventory,
        inventory_path=inventory_path,
        config_path=config_path,
        protocol_path=protocol_path,
        overrides_path=overrides_path,
        config=config,
    )
    if set(inventory["aggregations"]) != {"A1", "A2"}:
        raise RuntimeError(
            "Both complete A1 and A2 aggregations are required before comparison"
        )
    comparison_path = (
        resolve_path(config["outputs"]["aggregate_dir"]) / "A1_vs_A2_fp32_v1"
    )
    if comparison_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {comparison_path}")
    comparison = compare(config_path, output=comparison_path)
    inventory["comparison"] = str(comparison)
    report_path = (
        DEFAULT_CONFIG.parents[1]
        / "reports"
        / "transformer_final_fp32_A1_A2_v1.md"
    )
    report = write_research_report(
        config_path=config_path,
        protocol_path=protocol_path,
        inventory_path=inventory_path,
        output_path=report_path,
    )
    inventory["research_report"] = str(report)
    inventory["active_cell"] = None
    _atomic_write_json(inventory_path, inventory)
    print(f"Paired comparison written: {comparison}", flush=True)
    print(f"Final research report written: {report}", flush=True)
    return inventory_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-overrides", type=Path, default=DEFAULT_RUN_OVERRIDES)
    parser.add_argument(
        "--model-path", type=Path, default=Path(r"D:\RFC\models\deberta-v3-base")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = run_experiment(
        config_path=args.config,
        protocol_path=args.protocol,
        overrides_path=args.run_overrides,
        model_path=args.model_path,
    )
    print(f"Experiment completion inventory: {inventory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
