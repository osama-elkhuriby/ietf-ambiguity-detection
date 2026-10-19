"""Run the frozen recall study sequentially with immutable cell outputs."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors import safe_open

from experiments.transformer_final.recall_improvement_study_v1.study_protocol import (
    EXPECTED_BASE_PROTOCOL,
    PROTOCOL_PATH,
    SPLITS_PATH,
    STUDY_ROOT,
    _study_source_hashes,
)
from models.transformer_final.common import (
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import load_annotation_ready_protocol

INVENTORY_PATH = STUDY_ROOT / "EXPERIMENT_INVENTORY.json"
TIME_BUDGET_PATH = STUDY_ROOT / "time_budget.json"
MIN_FREE_VRAM_MIB = 18_000
CELL_TIMEOUT_SECONDS = 900
WINDOWS_GRAPHICS_PROCESSES = frozenset(
    {
        "anydesk",
        "chrome",
        "code",
        "crossdeviceresume",
        "dwm",
        "explorer",
        "msedgewebview2",
        "msedge",
        "notepad",
        "searchhost",
        "shellexperiencehost",
        "shellhost",
        "startmenuexperiencehost",
        "taskmgr",
        "textinputhost",
        "windowsterminal",
    }
)
STAGE_ORDER = (
    "B_nested_threshold_calibration",
    "C_local_context",
    "G_partial_4",
    "E_error_analysis",
    "D_focal_loss",
    "G2_partial_2",
)
STAGE_ESTIMATES = {
    "B_nested_threshold_calibration": 4500,
    "C_local_context": 2250,
    "G_partial_4": 1500,
    "E_error_analysis": 0,
    "D_focal_loss": 1500,
    "G2_partial_2": 1500,
}
STAGE_PLAN = {
    "B_nested_threshold_calibration": ("nested", 15, "target_only"),
    "C_local_context": ("context", 5, "local_context"),
    "G_partial_4": ("partial_4", 5, "target_only"),
    "D_focal_loss": ("focal", 5, "target_only"),
    "G2_partial_2": ("partial_2", 5, "target_only"),
}


def _deadline(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("The study deadline must have an explicit timezone")
    return result


def _remaining_seconds(deadline: datetime) -> float:
    return (deadline - datetime.now(deadline.tzinfo)).total_seconds()


def _write_inventory(inventory: dict[str, Any]) -> None:
    write_json(INVENTORY_PATH, inventory)


def _reset_time_skipped_stages(inventory: dict[str, Any]) -> None:
    for stage in inventory["stages"].values():
        if stage.get("status") == "SKIPPED_TIME_BUDGET":
            stage["status"] = "NOT_STARTED"


def _save_budget(budget: dict[str, Any]) -> None:
    budget["updated_at"] = datetime.now().astimezone().isoformat()
    budget["deadline_remaining_seconds"] = max(
        0,
        int(_remaining_seconds(_deadline(budget["authorization_window_deadline"]))),
    )
    write_json(TIME_BUDGET_PATH, budget)


def _verify_protocol_inputs(protocol: dict[str, Any]) -> Any:
    config = load_config(Path(protocol["config_path"]))
    baseline, base_protocol_hash = load_annotation_ready_protocol(
        Path(protocol["baseline_protocol_path"]), config
    )
    if base_protocol_hash != EXPECTED_BASE_PROTOCOL:
        raise RuntimeError("The frozen Experiment 2 protocol hash changed")
    if protocol["baseline_protocol_sha256"] != base_protocol_hash:
        raise RuntimeError("Study protocol points to a different baseline protocol")
    if (
        protocol.get("study_id") != "recall_improvement_study_v1"
        or protocol.get("schema_version") != 1
        or protocol.get("training") != baseline["training"]
        or protocol["dataset"]["source_sha256"] != baseline["dataset"]["source_sha256"]
        or protocol["dataset"]["family_fold_source_sha256"]
        != baseline["dataset"]["fold_source_sha256"]
        or protocol["model"]["local_path"] != baseline["model"]["local_path"]
        or protocol["model"]["source_sha256"] != baseline["model"]["source_sha256"]
        or protocol["model"]["model_file_hashes"]
        != baseline["model"]["model_file_hashes"]
    ):
        raise RuntimeError("Frozen study protocol diverges from Experiment 2 identity")
    if sha256_text(stable_json(protocol["source_snapshot"])) != protocol.get(
        "source_snapshot_sha256"
    ):
        raise RuntimeError("Frozen source-snapshot digest mismatch")
    if (
        sha256_file(Path(protocol["config_path"])) != protocol["config_sha256"]
        or config["_config_sha256"] != protocol["config_sha256"]
        or sha256_file(resolve_path(config["data"]["source_csv"]))
        != protocol["dataset"]["source_sha256"]
    ):
        raise RuntimeError("Study configuration hash changed")
    if source_file_hashes() != protocol["source_snapshot"]["core_files"]:
        raise RuntimeError("Core training source changed after protocol freeze")
    if _study_source_hashes() != protocol["source_snapshot"]["study_files"]:
        raise RuntimeError("Study source changed after protocol freeze")
    frame, dataset_manifest, dataset_dir = load_effective_dataset(config)
    if (
        dataset_manifest["effective_dataset_sha256"]
        != protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(dataset_dir / "manifest.json")
        != protocol["dataset"]["dataset_manifest_sha256"]
        or len(frame) != protocol["dataset"]["rows"]
        or sha256_file(SPLITS_PATH) != protocol["inner_cv"]["manifests_sha256"]
    ):
        raise RuntimeError("Frozen dataset or nested-split identity changed")
    for fold, expected_hash in protocol["dataset"]["fold_manifest_sha256"].items():
        if sha256_file(dataset_dir / "folds" / f"fold_{fold}.json") != expected_hash:
            raise RuntimeError(f"Frozen outer fold {fold} changed")
    model_path = Path(protocol["model"]["local_path"])
    model_hash, model_files = hash_directory(model_path)
    if (
        model_hash != protocol["model"]["source_sha256"]
        or model_files != protocol["model"]["model_file_hashes"]
    ):
        raise RuntimeError("Local model or tokenizer files changed after freeze")
    return frame


def _gpu_preflight() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The study requires exactly one available CUDA GPU")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one NVIDIA GPU, found {len(lines)}")
    name, free_text = [item.strip() for item in lines[0].split(",", maxsplit=1)]
    free_mib = int(free_text)
    if name != protocol_gpu_name() or free_mib < MIN_FREE_VRAM_MIB:
        raise RuntimeError(
            f"GPU preflight failed: name={name!r}, free={free_mib} MiB, "
            f"required_name={protocol_gpu_name()!r}, minimum_free={MIN_FREE_VRAM_MIB} MiB"
        )
    process_table = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    process_types = _parse_nvidia_smi_process_types(process_table.stdout)
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    active: list[str] = []
    for value in processes.stdout.splitlines():
        if not value.strip():
            continue
        pid_text, process_name = value.split(",", maxsplit=1)
        pid = int(pid_text.strip())
        process_name = process_name.strip()
        if pid == os.getpid():
            continue
        process_type = process_types.get(pid)
        if process_type == "C":
            active.append(value.strip())
        elif process_type in {"C+G", "G"}:
            if not _is_known_windows_graphics_process(process_name):
                active.append(value.strip())
        elif os.name == "nt":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        f"$process = Get-Process -Id {pid} "
                        "-ErrorAction SilentlyContinue; "
                        "if ($process) { $process.ProcessName }"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if not _is_known_windows_graphics_process(result.stdout.strip()):
                active.append(value.strip())
        else:
            active.append(value.strip())
    if active:
        raise RuntimeError(
            "GPU preflight found an existing compute process; refusing to overlap: "
            + "; ".join(active[:5])
        )
    return {
        "name": name,
        "free_memory_mib": free_mib,
        "minimum_free_memory_mib": MIN_FREE_VRAM_MIB,
        "active_compute_processes": active,
    }


def _parse_nvidia_smi_process_types(output: str) -> dict[int, str]:
    process_types: dict[int, str] = {}
    in_process_table = False
    for line in output.splitlines():
        if "Processes:" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        fields = line.strip().strip("|").split()
        if (
            len(fields) >= 5
            and fields[0].isdigit()
            and fields[3].isdigit()
            and fields[4] in {"C", "C+G", "G"}
        ):
            process_types[int(fields[3])] = fields[4]
    return process_types


def _is_known_windows_graphics_process(process_name: str) -> bool:
    name = Path(process_name.strip()).name.casefold().removesuffix(".exe")
    return name in WINDOWS_GRAPHICS_PROCESSES


def protocol_gpu_name() -> str:
    return str(read_json(PROTOCOL_PATH)["environment"]["gpu"])


def _cells_for_stage(
    stage: str, protocol: dict[str, Any], splits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    method, expected_count, input_mode = STAGE_PLAN[stage]
    specs: list[dict[str, Any]] = []
    seed = int(protocol["training"]["seed"])
    config_path = str(Path(protocol["config_path"]).resolve())
    model_path = str(Path(protocol["model"]["local_path"]).resolve())
    run_root = STUDY_ROOT / "runs" / stage
    audit_root = STUDY_ROOT / "audits" / stage
    if method == "nested":
        for split in splits:
            outer = int(split["outer_fold"])
            inner = int(split["inner_fold"])
            identity = f"outer_{outer}_inner_{inner}"
            specs.append(
                {
                    "cell_id": f"{stage}/{identity}",
                    "method": method,
                    "train_fold": inner,
                    "outer_fold": outer,
                    "seed": seed,
                    "config_path": config_path,
                    "model_path": model_path,
                    "input_mode": input_mode,
                    "fold_manifest": split,
                    "fold_manifest_sha256": sha256_text(stable_json(split)),
                    "fitting_record_ids": split["fitting_record_ids"],
                    "scoring_record_ids": split["scoring_record_ids"],
                    "run_path": str(run_root / identity),
                    "audit_path": str(audit_root / f"{identity}.json"),
                    "expected_score_ids": split["scoring_record_ids"],
                }
            )
    else:
        for fold in range(1, 6):
            specs.append(
                {
                    "cell_id": f"{stage}/fold_{fold}/seed_{seed}",
                    "method": method,
                    "train_fold": fold,
                    "seed": seed,
                    "config_path": config_path,
                    "model_path": model_path,
                    "input_mode": input_mode,
                    "run_path": str(run_root / f"fold_{fold}" / f"seed_{seed}"),
                    "audit_path": str(audit_root / f"fold_{fold}_seed_{seed}.json"),
                    "expected_score_ids": None,
                }
            )
    if len(specs) != expected_count:
        raise RuntimeError(f"Unexpected cell count for stage {stage}")
    return specs


def _validate_cell(
    spec: dict[str, Any],
    protocol: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    run = Path(spec["run_path"]).resolve()
    manifest = read_json(run / "run_manifest.json")
    run_config = read_json(run / "run_config.json")
    if (
        manifest.get("condition") != "A2"
        or int(manifest.get("fold", 0)) != int(spec["train_fold"])
        or int(manifest.get("seed", 0)) != int(spec["seed"])
        or manifest.get("loss")
        != ("focal_gamma_2" if spec["method"] == "focal" else "weighted_ce")
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != spec["input_mode"]
        or manifest.get("epochs_completed") != 5
        or manifest.get("resume_checkpoint") is not None
        or manifest.get("fresh_base_initialization") is not True
    ):
        raise RuntimeError(f"Run protocol mismatch: {spec['cell_id']}")
    is_partial = spec["method"] in {"partial_4", "partial_2"}
    expected_full_encoder = not is_partial
    if manifest.get("full_encoder_fine_tuned") is not expected_full_encoder:
        raise RuntimeError(f"Fine-tuning mode mismatch: {spec['cell_id']}")
    if spec["method"] == "focal" and (
        manifest.get("focal_gamma") != 2.0
        or manifest.get("focal_alpha") != "existing fitting-side class weights"
    ):
        raise RuntimeError(f"Focal-loss metadata mismatch: {spec['cell_id']}")
    if is_partial:
        expected_layers = 4 if spec["method"] == "partial_4" else 2
        if (
            manifest.get("fine_tuning_policy") != spec["method"]
            or manifest.get("trainable_last_layers") != expected_layers
            or not isinstance(manifest.get("optimizer_parameter_elements"), int)
            or manifest.get("optimizer_parameter_elements", 0) <= 0
        ):
            raise RuntimeError(
                f"Partial fine-tuning metadata mismatch: {spec['cell_id']}"
            )
    if (
        manifest.get("config_sha256") != protocol["config_sha256"]
        or manifest.get("effective_dataset_sha256")
        != protocol["dataset"]["effective_dataset_sha256"]
        or manifest.get("dataset_manifest_sha256")
        != protocol["dataset"]["dataset_manifest_sha256"]
        or manifest.get("model_source_sha256") != protocol["model"]["source_sha256"]
        or manifest.get("model_file_hashes") != protocol["model"]["model_file_hashes"]
        or manifest.get("source_file_hashes")
        != protocol["source_snapshot"]["core_files"]
        or manifest.get("fold_manifest_sha256")
        != protocol["dataset"]["fold_manifest_sha256"][str(spec["train_fold"])]
    ):
        raise RuntimeError(f"Run provenance mismatch: {spec['cell_id']}")
    resolved = run_config.get("resolved", {})
    expected_loss = "focal_gamma_2" if spec["method"] == "focal" else "weighted_ce"
    if (
        resolved.get("condition") != "A2"
        or resolved.get("loss") != expected_loss
        or resolved.get("precision") != "fp32"
        or resolved.get("input_mode") != spec["input_mode"]
        or int(resolved.get("fold", 0)) != int(spec["train_fold"])
        or int(resolved.get("seed", 0)) != int(spec["seed"])
    ):
        raise RuntimeError(f"Resolved run configuration mismatch: {spec['cell_id']}")
    if spec["method"] == "focal" and (
        resolved.get("focal_gamma") != 2.0
        or resolved.get("focal_alpha") != "existing fitting-side class weights"
    ):
        raise RuntimeError(f"Resolved focal-loss metadata mismatch: {spec['cell_id']}")
    if is_partial and (
        resolved.get("fine_tuning_policy") != spec["method"]
        or resolved.get("trainable_last_layers")
        != (4 if spec["method"] == "partial_4" else 2)
    ):
        raise RuntimeError(
            f"Resolved partial fine-tuning metadata mismatch: {spec['cell_id']}"
        )

    artifact_hashes = manifest.get("artifact_hashes", {})
    if not artifact_hashes:
        raise RuntimeError(f"Run has no artifact hashes: {spec['cell_id']}")
    for value in artifact_hashes.values():
        path = (run / value["path"]).resolve()
        if (
            run not in path.parents
            or not path.is_file()
            or sha256_file(path) != value["sha256"]
        ):
            raise RuntimeError(f"Artifact hash validation failed: {spec['cell_id']}")
    checkpoint = run / "checkpoints" / "final_epoch"
    checkpoint_metadata = read_json(run / "checkpoints" / "checkpoint_metadata.json")
    checkpoint_hash, checkpoint_files = hash_directory(checkpoint)
    if (
        checkpoint_metadata.get("epoch") != 5
        or checkpoint_metadata.get("sha256") != checkpoint_hash
        or checkpoint_metadata.get("files") != checkpoint_files
    ):
        raise RuntimeError(f"Checkpoint hash validation failed: {spec['cell_id']}")
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as tensors:
        tensor_keys = tensors.keys()
        if {tensors.get_slice(key).get_dtype() for key in tensor_keys} != {"F32"}:
            raise RuntimeError(f"Non-FP32 checkpoint: {spec['cell_id']}")

    predictions = pd.read_csv(
        run / "predictions" / "held_out_predictions.csv",
        dtype={"record_id": str},
        keep_default_na=False,
    )
    expected_ids = spec["expected_score_ids"]
    if expected_ids is None:
        held_out = frame.loc[frame["fold"].astype(int) == int(spec["train_fold"])]
        expected_ids = held_out["record_id"].astype(str).tolist()
    if spec["method"] == "nested":
        fitting_ids = set(spec["fitting_record_ids"])
        scoring_ids = set(spec["scoring_record_ids"])
        if (
            len(fitting_ids) != int(manifest["fitting_rows"])
            or len(scoring_ids) != int(manifest["held_out_rows"])
            or fitting_ids & scoring_ids
        ):
            raise RuntimeError(f"Nested fitting/scoring ID mismatch: {spec['cell_id']}")
        family_by_id = frame.set_index(frame["record_id"].astype(str))[
            "family_id"
        ].astype(str)
        fitting_families = set(family_by_id.loc[list(fitting_ids)])
        scoring_families = set(family_by_id.loc[list(scoring_ids)])
        if fitting_families & scoring_families:
            raise RuntimeError(f"Nested family leakage: {spec['cell_id']}")
        if (fitting_families | scoring_families) & set(
            protocol["dataset"]["reserved_families"]
        ):
            raise RuntimeError(f"Reserved family in nested data: {spec['cell_id']}")
    elif len(expected_ids) != int(manifest["held_out_rows"]):
        raise RuntimeError(f"Held-out row count mismatch: {spec['cell_id']}")
    if (
        predictions["record_id"].duplicated().any()
        or set(predictions["record_id"].astype(str)) != set(expected_ids)
        or len(predictions) != len(expected_ids)
    ):
        raise RuntimeError(
            f"Prediction IDs do not match frozen scoring IDs: {spec['cell_id']}"
        )
    indexed = frame.set_index(frame["record_id"].astype(str))
    record_ids = predictions["record_id"].astype(str)
    labels = record_ids.map(indexed["label"].astype(int))
    folds = record_ids.map(indexed["fold"].astype(int))
    families = record_ids.map(indexed["family_id"].astype(str))
    if labels.isna().any() or not np.array_equal(
        labels.to_numpy(), predictions["label"].astype(int)
    ) or not np.array_equal(folds.to_numpy(), predictions["fold"].astype(int)) or not np.array_equal(
        families.to_numpy(), predictions["family_id"].astype(str)
    ):
        raise RuntimeError(
            f"Prediction labels/folds/families differ from frozen dataset: "
            f"{spec['cell_id']}"
        )
    scores = pd.to_numeric(predictions["probability_yes"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError(f"Invalid held-out probability values: {spec['cell_id']}")
    reserved = set(protocol["dataset"]["reserved_families"])
    if set(predictions["family_id"].astype(str)) & reserved:
        raise RuntimeError(
            f"Reserved-family data entered a scoring fold: {spec['cell_id']}"
        )

    history = pd.read_csv(run / "history" / "training_history.csv")
    fit_count = int(manifest["fitting_rows"])
    fit_positive = int(manifest["fitting_positive_labels"])
    fit_negative = int(manifest["fitting_negative_labels"])
    if fit_positive <= 0:
        raise RuntimeError(f"No fitting positives in {spec['cell_id']}")
    expected_class_weights = {
        "0": 1.0,
        "1": fit_negative / fit_positive,
    }
    recorded_class_weights = manifest.get("actual_class_weights", {})
    if (
        fit_positive + fit_negative != fit_count
        or set(recorded_class_weights) != {"0", "1"}
        or any(
            not math.isclose(
                float(recorded_class_weights[key]),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key, value in expected_class_weights.items()
        )
    ):
        raise RuntimeError(f"Fitting-side class weights mismatch: {spec['cell_id']}")
    expected_steps = int(
        np.ceil(fit_count / int(protocol["training"]["train_batch_size"]))
    )
    if (
        len(history) != 5
        or history["epoch"].astype(int).tolist() != [1, 2, 3, 4, 5]
        or (history["rows_seen"].astype(int) != fit_count).any()
        or (history["optimizer_steps"].astype(int) != expected_steps).any()
        or not np.isfinite(history["training_loss"].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"Training history validation failed: {spec['cell_id']}")

    audit = read_json(Path(spec["audit_path"]))
    if (
        audit.get("cell_id") != spec["cell_id"]
        or audit.get("method") != spec["method"]
        or float(audit.get("elapsed_seconds", 0)) <= 0
        or int(audit.get("optimizer_parameter_count", 0)) <= 0
        or int(audit.get("optimizer_parameter_elements", 0)) <= 0
    ):
        raise RuntimeError(
            f"Treatment audit identity/content mismatch: {spec['cell_id']}"
        )
    if spec["method"] in {"partial_4", "partial_2"}:
        expected_layers = 4 if spec["method"] == "partial_4" else 2
        if (
            audit.get("last_trainable_layers") != expected_layers
            or audit.get("frozen_parameters_unchanged") is not True
            or audit.get("frozen_parameters_have_no_gradients") is not True
            or audit.get("trainable_parameters_finite") is not True
            or audit.get("optimizer_parameter_elements")
            != manifest.get("optimizer_parameter_elements")
        ):
            raise RuntimeError(f"Partial fine-tuning audit failed: {spec['cell_id']}")
    if (
        spec["method"] == "nested"
        and audit.get("custom_split_manifest_sha256")
        != spec["fold_manifest_sha256"]
    ):
        raise RuntimeError(f"Nested split audit mismatch: {spec['cell_id']}")


def _run_cell(
    spec: dict[str, Any],
    protocol: dict[str, Any],
    frame: pd.DataFrame,
    inventory: dict[str, Any],
    budget: dict[str, Any],
    timeout: int,
) -> None:
    run = Path(spec["run_path"])
    spec_path = STUDY_ROOT / "cell_specs" / f"{sha256_text(spec['cell_id'])}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    if spec_path.exists():
        if read_json(spec_path) != spec:
            raise RuntimeError(f"Frozen cell specification changed: {spec_path}")
    else:
        write_json(spec_path, spec)
    stdout_path = STUDY_ROOT / "logs" / f"{sha256_text(spec['cell_id'])}.stdout.log"
    stderr_path = STUDY_ROOT / "logs" / f"{sha256_text(spec['cell_id'])}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    record = inventory.setdefault("cells", {}).get(spec["cell_id"])
    if run.exists():
        try:
            _validate_cell(spec, protocol, frame)
        except Exception:
            record["status"] = "FAILED_VALIDATION"
            _record_cell_time(budget, spec, record, completed=False)
            _write_inventory(inventory)
            raise
        if record is None:
            record = {
                "cell_id": spec["cell_id"],
                "status": "COMPLETE",
                "spec_sha256": sha256_file(spec_path),
                "run_path": str(run),
                "reused_existing_complete_run": True,
                "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
            }
            inventory["cells"][spec["cell_id"]] = record
            inventory["gpu_cells_started"] = (
                int(inventory.get("gpu_cells_started", 0)) + 1
            )
            inventory["gpu_cells_completed"] = (
                int(inventory.get("gpu_cells_completed", 0)) + 1
            )
        elif record.get("status") not in {"COMPLETE", "IN_PROGRESS"}:
            raise RuntimeError(
                f"Refusing to reuse cell with state {record.get('status')}: "
                f"{spec['cell_id']}"
            )
        else:
            was_complete = record.get("status") == "COMPLETE"
            record["status"] = "COMPLETE"
            record["reused_existing_complete_run"] = True
            record["run_manifest_sha256"] = sha256_file(run / "run_manifest.json")
            if not was_complete:
                inventory["gpu_cells_completed"] = (
                    int(inventory.get("gpu_cells_completed", 0)) + 1
                )
                if "elapsed_seconds" in record:
                    _record_cell_time(budget, spec, record, completed=True)
        _write_inventory(inventory)
        return
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(
            f"Prior attempt logs exist without a completed immutable run; "
            f"no retry will be attempted: {spec['cell_id']}"
        )
    run.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(spec["audit_path"])
    if audit_path.exists():
        raise FileExistsError(
            f"Prior treatment audit exists without a complete run; no retry will "
            f"be attempted: {audit_path}"
        )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cell_id": spec["cell_id"],
        "status": "IN_PROGRESS",
        "spec_sha256": sha256_file(spec_path),
        "run_path": str(run),
        "started_at": datetime.now().astimezone().isoformat(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    inventory["cells"][spec["cell_id"]] = record
    _write_inventory(inventory)
    try:
        frame = _verify_protocol_inputs(protocol)
        gpu = _gpu_preflight()
    except Exception:
        record["status"] = "BLOCKED_PRECHECK"
        _write_inventory(inventory)
        raise
    record["gpu_preflight"] = gpu
    record["status"] = "IN_PROGRESS"
    inventory["gpu_cells_started"] = int(inventory.get("gpu_cells_started", 0)) + 1
    budget.setdefault(
        "actual",
        {
            "gpu_cells_started": 0,
            "gpu_cells_completed": 0,
            "gpu_cell_seconds": 0.0,
            "seconds_by_stage": {},
            "cell_ids": {},
        },
    )
    budget["actual"]["gpu_cells_started"] += 1
    _save_budget(budget)
    _write_inventory(inventory)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    conda_library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
    environment["PATH"] = (
        str(conda_library_bin) + os.pathsep + environment.get("PATH", "")
    )
    command = [
        sys.executable,
        "-m",
        "experiments.transformer_final.recall_improvement_study_v1.cell_launcher",
        "--spec",
        str(spec_path.resolve()),
    ]
    started = time.monotonic()
    try:
        with (
            stdout_path.open("x", encoding="utf-8") as stdout,
            stderr_path.open("x", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        record["status"] = "FAILED_TIMEOUT"
        record["elapsed_seconds"] = time.monotonic() - started
        record["timeout_seconds"] = timeout
        _record_cell_time(budget, spec, record, completed=False)
        _write_inventory(inventory)
        raise TimeoutError(
            f"Study cell {spec['cell_id']} exceeded {timeout} seconds; no retry "
            f"will be attempted"
        ) from exc
    record["elapsed_seconds"] = time.monotonic() - started
    record["return_code"] = result.returncode
    if result.returncode != 0:
        record["status"] = "FAILED"
        _record_cell_time(budget, spec, record, completed=False)
        _write_inventory(inventory)
        raise RuntimeError(
            f"Study cell {spec['cell_id']} failed with exit code "
            f"{result.returncode}; see {stderr_path}"
        )
    _validate_cell(spec, protocol, frame)
    record["status"] = "COMPLETE"
    record["completed_at"] = datetime.now().astimezone().isoformat()
    record["run_manifest_sha256"] = sha256_file(run / "run_manifest.json")
    inventory["gpu_cells_completed"] = int(inventory.get("gpu_cells_completed", 0)) + 1
    _record_cell_time(budget, spec, record, completed=True)
    _write_inventory(inventory)


def _record_cell_time(
    budget: dict[str, Any],
    spec: dict[str, Any],
    record: dict[str, Any],
    *,
    completed: bool,
) -> None:
    actual = budget["actual"]
    if spec["cell_id"] in actual["cell_ids"]:
        return
    elapsed = float(record["elapsed_seconds"])
    actual["cell_ids"][spec["cell_id"]] = {
        "stage": spec["cell_id"].split("/", 1)[0],
        "elapsed_seconds": elapsed,
        "completed": completed,
    }
    actual["gpu_cell_seconds"] += elapsed
    stage = spec["cell_id"].split("/", 1)[0]
    actual["seconds_by_stage"][stage] = (
        actual["seconds_by_stage"].get(stage, 0.0) + elapsed
    )
    if completed:
        actual["gpu_cells_completed"] += 1
    _save_budget(budget)


def run_study() -> None:
    if not PROTOCOL_PATH.is_file() or not SPLITS_PATH.is_file():
        raise FileNotFoundError(
            "Freeze the study protocol and nested splits before running GPU cells"
        )
    protocol = read_json(PROTOCOL_PATH)
    inventory = read_json(INVENTORY_PATH)
    protocol_digest = sha256_file(PROTOCOL_PATH)
    if inventory.get("protocol_sha256") != protocol_digest:
        raise RuntimeError("Study protocol does not match its inventory-pinned hash")
    frame = _verify_protocol_inputs(protocol)
    budget = read_json(TIME_BUDGET_PATH)
    budget.setdefault(
        "actual",
        {
            "gpu_cells_started": 0,
            "gpu_cells_completed": 0,
            "gpu_cell_seconds": 0.0,
            "seconds_by_stage": {},
            "cell_ids": {},
        },
    )
    _save_budget(budget)
    deadline = _deadline(budget["authorization_window_deadline"])
    if inventory.get("overall_status") not in {"IN_PROGRESS", "PAUSED"}:
        raise RuntimeError("Study inventory is not in a runnable state")
    inventory["stages"]["A_family_recall_diagnostics"]["status"] = "COMPLETE"
    inventory["study_started_at"] = inventory.get(
        "study_started_at", datetime.now().astimezone().isoformat()
    )
    _reset_time_skipped_stages(inventory)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    if len(splits) != 15:
        raise RuntimeError("Frozen nested split file must contain exactly 15 cells")
    _write_inventory(inventory)
    completed_stages: list[str] = []
    for index, stage in enumerate(STAGE_ORDER):
        if stage == "E_error_analysis":
            if inventory["stages"][stage]["status"] == "COMPLETE":
                if not (
                    STUDY_ROOT
                    / "analysis"
                    / "error_analysis"
                    / "analysis_manifest.json"
                ).is_file():
                    raise RuntimeError(
                        "Error-analysis stage is marked complete but its manifest is missing"
                    )
                completed_stages.append(stage)
                continue
            from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
                build_error_analysis,
            )

            build_error_analysis()
            inventory["stages"][stage]["status"] = "COMPLETE"
            completed_stages.append(stage)
            _write_inventory(inventory)
            continue
        if inventory["stages"][stage]["status"] == "SKIPPED_TIME_BUDGET":
            continue
        if inventory["stages"][stage]["status"] == "COMPLETE":
            completed_stages.append(stage)
            specs = _cells_for_stage(stage, protocol, splits)
            for spec in specs:
                _validate_cell(spec, protocol, frame)
            if (
                stage == "B_nested_threshold_calibration"
                and not (
                    STUDY_ROOT
                    / "analysis"
                    / "nested_threshold_calibration"
                    / "analysis_manifest.json"
                ).is_file()
            ):
                from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
                    build_nested_threshold_analysis,
                )

                build_nested_threshold_analysis()
            continue
        specs = _cells_for_stage(stage, protocol, splits)
        per_cell_estimate = STAGE_ESTIMATES[stage] / len(specs)
        pending_count = sum(not Path(spec["run_path"]).exists() for spec in specs)
        remaining = _remaining_seconds(deadline)
        reserve = int(
            budget["planning_estimates"]["validation_and_reporting"]["seconds"]
        )
        if stage in {
            "B_nested_threshold_calibration",
            "C_local_context",
            "G_partial_4",
        }:
            future_mandatory = sum(
                STAGE_ESTIMATES[value]
                for value in STAGE_ORDER[index + 1 :]
                if value
                in {
                    "B_nested_threshold_calibration",
                    "C_local_context",
                    "G_partial_4",
                }
            )
            required = (
                int(pending_count * per_cell_estimate) + future_mandatory + reserve
            )
        else:
            required = int(pending_count * per_cell_estimate) + reserve
        if remaining <= required:
            for pending in STAGE_ORDER[index:]:
                if (
                    pending != "E_error_analysis"
                    and inventory["stages"].get(pending, {}).get("status") != "COMPLETE"
                ):
                    inventory["stages"][pending]["status"] = "SKIPPED_TIME_BUDGET"
            inventory["overall_status"] = "PAUSED"
            inventory["budget_stop_remaining_seconds"] = max(0, int(remaining))
            _write_inventory(inventory)
            continue
        inventory["stages"][stage]["status"] = "IN_PROGRESS"
        _write_inventory(inventory)
        stage_paused = False
        for cell_index, spec in enumerate(specs):
            remaining = _remaining_seconds(deadline)
            remaining_stage_estimate = int(
                sum(not Path(item["run_path"]).exists() for item in specs[cell_index:])
                * per_cell_estimate
            )
            if stage in {
                "B_nested_threshold_calibration",
                "C_local_context",
                "G_partial_4",
            }:
                future_mandatory = sum(
                    STAGE_ESTIMATES[value]
                    for value in STAGE_ORDER[index + 1 :]
                    if value
                    in {
                        "B_nested_threshold_calibration",
                        "C_local_context",
                        "G_partial_4",
                    }
                )
            else:
                future_mandatory = 0
            if remaining <= remaining_stage_estimate + future_mandatory + reserve:
                inventory["stages"][stage]["status"] = "INCOMPLETE_TIME_BUDGET"
                inventory["budget_stop_remaining_seconds"] = max(0, int(remaining))
                for pending in STAGE_ORDER[index + 1 :]:
                    if (
                        pending != "E_error_analysis"
                        and inventory["stages"].get(pending, {}).get("status")
                        != "COMPLETE"
                    ):
                        inventory["stages"][pending]["status"] = "SKIPPED_TIME_BUDGET"
                stage_paused = True
                _write_inventory(inventory)
                break
            cell_timeout = min(
                CELL_TIMEOUT_SECONDS,
                max(
                    1,
                    int(
                        remaining
                        - reserve
                        - sum(
                            STAGE_ESTIMATES[value]
                            for value in STAGE_ORDER[index + 1 :]
                            if value
                            in {
                                "B_nested_threshold_calibration",
                                "C_local_context",
                                "G_partial_4",
                            }
                        )
                    ),
                ),
            )
            _run_cell(
                spec,
                protocol,
                frame,
                inventory,
                budget,
                cell_timeout,
            )
        if stage_paused:
            inventory["overall_status"] = "PAUSED"
            _write_inventory(inventory)
            continue
        inventory["stages"][stage]["status"] = "COMPLETE"
        completed_stages.append(stage)
        _write_inventory(inventory)
        if stage == "B_nested_threshold_calibration":
            from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
                build_nested_threshold_analysis,
            )

            build_nested_threshold_analysis()

    if all(
        inventory["stages"].get(stage, {}).get("status") == "COMPLETE"
        for stage in STAGE_ORDER
    ):
        inventory["overall_status"] = "COMPLETE"
    else:
        inventory["overall_status"] = "PAUSED"
    inventory["completed_stages"] = completed_stages
    from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
        build_final_analysis,
    )

    if not (STUDY_ROOT / "analysis" / "final").exists():
        build_final_analysis()
    inventory["study_finished_at"] = datetime.now().astimezone().isoformat()
    _write_inventory(inventory)


if __name__ == "__main__":
    run_study()
