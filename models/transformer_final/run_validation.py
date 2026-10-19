"""Integrity and protocol checks for one completed Experiment 1 cell."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from safetensors import safe_open

from .common import (
    hash_directory,
    read_json,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
)
from .config import public_config


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_completed_run(
    *,
    run: Path,
    condition: str,
    fold: int,
    seed: int,
    config: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    effective: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    dataset_dir: Path,
    model_source_sha256: str,
    model_file_hashes: dict[str, str],
    legacy_protocol_compatible: bool = False,
) -> pd.DataFrame:
    run = run.resolve()
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if (
        manifest.get("condition") != condition
        or int(manifest.get("fold", 0)) != fold
        or int(manifest.get("seed", 0)) != seed
    ):
        raise RuntimeError(f"Run identity mismatch: {run}")
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise RuntimeError(f"Base configuration hash mismatch: {run}")
    if manifest.get("protocol_id") not in {None, protocol["protocol_id"]}:
        raise RuntimeError(f"Run protocol identity mismatch: {run}")
    if manifest.get("protocol_sha256") not in {None, protocol_sha256}:
        raise RuntimeError(f"Run protocol hash mismatch: {run}")
    if manifest.get("protocol_sha256") is None and not legacy_protocol_compatible:
        raise RuntimeError(f"Run has no frozen-protocol provenance: {run}")
    if (
        manifest.get("original_dataset_sha256") != protocol["dataset"]["source_sha256"]
        or manifest.get("effective_dataset_sha256")
        != dataset_manifest["effective_dataset_sha256"]
        or manifest.get("effective_dataset_sha256")
        != protocol["dataset"]["effective_dataset_sha256"]
        or manifest.get("dataset_manifest_sha256")
        != protocol["dataset"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError(f"Dataset identity mismatch: {run}")
    if (
        manifest.get("model_repository") != protocol["model"]["repository"]
        or manifest.get("model_revision") != protocol["model"]["revision"]
        or manifest.get("model_source_sha256") != model_source_sha256
        or manifest.get("model_source_sha256") != protocol["model"]["source_sha256"]
        or manifest.get("model_file_hashes") != model_file_hashes
        or manifest.get("model_path") != protocol["model"]["local_path"]
    ):
        raise RuntimeError(f"Pretrained model identity mismatch: {run}")
    fold_manifest_path = dataset_dir / "folds" / f"fold_{fold}.json"
    if (
        manifest.get("fold_manifest_sha256")
        != protocol["dataset"]["fold_manifest_sha256"][str(fold)]
        or sha256_file(fold_manifest_path)
        != manifest.get("fold_manifest_sha256")
    ):
        raise RuntimeError(f"Fold manifest hash mismatch: {run}")
    if (
        manifest.get("precision") != "fp32"
        or manifest.get("loss") != config["conditions"][condition]
        or manifest.get("input_mode") != "target_only"
        or manifest.get("epoch_policy") != "fixed_5_no_early_stopping"
        or int(manifest.get("epochs_completed", 0)) != 5
        or int(manifest.get("outer_fold_evaluations_during_training", -1)) != 0
        or int(manifest.get("outer_fold_evaluations_after_training", -1)) != 1
        or manifest.get("resume_checkpoint") is not None
    ):
        raise RuntimeError(f"Training protocol mismatch: {run}")
    if manifest.get("protocol_sha256") is None:
        saved_versions = {
            "python": manifest.get("python"),
            "torch": manifest.get("torch"),
            "transformers": manifest.get("transformers"),
            "numpy": manifest.get("numpy"),
            "pandas": manifest.get("pandas"),
            "cuda_runtime": manifest.get("cuda_runtime"),
            "cudnn": manifest.get("cudnn"),
            "gpu": manifest.get("gpu"),
        }
        expected_environment = protocol["environment"]
        if any(
            saved_versions[key] != expected_environment[key]
            for key in saved_versions
        ):
            raise RuntimeError(f"Legacy run environment differs from FP32 protocol: {run}")
    else:
        source_hashes = manifest.get("source_file_hashes")
        if (
            not isinstance(source_hashes, dict)
            or source_hashes != source_file_hashes()
            or manifest.get("source_files_sha256")
            != sha256_text(stable_json(source_hashes))
            or manifest.get("python_executable")
            != protocol["environment"]["interpreter"]
            or manifest.get("forward_autocast") != "disabled"
            or manifest.get("cross_entropy_input_dtype") != "float32"
            or manifest.get("cuda_driver") != protocol["environment"]["driver"]
            or manifest.get("dependency_versions")
            != {
                "torch": protocol["environment"]["torch"],
                "transformers": protocol["environment"]["transformers"],
                "numpy": protocol["environment"]["numpy"],
                "pandas": protocol["environment"]["pandas"],
                "scikit-learn": protocol["environment"]["scikit-learn"],
                "tokenizers": protocol["environment"]["tokenizers"],
                "sentencepiece": protocol["environment"]["sentencepiece"],
                "safetensors": protocol["environment"]["safetensors"],
            }
        ):
            raise RuntimeError(f"Run source or dependency provenance mismatch: {run}")

    run_config = read_json(run / "run_config.json")
    if stable_json(run_config.get("base_configuration")) != stable_json(
        public_config(config)
    ):
        raise RuntimeError(f"Embedded base configuration mismatch: {run}")
    resolved = run_config.get("resolved", {})
    if (
        resolved.get("condition") != condition
        or int(resolved.get("fold", 0)) != fold
        or int(resolved.get("seed", 0)) != seed
        or resolved.get("precision") != "fp32"
        or resolved.get("input_mode") != "target_only"
    ):
        raise RuntimeError(f"Resolved run configuration mismatch: {run}")

    artifact_hashes = manifest.get("artifact_hashes", {})
    if not artifact_hashes:
        raise RuntimeError(f"Run has no artifact hashes: {run}")
    for artifact_name, metadata in artifact_hashes.items():
        artifact_path = run / metadata["path"]
        if (
            not _is_inside(artifact_path, run)
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != metadata.get("sha256")
        ):
            raise RuntimeError(
                f"Artifact hash mismatch ({artifact_name}): {artifact_path}"
            )

    checkpoint_dir = run / "checkpoints" / "final_epoch"
    checkpoint_metadata = read_json(run / "checkpoints" / "checkpoint_metadata.json")
    checkpoint_hash, checkpoint_files = hash_directory(checkpoint_dir)
    if (
        checkpoint_metadata.get("epoch") != 5
        or checkpoint_metadata.get("files") != checkpoint_files
        or checkpoint_metadata.get("sha256") != checkpoint_hash
    ):
        raise RuntimeError(f"Final checkpoint hash mismatch: {run}")
    checkpoint_dtypes: set[str] = set()
    with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt", device="cpu") as file:
        checkpoint_dtypes = {file.get_slice(key).get_dtype() for key in file.keys()}
    if checkpoint_dtypes != {"F32"}:
        raise RuntimeError(
            f"Final checkpoint is not entirely FP32 ({checkpoint_dtypes}): {run}"
        )
    if manifest.get("parameter_dtypes") not in (
        None,
        {"torch.float32": sum(manifest.get("parameter_dtypes", {}).values())},
    ):
        raise RuntimeError(f"Recorded model parameter dtype is not FP32: {run}")

    fold_manifest = read_json(fold_manifest_path)
    fitting_ids = set(fold_manifest["fitting_record_ids"])
    scoring_ids = set(fold_manifest["scoring_record_ids"])
    if (
        int(manifest.get("fitting_rows", -1)) != len(fitting_ids)
        or int(manifest.get("held_out_rows", -1)) != len(scoring_ids)
        or manifest.get("held_out_families") != fold_manifest["held_out_families"]
    ):
        raise RuntimeError(f"Fold row counts or families mismatch: {run}")
    fitting = effective.loc[effective["record_id"].astype(str).isin(fitting_ids)]
    if condition == "A1":
        expected_weights = {"0": 1.0, "1": 1.0}
    else:
        positive = int(fitting["label"].sum())
        negative = len(fitting) - positive
        if positive == 0 or negative == 0:
            raise RuntimeError(f"Invalid A2 fitting labels: {run}")
        expected_weights = {"0": 1.0, "1": negative / positive}
    if manifest.get("actual_class_weights") != expected_weights:
        raise RuntimeError(f"Class weights do not match fitting partition: {run}")

    expected_steps = math.ceil(
        len(fitting_ids)
        / int(config["training"]["train_batch_size"])
        / int(config["training"]["gradient_accumulation_steps"])
    )
    history = pd.read_csv(run / "history" / "training_history.csv")
    if (
        len(history) != 5
        or history["epoch"].astype(int).tolist() != [1, 2, 3, 4, 5]
        or history["optimizer_steps"].astype(int).tolist()
        != [expected_steps] * 5
        or history["rows_seen"].astype(int).tolist() != [len(fitting_ids)] * 5
    ):
        raise RuntimeError(f"Training history is incomplete or inconsistent: {run}")

    prediction_path = run / "predictions" / "held_out_predictions.csv"
    prediction_hash = artifact_hashes.get("held_out_predictions", {})
    if (
        prediction_hash.get("path") != "predictions/held_out_predictions.csv"
        or prediction_hash.get("sha256") != sha256_file(prediction_path)
    ):
        raise RuntimeError(f"Held-out prediction hash mismatch: {run}")
    frame = pd.read_csv(
        prediction_path,
        dtype={"item_id": str, "record_id": str},
        keep_default_na=False,
    )
    frame["label"] = frame["label"].astype(int)
    for field in ("logit_no", "logit_yes", "probability_no", "probability_yes"):
        frame[field] = frame[field].astype(float)
    expected_fold = effective.loc[
        effective["fold"].astype(int) == fold,
        ["record_id", "family_id", "label"],
    ].copy()
    expected_fold["record_id"] = expected_fold["record_id"].astype(str)
    if (
        frame["record_id"].duplicated().any()
        or set(frame["record_id"].astype(str)) != scoring_ids
        or set(frame["record_id"].astype(str)) != set(expected_fold["record_id"])
    ):
        raise RuntimeError(f"Prediction records do not match held-out fold: {run}")
    aligned = frame[["record_id", "family_id", "label"]].merge(
        expected_fold,
        on="record_id",
        suffixes=("_run", "_dataset"),
        validate="one_to_one",
    )
    if not (
        (aligned["family_id_run"] == aligned["family_id_dataset"]).all()
        and (aligned["label_run"] == aligned["label_dataset"]).all()
    ):
        raise RuntimeError(f"Prediction labels/families do not align: {run}")
    return frame
