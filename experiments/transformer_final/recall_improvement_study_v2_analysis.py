"""Validate completed V2 outputs and create an immutable consolidated report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.transformer_final.recall_improvement_study_v2.prepare_v2 import (
    V2_COMPATIBILITY_PATH,
    V2_INVENTORY_PATH,
    V2_PROTOCOL_PATH,
    V2_ROOT,
)
from models.transformer_final.common import (
    hash_directory,
    immutable_directory,
    read_json,
    resolve_path,
    sha256_file,
    sha256_text,
    stable_json,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import TOP_K_VALUES
from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
    _choose_threshold,
    _deployed_metric_row,
)

BASELINE_ROOT = resolve_path(
    "results/transformer/final_studies/final_annotation_ready_pilot_v1/aggregate_metrics"
)
METHODS = {
    "target_only_full": "Full fine-tuning - target-only",
    "local_context": "Full fine-tuning - local context",
    "partial_4": "Partial-4 fine-tuning",
    "focal_loss": "Focal loss",
    "partial_2": "Partial-2 fine-tuning",
}


def _verify_reused_run(
    record: dict[str, Any], expected_cell_id: str
) -> tuple[Path, dict[str, Any]]:
    if (
        record.get("v1_cell_id") != expected_cell_id
        or record.get("v2_status") != "REUSED_BY_REFERENCE_AFTER_COMPATIBILITY_CHECK"
    ):
        raise RuntimeError(f"Invalid V1 reuse record: {expected_cell_id}")
    run = Path(record["v1_run_path"]).resolve()
    manifest_path = run / "run_manifest.json"
    if (
        sha256_file(manifest_path) != record["v1_run_manifest_sha256"]
        or not run.is_dir()
    ):
        raise RuntimeError(f"V1 run-manifest digest mismatch: {expected_cell_id}")
    manifest = read_json(manifest_path)
    for artifact in record["v1_artifact_hashes"].values():
        path = (run / artifact["path"]).resolve()
        if run not in path.parents or not path.is_file():
            raise RuntimeError(f"V1 artifact missing or outside run: {expected_cell_id}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"V1 artifact digest mismatch: {expected_cell_id}")
    return run, manifest


def _verify_v2_run(
    *,
    run: Path,
    cell_id: str,
    method: str,
    fold: int,
    protocol: dict[str, Any],
    expected_manifest_sha256: str,
    attempt_id: str | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    run = run.resolve()
    manifest_path = run / "run_manifest.json"
    if not run.is_dir() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError(f"V2 run manifest missing or changed: {cell_id}")
    manifest = read_json(manifest_path)
    expected_mode = "local_context" if method == "context" else "target_only"
    expected_full_fine_tuning = method not in {"partial_4", "partial_2"}
    if (
        manifest.get("condition") != "A2"
        or int(manifest.get("fold", 0)) != fold
        or int(manifest.get("seed", 0)) != int(protocol["training"]["seed"])
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != expected_mode
        or manifest.get("epochs_completed") != 5
        or manifest.get("config_sha256") != protocol["config_sha256"]
        or manifest.get("effective_dataset_sha256")
        != protocol["dataset"]["effective_dataset_sha256"]
        or manifest.get("dataset_manifest_sha256")
        != protocol["dataset"]["dataset_manifest_sha256"]
        or manifest.get("fold_manifest_sha256")
        != protocol["dataset"]["fold_manifest_sha256"][str(fold)]
        or manifest.get("model_source_sha256") != protocol["model"]["source_sha256"]
        or manifest.get("model_file_hashes") != protocol["model"]["model_file_hashes"]
        or manifest.get("source_file_hashes")
        != protocol["source_snapshot"]["core_files"]
        or manifest.get("full_encoder_fine_tuned") is not expected_full_fine_tuning
        or manifest.get("study_protocol_sha256")
        != sha256_file(V2_PROTOCOL_PATH)
        or manifest.get("study_cell_id") != cell_id
    ):
        raise RuntimeError(f"V2 run provenance mismatch: {cell_id}")
    if attempt_id and manifest.get("study_attempt_id") != attempt_id:
        raise RuntimeError(f"V2 attempt identity mismatch: {cell_id}")
    if method == "focal" and (
        manifest.get("loss") != "focal_gamma_2"
        or manifest.get("focal_gamma") != 2.0
    ):
        raise RuntimeError(f"Focal-loss settings mismatch: {cell_id}")
    if method in {"partial_4", "partial_2"}:
        layers = 4 if method == "partial_4" else 2
        if (
            manifest.get("fine_tuning_policy") != method
            or manifest.get("trainable_last_layers") != layers
        ):
            raise RuntimeError(f"Partial fine-tuning settings mismatch: {cell_id}")
        if audit_path is None:
            raise RuntimeError(f"Partial-treatment audit path missing: {cell_id}")
        audit = read_json(audit_path)
        if (
            audit.get("frozen_parameters_unchanged") is not True
            or audit.get("frozen_parameters_have_no_gradients") is not True
            or audit.get("trainable_parameters_finite") is not True
            or audit.get("last_trainable_layers") != layers
        ):
            raise RuntimeError(f"Partial-treatment audit failed: {cell_id}")
    artifacts = manifest.get("artifact_hashes", {})
    if not artifacts:
        raise RuntimeError(f"V2 run does not list artifact hashes: {cell_id}")
    for artifact in artifacts.values():
        path = (run / artifact["path"]).resolve()
        if run not in path.parents or not path.is_file():
            raise RuntimeError(f"V2 artifact missing or outside run: {cell_id}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"V2 artifact digest mismatch: {cell_id}")
    checkpoint = run / "checkpoints" / "final_epoch"
    checkpoint_hash, checkpoint_files = hash_directory(checkpoint)
    checkpoint_metadata = read_json(run / "checkpoints" / "checkpoint_metadata.json")
    if (
        checkpoint_metadata.get("epoch") != 5
        or checkpoint_metadata.get("sha256") != checkpoint_hash
        or checkpoint_metadata.get("files") != checkpoint_files
    ):
        raise RuntimeError(f"V2 checkpoint validation failed: {cell_id}")
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as tensors:
        if {
            tensors.get_slice(key).get_dtype() for key in tensors.keys()
        } != {"F32"}:
            raise RuntimeError(f"V2 checkpoint is not FP32: {cell_id}")
    return manifest


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_ids: set[str],
    expected_fold: int | None,
    effective: pd.DataFrame,
    identity: str,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    frame["probability_yes"] = frame["probability_yes"].astype(float)
    if (
        frame["record_id"].duplicated().any()
        or set(frame["record_id"]) != expected_ids
        or not np.isfinite(frame["probability_yes"].to_numpy()).all()
        or frame["probability_yes"].lt(0).any()
        or frame["probability_yes"].gt(1).any()
        or (
            expected_fold is not None
            and not frame["fold"].eq(expected_fold).all()
        )
    ):
        raise RuntimeError(f"Prediction coverage or values invalid: {identity}")
    verified = frame.merge(
        effective[["record_id", "label", "fold", "family_id"]],
        on="record_id",
        suffixes=("", "_dataset"),
        validate="one_to_one",
    )
    if (
        not verified["label"].eq(verified["label_dataset"]).all()
        or not verified["fold"].eq(verified["fold_dataset"]).all()
    ):
        raise RuntimeError(f"Prediction labels/folds differ from dataset: {identity}")
    frame["family_id"] = verified["family_id"].astype(str).to_numpy()
    frame["fold"] = verified["fold_dataset"].astype(int).to_numpy()
    frame["label"] = verified["label_dataset"].astype(int).to_numpy()
    return frame


def _read_predictions(
    *,
    run: Path,
    expected_ids: set[str],
    expected_fold: int | None,
    effective: pd.DataFrame,
    identity: str,
) -> pd.DataFrame:
    predictions = pd.read_csv(
        run / "predictions" / "held_out_predictions.csv",
        dtype={"record_id": str},
        keep_default_na=False,
    )
    return _validate_predictions(
        predictions,
        expected_ids=expected_ids,
        expected_fold=expected_fold,
        effective=effective,
        identity=identity,
    )


def _collect_nested_thresholds(
    *,
    protocol: dict[str, Any],
    compatibility: dict[str, Any],
    effective: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float], dict[str, str]]:
    reused = {record["v1_cell_id"]: record for record in compatibility["reusable_v1_runs"]}
    splits = json.loads(
        Path(protocol["dataset"]["study_nested_splits_path"]).read_text(
            encoding="utf-8"
        )
    )
    if sha256_file(Path(protocol["dataset"]["study_nested_splits_path"])) != (
        protocol["dataset"]["study_nested_splits_sha256"]
    ):
        raise RuntimeError("V2 nested-split manifest digest mismatch")
    all_oof: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    thresholds: dict[int, float] = {}
    manifest_hashes: dict[str, str] = {}
    for outer_fold in range(1, 6):
        expected_ids = set(
            effective.loc[effective["fold"] != outer_fold, "record_id"].astype(str)
        )
        outer_frames: list[pd.DataFrame] = []
        seen: set[str] = set()
        for split in [item for item in splits if int(item["outer_fold"]) == outer_fold]:
            cell_id = (
                f"B_nested_threshold_calibration/outer_{outer_fold}_inner_"
                f"{int(split['inner_fold'])}"
            )
            record = reused.get(cell_id)
            if record is None:
                raise RuntimeError(f"Nested calibration run is not reusable: {cell_id}")
            run, manifest = _verify_reused_run(record, cell_id)
            if (
                manifest.get("fold") != int(split["inner_fold"])
                or manifest.get("fold_manifest_sha256")
                != protocol["dataset"]["fold_manifest_sha256"][
                    str(split["inner_fold"])
                ]
            ):
                raise RuntimeError(f"Nested run provenance mismatch: {cell_id}")
            split_ids = set(split["scoring_record_ids"])
            frame = _read_predictions(
                run=run,
                expected_ids=split_ids,
                expected_fold=None,
                effective=effective,
                identity=cell_id,
            )
            if seen & split_ids:
                raise RuntimeError(f"Nested OOF folds overlap: outer={outer_fold}")
            seen.update(split_ids)
            frame["outer_fold"] = outer_fold
            frame["inner_fold"] = int(split["inner_fold"])
            outer_frames.append(frame)
            manifest_hashes[cell_id] = record["v1_run_manifest_sha256"]
        if seen != expected_ids:
            raise RuntimeError(f"Nested OOF coverage mismatch: outer={outer_fold}")
        oof = pd.concat(outer_frames, ignore_index=True)
        selected = _choose_threshold(
            oof,
            [float(value) for value in protocol["inner_cv"]["threshold_candidates"]],
        )
        thresholds[outer_fold] = float(selected["threshold"])
        oof["outer_fold"] = outer_fold
        all_oof.append(oof)
        threshold_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_oof_rows": len(oof),
                "inner_precision": selected["precision"],
                "inner_recall": selected["recall"],
                "inner_f2": selected["f2"],
                "selected_threshold": selected["threshold"],
                "selection_policy": selected["selection_policy"],
            }
        )
    return (
        pd.concat(all_oof, ignore_index=True),
        pd.DataFrame(threshold_rows),
        thresholds,
        manifest_hashes,
    )


def _read_v2_method_predictions(
    *,
    stage: str,
    method: str,
    inventory: dict[str, Any],
    protocol: dict[str, Any],
    effective: pd.DataFrame,
    compatibility: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    records: dict[str, dict[str, Any]] = {}
    run_manifests: dict[str, str] = {}
    reused = {record["v1_cell_id"]: record for record in compatibility["reusable_v1_runs"]}
    for fold in range(1, 6):
        cell_id = f"{stage}/fold_{fold}/seed_{protocol['training']['seed']}"
        if method == "context" and fold < 3:
            reuse = reused.get(cell_id)
            if reuse is None:
                raise RuntimeError(f"Compatible V1 context run missing: {cell_id}")
            run, manifest = _verify_reused_run(reuse, cell_id)
            run_hash = reuse["v1_run_manifest_sha256"]
        elif method == "context" and fold == 3:
            record = inventory["authorized_attempts"]["C_local_context/fold_3/seed_2718"]
            if record.get("status") != "COMPLETE":
                raise RuntimeError("Authorized Fold 3 context attempt is incomplete")
            run = Path(record["run_path"])
            attempt_id = record["attempt_id"]
            manifest = _verify_v2_run(
                run=run,
                cell_id=cell_id,
                method=method,
                fold=fold,
                protocol=protocol,
                expected_manifest_sha256=record["run_manifest_sha256"],
                attempt_id=attempt_id,
            )
            run_hash = record["run_manifest_sha256"]
        else:
            record = inventory["pending_v2_cells"].get(cell_id)
            if record is None or record.get("status") != "COMPLETE":
                raise RuntimeError(f"V2 experiment cell is incomplete: {cell_id}")
            run = Path(record["run_path"])
            manifest = _verify_v2_run(
                run=run,
                cell_id=cell_id,
                method=method,
                fold=fold,
                protocol=protocol,
                expected_manifest_sha256=record["run_manifest_sha256"],
                audit_path=Path(record["audit_path"]),
            )
            run_hash = record["run_manifest_sha256"]
        if (
            int(manifest.get("fold", 0)) != fold
            or int(manifest.get("seed", 0)) != int(protocol["training"]["seed"])
        ):
            raise RuntimeError(f"Method run fold/seed mismatch: {cell_id}")
        ids = set(effective.loc[effective["fold"] == fold, "record_id"].astype(str))
        frame = _read_predictions(
            run=run,
            expected_ids=ids,
            expected_fold=fold,
            effective=effective,
            identity=cell_id,
        )
        frame["method"] = method
        records[cell_id] = frame.to_dict(orient="records")
        run_manifests[cell_id] = run_hash
    predictions = pd.DataFrame(
        [item for records_for_fold in records.values() for item in records_for_fold]
    )
    if predictions["record_id"].duplicated().any():
        raise RuntimeError(f"Duplicate OOF predictions for {method}")
    return predictions, run_manifests


def _read_baseline_predictions(
    effective: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    path = BASELINE_ROOT / "pooled_oof_predictions.csv"
    frame = pd.read_csv(path, dtype={"record_id": str}, keep_default_na=False)
    expected = set(effective["record_id"].astype(str))
    baseline = _validate_predictions(
        frame,
        expected_ids=expected,
        expected_fold=None,
        effective=effective,
        identity="full_fine_tuning_target_only",
    )
    return baseline, sha256_file(path)


def _metrics_at_threshold(
    predictions: pd.DataFrame, threshold: float, *, model: str
) -> dict[str, Any]:
    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["probability_yes"].to_numpy(dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    tn = int(np.count_nonzero(~predicted & (labels == 0)))
    fn = int(np.count_nonzero(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {
        "model": model,
        "threshold": threshold,
        "rows": len(predictions),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "predicted_positive_workload": tp + fp,
        "false_positives_per_1000_negatives": (
            1000 * fp / (fp + tn) if fp + tn else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _top_k_rows(predictions: pd.DataFrame, *, model: str) -> list[dict[str, Any]]:
    ranked = predictions.sort_values(
        ["probability_yes", "record_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    positives = int(predictions["label"].sum())
    prevalence = float(predictions["label"].mean())
    rows = []
    for k in TOP_K_VALUES:
        selected = ranked.head(k)
        retrieved = int(selected["label"].sum())
        rows.append(
            {
                "model": model,
                "k": int(k),
                "true_positives_retrieved": retrieved,
                "false_positives_at_k": len(selected) - retrieved,
                "precision_at_k": retrieved / len(selected) if len(selected) else 0.0,
                "recall_at_k": retrieved / positives if positives else 0.0,
                "lift_over_prevalence": (
                    retrieved / len(selected) / prevalence
                    if len(selected) and prevalence
                    else 0.0
                ),
            }
        )
    return rows


def build_report() -> Path:
    protocol = read_json(V2_PROTOCOL_PATH)
    inventory = read_json(V2_INVENTORY_PATH)
    compatibility = read_json(V2_COMPATIBILITY_PATH)
    if (
        sha256_file(V2_PROTOCOL_PATH) != inventory["protocol_sha256"]
        or sha256_file(V2_COMPATIBILITY_PATH)
        != inventory["compatibility_manifest_sha256"]
        or len(compatibility["reusable_v1_runs"]) != 17
        or compatibility["reused_runs_copied_or_modified"] is not False
    ):
        raise RuntimeError("V2 protocol or V1 reuse compatibility validation failed")
    for stage in ("C_local_context", "G_partial_4"):
        for fold in range(1, 6):
            if stage == "C_local_context" and fold == 3:
                continue
            cell = inventory["pending_v2_cells"][
                f"{stage}/fold_{fold}/seed_{protocol['training']['seed']}"
            ]
            if cell.get("status") != "COMPLETE":
                raise RuntimeError(f"Cannot report incomplete required stage {stage}")
    attempt = inventory["authorized_attempts"]["C_local_context/fold_3/seed_2718"]
    if attempt.get("status") != "COMPLETE":
        raise RuntimeError("Cannot report incomplete Fold 3 replacement attempt")
    optional_stages = (
        ("D_focal_loss", "focal"),
        ("G2_partial_2", "partial_2"),
    )
    include_optional: set[str] = set()
    incomplete_optional: set[str] = set()
    for stage, method in optional_stages:
        statuses = [
            inventory["pending_v2_cells"][
                f"{stage}/fold_{fold}/seed_{protocol['training']['seed']}"
            ]["status"]
            for fold in range(1, 6)
        ]
        if all(status == "COMPLETE" for status in statuses):
            include_optional.add(method)
        elif not all(
            status == "PENDING_SEPARATE_GPU_AUTHORIZATION"
            for status in statuses
        ):
            incomplete_optional.add(method)
    config = load_config(protocol["config_path"])
    effective, dataset_manifest, dataset_dir = load_effective_dataset(config)
    dataset = protocol["dataset"]
    if (
        sha256_file(dataset_dir / "manifest.json")
        != dataset["dataset_manifest_sha256"]
        or dataset_manifest["effective_dataset_sha256"]
        != dataset["effective_dataset_sha256"]
    ):
        raise RuntimeError("V2 analysis dataset identity mismatch")
    effective["record_id"] = effective["record_id"].astype(str)
    effective["label"] = effective["label"].astype(int)
    effective["fold"] = effective["fold"].astype(int)
    inner_oof, thresholds, selected_thresholds, nested_hashes = (
        _collect_nested_thresholds(
            protocol=protocol,
            compatibility=compatibility,
            effective=effective,
        )
    )
    baseline, baseline_hash = _read_baseline_predictions(effective)
    pooled: dict[str, pd.DataFrame] = {"target_only_full": baseline}
    run_hashes: dict[str, str] = {}
    for method, stage in (
        ("local_context", "C_local_context"),
        ("partial_4", "G_partial_4"),
        ("focal", "D_focal_loss"),
        ("partial_2", "G2_partial_2"),
    ):
        if method in {"focal", "partial_2"} and method not in include_optional:
            continue
        predictions, hashes = _read_v2_method_predictions(
            stage=stage,
            method=method,
            inventory=inventory,
            protocol=protocol,
            effective=effective,
            compatibility=compatibility,
        )
        display_key = {
            "local_context": "local_context",
            "partial_4": "partial_4",
            "focal": "focal_loss",
            "partial_2": "partial_2",
        }[method]
        pooled[display_key] = predictions
        run_hashes.update(hashes)

    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    top_k: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    deployed_predictions: list[pd.DataFrame] = []
    for model, predictions in pooled.items():
        metric_rows.append(_metrics_at_threshold(predictions, 0.5, model=model))
        selected = predictions.copy()
        selected["selected_threshold"] = selected["fold"].map(selected_thresholds)
        if selected["selected_threshold"].isna().any():
            raise RuntimeError(f"Missing nested threshold for method {model}")
        selected["predicted_label_nested_threshold"] = (
            selected["probability_yes"] >= selected["selected_threshold"]
        ).astype(int)
        deployed_predictions.append(selected)
        for fold, group in selected.groupby("fold", sort=True):
            row = _deployed_metric_row(
                group,
                model=model,
                operating_point="nested_fold_threshold",
                fold=int(fold),
            )
            row["selected_threshold"] = selected_thresholds[int(fold)]
            row["false_positives_per_1000_negatives"] = (
                1000 * row["false_positives"] / row["actual_no"]
                if row["actual_no"]
                else 0.0
            )
            threshold_rows.append(row)
        pooled_row = _deployed_metric_row(
            selected,
            model=model,
            operating_point="nested_fold_threshold",
        )
        pooled_row["selected_threshold"] = "fold_specific"
        pooled_row["false_positives_per_1000_negatives"] = (
            1000 * pooled_row["false_positives"] / pooled_row["actual_no"]
            if pooled_row["actual_no"]
            else 0.0
        )
        threshold_rows.append(pooled_row)
        top_k.extend(_top_k_rows(predictions, model=model))
        for op, threshold, deployed in (
            ("threshold_0.5", 0.5, predictions["probability_yes"] >= 0.5),
            (
                "nested_fold_threshold",
                "fold_specific",
                selected["predicted_label_nested_threshold"].astype(bool),
            ),
        ):
            labels = predictions["label"].to_numpy(dtype=np.int64)
            false_positives = int(np.count_nonzero(deployed.to_numpy() & (labels == 0)))
            true_negatives = int(np.count_nonzero(~deployed.to_numpy() & (labels == 0)))
            false_negatives = int(np.count_nonzero(~deployed.to_numpy() & (labels == 1)))
            workload_rows.append(
                {
                    "model": model,
                    "operating_point": op,
                    "threshold": threshold,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "predicted_positive_workload": int(deployed.sum()),
                    "false_positives_per_1000_negatives": (
                        1000 * false_positives / (false_positives + true_negatives)
                        if false_positives + true_negatives
                        else 0.0
                    ),
                }
            )

    nested_metric_frame = pd.DataFrame(threshold_rows)
    planned_comparison_rows: list[dict[str, Any]] = []
    comparisons = (
        ("local_context", "target_only_full", "Context vs target-only"),
        ("partial_4", "target_only_full", "Partial-4 vs full fine-tuning"),
    )
    for operating_point in ("threshold_0.5", "nested_fold_threshold"):
        source = (
            {row["model"]: row for row in metric_rows}
            if operating_point == "threshold_0.5"
            else {
                row["model"]: row
                for _, row in nested_metric_frame.loc[
                    nested_metric_frame["fold"].isna()
                ].iterrows()
            }
        )
        for treatment, reference, label in comparisons:
            measured = source.get(treatment)
            baseline_row = source.get(reference)
            if measured is None or baseline_row is None:
                raise RuntimeError(f"Missing planned comparison: {label}")
            planned_comparison_rows.append(
                {
                    "comparison": label,
                    "operating_point": operating_point,
                    "treatment": treatment,
                    "reference": reference,
                    "delta_precision": measured["precision"]
                    - baseline_row["precision"],
                    "delta_recall": measured["recall"] - baseline_row["recall"],
                    "delta_f2": measured["f2"] - baseline_row["f2"],
                    "delta_average_precision": measured["average_precision"]
                    - baseline_row["average_precision"],
                    "delta_roc_auc": measured["roc_auc"] - baseline_row["roc_auc"],
                    "delta_false_positives": measured["false_positives"]
                    - baseline_row["false_positives"],
                    "delta_predicted_positive_workload": measured[
                        "predicted_positive_workload"
                    ]
                    - baseline_row["predicted_positive_workload"],
                }
            )

    deployed = pd.concat(deployed_predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metric_rows)
    nested_rows = nested_metric_frame
    top_k_frame = pd.DataFrame(top_k)
    workload_frame = pd.DataFrame(workload_rows)
    planned_comparisons = pd.DataFrame(planned_comparison_rows)
    session_id = _budget_session_id()
    destination = V2_ROOT / "analysis" / f"final_authorized_{session_id}"
    with immutable_directory(destination) as output:
        metric_frame.to_csv(
            output / "metrics_at_threshold_0.5.csv",
            index=False,
            lineterminator="\n",
        )
        nested_rows.to_csv(
            output / "metrics_at_nested_thresholds.csv",
            index=False,
            lineterminator="\n",
        )
        inner_oof.to_csv(
            output / "nested_inner_oof_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        thresholds.to_csv(
            output / "nested_thresholds_by_outer_fold.csv",
            index=False,
            lineterminator="\n",
        )
        top_k_frame.to_csv(
            output / "top_k_metrics.csv", index=False, lineterminator="\n"
        )
        workload_frame.to_csv(
            output / "false_positive_workload.csv",
            index=False,
            lineterminator="\n",
        )
        planned_comparisons.to_csv(
            output / "planned_comparisons.csv",
            index=False,
            lineterminator="\n",
        )
        deployed.to_csv(
            output / "outer_oof_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        report = [
            "# Recall Improvement Study V2 - Consolidated Report",
            "",
            "## Protocol and evaluation",
            "",
            (
                "The V2 protocol, configuration, source snapshot, dataset, outer folds, "
                "and leakage audit were verified before reporting. The 15 nested "
                "calibration runs and two earlier local-context runs are reused by "
                "reference from V1; they were not copied or relabeled. Fold-specific "
                "thresholds were selected only from the nested inner OOF predictions "
                "for that outer fold, then evaluated on its untouched outer fold."
            ),
            "",
            f"- Protocol SHA-256: `{sha256_file(V2_PROTOCOL_PATH)}`",
            f"- Dataset SHA-256: `{dataset['effective_dataset_sha256']}`",
            f"- V1 compatible reused runs: {len(compatibility['reusable_v1_runs'])}",
            f"- Fold 3 replacement attempt: `{attempt['attempt_id']}`",
            f"- Threshold selection: `{protocol['inner_cv']['primary']}`",
            "",
            "## Pooled metrics at threshold 0.5",
            "",
            "| Method | Precision | Recall | F2 | AP | ROC-AUC | False positives | Review workload |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in metric_rows:
            report.append(
                f"| {METHODS[row['model']]} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f2']:.4f} | "
                f"{row['average_precision']:.4f} | {row['roc_auc']:.4f} | "
                f"{row['false_positives']} | {row['predicted_positive_workload']} |"
            )
        report.extend(
            [
                "",
                "## Pooled metrics with nested fold thresholds",
                "",
                "| Method | Precision | Recall | F2 | AP | ROC-AUC | False positives | FP / 1,000 negatives |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in nested_rows.loc[
            nested_rows["fold"].isna()
        ].iterrows():
            report.append(
                f"| {METHODS[row['model']]} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f2']:.4f} | "
                f"{row['average_precision']:.4f} | {row['roc_auc']:.4f} | "
                f"{row['false_positives']} | "
                f"{row['false_positives_per_1000_negatives']:.2f} |"
            )
        report.extend(
            [
                "",
                "## Per-fold selected thresholds",
                "",
                "| Outer fold | Inner OOF rows | Precision | Recall | F2 | Threshold | Selection policy |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in thresholds.to_dict(orient="records"):
            report.append(
                f"| {row['outer_fold']} | {row['inner_oof_rows']} | "
                f"{row['inner_precision']:.4f} | {row['inner_recall']:.4f} | "
                f"{row['inner_f2']:.4f} | {row['selected_threshold']:.4f} | "
                f"{row['selection_policy']} |"
            )
        report.extend(
            [
                "",
                "## Planned comparison deltas",
                "",
                "| Comparison | Operating point | Delta precision | Delta recall | Delta F2 | Delta AP | Delta ROC-AUC | Delta FP | Delta workload |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in planned_comparison_rows:
            report.append(
                f"| {row['comparison']} | {row['operating_point']} | "
                f"{row['delta_precision']:+.4f} | {row['delta_recall']:+.4f} | "
                f"{row['delta_f2']:+.4f} | {row['delta_average_precision']:+.4f} | "
                f"{row['delta_roc_auc']:+.4f} | {row['delta_false_positives']:+d} | "
                f"{row['delta_predicted_positive_workload']:+d} |"
            )
        report.extend(
            [
                "",
                "## Retrieval at review budgets",
                "",
                "| Method | K | Precision@K | Recall@K | FP@K | Lift |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in top_k:
            report.append(
                f"| {METHODS[row['model']]} | {row['k']} | "
                f"{row['precision_at_k']:.4f} | {row['recall_at_k']:.4f} | "
                f"{row['false_positives_at_k']} | "
                f"{row['lift_over_prevalence']:.2f} |"
            )
        report.extend(
            [
                "",
                "## Planned comparisons",
                "",
                (
                    "Context vs target-only compares `local_context` with "
                    "`target_only_full`; Partial-4 vs full fine-tuning compares "
                    "`partial_4` with `target_only_full`. Focal Loss and Partial-2 "
                    + (
                        "are included when their complete five-fold stages are present."
                        if not incomplete_optional
                        else "are omitted where the full five-fold stage did not complete."
                    )
                ),
                "",
                "## Interpretation limits",
                "",
                (
                    "Results are exploratory and use one seed over five family-aware "
                    "outer folds. AP and ROC-AUC are ranking metrics; they do not "
                    "change with the deployed threshold. Fold-specific precision, "
                    "recall, F2, and false-positive workload use the independently "
                    "selected nested thresholds. Top-K is a pooled ranking summary."
                ),
                "",
                "## Machine-readable outputs",
                "",
                "- `metrics_at_threshold_0.5.csv`",
                "- `metrics_at_nested_thresholds.csv`",
                "- `nested_thresholds_by_outer_fold.csv`",
                "- `nested_inner_oof_predictions.csv`",
                "- `top_k_metrics.csv`",
                "- `false_positive_workload.csv`",
                "- `planned_comparisons.csv`",
                "- `outer_oof_predictions.csv`",
            ]
        )
        (output / "REPORT.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        }
        write_json(
            output / "analysis_manifest.json",
            {
                "study_id": protocol["study_id"],
                "protocol_sha256": sha256_file(V2_PROTOCOL_PATH),
                "source_snapshot_sha256": protocol["source_snapshot_sha256"],
                "compatibility_manifest_sha256": sha256_file(
                    V2_COMPATIBILITY_PATH
                ),
                "baseline_predictions_sha256": baseline_hash,
                "nested_run_manifest_sha256": nested_hashes,
                "method_run_manifest_sha256": run_hashes,
                "fold3_attempt_id": attempt["attempt_id"],
                "v1_reused_runs": 17,
                "thresholds_use_outer_labels": False,
                "rows_per_method": {
                    key: len(value) for key, value in pooled.items()
                },
                "output_sha256": output_hashes,
            },
        )
    return destination


def _budget_session_id() -> str:
    budget = read_json(V2_ROOT / "time_budget.json")
    session_id = budget.get("authorization_session_id")
    if not session_id:
        raise RuntimeError("V2 authorization session identity is missing")
    return str(session_id)


def main() -> int:
    destination = build_report()
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
