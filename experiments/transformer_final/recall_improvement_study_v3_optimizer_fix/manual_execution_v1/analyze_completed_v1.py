"""CPU-only analysis of completed Recall Improvement Study V3 runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_BOOTSTRAP))

from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
    _choose_threshold,
)
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from manual_cell_launcher import (  # noqa: E402
    EXPECTED_V3_PROTOCOL_SHA256,
    EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
    V2_INVENTORY_PATH,
    V2_PROTOCOL_PATH,
    V3_INVENTORY_PATH,
    V3_PROTOCOL_PATH,
    V3_ROOT,
    _legacy_v3_protocol,
    validate_package,
)
from models.transformer_final.common import (
    immutable_directory,
    read_json,
    sha256_file,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import TOP_K_VALUES

REPO_ROOT = Path(__file__).resolve().parents[4]
V1_ROOT = V3_ROOT.parent / "recall_improvement_study_v1"
V2_ROOT = V3_ROOT.parent / "recall_improvement_study_v2"
V2_COMPATIBILITY_PATH = V2_ROOT / "V1_TO_V2_COMPATIBILITY.json"
V3_COMPATIBILITY_PATH = V3_ROOT / "COMPATIBILITY_REPORT.json"
BASELINE_ROOT = (
    REPO_ROOT
    / "experiments"
    / "transformer_final"
    / "final_annotation_ready_pilot_v1"
)
BASELINE_ANALYSIS_ROOT = BASELINE_ROOT / "aggregate_metrics"
METHOD_STAGES = {
    "partial_4": "G_partial_4",
    "focal_loss": "D_focal_loss",
    "partial_2": "G2_partial_2",
}


def _effective(protocol: dict[str, Any]) -> pd.DataFrame:
    config = load_config(protocol["config_path"])
    frame, manifest, dataset_dir = load_effective_dataset(config)
    if (
        manifest["effective_dataset_sha256"]
        != protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(dataset_dir / "manifest.json")
        != protocol["dataset"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("Analysis dataset differs from the frozen protocol")
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def _prediction_artifact(run: Path, manifest: dict[str, Any]) -> Path:
    artifact = manifest.get("artifact_hashes", {}).get("held_out_predictions")
    if not artifact:
        raise RuntimeError(f"Run lacks prediction artifact metadata: {run}")
    path = (run / artifact["path"]).resolve()
    if (
        run.resolve() not in path.parents
        or not path.is_file()
        or sha256_file(path) != artifact["sha256"]
    ):
        raise RuntimeError(f"Prediction artifact changed: {run}")
    return path


def _validated_predictions(
    path: Path,
    effective: pd.DataFrame,
    *,
    expected_ids: set[str],
    identity: str,
    require_dataset_fold: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"record_id": str}, keep_default_na=False)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["probability_yes"] = frame["probability_yes"].astype(float)
    if (
        frame["record_id"].duplicated().any()
        or set(frame["record_id"]) != expected_ids
        or not np.isfinite(frame["probability_yes"].to_numpy()).all()
        or frame["probability_yes"].lt(0).any()
        or frame["probability_yes"].gt(1).any()
    ):
        raise RuntimeError(f"Prediction coverage or scores invalid: {identity}")
    verified = frame.merge(
        effective[["record_id", "label", "fold", "family_id"]],
        on="record_id",
        suffixes=("", "_dataset"),
        validate="one_to_one",
    )
    if not verified["label"].eq(verified["label_dataset"]).all():
        raise RuntimeError(f"Prediction labels changed: {identity}")
    if require_dataset_fold and not verified["fold"].astype(int).eq(
        verified["fold_dataset"].astype(int)
    ).all():
        raise RuntimeError(f"Prediction folds changed: {identity}")
    frame["label"] = verified["label_dataset"].astype(int).to_numpy()
    frame["fold"] = verified["fold_dataset"].astype(int).to_numpy()
    frame["family_id"] = verified["family_id_dataset"].astype(str).to_numpy()
    return frame


def _read_baseline(effective: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis_manifest = read_json(BASELINE_ANALYSIS_ROOT / "analysis_manifest.json")
    path = BASELINE_ANALYSIS_ROOT / "pooled_oof_predictions.csv"
    if (
        analysis_manifest["artifact_hashes"].get("pooled_oof_predictions.csv")
        != sha256_file(path)
    ):
        raise RuntimeError("Frozen full-fine-tuning OOF predictions changed")
    frame = _validated_predictions(
        path,
        effective,
        expected_ids=set(effective["record_id"]),
        identity="full_fine_tuning",
    )
    return frame, {
        "origin": "Experiment 2 full fine-tuning",
        "protocol_sha256": analysis_manifest["protocol_sha256"],
        "source_snapshot_sha256": analysis_manifest["source_snapshot_sha256"],
        "predictions_sha256": sha256_file(path),
    }


def _collect_nested_thresholds(
    effective: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, dict[int, float], dict[str, str]]:
    compatibility = read_json(V2_COMPATIBILITY_PATH)
    reused = {
        record["v1_cell_id"]: record
        for record in compatibility["reusable_v1_runs"]
    }
    splits_path = Path(protocol["dataset"]["study_nested_splits_path"])
    if sha256_file(splits_path) != protocol["dataset"]["study_nested_splits_sha256"]:
        raise RuntimeError("Frozen nested split file changed")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    threshold_rows: list[dict[str, Any]] = []
    thresholds: dict[int, float] = {}
    provenance: dict[str, str] = {}
    for outer_fold in range(1, 6):
        expected_outer = set(
            effective.loc[effective["fold"] != outer_fold, "record_id"]
        )
        pieces: list[pd.DataFrame] = []
        seen: set[str] = set()
        for split in [
            item for item in splits if int(item["outer_fold"]) == outer_fold
        ]:
            inner = int(split["inner_fold"])
            cell_id = (
                f"B_nested_threshold_calibration/outer_{outer_fold}_inner_{inner}"
            )
            record = reused.get(cell_id)
            if (
                record is None
                or record.get("v2_status")
                != "REUSED_BY_REFERENCE_AFTER_COMPATIBILITY_CHECK"
            ):
                raise RuntimeError(f"Nested V1 run is not reusable: {cell_id}")
            run = Path(record["v1_run_path"]).resolve()
            manifest_path = run / "run_manifest.json"
            if sha256_file(manifest_path) != record["v1_run_manifest_sha256"]:
                raise RuntimeError(f"Nested V1 manifest changed: {cell_id}")
            manifest = read_json(manifest_path)
            split_ids = set(str(value) for value in split["scoring_record_ids"])
            if seen & split_ids:
                raise RuntimeError(f"Nested scoring overlap: outer fold {outer_fold}")
            frame = _validated_predictions(
                _prediction_artifact(run, manifest),
                effective,
                expected_ids=split_ids,
                identity=cell_id,
                require_dataset_fold=False,
            )
            seen.update(split_ids)
            pieces.append(frame)
            provenance[cell_id] = record["v1_run_manifest_sha256"]
        if seen != expected_outer:
            raise RuntimeError(f"Nested OOF coverage failed: outer fold {outer_fold}")
        oof = pd.concat(pieces, ignore_index=True)
        selected = _choose_threshold(
            oof,
            [float(value) for value in protocol["inner_cv"]["threshold_candidates"]],
        )
        thresholds[outer_fold] = float(selected["threshold"])
        threshold_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_oof_rows": len(oof),
                "inner_precision": selected["precision"],
                "inner_recall": selected["recall"],
                "inner_f1": selected["f1"],
                "inner_f2": selected["f2"],
                "inner_true_positives": selected["true_positives"],
                "inner_false_positives": selected["false_positives"],
                "inner_true_negatives": selected["true_negatives"],
                "inner_false_negatives": selected["false_negatives"],
                "selected_threshold": selected["threshold"],
                "selection_policy": selected["selection_policy"],
            }
        )
    return pd.DataFrame(threshold_rows), thresholds, provenance


def _read_context(
    effective: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    v2_inventory = read_json(V2_INVENTORY_PATH)
    compatibility = read_json(V2_COMPATIBILITY_PATH)
    reused = {
        item["v1_cell_id"]: item for item in compatibility["reusable_v1_runs"]
    }
    v3_compatibility = read_json(V3_COMPATIBILITY_PATH)
    expected_hashes = {
        item["cell_id"]: item["manifest_sha256"]
        for item in v3_compatibility["historical_reuse"]["context_runs"]
    }
    pieces: list[pd.DataFrame] = []
    runs: dict[str, str] = {}
    times: list[dict[str, Any]] = []
    for fold in range(1, 6):
        cell_id = f"C_local_context/fold_{fold}/seed_2718"
        if fold in (1, 2):
            record = reused[cell_id]
            run = Path(record["v1_run_path"])
            expected_hash = record["v1_run_manifest_sha256"]
            origin = "V1_reused_by_reference"
        elif fold == 3:
            record = v2_inventory["authorized_attempts"][cell_id]
            run = Path(record["run_path"])
            expected_hash = record["run_manifest_sha256"]
            origin = "V2_replacement_attempt"
        else:
            record = v2_inventory["pending_v2_cells"][cell_id]
            run = Path(record["run_path"])
            expected_hash = record["run_manifest_sha256"]
            origin = "V2"
        manifest_path = run / "run_manifest.json"
        if (
            expected_hash != expected_hashes[cell_id]
            or sha256_file(manifest_path) != expected_hash
        ):
            raise RuntimeError(f"Context provenance changed: {cell_id}")
        manifest = read_json(manifest_path)
        ids = set(effective.loc[effective["fold"] == fold, "record_id"])
        pieces.append(
            _validated_predictions(
                _prediction_artifact(run, manifest),
                effective,
                expected_ids=ids,
                identity=cell_id,
            )
        )
        runs[cell_id] = expected_hash
        times.append(
            {
                "model": "local_context",
                "fold": fold,
                "runtime_seconds": float(manifest["runtime_seconds"]),
                "origin": origin,
                "run_manifest_sha256": expected_hash,
            }
        )
    return pd.concat(pieces, ignore_index=True), {"run_manifests": runs}, times


def _read_v3_stage(
    method: str,
    stage: str,
    effective: pd.DataFrame,
    inventory: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any], list[dict[str, Any]]]:
    records = [
        inventory["attempts"][f"{stage}/fold_{fold}/seed_2718"]
        for fold in range(1, 6)
    ]
    statuses = [record["status"] for record in records]
    provenance: dict[str, Any] = {
        "status": "COMPLETE" if all(value == "COMPLETE" for value in statuses) else "INCOMPLETE",
        "cell_statuses": {
            f"{stage}/fold_{fold}/seed_2718": records[fold - 1]["status"]
            for fold in range(1, 6)
        },
        "run_manifests": {},
    }
    times: list[dict[str, Any]] = []
    pieces: list[pd.DataFrame] = []
    for fold, record in enumerate(records, start=1):
        if record["status"] != "COMPLETE":
            continue
        run = Path(record["run_path"])
        manifest_path = run / "run_manifest.json"
        if sha256_file(manifest_path) != record["run_manifest_sha256"]:
            raise RuntimeError(f"Completed V3 manifest changed: {stage} fold {fold}")
        manifest = read_json(manifest_path)
        ids = set(effective.loc[effective["fold"] == fold, "record_id"])
        pieces.append(
            _validated_predictions(
                _prediction_artifact(run, manifest),
                effective,
                expected_ids=ids,
                identity=f"{stage}/fold_{fold}",
            )
        )
        provenance["run_manifests"][record["cell_id"]] = record[
            "run_manifest_sha256"
        ]
        times.append(
            {
                "model": method,
                "fold": fold,
                "runtime_seconds": float(manifest["runtime_seconds"]),
                "origin": "V3_manual_execution_v1",
                "run_manifest_sha256": record["run_manifest_sha256"],
            }
        )
    if provenance["status"] != "COMPLETE":
        return None, provenance, times
    predictions = pd.concat(pieces, ignore_index=True)
    if predictions["record_id"].duplicated().any() or len(predictions) != len(effective):
        raise RuntimeError(f"V3 stage OOF coverage failed: {stage}")
    return predictions, provenance, times


def _baseline_times() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        run = BASELINE_ROOT / "training_outputs" / f"fold_{fold}" / "seed_2718"
        manifest_path = run / "run_manifest.json"
        manifest = read_json(manifest_path)
        if (
            int(manifest.get("fold", 0)) != fold
            or int(manifest.get("seed", 0)) != 2718
            or manifest.get("epochs_completed") != 5
            or manifest.get("full_encoder_fine_tuned") is not True
        ):
            raise RuntimeError(f"Full fine-tuning run identity failed: fold {fold}")
        rows.append(
            {
                "model": "full_fine_tuning",
                "fold": fold,
                "runtime_seconds": float(manifest["runtime_seconds"]),
                "origin": "Experiment_2",
                "run_manifest_sha256": sha256_file(manifest_path),
            }
        )
    return rows


def _metric_row(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    operating_point: str,
    fold: int | None = None,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    scores = frame["probability_yes"].to_numpy(dtype=np.float64)
    predicted = predicted.astype(np.int64)
    tp = int(np.count_nonzero((labels == 1) & (predicted == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predicted == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predicted == 0)))
    fn = int(np.count_nonzero((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    both_classes = len(np.unique(labels)) == 2
    return {
        "model": model,
        "operating_point": operating_point,
        "fold": fold,
        "rows": len(frame),
        "average_precision": (
            float(average_precision_score(labels, scores)) if labels.sum() else None
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _analysis_tables(
    methods: dict[str, pd.DataFrame], thresholds: dict[int, float]
) -> dict[str, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    top_k_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for model, frame in methods.items():
        frame = frame.reset_index(drop=True)
        selected_threshold = frame["fold"].map(thresholds).to_numpy(float)
        predictions_by_op = {
            "threshold_0.5": frame["probability_yes"].to_numpy(float) >= 0.5,
            "nested_fold_threshold": (
                frame["probability_yes"].to_numpy(float) >= selected_threshold
            ),
        }
        for operating_point, predicted in predictions_by_op.items():
            pooled_rows.append(
                _metric_row(
                    frame,
                    predicted,
                    model=model,
                    operating_point=operating_point,
                )
            )
            for fold, positions in frame.groupby("fold", sort=True).groups.items():
                index = frame.index.get_indexer(positions)
                fold_frame = frame.loc[positions]
                fold_rows.append(
                    _metric_row(
                        fold_frame,
                        predicted[index],
                        model=model,
                        operating_point=operating_point,
                        fold=int(fold),
                    )
                )
            for family, positions in frame.groupby("family_id", sort=True).groups.items():
                index = frame.index.get_indexer(positions)
                family_frame = frame.loc[positions]
                row = _metric_row(
                    family_frame,
                    predicted[index],
                    model=model,
                    operating_point=operating_point,
                )
                row["family_id"] = str(family)
                family_rows.append(row)
        ranked = frame.sort_values(
            ["probability_yes", "record_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        positives = int(frame["label"].sum())
        prevalence = float(frame["label"].mean())
        for k in TOP_K_VALUES:
            top = ranked.head(int(k))
            retrieved = int(top["label"].sum())
            top_k_rows.append(
                {
                    "model": model,
                    "k": int(k),
                    "true_positives_retrieved": retrieved,
                    "false_positives_at_k": len(top) - retrieved,
                    "precision_at_k": retrieved / len(top) if len(top) else 0.0,
                    "recall_at_k": retrieved / positives if positives else 0.0,
                    "lift_over_prevalence": (
                        retrieved / len(top) / prevalence
                        if len(top) and prevalence
                        else None
                    ),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)
    mean_rows: list[dict[str, Any]] = []
    metric_names = ("average_precision", "roc_auc", "precision", "recall", "f1", "f2")
    for (model, operating_point), group in fold_metrics.groupby(
        ["model", "operating_point"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model,
            "operating_point": operating_point,
            "folds": len(group),
        }
        for metric in metric_names:
            row[f"mean_fold_{metric}"] = float(group[metric].mean())
            row[f"std_fold_{metric}"] = float(group[metric].std(ddof=1))
        mean_rows.append(row)
    return {
        "pooled_metrics": pd.DataFrame(pooled_rows),
        "fold_metrics": fold_metrics,
        "mean_fold_metrics": pd.DataFrame(mean_rows),
        "top_k_retrieval": pd.DataFrame(top_k_rows),
        "family_metrics": pd.DataFrame(family_rows),
    }


def collect() -> dict[str, Any]:
    package_validation = validate_package()
    _legacy_v3_protocol()
    parent_protocol = read_json(V2_PROTOCOL_PATH)
    effective = _effective(parent_protocol)
    thresholds_frame, thresholds, nested_provenance = _collect_nested_thresholds(
        effective, parent_protocol
    )
    baseline, baseline_provenance = _read_baseline(effective)
    context, context_provenance, context_times = _read_context(effective)
    methods: dict[str, pd.DataFrame] = {
        "full_fine_tuning": baseline,
        "local_context": context,
    }
    inventory = read_json(V3_INVENTORY_PATH)
    stage_provenance: dict[str, Any] = {}
    time_rows = _baseline_times() + context_times
    for method, stage in METHOD_STAGES.items():
        predictions, provenance, times = _read_v3_stage(
            method, stage, effective, inventory
        )
        stage_provenance[method] = provenance
        time_rows.extend(times)
        if predictions is not None:
            methods[method] = predictions
    tables = _analysis_tables(methods, thresholds)
    pooled = tables["pooled_metrics"]
    comparisons: list[dict[str, Any]] = []
    for treatment in ("partial_4", "partial_2"):
        if treatment not in methods:
            continue
        for operating_point in ("threshold_0.5", "nested_fold_threshold"):
            full = pooled.loc[
                (pooled["model"] == "full_fine_tuning")
                & (pooled["operating_point"] == operating_point)
            ].iloc[0]
            partial = pooled.loc[
                (pooled["model"] == treatment)
                & (pooled["operating_point"] == operating_point)
            ].iloc[0]
            row: dict[str, Any] = {
                "comparison": f"{treatment}_vs_full_fine_tuning",
                "operating_point": operating_point,
                "treatment": treatment,
                "reference": "full_fine_tuning",
            }
            for metric in (
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "f2",
                "tp",
                "fp",
                "tn",
                "fn",
            ):
                row[f"delta_{metric}"] = partial[metric] - full[metric]
            comparisons.append(row)
    time_frame = pd.DataFrame(time_rows)
    time_summary = (
        time_frame.groupby("model", as_index=False)["runtime_seconds"]
        .agg(["count", "sum", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "completed_folds",
                "sum": "total_runtime_seconds",
                "mean": "mean_runtime_seconds",
                "min": "min_runtime_seconds",
                "max": "max_runtime_seconds",
            }
        )
    )
    full_total = float(
        time_summary.loc[
            time_summary["model"] == "full_fine_tuning", "total_runtime_seconds"
        ].iloc[0]
    )
    time_summary["total_time_ratio_vs_full"] = (
        time_summary["total_runtime_seconds"] / full_total
    )
    tables.update(
        {
            "threshold_calibration": thresholds_frame,
            "threshold_calibration_performance": tables["fold_metrics"].loc[
                tables["fold_metrics"]["operating_point"]
                == "nested_fold_threshold"
            ].reset_index(drop=True),
            "planned_comparisons": pd.DataFrame(comparisons),
            "gpu_training_time_by_fold": time_frame,
            "gpu_training_time_summary": time_summary,
        }
    )
    return {
        "tables": tables,
        "methods": methods,
        "stage_provenance": stage_provenance,
        "provenance": {
            "v1_nested_run_manifests": nested_provenance,
            "v2_context": context_provenance,
            "v3_protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
            "v3_source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
            "baseline": baseline_provenance,
        },
        "package_validation": package_validation,
    }


def write_analysis(destination: Path, *, require_all: bool) -> Path:
    collected = collect()
    incomplete = [
        method
        for method, value in collected["stage_provenance"].items()
        if value["status"] != "COMPLETE"
    ]
    if require_all and incomplete:
        raise RuntimeError(
            "Final analysis requires all V3 stages complete; incomplete: "
            + ", ".join(incomplete)
        )
    with immutable_directory(destination) as output:
        for name, frame in collected["tables"].items():
            frame.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
        pooled = collected["tables"]["pooled_metrics"]
        report = [
            "# Recall Improvement Study V3 — Completed-Run Analysis",
            "",
            f"Frozen V3 protocol SHA-256: `{EXPECTED_V3_PROTOCOL_SHA256}`.",
            "",
            "Only completed five-fold stages are included in aggregate performance tables. "
            "Incomplete or failed stages remain listed as incomplete in the analysis manifest.",
            "",
            "## Pooled results",
            "",
            "| Model | Operating point | AP | ROC-AUC | Precision | Recall | F1 | F2 | TP | FP | TN | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in pooled.to_dict(orient="records"):
            report.append(
                f"| {row['model']} | {row['operating_point']} | "
                f"{row['average_precision']:.4f} | {row['roc_auc']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['f2']:.4f} | {row['tp']} | "
                f"{row['fp']} | {row['tn']} | {row['fn']} |"
            )
        report.extend(
            [
                "",
                "## Included outputs",
                "",
                "- `pooled_metrics.csv`: pooled AP, ROC-AUC, precision, recall, F1, F2, and confusion counts.",
                "- `mean_fold_metrics.csv`: mean-fold AP and the other fold-level metrics.",
                "- `top_k_retrieval.csv`: frozen Top-K review budgets.",
                "- `family_metrics.csv`: family-level ranking and classification performance.",
                "- `threshold_calibration.csv` and `threshold_calibration_performance.csv`: fitting-side threshold selection and untouched outer-fold performance.",
                "- `gpu_training_time_by_fold.csv` and `gpu_training_time_summary.csv`: recorded GPU runtimes.",
                "- `planned_comparisons.csv`: full versus Partial-4 and Partial-2 deltas when complete.",
                "",
                "## Stage status",
                "",
            ]
        )
        for method, value in collected["stage_provenance"].items():
            report.append(f"- {method}: {value['status']}")
        (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        }
        write_json(
            output / "analysis_manifest.json",
            {
                "schema_version": 1,
                "study_id": "recall_improvement_study_v3_optimizer_fix",
                "protocol_sha256": sha256_file(V3_PROTOCOL_PATH),
                "source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
                "methods_included": list(collected["methods"]),
                "stage_provenance": collected["stage_provenance"],
                "historical_provenance": collected["provenance"],
                "missing_stages_reported_complete": False,
                "require_all": require_all,
                "output_sha256": output_hashes,
            },
        )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if args.validate_only == (args.output is not None):
        parser.error("choose exactly one of --validate-only or --output")
    return args


def main() -> int:
    args = parse_args()
    collected = collect() if args.validate_only else None
    if collected is not None:
        print(
            json.dumps(
                {
                    "cuda_used": False,
                    "methods_available": list(collected["methods"]),
                    "stage_status": {
                        key: value["status"]
                        for key, value in collected["stage_provenance"].items()
                    },
                    "protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(write_analysis(args.output.resolve(), require_all=args.require_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
