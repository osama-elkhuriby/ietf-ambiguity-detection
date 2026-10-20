"""Evaluate fixed rank-average ensembles of the final V4, DeBERTa, and ML OOF scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V4_PREDICTIONS = (
    ROOT
    / "results/LLM/qwen3-32b-final-annotation-ready-v4/analysis-final/oof_predictions.csv"
)
DEFAULT_TRANSFORMER_PREDICTIONS = (
    ROOT
    / "results/transformer/final_studies/final_annotation_ready_pilot_v1/"
    "aggregate_metrics/pooled_oof_predictions.csv"
)
DEFAULT_ML_PREDICTIONS = ROOT / "results/ML/v2_engineered_tfidf/oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "results/hybrid/final_rank_ensemble_v2"

_TOP_FRACTIONS = (0.05, 0.10, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_predictions(
    v4_path: Path, transformer_path: Path, ml_path: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    v4 = pd.read_csv(v4_path)
    transformer = pd.read_csv(transformer_path)
    ml = pd.read_csv(ml_path)
    v4_required = {
        "source_item_id",
        "family_id",
        "fold",
        "label",
        "ambiguity_probability",
    }
    transformer_required = {
        "item_id",
        "family_id",
        "fold",
        "label",
        "probability_yes",
    }
    ml_required = {
        "item_id",
        "final_label",
        "validation_group",
        "oof_probability_yes",
    }
    if not v4_required.issubset(v4.columns):
        raise ValueError(f"V4 predictions lack columns: {sorted(v4_required - set(v4))}")
    if not transformer_required.issubset(transformer.columns):
        raise ValueError(
            "Transformer predictions lack columns: "
            f"{sorted(transformer_required - set(transformer))}"
        )
    if not ml_required.issubset(ml.columns):
        raise ValueError(f"ML predictions lack columns: {sorted(ml_required - set(ml))}")

    v4 = v4[
        [
            "source_item_id",
            "family_id",
            "fold",
            "label",
            "ambiguity_probability",
        ]
    ].copy()
    transformer = transformer[
        ["item_id", "family_id", "fold", "label", "probability_yes"]
    ].copy()
    ml = ml[
        ["item_id", "final_label", "validation_group", "oof_probability_yes"]
    ].rename(columns={"item_id": "ml_item_id"})
    v4["_item_key"] = v4["source_item_id"].astype("string")
    transformer["_item_key"] = transformer["item_id"].astype("string")
    ml["_item_key"] = ml["ml_item_id"].astype("string")

    if (
        v4["_item_key"].isna().any()
        or transformer["_item_key"].isna().any()
        or ml["_item_key"].isna().any()
    ):
        raise ValueError("Prediction artifacts contain missing item IDs")
    if (
        v4["_item_key"].duplicated().any()
        or transformer["_item_key"].duplicated().any()
        or ml["_item_key"].duplicated().any()
    ):
        raise ValueError("Prediction artifacts must contain unique item IDs")
    if set(v4["_item_key"]) != set(transformer["_item_key"]):
        raise ValueError("V4 and transformer predictions do not cover the same item IDs")

    predictions = v4.merge(
        transformer,
        on="_item_key",
        how="inner",
        suffixes=("_v4", "_transformer"),
        validate="one_to_one",
        sort=False,
    )
    for column in ("family_id", "fold", "label"):
        if not predictions[f"{column}_v4"].equals(
            predictions[f"{column}_transformer"]
        ):
            raise ValueError(f"Model predictions disagree on {column}")

    predictions["_item_key"] = predictions["source_item_id"].astype("string")
    predictions = predictions.merge(
        ml,
        on="_item_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    ml_coverage = predictions["ml_item_id"].notna()
    common = predictions.loc[ml_coverage].copy()
    if common.empty:
        raise ValueError("ML predictions do not overlap the V4 development cohort")
    if common["validation_group"].isna().any():
        raise ValueError("ML validation-group assignments cannot be missing")
    if common.groupby("family_id_v4")["validation_group"].nunique().gt(1).any():
        raise ValueError("ML OOF predictions split a document family across groups")
    if common.groupby("validation_group")["family_id_v4"].nunique().gt(1).any():
        raise ValueError("An ML validation group contains multiple document families")

    labels = pd.to_numeric(common["label_v4"], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1 values")
    common["label"] = labels.astype(int)
    common["fold"] = pd.to_numeric(common["fold_v4"], errors="raise")
    if common["fold"].isna().any():
        raise ValueError("Fold assignments cannot be missing")
    if common.groupby("family_id_v4")["fold"].nunique().gt(1).any():
        raise ValueError("Each document family must be assigned to a single fold")

    ml_labels = common["final_label"].astype("string").str.casefold().map(
        {"yes": 1, "no": 0}
    )
    if ml_labels.isna().any():
        raise ValueError("ML labels must contain only Yes/No values")
    ml_scores = pd.to_numeric(common["oof_probability_yes"], errors="raise")
    if not np.isfinite(ml_scores).all() or not ml_scores.between(0, 1).all():
        raise ValueError("ML OOF probabilities must be finite and in [0, 1]")
    compatible_labels = common["label"].eq(ml_labels)
    mismatched_labels = int((~compatible_labels).sum())
    common = common.loc[compatible_labels].copy()
    common["ml_probability"] = ml_scores.loc[compatible_labels].astype(float)
    if common.empty or common["label"].nunique() != 2:
        raise ValueError("Label-compatible model predictions must contain both classes")

    for source_column, target_column in (
        ("ambiguity_probability", "v4_probability"),
        ("probability_yes", "transformer_probability"),
    ):
        values = pd.to_numeric(common[source_column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"{source_column} must contain finite probabilities in [0, 1]")
        common[target_column] = values.astype(float)

    audit = {
        "v4_rows": len(v4),
        "ml_rows": len(ml),
        "shared_item_rows": int(ml_coverage.sum()),
        "v4_rows_missing_from_ml": int((~ml_coverage).sum()),
        "label_mismatch_rows_excluded": mismatched_labels,
        "compatible_rows": len(common),
    }
    return common[
        [
            "source_item_id",
            "family_id_v4",
            "fold",
            "label",
            "v4_probability",
            "transformer_probability",
            "ml_probability",
            "validation_group",
        ]
    ].rename(columns={"family_id_v4": "family_id"}), audit


def _rank_average(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    score_columns = {
        "v4": "v4_probability",
        "transformer": "transformer_probability",
        "ml": "ml_probability",
    }
    for name, column in score_columns.items():
        result[f"{name}_rank"] = result[column].rank(method="average", pct=True)
    result["v4_transformer_rank_score"] = (
        result["v4_rank"] + result["transformer_rank"]
    ) / 2
    result["v4_ml_rank_score"] = (result["v4_rank"] + result["ml_rank"]) / 2
    result["hybrid_rank_score"] = (
        result["v4_rank"] + result["transformer_rank"] + result["ml_rank"]
    ) / 3
    return result


def _top_k_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positive_count = int(labels.sum())
    metrics: dict[str, float] = {}
    for fraction in _TOP_FRACTIONS:
        count = max(1, int(np.ceil(len(labels) * fraction)))
        selected = np.argsort(-scores, kind="stable")[:count]
        retrieved = int(labels[selected].sum())
        suffix = f"{int(fraction * 100)}pct"
        metrics[f"precision_at_{suffix}"] = retrieved / count
        metrics[f"recall_at_{suffix}"] = (
            retrieved / positive_count if positive_count else 0.0
        )
    return metrics


def _metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    labels = predictions["label"].to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required to calculate evaluation metrics")
    prevalence = float(labels.mean())
    model_scores = (
        ("V4", predictions["v4_probability"].to_numpy(dtype=float)),
        (
            "DeBERTa",
            predictions["transformer_probability"].to_numpy(dtype=float),
        ),
        ("ML logistic regression", predictions["ml_probability"].to_numpy(dtype=float)),
        (
            "V4 + DeBERTa",
            predictions["v4_transformer_rank_score"].to_numpy(dtype=float),
        ),
        ("V4 + ML", predictions["v4_ml_rank_score"].to_numpy(dtype=float)),
        (
            "V4 + DeBERTa + ML",
            predictions["hybrid_rank_score"].to_numpy(dtype=float),
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, scores in model_scores:
        row: dict[str, Any] = {
            "model": name,
            "rows": len(labels),
            "positives": int(labels.sum()),
            "prevalence": prevalence,
            "average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        }
        row["ap_lift_over_prevalence"] = row["average_precision"] / prevalence
        row.update(_top_k_metrics(labels, scores))
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    v4_path: Path, transformer_path: Path, ml_path: Path, output_dir: Path
) -> pd.DataFrame:
    aligned_predictions, audit = _read_predictions(v4_path, transformer_path, ml_path)
    predictions = _rank_average(aligned_predictions)
    metrics = _metrics_table(predictions)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {output_dir}")
    output_dir.mkdir(parents=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    manifest = {
        "experiment": "final_rank_ensemble_v2",
        "method": "equal-weight mean of within-cohort percentile ranks",
        "component_models": [
            "V4 Qwen/MiniLM semantic ensemble",
            "DeBERTa-v3-base",
            "class-balanced LogisticRegression, engineered features + TF-IDF",
        ],
        "threshold_tuning": False,
        "weight_tuning": False,
        "cohort_audit": audit,
        "rows": len(predictions),
        "positives": int(predictions["label"].sum()),
        "families": sorted(predictions["family_id"].unique().tolist()),
        "folds": sorted(int(fold) for fold in predictions["fold"].unique()),
        "inputs": {
            "v4_oof_predictions_sha256": _sha256(v4_path),
            "transformer_oof_predictions_sha256": _sha256(transformer_path),
            "ml_oof_predictions_sha256": _sha256(ml_path),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "interpretation": (
            "Exploratory development OOF comparison only. The component models "
            "and V4 design were evaluated during adaptive development. The ML "
            "component was trained using an earlier annotation artifact; rows "
            "missing from that artifact or with conflicting labels were excluded. "
            "This is not independent confirmation or a sealed-test result."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-predictions", type=Path, default=DEFAULT_V4_PREDICTIONS)
    parser.add_argument(
        "--transformer-predictions",
        type=Path,
        default=DEFAULT_TRANSFORMER_PREDICTIONS,
    )
    parser.add_argument("--ml-predictions", type=Path, default=DEFAULT_ML_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = run_experiment(
        args.v4_predictions,
        args.transformer_predictions,
        args.ml_predictions,
        args.output_dir,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
