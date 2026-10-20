"""Analyze frozen Experiment 2 OOF scores without touching training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from models.transformer_final.common import resolve_path, sha256_file
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PROTOCOL,
    load_annotation_ready_protocol,
)
from models.transformer_final.experiment2_analysis import (
    _threshold_metrics,
    _top_k_table,
)

STUDY_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = resolve_path(
    "results/transformer/final_studies/final_annotation_ready_pilot_v1/aggregate_metrics"
)
THRESHOLDS = (
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
)


def _distribution(values: pd.Series) -> dict[str, float]:
    data = values.to_numpy(dtype=np.float64)
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "maximum": float(np.max(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
    }


def build_stage_a() -> dict[str, Any]:
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    protocol, protocol_hash = load_annotation_ready_protocol(DEFAULT_PROTOCOL, config)
    if protocol_hash != "242f98c56d88a11dea3b355c4c3b14c693029e3d8a607fdd7a067f29af630b36":
        raise RuntimeError("Experiment 2 baseline protocol hash differs")
    effective, _, _ = load_effective_dataset(config)
    prediction_path = BASELINE_ROOT / "pooled_oof_predictions.csv"
    predictions = pd.read_csv(
        prediction_path, dtype={"record_id": str}, keep_default_na=False
    )
    required = {"record_id", "label", "probability_yes", "fold", "family_id"}
    if required - set(predictions.columns):
        raise RuntimeError("Baseline OOF predictions have an unexpected schema")
    if predictions["record_id"].duplicated().any() or len(predictions) != len(effective):
        raise RuntimeError("Baseline OOF predictions are not a unique effective cohort")
    expected = effective[
        ["record_id", "label", "fold", "target_clause"]
    ].rename(columns={"label": "dataset_label", "fold": "dataset_fold"})
    aligned = predictions.merge(
        expected,
        on="record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise RuntimeError("Baseline prediction IDs do not align to frozen cohort")
    if (
        (aligned["label"].astype(int) != aligned["dataset_label"].astype(int)).any()
        or (aligned["fold"].astype(int) != aligned["dataset_fold"].astype(int)).any()
    ):
        raise RuntimeError("Baseline predictions disagree with frozen labels or folds")
    predictions["label"] = predictions["label"].astype(int)
    predictions["fold"] = predictions["fold"].astype(int)
    predictions["probability_yes"] = predictions["probability_yes"].astype(float)
    rows: list[dict[str, Any]] = []
    per_fold_tables: list[pd.DataFrame] = []
    plot_dir = STUDY_ROOT / "plots" / "stage_a"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for fold in sorted(predictions["fold"].unique()):
        frame = predictions.loc[predictions["fold"] == fold].copy()
        positives = frame.loc[frame["label"] == 1, "probability_yes"]
        negatives = frame.loc[frame["label"] == 0, "probability_yes"]
        metrics = {
            "fold": int(fold),
            "rows": len(frame),
            "positive_count": int(frame["label"].sum()),
            "negative_count": int((frame["label"] == 0).sum()),
            "positive_prevalence": float(frame["label"].mean()),
            "average_precision": float(
                average_precision_score(frame["label"], frame["probability_yes"])
            ),
            "roc_auc": float(roc_auc_score(frame["label"], frame["probability_yes"])),
            "positive_scores": _distribution(positives),
            "negative_scores": _distribution(negatives),
            "predicted_yes_at_0_5": int((frame["probability_yes"] >= 0.5).sum()),
        }
        rows.append(metrics)
        for threshold in THRESHOLDS:
            per_fold_tables.append(
                pd.DataFrame([{"fold": int(fold), **_threshold_metrics(frame, threshold)}])
            )
        bins = np.linspace(0, 1, 51)
        negative_density, _ = np.histogram(
            negatives.to_numpy(dtype=np.float64), bins=bins, density=True
        )
        positive_density, _ = np.histogram(
            positives.to_numpy(dtype=np.float64), bins=bins, density=True
        )
        centers = (bins[:-1] + bins[1:]) / 2
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        axis.plot(centers, negative_density, label=f"No (n={len(negatives)})")
        axis.plot(centers, positive_density, label=f"Yes (n={len(positives)})")
        axis.axvline(0.5, color="black", linestyle="--", label="0.5 threshold")
        axis.set(
            title=f"Experiment 2 OOF Score Distributions — Fold {fold}",
            xlabel="Predicted probability of Yes",
            ylabel="Density",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"fold_{fold}_score_distribution.png", dpi=160)
        plt.close(fig)

    threshold_rows = pd.concat(per_fold_tables, ignore_index=True)
    threshold_rows.to_csv(
        STUDY_ROOT / "stage_a_per_fold_thresholds.csv",
        index=False,
        lineterminator="\n",
    )
    overall_top_k = _top_k_table(predictions)
    overall_top_k.to_csv(
        STUDY_ROOT / "stage_a_top_k.csv", index=False, lineterminator="\n"
    )
    report_path = STUDY_ROOT / "stage_a_family_recall_diagnostics.md"
    folds_by_number = {item["fold"]: item for item in rows}
    fold_2 = folds_by_number[2]
    report = [
        "# Stage A — Family-Specific Recall Diagnostics",
        "",
        "## Provenance and scope",
        "",
        f"- Baseline: Experiment 2, protocol SHA-256 `{protocol_hash}`.",
        f"- Prediction artifact SHA-256: `{sha256_file(prediction_path)}`.",
        f"- Effective cohort SHA-256: `{protocol['dataset']['effective_dataset_sha256']}`.",
        f"- Rows verified: {len(predictions)} unique records, one prediction per frozen cohort record.",
        "- Read-only analysis; historical predictions and checkpoints were not modified.",
        "",
        "## Per-fold ranking and class score distributions",
        "",
        "| Fold | Rows | Yes | Prevalence | AP | ROC-AUC | Predicted Yes @ 0.5 | Positive scores (min / median / mean / max) | Negative scores (min / median / mean / max) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in rows:
        pos = item["positive_scores"]
        neg = item["negative_scores"]
        report.append(
            f"| {item['fold']} | {item['rows']} | {item['positive_count']} | "
            f"{item['positive_prevalence']:.4f} | {item['average_precision']:.4f} | "
            f"{item['roc_auc']:.4f} | {item['predicted_yes_at_0_5']} | "
            f"{pos['minimum']:.4f} / {pos['median']:.4f} / {pos['mean']:.4f} / {pos['maximum']:.4f} | "
            f"{neg['minimum']:.4f} / {neg['median']:.4f} / {neg['mean']:.4f} / {neg['maximum']:.4f} |"
        )
    report.extend(
        [
            "",
            "P90, P95, and P99 values for both classes are included in `stage_a_fold_diagnostics.json`; every requested threshold's TP/FP/TN/FN, precision, recall, F1, and F2 is in `stage_a_per_fold_thresholds.csv`. The score-distribution charts are in `plots/stage_a/`.",
            "",
            "## Fold 2 diagnosis",
            "",
            f"Fold 2 contains {fold_2['positive_count']} positives among {fold_2['rows']} records (prevalence {fold_2['positive_prevalence']:.2%}), AP {fold_2['average_precision']:.4f}, and ROC-AUC {fold_2['roc_auc']:.4f}. Its positive score median is {fold_2['positive_scores']['median']:.4f} and maximum is {fold_2['positive_scores']['maximum']:.4f}; the frozen 0.5 rule predicts {fold_2['predicted_yes_at_0_5']} positives.",
            "",
            "Evidence: the reported AP/ROC-AUC and separated score ranks indicate useful ranking within this fold, while all positive scores below 0.5 explain why the global threshold yields zero recall there. This is threshold-induced zero recall on a fold with ranking signal; it does not by itself establish well-calibrated probabilities or prove a single global threshold transfers across families.",
            "",
            "## Interpretation and limitations",
            "",
            "- **Threshold versus ranking:** Fold-level ranking is materially better than chance in some families, while the 0.5 decision rule yields zero recall in Fold 2. The held-out threshold sweep is descriptive only and cannot choose a threshold for this study.",
            "- **Family-dependent score behavior:** Variations in positive score summaries and fixed-threshold counts are consistent with family-dependent score scales; these observations do not prove calibration error without fitting-side reliability analysis.",
            "- **Score compression:** Any concentration at low probabilities is directly visible in the per-fold histograms and score quantiles. Do not interpret low scores as evidence of absent ranking signal.",
            "- **Weak ranking and minority learning:** AP and class-conditional score overlap quantify remaining discrimination limits. The small positive counts per family make these estimates variable.",
            "- **Missing semantic context:** Target-only predictions cannot establish that context would help; that requires the frozen Stage C ablation.",
            "",
            "## Overall retrieval baseline",
            "",
            "The top-K table in `stage_a_top_k.csv` reports actual-positive retrieval at the fixed review budgets K=10, 25, 50, 100, and 200. No operating threshold was selected.",
            "",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")
    result = {
        "schema_version": 1,
        "baseline_protocol_sha256": protocol_hash,
        "baseline_predictions_sha256": sha256_file(prediction_path),
        "effective_dataset_sha256": protocol["dataset"]["effective_dataset_sha256"],
        "rows": len(predictions),
        "unique_records": int(predictions["record_id"].nunique()),
        "folds": rows,
        "overall_top_k": overall_top_k.to_dict(orient="records"),
    }
    (STUDY_ROOT / "stage_a_fold_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build_stage_a(), indent=2, sort_keys=True))
