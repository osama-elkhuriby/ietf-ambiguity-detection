"""Aggregate and report the exploratory threshold pilot from verified OOF runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .common import (
    hash_directory,
    immutable_directory,
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)
from .config import load_config
from .dataset import load_effective_dataset
from .experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PROTOCOL,
    TOP_K_VALUES,
    THRESHOLDS,
    load_annotation_ready_protocol,
)
from .metrics import binary_metrics
from .run_validation import validate_completed_run

ERROR_THRESHOLD_VALUES = (0.01, 0.05, 0.10, 0.50)
ERROR_SAMPLE_SIZE = 5


def _threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    scores = frame["probability_yes"].to_numpy(dtype=np.float64)
    predicted = scores >= threshold
    actual = labels == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2_denominator = 4 * precision + recall
    f2 = 5 * precision * recall / f2_denominator if f2_denominator else 0.0
    balanced_accuracy = (recall + specificity) / 2
    mcc = float(
        binary_metrics(labels, scores, threshold=threshold)["mcc"]
    )
    predicted_yes = tp + fp
    return {
        "threshold": float(threshold),
        "rows": len(frame),
        "actual_yes": int(actual.sum()),
        "actual_no": int((~actual).sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "predicted_yes": predicted_yes,
        "predicted_no": len(frame) - predicted_yes,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def _threshold_table(frame: pd.DataFrame, *, fold: int | None = None) -> pd.DataFrame:
    subset = frame if fold is None else frame.loc[frame["fold"] == fold]
    rows = [_threshold_metrics(subset, threshold) for threshold in THRESHOLDS]
    if fold is not None:
        for row in rows:
            row["fold"] = fold
    return pd.DataFrame(rows)


def _score_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _distribution_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.loc[frame["label"] == 1, "probability_yes"]
    negative = frame.loc[frame["label"] == 0, "probability_yes"]
    actual_prevalence = float(frame["label"].mean())
    scores = frame["probability_yes"].to_numpy(dtype=np.float64)
    actual = frame["label"].to_numpy(dtype=np.float64)
    brier = float(np.mean((scores - actual) ** 2))
    brier_baseline = float(actual_prevalence * (1 - actual_prevalence))
    bins = np.linspace(0.0, 1.0, 51)
    pos_density, _ = np.histogram(positive, bins=bins, density=True)
    neg_density, _ = np.histogram(negative, bins=bins, density=True)
    overlap = float(np.minimum(pos_density, neg_density).sum() * (bins[1] - bins[0]))
    return {
        "actual_yes_score_distribution": _score_summary(positive),
        "actual_no_score_distribution": _score_summary(negative),
        "probability_below_0_01_percent": float(np.mean(scores < 0.01) * 100),
        "probability_below_0_05_percent": float(np.mean(scores < 0.05) * 100),
        "probability_below_0_10_percent": float(np.mean(scores < 0.10) * 100),
        "probability_above_0_50_percent": float(np.mean(scores > 0.50) * 100),
        "average_predicted_positive_probability": float(np.mean(scores)),
        "actual_positive_prevalence": actual_prevalence,
        "brier_score": brier,
        "prevalence_only_brier_baseline_descriptive": brier_baseline,
        "positive_negative_histogram_overlap_coefficient_50_bins": overlap,
        "overlap_interpretation": (
            "Descriptive histogram intersection on fixed 0.02-wide bins; values "
            "closer to 1 indicate more score-distribution overlap. No cutoff is "
            "treated as a pre-registered inferential criterion."
        ),
    }


def _top_k_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["probability_yes", "record_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    prevalence = float(frame["label"].mean())
    rows = []
    for k in TOP_K_VALUES:
        top = ranked.head(k)
        true_positive = int(top["label"].sum())
        precision = true_positive / len(top)
        recall = true_positive / int(frame["label"].sum())
        rows.append(
            {
                "k": k,
                "true_ambiguous_retrieved": true_positive,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_over_positive_prevalence": (
                    precision / prevalence if prevalence else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _family_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_id, group in frame.groupby("family_id", sort=True):
        metrics = binary_metrics(
            group["label"].to_numpy(),
            group["probability_yes"].to_numpy(),
            threshold=0.5,
        )
        rows.append(
            {
                "family_id": str(family_id),
                "rows": len(group),
                "positive_labels": int(group["label"].sum()),
                "positive_prevalence": metrics["positive_prevalence"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "ap_prevalence_lift": metrics["ap_prevalence_lift"],
                "predicted_yes_at_0_5": int(
                    (group["probability_yes"] >= 0.5).sum()
                ),
                "true_positives_at_0_5": int(
                    (
                        (group["label"] == 1)
                        & (group["probability_yes"] >= 0.5)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _error_examples(frame: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "item_id",
        "family_id",
        "target_clause",
        "label",
        "probability_yes",
        "fold",
    ]
    parts: list[pd.DataFrame] = []
    for category, label, ascending in (
        ("highest_scoring_actual_no", 0, False),
        ("highest_scoring_actual_yes", 1, False),
        ("lowest_scoring_actual_yes", 1, True),
    ):
        subset = frame.loc[frame["label"] == label].sort_values(
            ["probability_yes", "record_id"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        selected = subset.head(10)[fields].copy()
        selected.insert(0, "example_type", category)
        parts.append(selected)
    for threshold in ERROR_THRESHOLD_VALUES:
        false_positives = frame.loc[
            (frame["label"] == 0) & (frame["probability_yes"] >= threshold)
        ].sort_values(
            ["probability_yes", "record_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected = false_positives.head(ERROR_SAMPLE_SIZE)[fields].copy()
        selected.insert(0, "example_type", f"false_positive_at_{threshold:g}")
        parts.append(selected)
    examples = pd.concat(parts, ignore_index=True)
    for threshold in ERROR_THRESHOLD_VALUES:
        column = f"predicted_yes_at_{threshold:g}"
        examples[column] = (examples["probability_yes"] >= threshold).astype(int)
    return examples


def _plot_artifacts(frame: pd.DataFrame, thresholds: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = frame["label"].to_numpy(dtype=np.int64)
    scores = frame["probability_yes"].to_numpy(dtype=np.float64)

    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(recall, precision, label="Pooled outer-fold OOF")
    axis.axhline(float(labels.mean()), color="grey", linestyle="--", label="Prevalence")
    axis.set(title="Precision–Recall Curve (Exploratory OOF)", xlabel="Recall", ylabel="Precision")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(fpr, tpr, label="Pooled outer-fold OOF")
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    axis.set(title="ROC Curve (Exploratory OOF)", xlabel="False-positive rate", ylabel="True-positive rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    bins = np.linspace(0, 1, 51)
    axis.hist(frame.loc[labels == 0, "probability_yes"], bins=bins, density=True, alpha=0.55, label="Actual No")
    axis.hist(frame.loc[labels == 1, "probability_yes"], bins=bins, density=True, alpha=0.55, label="Actual Yes")
    axis.set(title="Held-out Probability Distributions", xlabel="Predicted probability of Yes", ylabel="Density")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "probability_histogram.png", dpi=160)
    plt.close(fig)

    x = thresholds["threshold"]
    plots = (
        ("predicted_yes", "Number of Predicted Yes by Threshold", "Predicted Yes", "predicted_yes_vs_threshold.png"),
        ("precision_recall", "Precision and Recall by Fixed Threshold", "Metric", "precision_recall_vs_threshold.png"),
        ("f1_f2", "F1 and F2 by Fixed Threshold", "Score", "f1_f2_vs_threshold.png"),
    )
    for key, title, ylabel, filename in plots:
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        if key == "predicted_yes":
            axis.plot(x, thresholds["predicted_yes"], marker="o", label="Predicted Yes")
        elif key == "precision_recall":
            axis.plot(x, thresholds["precision"], marker="o", label="Precision")
            axis.plot(x, thresholds["recall"], marker="o", label="Recall")
        else:
            axis.plot(x, thresholds["f1"], marker="o", label="F1")
            axis.plot(x, thresholds["f2"], marker="o", label="F2")
        axis.set(
            title=title,
            xlabel="Fixed probability threshold",
            ylabel=ylabel,
            xscale="log",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    fold_ap = (
        frame.groupby("fold", sort=True)
        .apply(
            lambda group: binary_metrics(
                group["label"].to_numpy(),
                group["probability_yes"].to_numpy(),
                threshold=0.5,
            )["average_precision"],
            include_groups=False,
        )
        .rename("average_precision")
    )
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.bar(fold_ap.index.astype(str), fold_ap.values)
    axis.set(
        title="Average Precision by Held-out Fold (Exploratory OOF)",
        xlabel="Family-aware fold",
        ylabel="Average Precision",
    )
    fig.tight_layout()
    fig.savefig(output / "per_fold_average_precision.png", dpi=160)
    plt.close(fig)


def _exp1_comparison(aggregate_root: Path) -> dict[str, Any]:
    a2_root = aggregate_root / "A2"
    per_seed = pd.read_csv(a2_root / "per_seed_metrics.csv")
    per_fold = pd.read_csv(a2_root / "per_fold_seed_metrics.csv")
    matched_seed = per_seed.loc[per_seed["seed"] == 2718]
    if len(matched_seed) != 1:
        raise RuntimeError("Experiment 1 A2 seed 2718 reference is not unique")
    matched_folds = per_fold.loc[per_fold["seed"] == 2718]
    return {
        "caveat": (
            "Descriptive only: labels and prevalence differ between annotation "
            "schemes, and this pilot uses one seed versus three in Experiment 1."
        ),
        "experiment_1_a2_all_seed_mean": {
            "mean_fold_average_precision": float(
                per_seed["mean_fold_average_precision"].mean()
            ),
            "mean_pooled_average_precision": float(
                per_seed["pooled_average_precision"].mean()
            ),
            "mean_pooled_roc_auc": float(per_seed["pooled_roc_auc"].mean()),
            "predicted_yes_at_0_5_by_seed": (
                per_seed["pooled_confusion_tp"]
                + per_seed["pooled_confusion_fp"]
            ).astype(int).tolist(),
        },
        "experiment_1_a2_seed_2718": matched_seed.iloc[0].to_dict(),
        "experiment_1_a2_seed_2718_fold_average_precision": matched_folds[
            ["fold", "average_precision", "positive_labels"]
        ].to_dict(orient="records"),
    }


def _read_training_cells(
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    inventory_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective, dataset_manifest, dataset_dir = load_effective_dataset(config)
    model_hash, model_files = hash_directory(Path(protocol["model"]["local_path"]))
    if model_hash != protocol["model"]["source_sha256"]:
        raise RuntimeError("Pinned base model hash changed before analysis")
    inventory = read_json(inventory_path)
    rows: list[pd.DataFrame] = []
    cell_summaries: list[dict[str, Any]] = []
    seed = int(protocol["training"]["seed"])
    for fold in protocol["training"]["folds"]:
        key = f"fold_{fold}/seed_{seed}"
        if inventory.get("cells", {}).get(key, {}).get("status") != "complete":
            raise RuntimeError(f"Cannot analyze incomplete cell {key}")
        run = resolve_path(inventory["cells"][key]["run_path"])
        predictions = validate_completed_run(
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
        predictions["fold"] = fold
        rows.append(predictions)
        run_manifest = read_json(run / "run_manifest.json")
        history = pd.read_csv(run / "history" / "training_history.csv")
        cell_summaries.append(
            {
                "fold": fold,
                "seed": seed,
                "run_path": str(run),
                "held_out_families": "|".join(run_manifest["held_out_families"]),
                "fitting_rows": run_manifest["fitting_rows"],
                "fitting_positive_labels": run_manifest["fitting_positive_labels"],
                "fitting_negative_labels": run_manifest["fitting_negative_labels"],
                "held_out_rows": run_manifest["held_out_rows"],
                "held_out_positive_labels": run_manifest["held_out_positive_labels"],
                "held_out_negative_labels": run_manifest["held_out_negative_labels"],
                "fitting_exclusion_count": run_manifest["fitting_exclusion_count"],
                "class_weights": json.dumps(
                    run_manifest["actual_class_weights"], sort_keys=True
                ),
                "epochs_completed": run_manifest["epochs_completed"],
                "final_epoch_training_loss": float(history.iloc[-1]["training_loss"]),
                "final_epoch_lr": float(history.iloc[-1]["learning_rate_after_epoch"]),
                "precision": run_manifest["precision"],
                "gpu": run_manifest["gpu"],
                "runtime_seconds": run_manifest["runtime_seconds"],
                "peak_gpu_memory_allocated_mb": run_manifest[
                    "peak_gpu_memory_allocated_mb"
                ],
            }
        )
    pooled = pd.concat(rows, ignore_index=True)
    expected = set(effective["record_id"].astype(str))
    if (
        len(pooled) != len(effective)
        or pooled["record_id"].astype(str).duplicated().any()
        or set(pooled["record_id"].astype(str)) != expected
        or set(pooled["fold"].astype(int)) != set(protocol["training"]["folds"])
    ):
        raise RuntimeError("Pooled OOF predictions do not represent the unique cohort")
    aligned = pooled[["record_id", "label"]].merge(
        effective[["record_id", "label"]],
        on="record_id",
        suffixes=("_run", "_dataset"),
        validate="one_to_one",
    )
    if not (aligned["label_run"] == aligned["label_dataset"]).all():
        raise RuntimeError("Pooled OOF labels differ from the frozen Dataset B cohort")
    return pooled, pd.DataFrame(cell_summaries)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    headers = [column.replace("_", " ").title() for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered = "undefined"
            elif column in formats:
                rendered = format(float(value), formats[column])
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    destination: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    source_audit: dict[str, Any],
    pooled: pd.DataFrame,
    cells: pd.DataFrame,
    threshold_pooled: pd.DataFrame,
    distributions: dict[str, Any],
    top_k: pd.DataFrame,
    families: pd.DataFrame,
    comparison: dict[str, Any],
    analysis_dir: Path,
) -> None:
    ranking = binary_metrics(
        pooled["label"].to_numpy(),
        pooled["probability_yes"].to_numpy(),
        threshold=0.5,
    )
    first_positive = threshold_pooled.loc[threshold_pooled["predicted_yes"] > 0]
    first_threshold = first_positive.iloc[0] if not first_positive.empty else None
    thresholds_with_tp = threshold_pooled.loc[
        threshold_pooled["true_positives"] > 0, "threshold"
    ].tolist()
    best_f1 = threshold_pooled.sort_values(
        ["f1", "threshold"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    best_f2 = threshold_pooled.sort_values(
        ["f2", "threshold"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    per_fold = pd.DataFrame(
        [
            {
                "fold": fold,
                **binary_metrics(
                    group["label"].to_numpy(),
                    group["probability_yes"].to_numpy(),
                    threshold=0.5,
                ),
            }
            for fold, group in pooled.groupby("fold", sort=True)
        ]
    )
    per_fold["predicted_yes"] = (
        per_fold["confusion_matrix"].map(lambda item: item["tp"] + item["fp"])
    )
    threshold_display = threshold_pooled.rename(
        columns={
            "predicted_yes": "predicted_yes",
            "true_positives": "tp",
            "false_positives": "fp",
            "false_negatives": "fn",
        }
    )
    report = [
        "# Final Annotation-Ready DeBERTa Pilot — Exploratory Results",
        "",
        "## Execution status and frozen protocol",
        "",
        f"- Protocol: `{protocol['protocol_id']}`; SHA-256 `{protocol_sha256}`.",
        f"- Dataset source SHA-256: `{protocol['dataset']['source_sha256']}`.",
        f"- Effective cohort SHA-256: `{protocol['dataset']['effective_dataset_sha256']}`.",
        "- Five sequential cells completed: folds 1–5, seed 2718, weighted cross-entropy.",
        "- Every cell used a fresh local DeBERTa-v3-base initialization, FP32 parameters and forward pass, FP32 cross-entropy, five fixed epochs, and no outer-fold selection.",
        "- No downloads or Experiment 1 outputs were used as training inputs.",
        "",
        "## Dataset verification and labels",
        "",
        f"- Source: `{source_audit['source_path']}`; {source_audit['rows']:,} rows; {source_audit['document_versions']} document versions; {source_audit['duplicate_item_ids']} duplicate item IDs; {source_audit['blank_target_clauses']} blank target clauses.",
        "- `final_label` is the only binary Yes/No outcome column in the dataset schema; `final_source` and `final_status` are provenance/status fields.",
        f"- Resolved source labels: Yes {source_audit['resolved_label_counts']['Yes']:,}; No {source_audit['resolved_label_counts']['No']:,}.",
        f"- Unresolved: {source_audit['excluded_unresolved_count']} record (item_id 2161); its placeholder No value was excluded under the explicit unresolved policy.",
        f"- Effective cohort: {len(pooled):,} clauses; {int(pooled['label'].sum()):,} Yes; {int((pooled['label'] == 0).sum()):,} No; prevalence {pooled['label'].mean():.2%}.",
        f"- The 121 candidate records absent from the approved annotation cohort remain excluded. TCP ({source_audit['reserved_partition_counts']['tcp_rows']}) and DNS ({source_audit['reserved_partition_counts']['dns_rows']}) source rows were not used.",
        f"- Leakage audit: {json.dumps(source_audit['leakage_audit_counts'], sort_keys=True)}; cross-fold document/family blocking was false.",
        "- Existing family-to-fold assignment was unchanged; fitting-side exact/near/lineage/BCP14 exclusions and context sanitization remain in force.",
        "",
        "## Fold counts, class weights, and training",
        "",
        _markdown_table(
            cells,
            [
                "fold",
                "held_out_families",
                "fitting_rows",
                "fitting_positive_labels",
                "fitting_negative_labels",
                "held_out_rows",
                "held_out_positive_labels",
                "held_out_negative_labels",
                "fitting_exclusion_count",
                "class_weights",
                "final_epoch_training_loss",
                "runtime_seconds",
            ],
            {"final_epoch_training_loss": ".5f", "runtime_seconds": ".1f"},
        ),
        "",
        f"- Python {protocol['environment']['python']}; PyTorch {protocol['environment']['torch']}; CUDA {protocol['environment']['cuda_runtime']}; cuDNN {protocol['environment']['cudnn']}; GPU {protocol['environment']['gpu']}; driver {protocol['environment']['driver']}.",
        "",
        "## Threshold-independent ranking and calibration diagnostics",
        "",
        f"- AP / PR-AUC: **{ranking['average_precision']:.4f}**; new-cohort prevalence baseline: **{ranking['positive_prevalence']:.4f}**; AP/prevalence lift: **{ranking['ap_prevalence_lift']:.3f}**.",
        f"- ROC-AUC: **{ranking['roc_auc']:.4f}**; actual positives {ranking['positive_labels']}; actual negatives {ranking['negative_labels']}.",
        f"- Brier score: {distributions['brier_score']:.5f}; prevalence-only descriptive baseline: {distributions['prevalence_only_brier_baseline_descriptive']:.5f}.",
        f"- Histogram overlap coefficient for Yes/No score distributions: {distributions['positive_negative_histogram_overlap_coefficient_50_bins']:.3f}. This is a descriptive 50-bin overlap measure, not a hypothesis test.",
        "- AP above prevalence and ROC-AUC above 0.5 are evidence of ranking signal on this one-seed cohort, not proof of robust family-generalizable learning.",
        "",
        "## Fixed-threshold held-out results (exploratory)",
        "",
        "All thresholds below were applied to the pooled, unique 2,142-record OOF cohort with `probability_yes >= threshold`. These are exploratory outer-fold diagnostics, not an unbiased threshold-selection procedure.",
        "",
        _markdown_table(
            threshold_display,
            [
                "threshold",
                "predicted_yes",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "specificity",
                "f1",
                "f2",
                "balanced_accuracy",
                "mcc",
            ],
            {
                "threshold": ".3g",
                "precision": ".4f",
                "recall": ".4f",
                "specificity": ".4f",
                "f1": ".4f",
                "f2": ".4f",
                "balanced_accuracy": ".4f",
                "mcc": ".4f",
            },
        ),
        "",
        (
            f"- First requested threshold producing any predicted Yes: **{first_threshold['threshold']:g}**, "
            f"{int(first_threshold['predicted_yes'])} predictions, of which "
            f"{int(first_threshold['true_positives'])} were correct."
            if first_threshold is not None
            else "- No requested threshold produced a predicted Yes."
        ),
        f"- Thresholds with at least one true positive: {', '.join(f'{value:g}' for value in thresholds_with_tp) if thresholds_with_tp else 'none'}.",
        f"- At reference threshold 0.5: {int(threshold_pooled.iloc[-1]['predicted_yes'])} predicted Yes, {int(threshold_pooled.iloc[-1]['true_positives'])} correct.",
        f"- Highest observed F1 is {best_f1['f1']:.4f} at {best_f1['threshold']:g}; highest observed F2 is {best_f2['f2']:.4f} at {best_f2['threshold']:g}. These are post-hoc exploratory optima and must not be treated as selected operating thresholds.",
        "",
        "### Per-fold threshold-independent performance at the fixed 0.5 reference",
        "",
        _markdown_table(
            per_fold,
            ["fold", "rows", "positive_labels", "average_precision", "roc_auc", "predicted_yes", "false_positives", "false_negatives"],
            {"average_precision": ".4f", "roc_auc": ".4f"},
        ),
        "",
        "### Score distributions",
        "",
        f"- Actual Yes scores: {json.dumps(distributions['actual_yes_score_distribution'], sort_keys=True)}",
        f"- Actual No scores: {json.dumps(distributions['actual_no_score_distribution'], sort_keys=True)}",
        f"- Scores below 0.01 / 0.05 / 0.10: {distributions['probability_below_0_01_percent']:.2f}% / {distributions['probability_below_0_05_percent']:.2f}% / {distributions['probability_below_0_10_percent']:.2f}%; scores above 0.50: {distributions['probability_above_0_50_percent']:.2f}%.",
        f"- Mean predicted positive probability: {distributions['average_predicted_positive_probability']:.4f}; actual prevalence: {distributions['actual_positive_prevalence']:.4f}.",
        "",
        "## Human-review retrieval (exploratory)",
        "",
        _markdown_table(
            top_k,
            [
                "k",
                "true_ambiguous_retrieved",
                "precision_at_k",
                "recall_at_k",
                "lift_over_positive_prevalence",
            ],
            {
                "precision_at_k": ".4f",
                "recall_at_k": ".4f",
                "lift_over_positive_prevalence": ".3f",
            },
        ),
        "",
        "## Family-level results",
        "",
        _markdown_table(
            families,
            [
                "family_id",
                "rows",
                "positive_labels",
                "positive_prevalence",
                "average_precision",
                "roc_auc",
                "ap_prevalence_lift",
            ],
            {
                "positive_prevalence": ".3f",
                "average_precision": ".4f",
                "roc_auc": ".4f",
                "ap_prevalence_lift": ".3f",
            },
        ),
        "",
        "## Error examples and interpretation",
        "",
        "Complete example rows, target clauses, labels, probabilities, folds, and predictions at illustrative thresholds are in `error_analysis_examples.csv`. These are held-out examples only; no annotations were changed. Semantic categories (vagueness, subjective quantity, unspecified implementation choice, conditional complexity, semantic ambiguity, missing protocol context) require human interpretation and are not inferred from predicted scores.",
        "",
        "## Descriptive comparison with Experiment 1",
        "",
        comparison["caveat"],
        "",
        f"- Experiment 1 A2 all-seed means: mean-fold AP {comparison['experiment_1_a2_all_seed_mean']['mean_fold_average_precision']:.4f}; mean pooled AP {comparison['experiment_1_a2_all_seed_mean']['mean_pooled_average_precision']:.4f}; mean pooled ROC-AUC {comparison['experiment_1_a2_all_seed_mean']['mean_pooled_roc_auc']:.4f}.",
        f"- Experiment 1 A2 predicted-Yes counts at 0.5 by seed: {comparison['experiment_1_a2_all_seed_mean']['predicted_yes_at_0_5_by_seed']}. Matched Experiment 1 A2 seed-2718 pooled and fold metrics are in `experiment_1_reference.json`; this comparison remains confounded by changed labels and prevalence.",
        "",
        "## Interpretation and limitations",
        "",
        f"1. Positive predictions at threshold 0.5: {int(threshold_pooled.iloc[-1]['predicted_yes'])}.",
        f"2. Predictions first emerge at the lowest listed active threshold: {first_threshold['threshold']:g}." if first_threshold is not None else "2. No positive predictions emerge at the tested thresholds.",
        f"3. Nonzero true positives occur at: {', '.join(f'{value:g}' for value in thresholds_with_tp) if thresholds_with_tp else 'no tested threshold'}.",
        "4. Precision generally trades off against recall as the fixed threshold is lowered; use the complete table rather than the post-hoc maximum as a decision rule.",
        f"5. AP/prevalence lift is {ranking['ap_prevalence_lift']:.3f}; any apparent ranking signal remains a one-seed exploratory finding.",
        f"6. Family AP ranges from {families['average_precision'].min():.4f} to {families['average_precision'].max():.4f}; the family-wise sample sizes and positive counts vary sharply.",
        "7. The revised annotations contain more resolved Yes labels than the original Dataset A effective cohort, but the changed target definition and single-seed pilot prevent a definitive claim that this annotation scheme is more learnable.",
        "8. One appropriate next controlled experiment is a pre-registered nested family-aware calibration study: choose operating thresholds using fitting-side inner OOF predictions only, then evaluate once on untouched outer families; do not reuse this pilot's exploratory outer-fold optimum.",
        "",
        "## Plots",
        "",
        "- `plots/precision_recall_curve.png`",
        "- `plots/roc_curve.png`",
        "- `plots/probability_histogram.png`",
        "- `plots/predicted_yes_vs_threshold.png`",
        "- `plots/precision_recall_vs_threshold.png`",
        "- `plots/f1_f2_vs_threshold.png`",
        "- `plots/per_fold_average_precision.png`",
        "",
        "## Provenance",
        "",
        f"- Frozen source snapshot SHA-256: `{protocol['source_snapshot']['sha256']}`.",
        f"- Protocol hash: `{protocol_sha256}`.",
        f"- Inventory: `{analysis_dir.parent / 'audit_logs' / 'run_management' / 'completion_inventory.json'}`.",
        "- Per-run manifests, checkpoint hashes, source hashes, model/tokenizer hashes, and dependency/CUDA metadata are retained alongside their run outputs.",
        "",
    ]
    (destination / "research_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def create_experiment2_analysis(
    *,
    config_path: Path,
    protocol_path: Path,
    inventory_path: Path,
    analysis_path: Path,
) -> Path:
    config = load_config(config_path)
    protocol, protocol_sha256 = load_annotation_ready_protocol(protocol_path, config)
    expected_analysis_path = resolve_path(config["outputs"]["aggregate_dir"])
    if analysis_path.resolve() != expected_analysis_path:
        raise RuntimeError("Analysis output path differs from isolated Experiment 2 config")
    pooled, cells = _read_training_cells(
        config=config,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        inventory_path=inventory_path,
    )
    threshold_folds = pd.concat(
        [
            _threshold_table(pooled, fold=int(fold))
            for fold in sorted(protocol["training"]["folds"])
        ],
        ignore_index=True,
    )
    threshold_pooled = _threshold_table(pooled)
    threshold_overall = threshold_pooled.copy()
    distributions = _distribution_diagnostics(pooled)
    top_k = _top_k_table(pooled)
    families = _family_metrics(pooled)
    examples = _error_examples(pooled)
    original_root = resolve_path(
        "results/transformer/final_studies/original_annotation/aggregate_metrics"
    )
    comparison = _exp1_comparison(original_root)
    source_audit = read_json(
        resolve_path(config["outputs"]["audit_dir"]) / "dataset_source_audit.json"
    )
    with immutable_directory(analysis_path) as building:
        pooled.sort_values(["fold", "item_id"], kind="mergesort").to_csv(
            building / "pooled_oof_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        cells.to_csv(building / "training_cells.csv", index=False, lineterminator="\n")
        threshold_folds.to_csv(
            building / "thresholds_per_fold.csv", index=False, lineterminator="\n"
        )
        threshold_pooled.to_csv(
            building / "thresholds_pooled_oof.csv", index=False, lineterminator="\n"
        )
        threshold_overall.to_csv(
            building / "thresholds_overall_one_seed.csv",
            index=False,
            lineterminator="\n",
        )
        _plot_artifacts(pooled, threshold_pooled, building / "plots")
        families.to_csv(building / "family_metrics.csv", index=False, lineterminator="\n")
        top_k.to_csv(building / "top_k_retrieval.csv", index=False, lineterminator="\n")
        examples.to_csv(
            building / "error_analysis_examples.csv",
            index=False,
            lineterminator="\n",
        )
        write_json(building / "probability_distribution.json", distributions)
        write_json(
            building / "threshold_independent_metrics.json",
            binary_metrics(
                pooled["label"].to_numpy(),
                pooled["probability_yes"].to_numpy(),
                threshold=0.5,
            ),
        )
        write_json(building / "experiment_1_reference.json", comparison)
        write_json(
            building / "dataset_source_audit.json",
            source_audit,
        )
        _write_report(
            destination=building,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            source_audit=source_audit,
            pooled=pooled,
            cells=cells,
            threshold_pooled=threshold_pooled,
            distributions=distributions,
            top_k=top_k,
            families=families,
            comparison=comparison,
            analysis_dir=analysis_path.resolve(),
        )
        artifact_hashes = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        write_json(
            building / "analysis_manifest.json",
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "source_snapshot_sha256": protocol["source_snapshot"]["sha256"],
                "effective_dataset_sha256": protocol["dataset"][
                    "effective_dataset_sha256"
                ],
                "oof_rows": len(pooled),
                "unique_oof_record_ids": int(pooled["record_id"].nunique()),
                "thresholds": list(THRESHOLDS),
                "artifact_hashes": artifact_hashes,
            },
        )
    return analysis_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = create_experiment2_analysis(
        config_path=args.config,
        protocol_path=args.protocol,
        inventory_path=args.inventory,
        analysis_path=args.output,
    )
    print(f"Experiment 2 analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
