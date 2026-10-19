"""Generate the final report from verified FP32 aggregate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import resolve_path
from .config import load_config


def _metric_summary(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(float)
    if values.empty:
        return "undefined"
    return f"{values.mean():.4f} (SD {values.std(ddof=1):.4f})"


def write_research_report(
    *,
    config_path: Path,
    protocol_path: Path,
    inventory_path: Path,
    output_path: Path,
) -> Path:
    config = load_config(config_path)
    root = resolve_path(config["outputs"]["aggregate_dir"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    per_fold = {
        condition: pd.read_csv(root / condition / "per_fold_seed_metrics.csv")
        for condition in ("A1", "A2")
    }
    per_seed = {
        condition: pd.read_csv(root / condition / "per_seed_metrics.csv")
        for condition in ("A1", "A2")
    }
    family = {
        condition: pd.read_csv(root / condition / "per_family_seed_metrics.csv")
        for condition in ("A1", "A2")
    }
    comparison_path = root / "A1_vs_A2_fp32_v1" / "comparison_summary.json"
    comparison: dict[str, Any] = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )
    required_cells = {
        f"{condition}/fold_{fold}/seed_{seed}"
        for condition in ("A1", "A2")
        for seed in config["training"]["seeds"]
        for fold in range(1, int(config["folds"]["count"]) + 1)
    }
    if (
        len(per_fold["A1"]) != 15
        or len(per_fold["A2"]) != 15
        or inventory.get("protocol_id") != "original_annotation_experiment_1_fp32_v1"
        or set(inventory.get("cells", {})) != required_cells
        or any(
            entry.get("status") != "complete"
            for entry in inventory["cells"].values()
        )
        or set(inventory.get("aggregations", {})) != {"A1", "A2"}
    ):
        raise RuntimeError("Cannot report incomplete or unverified experiment results")

    metrics = (
        ("Mean-fold AP", "mean_fold_average_precision", None),
        ("Pooled OOF AP", "pooled_average_precision", None),
        ("Pooled OOF ROC-AUC", "pooled_roc_auc", None),
        ("Pooled precision @ 0.5", "pooled_precision", None),
        ("Pooled recall @ 0.5", "pooled_recall", None),
        ("Pooled F1 @ 0.5", "pooled_f1", None),
        ("Pooled F2 @ 0.5", "pooled_f2", None),
        ("Pooled MCC @ 0.5", "pooled_mcc", None),
        ("Pooled AP / prevalence", "pooled_ap_prevalence_lift", None),
        ("Pooled accuracy @ 0.5", "pooled_accuracy", None),
        ("Pooled false positives", "pooled_false_positives", None),
        ("Pooled false negatives", "pooled_false_negatives", None),
    )
    lines = [
        "# Dataset A Transformer Experiment 1 — Frozen FP32 Results",
        "",
        "## Execution status",
        "",
        "- Required cells: 30 (15 A1 and 15 A2).",
        f"- Verified completed cells: {len(inventory['cells'])}.",
        "- Failed cells: 0; remaining cells: 0.",
        "- A1 and A2 each contain five folds by three seeds.",
        "",
        "## Frozen protocol and environment",
        "",
        f"- Protocol: `{inventory['protocol_id']}`; SHA-256 `{inventory['protocol_sha256']}`.",
        f"- Protocol file: `{protocol_path}`.",
        f"- Dataset: Dataset A, source hash `{config['data']['source_sha256']}`.",
        "- Effective Dataset A development rows: 2,142; positive labels: 45.",
        "- Model: microsoft/deberta-v3-base, freshly initialized from the same local base checkpoint for every cell.",
        "- Parameter storage and forward/loss precision: FP32; autocast disabled.",
        "- Optimizer: AdamW; learning rate 1e-5; weight decay 0.01.",
        "- Scheduler: linear decay with 10% warmup; 5 fixed epochs; batch 8; accumulation 1; gradient clipping norm 1.0.",
        "- Seeds: 1729, 2718, 31415; folds: the existing five family-based splits.",
        "- GPU and software versions are recorded in each future run manifest and the frozen protocol.",
        "",
        "## Reuse of the completed diagnostic cell",
        "",
        "A1 fold 1 / seed 1729 was reused from the immutable `seed_1729_diagnostic_numerics_v2` directory. Its condition, standard cross-entropy, seed/fold, FP32 checkpoint tensors, data/fold/config hashes, local base-model hashes, predictions, history, and checkpoint hashes were verified. Its original configuration uses the experiment-v1 hash and a CLI FP32 override; the explicit mapping marks this exact legacy cell as compatible with this frozen FP32 protocol. No historical source-code hash is asserted: its executed code provenance remains incomplete because relevant source files were untracked.",
        "",
        "## Results",
        "",
        "Primary metric is average precision (PR-AUC). Pooled metrics are calculated from each seed's complete out-of-fold predictions; means and sample standard deviations below summarize the three seeds.",
        "",
        "| Metric | A1 standard CE | A2 weighted CE |",
        "|---|---:|---:|",
    ]
    for label, column, _ in metrics:
        lines.append(
            f"| {label} | {_metric_summary(per_seed['A1'], column)} | "
            f"{_metric_summary(per_seed['A2'], column)} |"
        )

    lines.extend(
        [
            "",
            "### Paired A2 minus A1 differences",
            "",
            "Intervals are descriptive normal 95% intervals over the 15 paired fold/seed cells or the three paired seed-level pooled metrics; they are not significance tests.",
            "",
            "| Metric | Mean difference | Sample SD | Descriptive 95% interval |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, stats in comparison["fold_seed_differences_A2_minus_A1"].items():
        interval = stats["descriptive_normal_95_percent_interval"]
        lines.append(
            f"| {metric} | {stats['mean']:.4f} | {stats['sample_std']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "### Fold positive counts and A2 fitting-only weights",
            "",
            "| Seed | Fold | Held-out positives | A1 AP | A2 AP | A2 class weights (No, Yes) |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    a1_by_cell = per_fold["A1"].set_index(["seed", "fold"])
    a2_by_cell = per_fold["A2"].set_index(["seed", "fold"])
    for seed in config["training"]["seeds"]:
        for fold in range(1, int(config["folds"]["count"]) + 1):
            a1row = a1_by_cell.loc[(seed, fold)]
            a2row = a2_by_cell.loc[(seed, fold)]
            lines.append(
                f"| {seed} | {fold} | {int(a1row['positive_labels'])} | "
                f"{a1row['average_precision']:.4f} | "
                f"{a2row['average_precision']:.4f} | `{a2row['class_weights']}` |"
            )
    family_ap = (
        family["A1"].groupby("family_id")["average_precision"].mean().sort_values()
    )
    lines.extend(
        [
            "",
            "## Scientific interpretation and limitations",
            "",
            f"- Dataset A has only 45 positive examples across nine development families; effective prevalence is 45/2,142 ({45 / 2142:.2%}). Some held-out folds have very few positives, so fold-level AP can be highly variable.",
            "- Accuracy is secondary and may appear high simply because the negative class dominates. Threshold-0.5 precision/recall/F1/F2 and confusion counts are reported alongside ranking metrics.",
            "- AP should be read relative to the effective positive prevalence; AP/prevalence lift is included. Raw AP should not be compared directly with Dataset B, whose prevalence differs.",
            "- Decreasing training loss is not evidence of held-out generalization. No outer-fold labels were used for tuning or threshold selection.",
            "- Results are descriptive across five family folds and three seeds. The intervals are not inferential significance tests; family heterogeneity and small positive counts limit certainty.",
            "- A2 class weights are verified from each fold's fitting partition only. A1 and A2 differ only in the pre-registered loss weighting under the shared FP32 protocol.",
            "- Dataset B, TCP, and DNS were excluded. Held-out predictions were validated against the Dataset A fold scoring IDs; the reserved families are absent from the effective development dataset.",
            "- A1 fold 1 / seed 1729 has incomplete executed-code provenance, as described above. New runs record deterministic source-file hashes, dependency versions, model/data/config/fold hashes, and CUDA environment details.",
            "",
            "## Artifact locations",
            "",
            f"- Durable completion inventory: `{inventory_path}`.",
            f"- A1 aggregate: `{root / 'A1'}`.",
            f"- A2 aggregate: `{root / 'A2'}`.",
            f"- Paired comparison: `{root / 'A1_vs_A2_fp32_v1'}`.",
            f"- A1 mean fold AP across seeds: {per_seed['A1']['mean_fold_average_precision'].mean():.4f}.",
            f"- A2 mean fold AP across seeds: {per_seed['A2']['mean_fold_average_precision'].mean():.4f}.",
            f"- Family AP means (A1, descriptive range): {family_ap.index[0]} {family_ap.iloc[0]:.4f} to {family_ap.index[-1]} {family_ap.iloc[-1]:.4f}.",
            "",
        ]
    )
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite research report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
