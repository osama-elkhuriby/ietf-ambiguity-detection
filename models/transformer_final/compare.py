"""Paired descriptive comparison of completed A1 and A2 outer-fold cells."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .common import immutable_directory, resolve_path, write_json
from .config import DEFAULT_CONFIG, load_config


def compare(config_path: Path, *, output: Path) -> Path:
    config = load_config(config_path)
    root = resolve_path(config["outputs"]["aggregate_dir"])
    a1_fold = pd.read_csv(root / "A1" / "per_fold_seed_metrics.csv")
    a2_fold = pd.read_csv(root / "A2" / "per_fold_seed_metrics.csv")
    paired_folds = a1_fold.merge(
        a2_fold,
        on=["seed", "fold"],
        suffixes=("_A1", "_A2"),
        validate="one_to_one",
    )
    metrics = (
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "accuracy",
        "ap_prevalence_lift",
        "false_positives",
        "false_negatives",
    )
    if len(paired_folds) != 15:
        raise RuntimeError(f"Expected 15 paired fold/seed cells, found {len(paired_folds)}")
    for metric in metrics:
        paired_folds[f"difference_{metric}_A2_minus_A1"] = (
            paired_folds[f"{metric}_A2"] - paired_folds[f"{metric}_A1"]
        )

    def describe_difference(values: pd.Series) -> dict[str, object]:
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        standard_error = std / math.sqrt(len(values))
        return {
            "mean": mean,
            "sample_std": std,
            "standard_error": standard_error,
            "descriptive_normal_95_percent_interval": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
        }

    pooled_a1 = pd.read_csv(root / "A1" / "per_seed_metrics.csv")
    pooled_a2 = pd.read_csv(root / "A2" / "per_seed_metrics.csv")
    paired_seeds = pooled_a1.merge(
        pooled_a2,
        on="seed",
        suffixes=("_A1", "_A2"),
        validate="one_to_one",
    )
    pooled_metrics = tuple(f"pooled_{metric}" for metric in metrics)
    for metric in pooled_metrics:
        paired_seeds[f"difference_{metric}_A2_minus_A1"] = (
            paired_seeds[f"{metric}_A2"] - paired_seeds[f"{metric}_A1"]
        )

    family_a1 = pd.read_csv(root / "A1" / "per_family_seed_metrics.csv")
    family_a2 = pd.read_csv(root / "A2" / "per_family_seed_metrics.csv")
    paired_families = family_a1.merge(
        family_a2,
        on=["seed", "family_id"],
        suffixes=("_A1", "_A2"),
        validate="one_to_one",
    )
    if len(paired_seeds) != len(config["training"]["seeds"]):
        raise RuntimeError("Pooled seed comparison does not contain each seed once")
    if len(paired_families) != len(config["training"]["seeds"]) * len(
        config["partitions"]["train_families"]
    ):
        raise RuntimeError("Family comparison is incomplete or contains duplicate cells")
    family_metric = "average_precision"
    paired_families[f"difference_{family_metric}_A2_minus_A1"] = (
        paired_families[f"{family_metric}_A2"]
        - paired_families[f"{family_metric}_A1"]
    )
    fold_metric_differences = {
        metric: describe_difference(
            paired_folds[f"difference_{metric}_A2_minus_A1"]
        )
        for metric in metrics
    }
    seed_metric_differences = {
        metric: describe_difference(
            paired_seeds[f"difference_{metric}_A2_minus_A1"]
        )
        for metric in pooled_metrics
    }
    report = {
        "schema_version": 1,
        "interpretation": (
            "Paired descriptive differences only; intervals are descriptive and "
            "do not establish statistical significance or superiority."
        ),
        "primary_metric": "average_precision",
        "threshold": float(config["training"]["threshold"]),
        "paired_cells": len(paired_folds),
        "paired_seeds": len(paired_seeds),
        "paired_seed_family_cells": len(paired_families),
        "fold_seed_differences_A2_minus_A1": fold_metric_differences,
        "pooled_oof_seed_differences_A2_minus_A1": seed_metric_differences,
        "fold_level_average_precision_differences": paired_folds[
            ["seed", "fold", "average_precision_A1", "average_precision_A2",
             "difference_average_precision_A2_minus_A1",
             "positive_labels_A1", "positive_labels_A2"]
        ].to_dict(orient="records"),
        "seed_level_pooled_metrics": paired_seeds[
            [
                "seed",
                "pooled_average_precision_A1",
                "pooled_average_precision_A2",
                "difference_pooled_average_precision_A2_minus_A1",
                "pooled_recall_A1",
                "pooled_recall_A2",
                "pooled_precision_A1",
                "pooled_precision_A2",
                "pooled_f1_A1",
                "pooled_f1_A2",
                "pooled_false_positives_A1",
                "pooled_false_positives_A2",
                "pooled_false_negatives_A1",
                "pooled_false_negatives_A2",
            ]
        ].to_dict(orient="records"),
        "family_level_average_precision_differences": paired_families[
            [
                "seed",
                "family_id",
                "average_precision_A1",
                "average_precision_A2",
                "difference_average_precision_A2_minus_A1",
                "positive_labels_A1",
            ]
        ].to_dict(orient="records"),
        "a2_fitting_only_class_weights": a2_fold[
            ["seed", "fold", "class_weights", "positive_labels", "negative_labels"]
        ].to_dict(orient="records"),
    }
    with immutable_directory(output.resolve()) as building:
        paired_folds.to_csv(
            building / "paired_fold_seed_comparison.csv",
            index=False,
            lineterminator="\n",
        )
        paired_seeds.to_csv(
            building / "paired_pooled_seed_comparison.csv",
            index=False,
            lineterminator="\n",
        )
        paired_families.to_csv(
            building / "paired_family_seed_comparison.csv",
            index=False,
            lineterminator="\n",
        )
        write_json(building / "comparison_summary.json", report)
    return output.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Comparison written to {compare(args.config, output=args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
