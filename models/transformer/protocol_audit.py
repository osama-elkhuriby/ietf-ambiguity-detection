"""Generate the pre-ablation nested-CV protocol audit from development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import (
    WeightStrategy,
    compute_training_weights,
    load_development_split,
    load_frozen_folds,
    load_nested_folds,
)

FOLDS_PATH = ROOT / "experiments/transformer/configs/folds_v1.json"
INNER_PATH = ROOT / "experiments/transformer/configs/inner_selection_v1.json"
DEVELOPMENT_DIR = ROOT / "corpus/processed/transformer_partitioned/v1/development"
V1_ROOT = ROOT / "experiments/transformer/results/lr-screen-v1"


def history_checkpoint_record(
    history: pd.DataFrame, *, best_epoch: int
) -> dict[str, float | int | None]:
    """Extract selected/final/fixed-epoch held-out PR-AUC from a v1 history."""

    best = history.loc[history["epoch"] == best_epoch, "pr_auc"]
    if len(best) != 1:
        raise ValueError("Best epoch is absent or duplicated in training history")
    final = history.iloc[-1]

    def at_epoch(epoch: int) -> float | None:
        values = history.loc[history["epoch"] == epoch, "pr_auc"]
        return float(values.iloc[0]) if len(values) == 1 else None

    best_value = float(best.iloc[0])
    final_value = float(final["pr_auc"])
    epoch3 = at_epoch(3)
    epoch5 = at_epoch(5)
    return {
        "best_epoch": best_epoch,
        "best_pr_auc": best_value,
        "final_epoch": int(final["epoch"]),
        "final_pr_auc": final_value,
        "best_minus_final": best_value - final_value,
        "epoch_3_pr_auc": epoch3,
        "best_minus_epoch_3": best_value - epoch3 if epoch3 is not None else None,
        "epoch_5_pr_auc": epoch5,
        "best_minus_epoch_5": best_value - epoch5 if epoch5 is not None else None,
    }


def _weight_record(
    frame: pd.DataFrame, scope: str, fold: int | None, strategy: WeightStrategy
) -> dict[str, Any]:
    weights = compute_training_weights(frame, strategy)
    labels = frame["label"].to_numpy(dtype=np.int64)
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    return {
        "scope": scope,
        "outer_fold": fold,
        "weight_strategy": strategy,
        "rows": len(frame),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "positive_effective_weight": positive,
        "negative_effective_weight": negative,
        "positive_weight_share": positive / (positive + negative),
        "positive_to_negative_weight_ratio": positive / negative,
        "mean_weight": float(weights.mean()),
    }


def run(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Protocol audit already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    train = load_development_split(DEVELOPMENT_DIR, "train")
    train_hash = sha256_file(DEVELOPMENT_DIR / "train.csv")
    outer = load_frozen_folds(
        train,
        FOLDS_PATH,
        expected_sha256=sha256_file(FOLDS_PATH),
        expected_train_sha256=train_hash,
    )
    nested = load_nested_folds(
        train,
        outer,
        INNER_PATH,
        expected_sha256=sha256_file(INNER_PATH),
        expected_train_sha256=train_hash,
    )

    history_rows: list[dict[str, Any]] = []
    for learning_rate in ("1e-5", "2e-5", "3e-5"):
        for fold in range(1, 6):
            directory = V1_ROOT / f"lr-{learning_rate}" / f"fold-{fold}-seed-1729"
            history = pd.read_csv(directory / "training_history.csv")
            manifest = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            history_rows.append(
                {
                    "learning_rate": learning_rate,
                    "fold": fold,
                    **history_checkpoint_record(
                        history, best_epoch=int(manifest["best_epoch"])
                    ),
                }
            )
    histories = pd.DataFrame(history_rows)
    histories.to_csv(
        output_dir / "v1_checkpoint_selection.csv", index=False, lineterminator="\n"
    )

    tier_rows: list[dict[str, Any]] = []
    for tier, group in train.groupby("label_tier", sort=True):
        tier_rows.append(
            {
                "scope": "full_train",
                "outer_fold": None,
                "label_tier": tier,
                "rows": len(group),
                "positives": int(group["label"].sum()),
            }
        )
    for fold in outer:
        scored = train.iloc[fold.scoring_indices]
        for tier, group in scored.groupby("label_tier", sort=True):
            tier_rows.append(
                {
                    "scope": "outer_scoring_fold",
                    "outer_fold": fold.number,
                    "label_tier": tier,
                    "rows": len(group),
                    "positives": int(group["label"].sum()),
                }
            )
    pd.DataFrame(tier_rows).to_csv(
        output_dir / "tier_positive_counts.csv", index=False, lineterminator="\n"
    )

    weight_rows: list[dict[str, Any]] = []
    strategies: tuple[WeightStrategy, ...] = (
        "unweighted",
        "class",
        "confidence",
        "combined",
    )
    for strategy in strategies:
        weight_rows.append(_weight_record(train, "full_train", None, strategy))
    for nested_fold in nested:
        fitting = train.iloc[nested_fold.fitting_indices]
        for strategy in strategies:
            weight_rows.append(
                _weight_record(
                    fitting,
                    "nested_outer_fitting",
                    nested_fold.outer.number,
                    strategy,
                )
            )
    pd.DataFrame(weight_rows).to_csv(
        output_dir / "effective_weight_totals.csv", index=False, lineterminator="\n"
    )

    summary = {
        "schema_version": 1,
        "v1_protocol_finding": {
            "early_stopping_partition": "held_out_outer_fold",
            "checkpoint_selection_partition": "held_out_outer_fold",
            "checkpoint_metric": "outer_fold_pr_auc",
            "outer_fold_reused_for_reported_score": True,
        },
        "tfidf_protocol_finding": {
            "held_out_fold_used_for_hyperparameter_tuning": False,
            "fixed_hyperparameters": {
                "C": 1.0,
                "solver": "liblinear",
                "min_df": 2,
                "max_features": 20000,
                "ngram_range": [1, 2],
            },
        },
        "mean_v1_best_minus_final_pr_auc": float(
            histories["best_minus_final"].mean()
        ),
        "mean_v1_best_minus_epoch_3_pr_auc": float(
            histories["best_minus_epoch_3"].mean(skipna=True)
        ),
        "mean_v1_best_minus_epoch_5_pr_auc": float(
            histories["best_minus_epoch_5"].mean(skipna=True)
        ),
        "epoch_3_available_cells": int(histories["epoch_3_pr_auc"].notna().sum()),
        "epoch_5_available_cells": int(histories["epoch_5_pr_auc"].notna().sum()),
        "fold_assignment_sha256": sha256_file(FOLDS_PATH),
        "inner_selection_assignment_sha256": sha256_file(INNER_PATH),
        "development_train_sha256": train_hash,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/transformer/governance/step2b_protocol_audit_v1",
    )
    args = parser.parse_args()
    print(run(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
