"""Generate paper figures for the final three-model rank-ensemble experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "results/hybrid/final_rank_ensemble_v2/oof_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper/figures"
_REVIEW_BUDGETS = np.arange(1, 51)

_MODELS = (
    ("ML logistic regression", "ml_probability", "#0072B2"),
    ("V4 semantic ensemble", "v4_probability", "#D55E00"),
    ("DeBERTa-v3-base", "transformer_probability", "#009E73"),
    ("V4 + ML", "v4_ml_rank_score", "#CC79A7"),
    ("V4 + DeBERTa", "v4_transformer_rank_score", "#E69F00"),
    ("V4 + DeBERTa + ML", "hybrid_rank_score", "#332288"),
)


def _load_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {"label", *(column for _, column, _ in _MODELS)}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Hybrid predictions lack columns: {sorted(missing)}")
    labels = pd.to_numeric(predictions["label"], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1 values")
    if labels.nunique() != 2:
        raise ValueError("Both classes are required to plot evaluation results")
    predictions["label"] = labels.astype(int)
    for name, column, _ in _MODELS:
        scores = pd.to_numeric(predictions[column], errors="raise")
        if not np.isfinite(scores).all():
            raise ValueError(f"{name} scores contain non-finite values")
        predictions[column] = scores.astype(float)
    return predictions


def plot_precision_recall(predictions: pd.DataFrame, output_dir: Path) -> None:
    labels = predictions["label"].to_numpy(dtype=int)
    prevalence = float(labels.mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for name, column, color in _MODELS:
        scores = predictions[column].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.0 if name == "V4 + DeBERTa + ML" else 1.5,
            label=f"{name} (AP={ap:.3f})",
        )
    ax.axhline(
        prevalence,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label=f"Prevalence ({prevalence:.3f})",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall on the label-compatible OOF cohort",
    )
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    _save_figure(fig, output_dir, "hybrid_pr_curves")


def plot_review_budget(predictions: pd.DataFrame, output_dir: Path) -> None:
    labels = predictions["label"].to_numpy(dtype=int)
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("At least one positive label is required for recall")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fractions = _REVIEW_BUDGETS / 100
    ax.plot(
        fractions * 100,
        fractions,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="Uniform-review expectation",
    )
    for name, column, color in _MODELS:
        scores = predictions[column].to_numpy(dtype=float)
        order = np.argsort(-scores, kind="stable")
        cumulative_positives = np.cumsum(labels[order])
        retrieved = [
            cumulative_positives[max(1, int(np.ceil(len(labels) * fraction))) - 1]
            / positive_count
            for fraction in fractions
        ]
        ax.plot(
            fractions * 100,
            retrieved,
            color=color,
            linewidth=2.0 if name == "V4 + DeBERTa + ML" else 1.5,
            label=name,
        )
    ax.set(
        xlim=(0, 50),
        ylim=(0, 1),
        xlabel="Top-ranked clauses reviewed (%)",
        ylabel="Recall of operational-reference positives",
        title="Positive retrieval versus review budget",
    )
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    _save_figure(fig, output_dir, "hybrid_review_budget")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        destination = output_dir / f"{stem}.{suffix}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite paper figure: {destination}")
        fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    predictions = _load_predictions(args.predictions)
    plot_precision_recall(predictions, args.output_dir)
    plot_review_budget(predictions, args.output_dir)
    print(
        f"Generated paper figures from {len(predictions)} OOF rows "
        f"({int(predictions['label'].sum())} positives) in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
