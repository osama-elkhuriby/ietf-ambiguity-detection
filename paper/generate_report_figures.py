"""Generate small report-specific figures from committed aggregate metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "figures"

RUNS = {
    "Logistic regression": (
        ROOT / "results/ML/v2_engineered_tfidf/metrics.json",
        ROOT / "results/ML/v2_engineered_tfidf_revision/metrics.json",
    ),
    "Linear SVM": (
        ROOT / "results/ML/additional_classifiers/linear_svm/engineered_tfidf/metrics.json",
        ROOT
        / "results/ML/additional_classifiers/linear_svm/engineered_tfidf_revision/metrics.json",
    ),
    "Random forest": (
        ROOT
        / "results/ML/additional_classifiers/random_forest/engineered_tfidf/metrics.json",
        ROOT
        / "results/ML/additional_classifiers/random_forest/engineered_tfidf_revision/metrics.json",
    ),
}


def pooled(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["validation"]["pooled_oof_metrics"]


def revision_ablation() -> None:
    labels = list(RUNS)
    metrics = (("pr_auc", "Average precision"), ("f2_at_0.5", "$F_2$ at 0.5"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(labels))
    width = 0.34
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        baseline = [pooled(RUNS[label][0])[metric] for label in labels]
        revision = [pooled(RUNS[label][1])[metric] for label in labels]
        left = axis.bar(positions - width / 2, baseline, width, label="Engineered + TF-IDF")
        right = axis.bar(
            positions + width / 2,
            revision,
            width,
            label="Engineered + TF-IDF + revision",
        )
        axis.bar_label(left, fmt="%.3f", padding=3, fontsize=8)
        axis.bar_label(right, fmt="%.3f", padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylim(0, max(baseline + revision) * 1.22)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "Effect of past-only revision features on the matched classical cohort",
        y=0.98,
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "revision_feature_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def classical_ap_summary() -> None:
    """Plot the best non-revision AP per classifier and the fixed heuristic."""
    heuristic_path = ROOT / "results/heuristic/final_annotation_v1/metrics.json"
    labels = ["Linear SVM", "Random forest", "Logistic regression", "Fixed heuristic"]
    values = [
        pooled(RUNS["Linear SVM"][0])["pr_auc"],
        pooled(RUNS["Random forest"][0])["pr_auc"],
        pooled(RUNS["Logistic regression"][0])["pr_auc"],
        pooled(heuristic_path)["pr_auc"],
    ]
    prevalence = json.loads(heuristic_path.read_text(encoding="utf-8"))[
        "positive_prevalence"
    ]

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=["#2878B5", "#5DA5DA", "#9ECAE1", "#D07C40"])
    axis.bar_label(bars, fmt="%.3f", padding=4)
    axis.axvline(
        prevalence,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"Positive prevalence = {prevalence:.4f}",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.22)
    axis.set_xlabel("Average precision (pooled out-of-fold)")
    axis.set_title("Best matched classical configuration per model family")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "classical_ap_summary.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def project_cohorts() -> None:
    """Render cohort counts from the current manifests and aggregate reports."""
    figure, axis = plt.subplots(figsize=(14.5, 8.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    navy = "#183b5b"
    text = "#1e3042"
    muted = "#60748a"

    axis.text(
        0.5,
        0.94,
        "Project data and evaluated cohorts",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=navy,
    )
    for x, value, label in (
        (0.17, "264", "draft revisions"),
        (0.50, "13", "RFCs"),
        (0.83, "31", "verified technical errata"),
    ):
        axis.text(x, 0.82, value, ha="center", va="center", fontsize=31, fontweight="bold", color=navy)
        axis.text(x, 0.765, label, ha="center", va="center", fontsize=16, color=text)

    rows = (
        (
            0.64,
            "Classical ML + heuristic",
            "2,425 clauses | 160 positives | prevalence 6.60%",
            "3-fold family-grouped OOF; includes TCP/DNS",
            "#4d8a5b",
            "#edf6ef",
            0.25,
        ),
        (
            0.49,
            "Qwen / hybrid LM",
            "2,142 clauses | 126 positives | prevalence 5.88%",
            "5-fold family-grouped OOF",
            "#bd8020",
            "#fff6e8",
            0.20,
        ),
        (
            0.34,
            "Earlier DeBERTa study",
            "2,125 clauses | 136 positives | prevalence 6.40%",
            "5-fold family-isolated OOF",
            "#7857b4",
            "#f4effb",
            0.235,
        ),
        (
            0.19,
            "Updated reviewed-label DeBERTa",
            "2,143 clauses | 138 positives | prevalence 6.44%",
            "5-fold family-isolated OOF; includes 100 error-selected human reviews",
            "#3974a5",
            "#eaf3fa",
            0.33,
        ),
    )
    for y, label, headline, detail, edge, face, width in rows:
        box = FancyBboxPatch(
            (0.04, y - 0.037),
            width,
            0.075,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=2,
            edgecolor=edge,
            facecolor=face,
        )
        axis.add_patch(box)
        axis.text(0.055, y, label, ha="left", va="center", fontsize=14, fontweight="bold", color=text)
        axis.text(0.45, y + 0.016, headline, ha="left", va="center", fontsize=14, color=text)
        axis.text(0.45, y - 0.030, detail, ha="left", va="center", fontsize=12.5, color=muted)

    axis.text(
        0.5,
        0.075,
        "Cohorts and protocols differ; cross-study scores are not a unified leaderboard.",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#a33636",
    )
    figure.tight_layout(pad=0.3)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "project_cohorts.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    revision_ablation()
    classical_ap_summary()
    project_cohorts()
