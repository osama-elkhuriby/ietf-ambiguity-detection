"""Plot pooled out-of-fold annotation-model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "results/ML/figures"
METRICS = (
    ("pr_auc", "PR-AUC"),
    ("precision_at_0.5", "Precision"),
    ("recall_at_0.5", "Recall"),
    ("f1_at_0.5", "F1"),
    ("f2_at_0.5", "F2"),
    ("roc_auc", "ROC-AUC"),
)
FEATURE_MODE_LABELS = {
    "engineered_tfidf": "Engineered + TF-IDF",
    "tfidf": "TF-IDF",
    "embeddings": "Embeddings",
    "tfidf_embeddings": "TF-IDF + embeddings",
    "engineered_tfidf_revision": "Engineered + TF-IDF + revision",
    "engineered_rules": "Engineered rules",
}
FEATURE_MODE_ORDER = (
    "engineered_tfidf",
    "engineered_tfidf_revision",
    "tfidf",
    "embeddings",
    "tfidf_embeddings",
    "engineered_rules",
)
MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random forest",
    "heuristic": "Heuristic",
}


@dataclass(frozen=True)
class ExperimentResult:
    path: Path
    model: str
    feature_mode: str
    rows: int
    positives: int
    threshold: float | None
    fold_signature: tuple[tuple[int | None, int | None, int | None], ...]
    metrics: dict[str, float]

    @property
    def cohort_signature(
        self,
    ) -> tuple[int, int, tuple[tuple[int | None, int | None, int | None], ...]]:
        return self.rows, self.positives, self.fold_signature


def load_oof_predictions(experiment: ExperimentResult) -> tuple[Any, Any]:
    """Load labels and positive-class scores for an experiment."""
    import numpy as np
    import pandas as pd

    if experiment.model == "heuristic":
        predictions_path = experiment.path.parent / "predictions.csv"
        score_columns = ("heuristic_score",)
    else:
        predictions_path = experiment.path.parent / "oof_predictions.csv"
        score_columns = ("oof_probability_yes", "oof_score_yes")

    frame = pd.read_csv(
        predictions_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    if "final_label" not in frame.columns:
        raise ValueError(
            f"{predictions_path} is missing the final_label column"
        )
    score_column = next(
        (column for column in score_columns if column in frame.columns),
        None,
    )
    if score_column is None:
        raise ValueError(
            f"{predictions_path} is missing a score column from "
            f"{list(score_columns)}"
        )
    normalized_labels = frame["final_label"].str.strip().str.casefold()
    if not normalized_labels.isin({"yes", "no"}).all():
        raise ValueError(f"{predictions_path} contains labels other than yes/no")
    labels = normalized_labels.eq("yes").to_numpy(dtype=np.int8)
    scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if len(frame) != experiment.rows or int(labels.sum()) != experiment.positives:
        raise ValueError(
            f"{predictions_path} does not match its metrics summary "
            f"({len(frame)} rows, {int(labels.sum())} positives; expected "
            f"{experiment.rows} rows, {experiment.positives} positives)"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{predictions_path} contains non-finite scores")
    if np.unique(labels).size != 2:
        raise ValueError(f"{predictions_path} must contain both label classes")
    return labels, scores


def _metric_files(results_root: Path) -> list[Path]:
    paths = [
        *results_root.glob("ML/v2_*/metrics.json"),
        *results_root.glob("ML/additional_classifiers/*/*/metrics.json"),
        results_root / "heuristic/final_annotation_v1/metrics.json",
    ]
    return sorted({path for path in paths if path.is_file()})


def _read_experiment(path: Path) -> ExperimentResult | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    validation = data.get("validation", {})
    pooled = validation.get("pooled_oof_metrics")
    if not isinstance(pooled, dict) or not all(
        name in pooled for name, _ in METRICS
    ):
        return None

    model_details = data.get("model")
    is_heuristic_path = "heuristic" in path.parts
    if isinstance(model_details, dict) and not is_heuristic_path:
        model = str(model_details.get("type", "unknown"))
        feature_mode = str(model_details.get("feature_set", "unknown"))
    else:
        model = "heuristic"
        feature_mode = "engineered_rules"
    folds = validation.get("folds", [])
    signature = tuple(
        (
            fold.get("validation_rows"),
            fold.get("validation_positives"),
            fold.get("validation_groups"),
        )
        for fold in folds
    )
    return ExperimentResult(
        path=path,
        model=model,
        feature_mode=feature_mode,
        rows=int(data["rows"]),
        positives=int(data["positive_rows"]),
        threshold=validation.get("prediction_threshold"),
        fold_signature=signature,
        metrics={name: float(pooled[name]) for name, _ in METRICS},
    )


def load_experiments(results_root: Path) -> list[ExperimentResult]:
    """Read pooled metric summaries from known annotation experiment folders."""
    experiments = []
    for path in _metric_files(results_root):
        result = _read_experiment(path)
        if result is not None:
            experiments.append(result)
    return experiments


def select_common_cohort(
    experiments: list[ExperimentResult],
) -> tuple[list[ExperimentResult], tuple[int, int, tuple[Any, ...]]]:
    """Select the largest exact match on dataset size, positives, and fold sizes."""
    if not experiments:
        raise ValueError("No experiment summaries with all pooled metrics were found")
    counts = Counter(experiment.cohort_signature for experiment in experiments)
    common_signature = counts.most_common(1)[0][0]
    selected = [
        experiment
        for experiment in experiments
        if experiment.cohort_signature == common_signature
    ]
    return selected, common_signature


def write_metrics_csv(
    output_path: Path,
    experiments: list[ExperimentResult],
    common_signature: tuple[int, int, tuple[Any, ...]],
) -> None:
    fieldnames = [
        "model",
        "feature_mode",
        "feature_mode_label",
        "rows",
        "positive_rows",
        "prediction_threshold",
        "common_cohort",
        "cohort_note",
        "metrics_path",
        *(name for name, _ in METRICS),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            writer.writerow(
                {
                    "model": experiment.model,
                    "feature_mode": experiment.feature_mode,
                    "feature_mode_label": FEATURE_MODE_LABELS.get(
                        experiment.feature_mode,
                        experiment.feature_mode,
                    ),
                    "rows": experiment.rows,
                    "positive_rows": experiment.positives,
                    "prediction_threshold": experiment.threshold,
                    "common_cohort": experiment.cohort_signature == common_signature,
                    "cohort_note": (
                        "matched"
                        if experiment.cohort_signature == common_signature
                        else (
                            "different row count, positive count, or fold composition; "
                            "excluded from plots"
                        )
                    ),
                    "metrics_path": experiment.path.as_posix(),
                    **experiment.metrics,
                }
            )


def _plot_grouped_metrics(
    axis: Any,
    series: list[tuple[str, list[ExperimentResult]]],
    title: str,
) -> None:
    import numpy as np

    metric_labels = [label for _, label in METRICS]
    positions = np.arange(len(METRICS))
    width = 0.82 / max(len(series), 1)
    for index, (label, experiments) in enumerate(series):
        values = [
            next(
                experiment.metrics[metric_name]
                for experiment in experiments
            )
            for metric_name, _ in METRICS
        ]
        offset = (index - (len(series) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=label,
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axis.set_title(title)
    axis.set_xticks(positions, metric_labels, rotation=20, ha="right")
    axis.set_ylim(0, 1.16)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def plot_feature_comparisons(
    output_dir: Path,
    experiments: list[ExperimentResult],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ("LogisticRegression", "linear_svm", "random_forest", "heuristic")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for axis, model in zip(axes.flat, models):
        model_results = [item for item in experiments if item.model == model]
        available_modes = {item.feature_mode for item in model_results}
        modes = [mode for mode in FEATURE_MODE_ORDER if mode in available_modes]
        series = [
            (
                FEATURE_MODE_LABELS.get(mode, mode),
                [item for item in model_results if item.feature_mode == mode],
            )
            for mode in modes
        ]
        if not series:
            axis.set_visible(False)
            continue
        _plot_grouped_metrics(
            axis,
            series,
            f"{MODEL_LABELS.get(model, model)} — feature setup",
        )
        if model == "heuristic":
            axis.text(
                0.5,
                -0.42,
                "Single rule-based feature setup; no feature ablation available.",
                transform=axis.transAxes,
                ha="center",
                fontsize=8,
            )
        axis.legend(fontsize=8, loc="upper left")

    figure.suptitle(
        "Annotation model metrics by feature setup\n"
        "Largest exactly matched grouped-CV cohort; threshold metrics use each run's saved threshold",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"feature_comparison_by_model.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_model_comparisons_by_feature(
    output_dir: Path,
    experiments: list[ExperimentResult],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_modes = (
        "engineered_tfidf",
        "engineered_tfidf_revision",
        "tfidf",
        "embeddings",
        "tfidf_embeddings",
    )
    model_order = ("LogisticRegression", "linear_svm", "random_forest")
    figure, axes = plt.subplots(2, 3, figsize=(21, 10), sharey=True)
    for axis, feature_mode in zip(axes.flat, feature_modes):
        series = [
            (
                MODEL_LABELS[model],
                [
                    item
                    for item in experiments
                    if item.model == model and item.feature_mode == feature_mode
                ],
            )
            for model in model_order
        ]
        series = [(label, items) for label, items in series if items]
        if not series:
            axis.set_visible(False)
            continue
        _plot_grouped_metrics(
            axis,
            series,
            FEATURE_MODE_LABELS[feature_mode],
        )
        if feature_mode == "engineered_tfidf" and not any(
            item.model == "LogisticRegression"
            and item.feature_mode == feature_mode
            for item in experiments
        ):
            axis.text(
                0.5,
                -0.42,
                "Logistic regression omitted: saved run has a different validation cohort.",
                transform=axis.transAxes,
                ha="center",
                fontsize=8,
            )
        axis.legend(fontsize=8, loc="upper left")
    for axis in axes.flat[len(feature_modes) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Classifier comparison within each feature setup\n"
        "All shown runs use the same dataset cohort and grouped folds",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"model_comparison_by_feature.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(figure)


def _best_by_model(
    experiments: list[ExperimentResult],
) -> list[ExperimentResult]:
    model_order = ("LogisticRegression", "linear_svm", "random_forest", "heuristic")
    best = []
    for model in model_order:
        candidates = [item for item in experiments if item.model == model]
        if candidates:
            best.append(max(candidates, key=lambda item: item.metrics["pr_auc"]))
    return best


def plot_model_comparison(output_dir: Path, experiments: list[ExperimentResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    best = _best_by_model(experiments)
    figure, axis = plt.subplots(figsize=(14, 7))
    metric_labels = [label for _, label in METRICS]
    positions = np.arange(len(METRICS))
    width = 0.8 / len(best)
    for index, experiment in enumerate(best):
        values = [experiment.metrics[name] for name, _ in METRICS]
        offset = (index - (len(best) - 1) / 2) * width
        label = (
            f"{MODEL_LABELS.get(experiment.model, experiment.model)} "
            f"({FEATURE_MODE_LABELS.get(experiment.feature_mode, experiment.feature_mode)})"
        )
        bars = axis.bar(positions + offset, values, width, label=label)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axis.set_title(
        "Best cohort-compatible feature setup per model (selected by pooled PR-AUC)\n"
        "All shown runs use the same dataset cohort and grouped folds"
    )
    axis.set_xticks(positions, metric_labels, rotation=20, ha="right")
    axis.set_ylim(0, 1.16)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9, loc="upper left")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"best_model_comparison.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_precision_recall_curves(
    output_dir: Path,
    experiments: list[ExperimentResult],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve

    columns = 4
    rows = (len(experiments) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 3.8 * rows),
        squeeze=False,
    )
    for axis, experiment in zip(axes.flat, experiments):
        labels, scores = load_oof_predictions(experiment)
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        prevalence = float(np.mean(labels))
        axis.step(
            recall,
            precision,
            where="post",
            label=f"AP = {average_precision:.3f}",
        )
        axis.axhline(
            prevalence,
            linestyle="--",
            color="gray",
            linewidth=1,
            label=f"Prevalence = {prevalence:.3f}",
        )
        axis.set_title(
            f"{MODEL_LABELS.get(experiment.model, experiment.model)}\n"
            f"{FEATURE_MODE_LABELS.get(experiment.feature_mode, experiment.feature_mode)}"
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    for axis in axes.flat[len(experiments) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Precision–recall curves from pooled out-of-fold predictions\n"
        "Matched annotation cohort and grouped folds",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"precision_recall_curves.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature and classifier comparisons from saved annotation metrics."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    experiments = load_experiments(args.results_root)
    selected, common_signature = select_common_cohort(experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(
        args.output_dir / "experiment_metrics.csv",
        experiments,
        common_signature,
    )
    selected = sorted(
        selected,
        key=lambda item: (
            ("LogisticRegression", "linear_svm", "random_forest", "heuristic").index(
                item.model
            ),
            FEATURE_MODE_ORDER.index(item.feature_mode),
        ),
    )
    plot_feature_comparisons(args.output_dir, selected)
    plot_model_comparisons_by_feature(args.output_dir, selected)
    plot_model_comparison(args.output_dir, selected)
    plot_precision_recall_curves(args.output_dir, selected)

    print(
        json.dumps(
            {
                "experiments_found": len(experiments),
                "matched_cohort_experiments": len(selected),
                "matched_rows": common_signature[0],
                "matched_positive_rows": common_signature[1],
                "excluded_experiments": [
                    {
                        "model": item.model,
                        "feature_mode": item.feature_mode,
                        "rows": item.rows,
                        "positive_rows": item.positives,
                        "path": item.path.as_posix(),
                    }
                    for item in experiments
                    if item.cohort_signature != common_signature
                ],
                "output_dir": args.output_dir.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
