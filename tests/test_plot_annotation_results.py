from __future__ import annotations

import json

from models.classical.plot_annotation_results import (
    load_experiments,
    select_common_cohort,
    write_metrics_csv,
)


METRICS = {
    "pr_auc": 0.4,
    "precision_at_0.5": 0.3,
    "recall_at_0.5": 0.5,
    "f1_at_0.5": 0.375,
    "f2_at_0.5": 0.44,
    "roc_auc": 0.7,
}


def _write_metrics(
    path,
    *,
    rows: int,
    positives: int,
    validation_rows: tuple[int, int, int],
    validation_positives: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True)
    payload = {
        "rows": rows,
        "positive_rows": positives,
        "model": {
            "type": "LogisticRegression",
            "feature_set": path.parent.name,
        },
        "validation": {
            "prediction_threshold": 0.5,
            "pooled_oof_metrics": METRICS,
            "folds": [
                {
                    "validation_rows": row_count,
                    "validation_positives": positive_count,
                    "validation_groups": index + 1,
                }
                for index, (row_count, positive_count) in enumerate(
                    zip(validation_rows, validation_positives)
                )
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_common_cohort_requires_matching_data_and_fold_composition(tmp_path) -> None:
    results_root = tmp_path / "results"
    _write_metrics(
        results_root / "ML/v2_tfidf/metrics.json",
        rows=20,
        positives=4,
        validation_rows=(6, 7, 7),
        validation_positives=(1, 1, 2),
    )
    _write_metrics(
        results_root / "ML/v2_embeddings/metrics.json",
        rows=20,
        positives=4,
        validation_rows=(6, 7, 7),
        validation_positives=(1, 1, 2),
    )
    _write_metrics(
        results_root / "ML/v2_engineered_tfidf/metrics.json",
        rows=20,
        positives=5,
        validation_rows=(6, 7, 7),
        validation_positives=(2, 1, 2),
    )

    experiments = load_experiments(results_root)
    selected, signature = select_common_cohort(experiments)

    assert len(selected) == 2
    assert signature[:2] == (20, 4)


def test_metrics_csv_marks_experiments_outside_matched_cohort(tmp_path) -> None:
    results_root = tmp_path / "results"
    _write_metrics(
        results_root / "ML/v2_tfidf/metrics.json",
        rows=20,
        positives=4,
        validation_rows=(6, 7, 7),
        validation_positives=(1, 1, 2),
    )
    _write_metrics(
        results_root / "ML/v2_engineered_tfidf/metrics.json",
        rows=20,
        positives=5,
        validation_rows=(6, 7, 7),
        validation_positives=(2, 1, 2),
    )
    experiments = load_experiments(results_root)
    _, signature = select_common_cohort(experiments)

    output_path = tmp_path / "metrics.csv"
    write_metrics_csv(output_path, experiments, signature)
    rows = output_path.read_text(encoding="utf-8").splitlines()

    assert "cohort_note" in rows[0]
    assert any("matched" in row for row in rows[1:])
    assert any("excluded from plots" in row for row in rows[1:])
