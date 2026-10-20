from __future__ import annotations

import pandas as pd
import pytest

from models.hybrid.final_rank_ensemble import (
    _metrics_table,
    _rank_average,
    _read_predictions,
)
from models.hybrid.plot_final_rank_ensemble import (
    plot_precision_recall,
    plot_review_budget,
)


def _write_prediction_inputs(tmp_path):
    v4 = pd.DataFrame(
        {
            "source_item_id": [1, 2, 3, 4],
            "family_id": ["a", "a", "b", "b"],
            "fold": [1, 1, 2, 2],
            "label": [1, 0, 1, 0],
            "ambiguity_probability": [0.9, 0.4, 0.7, 0.2],
        }
    )
    transformer = pd.DataFrame(
        {
            "item_id": [4, 3, 2, 1],
            "family_id": ["b", "b", "a", "a"],
            "fold": [2, 2, 1, 1],
            "label": [0, 1, 0, 1],
            "probability_yes": [0.1, 0.6, 0.3, 0.8],
        }
    )
    ml = pd.DataFrame(
        {
            "item_id": [4, 3, 2, 1],
            "final_label": ["No", "Yes", "No", "Yes"],
            "validation_group": [2, 2, 1, 1],
            "oof_probability_yes": [0.2, 0.7, 0.35, 0.85],
        }
    )
    v4_path = tmp_path / "v4.csv"
    transformer_path = tmp_path / "transformer.csv"
    ml_path = tmp_path / "ml.csv"
    v4.to_csv(v4_path, index=False)
    transformer.to_csv(transformer_path, index=False)
    ml.to_csv(ml_path, index=False)
    return v4_path, transformer_path, ml_path


def test_rank_average_aligns_rows_and_reports_metrics(tmp_path) -> None:
    v4_path, transformer_path, ml_path = _write_prediction_inputs(tmp_path)
    predictions, audit = _read_predictions(v4_path, transformer_path, ml_path)
    ranked = _rank_average(predictions)
    metrics = _metrics_table(ranked)

    assert predictions["source_item_id"].tolist() == [1, 2, 3, 4]
    assert ranked["hybrid_rank_score"].between(0, 1).all()
    assert audit["compatible_rows"] == 4
    assert metrics["model"].tolist() == [
        "V4",
        "DeBERTa",
        "ML logistic regression",
        "V4 + DeBERTa",
        "V4 + ML",
        "V4 + DeBERTa + ML",
    ]
    assert metrics["average_precision"].between(0, 1).all()
    assert metrics["recall_at_20pct"].between(0, 1).all()


def test_prediction_join_rejects_fold_mismatch(tmp_path) -> None:
    v4_path, transformer_path, ml_path = _write_prediction_inputs(tmp_path)
    transformer = pd.read_csv(transformer_path)
    transformer.loc[0, "fold"] = 1
    transformer.to_csv(transformer_path, index=False)

    with pytest.raises(ValueError, match="disagree on fold"):
        _read_predictions(v4_path, transformer_path, ml_path)


def test_prediction_join_excludes_missing_and_label_conflict_rows(tmp_path) -> None:
    v4_path, transformer_path, ml_path = _write_prediction_inputs(tmp_path)
    ml = pd.read_csv(ml_path)
    ml = ml.loc[ml["item_id"] != 4].copy()
    ml.loc[ml["item_id"] == 3, "final_label"] = "No"
    ml.to_csv(ml_path, index=False)

    predictions, audit = _read_predictions(v4_path, transformer_path, ml_path)

    assert len(predictions) == 2
    assert audit["v4_rows_missing_from_ml"] == 1
    assert audit["label_mismatch_rows_excluded"] == 1
    assert audit["compatible_rows"] == 2


def test_paper_plots_write_png_and_pdf(tmp_path) -> None:
    v4_path, transformer_path, ml_path = _write_prediction_inputs(tmp_path)
    aligned, _ = _read_predictions(v4_path, transformer_path, ml_path)
    predictions = _rank_average(aligned)
    output_dir = tmp_path / "figures"

    plot_precision_recall(predictions, output_dir)
    plot_review_budget(predictions, output_dir)

    for stem in ("hybrid_pr_curves", "hybrid_review_budget"):
        for suffix in ("png", "pdf"):
            assert (output_dir / f"{stem}.{suffix}").stat().st_size > 0
