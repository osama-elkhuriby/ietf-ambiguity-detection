from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from models.llm.in_context_evaluate_v2 import (
    _best_threshold,
    _cross_fitted_threshold_metrics,
    _load_protocol,
    _validate_config,
    select_demonstrations_v2,
)
from models.llm.prompting import system_prompt_for_contract


def _row(index: int, label: int, family: str, score: float) -> dict[str, object]:
    return {
        "record_id": f"row-{index}",
        "family_id": family,
        "label_tier": "manual_conflict_verdict" if index % 3 == 0 else "model_agreement",
        "label": label,
        "previous_clause": f"previous {index}",
        "target_clause": f"target {index}",
        "next_clause": f"next {index}",
        "source_row_sha256": f"{index:064x}",
        "v1_ambiguity_probability": score,
    }


def test_strict_v2_prompt_has_conservative_boundary() -> None:
    prompt = system_prompt_for_contract("ordinary_engineer_strict_v2")
    assert "default to no" in prompt
    assert "intentionally permitted choices" in prompt
    assert "two conflicting implementation outcomes" in prompt
    assert prompt.endswith("no punctuation or explanation.")


def test_v2_config_and_pinned_inputs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/llm/configs/qwen3_32b_final_annotation_ready_in_context_v2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(config)
    loaded, frame, folds, _, scores = _load_protocol(path)
    assert loaded["in_context"]["shot_counts"] == [8, 16, 24]
    assert len(frame) == len(scores) == 2142
    assert len(folds) == 5
    assert loaded["comparison"]["baseline_name"] == "qwen3_32b_in_context_v1"


def test_hard_negative_selection_obeys_composition_and_overlap() -> None:
    rows = [
        _row(index, 1 if index in {0, 1, 2} else 0, f"family-{index % 6}", index / 30)
        for index in range(30)
    ]
    fitting = pd.DataFrame(rows).drop(columns="v1_ambiguity_probability")
    scores = pd.DataFrame(rows)[["record_id", "v1_ambiguity_probability"]]
    selected = select_demonstrations_v2(
        fitting,
        scores,
        shots=8,
        fold=1,
        seed=9,
        composition={"positive": 1, "negative": 7},
        forbidden_normalized_texts={"target 29"},
    )
    assert len(selected) == 8
    assert int(selected["label"].sum()) == 1
    assert "row-29" not in set(selected["record_id"])
    assert selected.loc[selected["label"] == 0, "v1_ambiguity_probability"].min() >= 20 / 30
    assert selected["family_id"].nunique() >= 5


def test_threshold_selection_and_cross_fitting_are_fold_isolated() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(1, 6):
        for offset, (label, score) in enumerate(
            [(0, 0.05), (0, 0.2), (0, 0.7), (1, 0.65), (1, 0.9)]
        ):
            rows.append(
                {
                    "fold": fold,
                    "record_id": f"{fold}-{offset}",
                    "label": label,
                    "ambiguity_probability": score + fold * 0.001,
                }
            )
    frame = pd.DataFrame(rows)
    threshold = _best_threshold(
        frame["label"].to_numpy(), frame["ambiguity_probability"].to_numpy(), "f2"
    )
    assert 0.6 < threshold < 0.8
    metrics, predictions = _cross_fitted_threshold_metrics(frame, "f2")
    assert set(metrics["fold_thresholds"]) == {"1", "2", "3", "4", "5"}
    assert predictions["predicted_label_thresholded"].isin([0, 1]).all()
    assert np.isfinite(predictions["threshold_used"]).all()
    assert metrics["recall"] >= 0.8
