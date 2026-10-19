from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from models.llm.in_context_evaluate import (
    _confusion_svg,
    _select_shot_condition,
    _validate_config,
    classification_metrics,
    render_in_context_messages,
    select_demonstrations,
)


def demonstration_frame() -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for number in range(12):
            rows.append(
                {
                    "record_id": f"r-{label}-{number}",
                    "family_id": f"family-{number % 4}",
                    "target_clause": f"target {label} {number}",
                    "previous_clause": f"previous {label} {number}",
                    "next_clause": f"next {label} {number}",
                    "source_row_sha256": f"hash-{label}-{number}",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_demonstrations_are_skewed_diverse_and_do_not_overlap_queries() -> None:
    frame = demonstration_frame()
    forbidden = {"previous 0 0", "next 1 0"}
    first = select_demonstrations(
        frame,
        shots=8,
        fold=2,
        seed=20261018,
        forbidden_normalized_texts=forbidden,
    )
    second = select_demonstrations(
        frame,
        shots=8,
        fold=2,
        seed=20261018,
        forbidden_normalized_texts=forbidden,
    )
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert first["label"].value_counts().to_dict() == {0: 6, 1: 2}
    assert first.groupby("label")["family_id"].nunique().min() >= 2
    assert not {"r-0-0", "r-1-0"} & set(first["record_id"])


def test_messages_contain_examples_and_no_dataset_metadata() -> None:
    demonstrations = demonstration_frame().iloc[[0, 1, 12]].reset_index(drop=True)
    query = pd.Series(
        {
            "target_clause": "query target",
            "previous_clause": "query previous",
            "next_clause": "query next",
        }
    )
    messages = render_in_context_messages(
        query, demonstrations, prompt_contract="ordinary_engineer_v1"
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert rendered.count("LABELED EXAMPLE:") == 3
    assert [message["content"] for message in messages if message["role"] == "assistant"] == [
        "no",
        "no",
        "yes",
    ]
    assert "CLAUSE TO CLASSIFY:" in rendered
    assert "query target" in rendered
    assert "family_id" not in rendered and "source_row_sha256" not in rendered


def test_classification_metrics_use_generated_labels_and_logprob_scores() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    predicted = np.array([0, 0, 1, 1], dtype=np.int64)
    metrics = classification_metrics(labels, probabilities, predicted)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_shot_selection_uses_paired_one_standard_error_preference() -> None:
    rows = []
    values = {
        0: [0.50, 0.52, 0.48, 0.51, 0.49],
        4: [0.51, 0.53, 0.49, 0.52, 0.50],
        8: [0.70, 0.71, 0.69, 0.72, 0.68],
    }
    for shots, scores in values.items():
        rows.extend(
            {"shots": shots, "fold": fold, "pr_auc": score}
            for fold, score in enumerate(scores, start=1)
        )
    selection = _select_shot_condition(pd.DataFrame(rows))
    assert selection["top_shots"] == 8
    assert selection["selected_shots"] == 8


def test_confusion_svg_contains_all_counts() -> None:
    svg = _confusion_svg("four_shot", {"tn": 11, "fp": 2, "fn": 3, "tp": 5})
    assert svg.startswith("<svg")
    for value in (11, 2, 3, 5):
        assert f">{value}</text>" in svg


def test_current_config_is_ollama_zero_few_shot_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "experiments/llm/configs/qwen3_32b_final_annotation_ready_in_context.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["parameter_updates"] is False
    assert config["source_annotation"].endswith("final_annotation_ready.csv")
    assert config["model"]["name"] == "qwen3:32b"
    assert config["in_context"]["shot_counts"] == [0, 4, 8]
    _validate_config(config)
