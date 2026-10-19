from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from models.llm.hybrid_evaluate_v3 import (
    _load_protocol,
    _validate_config,
    blend_probabilities,
    select_candidate,
)


def test_v3_config_and_hash_pinned_inputs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root / "experiments/llm/configs/qwen3_32b_final_annotation_ready_hybrid_v3.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(config)
    loaded, frame, base, families = _load_protocol(path)
    assert loaded["base_signal"]["condition"] == "zero_shot"
    assert len(frame) == len(base) == 2142
    assert int(frame["label"].sum()) == 126
    assert len(families) == 9
    assert "draft-ietf-tcpm-rfc793bis" not in families
    assert "draft-ietf-dnsop-svcb-https" not in families


def test_log_odds_blend_respects_endpoints_and_order() -> None:
    lexical = np.array([0.1, 0.6, 0.9])
    qwen = np.array([0.8, 0.5, 0.2])
    assert np.allclose(blend_probabilities(lexical, qwen, 1.0), lexical)
    assert np.allclose(blend_probabilities(lexical, qwen, 0.0), qwen)
    middle = blend_probabilities(lexical, qwen, 0.5)
    assert np.isfinite(middle).all()
    assert ((middle > 0) & (middle < 1)).all()


def test_candidate_selection_uses_inner_predictions_only() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    qwen = np.full(6, 0.5)
    lexical = {
        0.1: np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
        1.0: np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]),
    }
    c_value, weight, table = select_candidate(labels, qwen, lexical, (0.5, 1.0))
    assert c_value == 0.1
    assert weight in {0.5, 1.0}
    assert len(table) == 4
    assert table.iloc[0]["inner_pr_auc"] == 1.0
