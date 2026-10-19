from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from models.llm.semantic_ensemble_v4 import (
    _load_protocol,
    _validate_config,
    blend_three_probabilities,
)


def test_v4_config_and_hash_pinned_inputs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "experiments/llm/configs/qwen3_32b_final_annotation_ready_semantic_v4.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(config)
    loaded, frame, families = _load_protocol(path)
    assert loaded["sentence_encoder"]["dimensions"] == 384
    assert len(frame) == 2142
    assert int(frame["label"].sum()) == 126
    assert len(families) == 9
    assert "draft-ietf-tcpm-rfc793bis" not in families
    assert "draft-ietf-dnsop-svcb-https" not in families


def test_three_way_log_odds_blend_respects_endpoints() -> None:
    first = np.array([0.1, 0.5, 0.9])
    second = np.array([0.2, 0.4, 0.8])
    third = np.array([0.3, 0.6, 0.7])
    assert np.allclose(
        blend_three_probabilities(first, second, third, (1.0, 0.0, 0.0)), first
    )
    assert np.allclose(
        blend_three_probabilities(first, second, third, (0.0, 1.0, 0.0)), second
    )
    mixed = blend_three_probabilities(first, second, third, (0.8, 0.1, 0.1))
    assert np.isfinite(mixed).all()
    assert ((mixed > 0) & (mixed < 1)).all()
