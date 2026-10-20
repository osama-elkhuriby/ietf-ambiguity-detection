from __future__ import annotations

import pandas as pd
import pytest

from models.classical.revision_features import (
    REVISION_FEATURE_COLUMNS,
    load_revision_features,
)


def _write_inputs(tmp_path, *, cached_document: str = "draft-example-02"):
    key_path = tmp_path / "key.csv"
    pd.DataFrame(
        {
            "item_id": ["7"],
            "record_id": ["record-7"],
            "family_id": ["draft-example"],
            # These deliberately tempting fields must never be loaded as features.
            "candidate_source": ["future-derived-source"],
            "votes_json": ['{"label": "Yes"}'],
        }
    ).to_csv(key_path, index=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd.DataFrame(
        {
            "record_id": ["record-7"],
            "family_id": ["draft-example"],
            "document_id": [cached_document],
            "H0_has_predecessor": [1],
            "H0_relation_type": ["MODIFIED"],
            "H0_similarity": [0.8],
            "H0_lexical_similarity": [0.7],
            "H1_has_predecessor": [1],
            "H1_relation_type": ["UNCHANGED"],
            "H1_similarity": [1.0],
            "H1_lexical_similarity": [1.0],
            "matcher": ["section_constrained_lexical_minilm_v3"],
        }
    ).to_parquet(cache_dir / "draft-example.parquet", index=False)
    return key_path, cache_dir


def test_revision_features_are_compact_numeric_change_signals(tmp_path) -> None:
    key_path, cache_dir = _write_inputs(tmp_path)
    frame = pd.DataFrame(
        {"item_id": ["7"], "document": ["draft-example-02"]}, index=[11]
    )

    features = load_revision_features(
        frame, key_path=key_path, cache_dir=cache_dir
    )

    assert tuple(features.columns) == REVISION_FEATURE_COLUMNS
    assert features.index.tolist() == [11]
    assert features.loc[11, "h0_modified"] == 1.0
    assert features.loc[11, "h0_match_change_distance"] == pytest.approx(0.2)
    assert features.loc[11, "h0_lexical_change_distance"] == pytest.approx(0.3)
    assert features.loc[11, "h1_unchanged"] == 1.0
    assert not any("source" in column or "label" in column for column in features)


def test_revision_features_reject_document_identity_mismatch(tmp_path) -> None:
    key_path, cache_dir = _write_inputs(
        tmp_path, cached_document="draft-example-03"
    )
    frame = pd.DataFrame(
        {"item_id": ["7"], "document": ["draft-example-02"]}
    )

    with pytest.raises(ValueError, match="document identity"):
        load_revision_features(frame, key_path=key_path, cache_dir=cache_dir)
