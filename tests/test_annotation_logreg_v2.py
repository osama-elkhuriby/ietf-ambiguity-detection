from __future__ import annotations

import numpy as np
import pandas as pd

from models.classical.annotation_logreg_experiment import (
    DEFAULT_OUTPUTS,
    _fit_components,
    _transform_components,
    V2_COLUMNS,
    build_validation_groups,
    extract_v2_features,
)
from models.classical.train_annotation_embeddings_only import (
    FEATURE_MODE as EMBEDDINGS_ONLY_MODE,
)
from models.classical.train_annotation_engineered_tfidf import (
    FEATURE_MODE as ENGINEERED_TFIDF_MODE,
)
from models.classical.train_annotation_tfidf_embeddings import (
    FEATURE_MODE as TFIDF_EMBEDDINGS_MODE,
)
from models.classical.train_annotation_tfidf_only import (
    FEATURE_MODE as TFIDF_ONLY_MODE,
)


def test_v2_features_cover_mixed_modals_numeric_context_and_references() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_clause": (
                    "If it sends three packets, the endpoint MUST NOT wait and "
                    "MAY retry using a reasonable delay in Section 3.1 [RFC9000]."
                ),
                "previous_clause": "The endpoint sends packets.",
                "next_clause": "Otherwise, it can continue.",
            }
        ]
    )

    features = extract_v2_features(frame).iloc[0]

    assert tuple(features.index) == V2_COLUMNS
    assert len(V2_COLUMNS) == 23
    assert features["has_strong_modal"] == 1
    assert features["has_may_modal"] == 1
    assert features["modal_level_count"] == 2
    assert features["has_no_modal"] == 0
    assert features["has_numeric_value"] == 1
    assert features["numeric_count"] >= 1
    assert features["hedge_term_count"] == 1
    assert features["has_if_without_otherwise"] == 0
    assert features["target_context_overlap"] > 0
    assert features["bracket_reference_count"] == 1
    assert features["section_reference_count"] == 1


def test_v2_conditional_and_initial_pronoun_indicators() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_clause": "If the server is ready, it sends a response.",
                "previous_clause": "",
                "next_clause": "The client waits.",
            },
            {
                "target_clause": "It is sent when available.",
                "previous_clause": "",
                "next_clause": "",
            },
        ]
    )

    features = extract_v2_features(frame)

    assert features.loc[0, "has_if_without_otherwise"] == 1
    assert features.loc[1, "pronoun_without_noun_before"] == 1


def test_validation_groups_join_revisions_of_the_same_family() -> None:
    frame = pd.DataFrame(
        {
            "document": [
                "draft-alpha-01",
                "draft-alpha-02",
                "draft-beta-01",
                "draft-gamma-01",
            ],
            "target_clause": [
                "The endpoint MUST send a response to the peer.",
                "The endpoint SHOULD send a response to the peer.",
                "The endpoint MUST send a response to the peer.",
                "A completely unrelated server clause describes another process.",
            ],
        }
    )

    groups = build_validation_groups(frame)

    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert groups[2] != groups[3]


def test_entry_points_select_the_requested_feature_combinations() -> None:
    assert ENGINEERED_TFIDF_MODE == "engineered_tfidf"
    assert TFIDF_ONLY_MODE == "tfidf"
    assert EMBEDDINGS_ONLY_MODE == "embeddings"
    assert TFIDF_EMBEDDINGS_MODE == "tfidf_embeddings"
    assert len(set(DEFAULT_OUTPUTS.values())) == 4


def test_engineered_tfidf_transform_has_only_requested_blocks() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_clause": "The endpoint MUST send a shared unique value.",
                "previous_clause": "",
                "next_clause": "",
            },
            {
                "target_clause": "If possible, the server SHOULD send a shared unique value.",
                "previous_clause": "A peer requests a value.",
                "next_clause": "",
            },
            {
                "target_clause": "A client MAY send a shared unique value.",
                "previous_clause": "The server listens.",
                "next_clause": "It then responds.",
            },
        ]
    )
    features = extract_v2_features(frame)
    components, train_matrix = _fit_components(
        "engineered_tfidf",
        frame["target_clause"].iloc[:2],
        features.iloc[:2],
        None,
    )
    validation_matrix = _transform_components(
        "engineered_tfidf",
        components,
        frame["target_clause"].iloc[2:],
        features.iloc[2:],
        None,
    )

    assert train_matrix.shape == (
        2,
        len(components["vectorizer"].get_feature_names_out()) + len(V2_COLUMNS),
    )
    assert validation_matrix.shape == (1, train_matrix.shape[1])
    assert train_matrix.nnz > 0
    assert validation_matrix.nnz > 0
    assert "embedding_scaler" not in components


def test_text_only_modes_exclude_engineered_feature_block() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_clause": "The endpoint MUST send a shared unique value.",
                "previous_clause": "",
                "next_clause": "",
            },
            {
                "target_clause": "If possible, the server SHOULD send a shared unique value.",
                "previous_clause": "A peer requests a value.",
                "next_clause": "",
            },
            {
                "target_clause": "The endpoint MAY send a shared unique value.",
                "previous_clause": "The server listens.",
                "next_clause": "",
            },
        ]
    )
    embeddings = np.arange(12, dtype=float).reshape(3, 4)
    for mode in ("tfidf", "embeddings", "tfidf_embeddings"):
        train_embeddings = embeddings[:2] if "embeddings" in mode else None
        validation_embeddings = embeddings[2:] if "embeddings" in mode else None
        components, train_matrix = _fit_components(
            mode,
            frame["target_clause"].iloc[:2],
            None,
            train_embeddings,
        )
        validation_matrix = _transform_components(
            mode,
            components,
            frame["target_clause"].iloc[2:],
            None,
            validation_embeddings,
        )
        expected_features = (
            len(components["vectorizer"].get_feature_names_out())
            if "tfidf" in mode
            else 0
        ) + (4 if "embeddings" in mode else 0)
        assert train_matrix.shape == (2, expected_features)
        assert validation_matrix.shape == (1, expected_features)
        assert "numerical_scaler" not in components
        assert ("vectorizer" in components) == ("tfidf" in mode)
        assert ("embedding_scaler" in components) == ("embeddings" in mode)
