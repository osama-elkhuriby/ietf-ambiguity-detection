from __future__ import annotations

import pandas as pd

from models.classical.annotation_logreg_experiment2 import (
    TRAINING_FEATURE_COLUMNS,
    V2_COLUMNS,
    _log_counts,
    extract_v2_features,
)


def test_exception_negation_interaction_is_extracted_but_not_used_for_training() -> None:
    extracted = extract_v2_features(
        pd.DataFrame(
            {
                "target_clause": ["The server MUST NOT retry unless instructed."],
                "previous_clause": [""],
                "next_clause": [""],
            }
        )
    )
    assert "exception_negation_interaction" in V2_COLUMNS
    assert extracted.loc[0, "exception_negation_interaction"] > 0

    training_features = _log_counts(extracted)
    assert tuple(training_features.columns) == TRAINING_FEATURE_COLUMNS
    assert "exception_negation_interaction" not in training_features.columns
