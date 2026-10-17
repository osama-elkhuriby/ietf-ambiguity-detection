from __future__ import annotations

import numpy as np
import pandas as pd

from models.classical.train_annotation_engineered_tfidf_holdout import (
    make_holdout_split,
)


def test_holdout_split_keeps_groups_disjoint_and_both_classes_present() -> None:
    groups = np.repeat(np.arange(8), 2)
    labels = np.tile(np.array([0, 1]), 8)
    frame = pd.DataFrame({"row": np.arange(len(labels))})

    train_indices, test_indices = make_holdout_split(frame, labels, groups)

    assert set(groups[train_indices]).isdisjoint(set(groups[test_indices]))
    assert set(labels[train_indices]) == {0, 1}
    assert set(labels[test_indices]) == {0, 1}
