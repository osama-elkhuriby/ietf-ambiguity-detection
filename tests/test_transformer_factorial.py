from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from models.transformer.experiment import DevelopmentFold, NestedDevelopmentFold
from models.transformer.factorial_artifacts import (
    apply_context_policy,
    build_context_policy,
    build_soft_targets,
)
from models.transformer.modeling import (
    weighted_cross_entropy,
    weighted_soft_cross_entropy,
)


def _votes(*values: str | None) -> str:
    names = ("gemma3:27b", "gpt-oss:20b", "qwen3:32b")
    return json.dumps(
        {
            name: (
                {"status": "valid", "vote": vote}
                if vote is not None
                else {"status": "invalid", "vote": None}
            )
            for name, vote in zip(names, values, strict=True)
        }
    )


def test_soft_targets_use_only_valid_votes(tmp_path) -> None:
    train = pd.DataFrame(
        {"record_id": ["a", "b"], "label": [1, 0]}
    )
    source = pd.DataFrame(
        {
            "record_id": ["a", "b", "sealed-not-requested"],
            "votes_json": [
                _votes("yes", "no", None),
                _votes("no", "no", "yes"),
                _votes("yes", "yes", "yes"),
            ],
        }
    )
    path = tmp_path / "consensus.csv"
    source.to_csv(path, index=False)
    result = build_soft_targets(train, path).set_index("record_id")
    assert result.loc["a", "valid_vote_count"] == 2
    assert result.loc["a", "soft_target"] == pytest.approx(0.5)
    assert result.loc["b", "soft_target"] == pytest.approx(1 / 3)
    assert "sealed-not-requested" not in result.index


def test_one_hot_soft_loss_matches_hard_loss_and_gradients() -> None:
    hard_logits = torch.tensor(
        [[1.2, -0.4], [-0.2, 0.9], [0.1, 0.4]], requires_grad=True
    )
    soft_logits = hard_logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([0.5, 1.5, 0.75])
    hard = weighted_cross_entropy(hard_logits, labels, weights)
    soft = weighted_soft_cross_entropy(soft_logits, labels.float(), weights)
    hard.backward()
    soft.backward()
    assert soft.item() == pytest.approx(hard.item())
    assert soft_logits.grad is not None
    assert hard_logits.grad is not None
    assert np.allclose(soft_logits.grad.numpy(), hard_logits.grad.numpy())


def test_context_policy_blanks_only_flagged_context() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "previous_clause": ["leak", "safe"],
            "next_clause": ["safe", "leak"],
            "target_clause": ["first", "second"],
        }
    )
    policy = pd.DataFrame(
        {
            "fold": [1, 1],
            "record_id": ["a", "b"],
            "partition_role": ["fitting", "fitting"],
            "previous_allowed": [False, True],
            "next_allowed": [True, False],
        }
    )
    sanitized, counts = apply_context_policy(
        frame, policy, fold=1, expected_role="fitting"
    )
    assert sanitized["previous_clause"].tolist() == ["", "safe"]
    assert sanitized["next_clause"].tolist() == ["safe", ""]
    assert sanitized["target_clause"].tolist() == ["first", "second"]
    assert counts == {"previous_removed": 1, "next_removed": 1}


def test_context_policy_flags_exact_and_near_cross_role_targets() -> None:
    long_target = " ".join(f"token{index}" for index in range(20))
    near_target = " ".join([*(f"token{index}" for index in range(19)), "changed"])
    frame = pd.DataFrame(
        {
            "record_id": ["fit", "inner", "outer"],
            "target_clause": [long_target, "inner target", "outer exact target"],
            "previous_clause": ["outer exact target", "", ""],
            "next_clause": ["", "", near_target],
        }
    )
    outer = DevelopmentFold(
        number=1,
        train_indices=np.array([0, 1]),
        validation_indices=np.array([2]),
        scoring_indices=np.array([2]),
        held_out_families=("outer",),
        excluded_exact_duplicate_record_ids=(),
    )
    nested = NestedDevelopmentFold(
        outer=outer,
        fitting_indices=np.array([0]),
        selection_indices=np.array([1]),
        selection_scoring_indices=np.array([1]),
        inner_selection_family="inner",
        excluded_inner_duplicate_record_ids=(),
    )
    policy = build_context_policy(frame, [nested], threshold=0.85).set_index(
        "record_id"
    )
    assert bool(policy.loc["fit", "previous_allowed"]) is False
    assert policy.loc["fit", "previous_max_similarity"] == 1.0
    assert bool(policy.loc["outer", "next_allowed"]) is False
    assert 0.85 <= policy.loc["outer", "next_max_similarity"] < 1.0
