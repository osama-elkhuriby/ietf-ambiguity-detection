from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.transformer.modeling import collate_clauses, weighted_cross_entropy


def test_collator_preserves_weights_positions_and_padding() -> None:
    batch = collate_clauses(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "token_type_ids": [0, 0],
                "labels": 0,
                "loss_weight": 0.5,
                "row_position": 3,
            },
            {
                "input_ids": [1, 4, 2],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
                "labels": 1,
                "loss_weight": 1.5,
                "row_position": 7,
            },
        ],
        pad_token_id=9,
    )
    assert batch["input_ids"].tolist() == [[1, 2, 9], [1, 4, 2]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch["loss_weight"].tolist() == pytest.approx([0.5, 1.5])
    assert batch["row_position"].tolist() == [3, 7]


def test_weighted_cross_entropy_applies_example_weights() -> None:
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1])
    equal = weighted_cross_entropy(logits, labels, torch.tensor([1.0, 1.0]))
    emphasize_error = weighted_cross_entropy(logits, labels, torch.tensor([0.5, 1.5]))
    assert emphasize_error > equal
