"""PyTorch dataset and loss helpers for ambiguity classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import Dataset

from models.transformer.tokenization import encode_example


class ClauseDataset(Dataset[dict[str, Any]]):
    """Pre-tokenized clauses with labels and per-example loss weights."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        *,
        variant: str,
        max_length: int,
        weights: np.ndarray | None = None,
    ) -> None:
        if weights is None:
            weights = np.ones(len(frame), dtype=np.float32)
        if len(weights) != len(frame):
            raise ValueError("One loss weight is required for every dataset row")
        self.examples: list[dict[str, Any]] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            encoded = encode_example(
                tokenizer,
                target=str(row.target_clause),
                previous=str(row.previous_clause),
                following=str(row.next_clause),
                variant=variant,
                max_length=max_length,
            )
            self.examples.append(
                {
                    "input_ids": encoded.input_ids,
                    "attention_mask": encoded.attention_mask,
                    "token_type_ids": encoded.token_type_ids,
                    "labels": int(row.label),
                    "loss_weight": float(weights[position]),
                    "row_position": position,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_clauses(
    examples: Sequence[dict[str, Any]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Dynamically pad a batch while preserving row positions and weights."""

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(item["input_ids"]) for item in examples)

    def padded(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (width - len(values))

    return {
        "input_ids": torch.tensor(
            [padded(item["input_ids"], pad_token_id) for item in examples],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [padded(item["attention_mask"], 0) for item in examples],
            dtype=torch.long,
        ),
        "token_type_ids": torch.tensor(
            [padded(item["token_type_ids"], 0) for item in examples],
            dtype=torch.long,
        ),
        "labels": torch.tensor([item["labels"] for item in examples], dtype=torch.long),
        "loss_weight": torch.tensor(
            [item["loss_weight"] for item in examples], dtype=torch.float32
        ),
        "row_position": torch.tensor(
            [item["row_position"] for item in examples], dtype=torch.long
        ),
    }


def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return the weighted mean CE without applying weights twice."""

    losses = functional.cross_entropy(logits, labels, reduction="none")
    if losses.shape != weights.shape:
        raise ValueError("Loss weights must have one value per example")
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise ValueError("Loss weights must have a positive finite sum")
    return (losses * weights).sum() / denominator
