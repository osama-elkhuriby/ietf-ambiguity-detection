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
        soft_targets: np.ndarray | None = None,
    ) -> None:
        if weights is None:
            weights = np.ones(len(frame), dtype=np.float32)
        if len(weights) != len(frame):
            raise ValueError("One loss weight is required for every dataset row")
        if soft_targets is not None:
            soft_targets = np.asarray(soft_targets, dtype=np.float32)
            if len(soft_targets) != len(frame):
                raise ValueError("One soft target is required for every dataset row")
            if not np.isfinite(soft_targets).all() or not (
                (soft_targets >= 0.0) & (soft_targets <= 1.0)
            ).all():
                raise ValueError("Soft targets must be finite values in [0, 1]")
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
            example = {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "token_type_ids": encoded.token_type_ids,
                "labels": int(row.label),
                "loss_weight": float(weights[position]),
                "row_position": position,
            }
            if soft_targets is not None:
                example["soft_target"] = float(soft_targets[position])
            self.examples.append(example)

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

    batch = {
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
    if "soft_target" in examples[0]:
        if not all("soft_target" in item for item in examples):
            raise ValueError("A batch cannot mix hard-only and soft-target examples")
        batch["soft_target"] = torch.tensor(
            [item["soft_target"] for item in examples], dtype=torch.float32
        )
    return batch


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


def weighted_soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted binary soft-label CE over a two-logit classifier."""

    if targets.shape != weights.shape:
        raise ValueError("Soft targets and loss weights must align")
    if not torch.isfinite(targets).all() or bool(
        ((targets < 0.0) | (targets > 1.0)).any()
    ):
        raise ValueError("Soft targets must be finite values in [0, 1]")
    log_probabilities = functional.log_softmax(logits, dim=1)
    losses = -(
        (1.0 - targets) * log_probabilities[:, 0]
        + targets * log_probabilities[:, 1]
    )
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise ValueError("Loss weights must have a positive finite sum")
    return (losses * weights).sum() / denominator
