"""Dataset, dynamic padding, and model-interface filtering."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenization import encode_example


def supported_model_fields(model: Any) -> set[str]:
    signature = inspect.signature(model.forward)
    return {
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in signature.parameters
    }


class ClauseDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        *,
        input_mode: str,
        max_length: int,
        model_fields: set[str],
    ) -> None:
        if "input_ids" not in model_fields:
            raise RuntimeError("Model forward method does not accept input_ids")
        self.examples: list[dict[str, Any]] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            encoded = encode_example(
                tokenizer,
                target=str(row.target_clause),
                section=str(row.section_heading),
                previous=str(row.previous_clause),
                following=str(row.next_clause),
                input_mode=input_mode,
                max_length=max_length,
            )
            example: dict[str, Any] = {
                "input_ids": encoded.input_ids,
                "labels": int(row.label),
                "row_position": position,
                "target_truncated": encoded.target_truncated,
                "context_truncated": encoded.context_truncated,
            }
            if "attention_mask" in model_fields:
                example["attention_mask"] = encoded.attention_mask
            if "token_type_ids" in model_fields and encoded.token_type_ids is not None:
                example["token_type_ids"] = encoded.token_type_ids
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_clauses(
    examples: Sequence[dict[str, Any]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(item["input_ids"]) for item in examples)

    def pad(values: list[int], value: int) -> list[int]:
        return values + [value] * (width - len(values))

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(
            [pad(item["input_ids"], pad_token_id) for item in examples],
            dtype=torch.long,
        ),
        "labels": torch.tensor([item["labels"] for item in examples], dtype=torch.long),
        "row_position": torch.tensor(
            [item["row_position"] for item in examples], dtype=torch.long
        ),
        "target_truncated": torch.tensor(
            [item["target_truncated"] for item in examples], dtype=torch.bool
        ),
        "context_truncated": torch.tensor(
            [item["context_truncated"] for item in examples], dtype=torch.bool
        ),
    }
    for field in ("attention_mask", "token_type_ids"):
        present = [field in item for item in examples]
        if any(present) and not all(present):
            raise RuntimeError(f"Batch mixes examples with and without {field}")
        if all(present):
            batch[field] = torch.tensor(
                [pad(item[field], 0) for item in examples], dtype=torch.long
            )
    return batch
