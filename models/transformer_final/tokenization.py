"""Target-preserving serialization and an offline tokenizer-length audit."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .common import immutable_directory, resolve_path, write_json
from .config import DEFAULT_CONFIG, load_config
from .dataset import load_effective_dataset
from .preflight import require_local_model_files


class TokenizerLike(Protocol):
    model_input_names: list[str]

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    target_token_count: int
    target_tokens_kept: int
    context_token_count: int
    context_tokens_kept: int
    special_token_count: int

    @property
    def target_truncated(self) -> bool:
        return self.target_tokens_kept < self.target_token_count

    @property
    def context_truncated(self) -> bool:
        return self.context_tokens_kept < self.context_token_count


def serialize_local_context(
    target: str, section: str, previous: str, following: str
) -> str:
    return (
        f"[TARGET]\n{target}\n\n[SECTION]\n{section}\n\n"
        f"[PREVIOUS]\n{previous}\n\n[NEXT]\n{following}"
    )


def _head_tail_indices(length: int, limit: int) -> list[int]:
    if length <= limit:
        return list(range(length))
    head = math.ceil(limit / 2)
    tail = limit - head
    return [*range(head), *range(length - tail, length)] if tail else list(range(head))


def _tokenize_public(
    tokenizer: TokenizerLike, text: str, *, add_special_tokens: bool
) -> dict[str, list[int]]:
    """Use the public tokenizer call contract and normalize one unpadded example."""
    wants_token_types = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=wants_token_types,
        return_special_tokens_mask=True,
    )
    result = {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
        "special_tokens_mask": list(encoded["special_tokens_mask"]),
    }
    if "token_type_ids" in encoded:
        result["token_type_ids"] = list(encoded["token_type_ids"])
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise RuntimeError("Tokenizer returned misaligned model input fields")
    if not add_special_tokens and any(result["special_tokens_mask"]):
        raise RuntimeError("Tokenizer inserted special tokens when they were disabled")
    return result


def _content_bounds(
    formatted: dict[str, list[int]], raw_content: dict[str, list[int]]
) -> tuple[int, int, int]:
    mask = formatted["special_tokens_mask"]
    content_positions = [
        index for index, is_special in enumerate(mask) if not is_special
    ]
    expected_content = [formatted["input_ids"][index] for index in content_positions]
    if expected_content != raw_content["input_ids"]:
        raise RuntimeError(
            "Tokenizer special-token encoding does not preserve the raw content sequence"
        )
    if not content_positions:
        raise RuntimeError(
            "Tokenizer produced no content positions for non-empty input"
        )
    if content_positions != list(
        range(content_positions[0], content_positions[-1] + 1)
    ):
        raise RuntimeError(
            "Tokenizer interleaves special tokens within a single sequence"
        )
    start = content_positions[0]
    return start, start + len(content_positions), sum(mask)


def _with_selected_content(
    formatted: dict[str, list[int]],
    raw_content: dict[str, list[int]],
    selected_indices: list[int],
) -> tuple[list[int], list[int], list[int] | None, int]:
    """Retain the public tokenizer's special-token template around selected tokens."""
    content_start, content_end, special_count = _content_bounds(formatted, raw_content)
    prefix = slice(0, content_start)
    suffix = slice(content_end, len(formatted["input_ids"]))
    selected_ids = [raw_content["input_ids"][index] for index in selected_indices]
    input_ids = [
        *formatted["input_ids"][prefix],
        *selected_ids,
        *formatted["input_ids"][suffix],
    ]
    selected_attention = [
        raw_content["attention_mask"][index] for index in selected_indices
    ]
    attention_mask = [
        *formatted["attention_mask"][prefix],
        *selected_attention,
        *formatted["attention_mask"][suffix],
    ]
    token_type_ids: list[int] | None = None
    if "token_type_ids" in formatted:
        if "token_type_ids" not in raw_content:
            raise RuntimeError("Tokenizer omitted raw token_type_ids")
        selected_types = [
            raw_content["token_type_ids"][index] for index in selected_indices
        ]
        token_type_ids = [
            *formatted["token_type_ids"][prefix],
            *selected_types,
            *formatted["token_type_ids"][suffix],
        ]
    return input_ids, attention_mask, token_type_ids, special_count


def _allocate_context(parts: list[list[int]], budget: int) -> list[list[int]]:
    kept = [[] for _ in parts]
    positions = [0] * len(parts)
    while budget > 0:
        progressed = False
        for index, part in enumerate(parts):
            if positions[index] < len(part) and budget > 0:
                kept[index].append(part[positions[index]])
                positions[index] += 1
                budget -= 1
                progressed = True
        if not progressed:
            break
    return kept


def encode_example(
    tokenizer: TokenizerLike,
    *,
    target: str,
    section: str = "",
    previous: str = "",
    following: str = "",
    input_mode: str = "target_only",
    max_length: int = 256,
) -> EncodedExample:
    if input_mode not in {"target_only", "local_context"}:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    if input_mode == "target_only":
        raw_target = _tokenize_public(tokenizer, target, add_special_tokens=False)
        formatted_target = _tokenize_public(tokenizer, target, add_special_tokens=True)
        special = sum(formatted_target["special_tokens_mask"])
        content_budget = max_length - special
        if content_budget < 1:
            raise ValueError("max_length leaves no room for target text")
        selected_indices = _head_tail_indices(
            len(raw_target["input_ids"]), content_budget
        )
        input_ids, attention_mask, token_type_ids, special = _with_selected_content(
            formatted_target, raw_target, selected_indices
        )
        target_count = len(raw_target["input_ids"])
        target_kept = len(selected_indices)
        context_total = context_kept = 0
    else:
        target_text = f"[TARGET]\n{target}"
        raw_target = _tokenize_public(tokenizer, target_text, add_special_tokens=False)
        formatted_target = _tokenize_public(
            tokenizer, target_text, add_special_tokens=True
        )
        special = sum(formatted_target["special_tokens_mask"])
        content_budget = max_length - special
        if content_budget < 1:
            raise ValueError("max_length leaves no room for target text")
        context_parts = [
            _tokenize_public(
                tokenizer, f"\n\n[SECTION]\n{section}", add_special_tokens=False
            ),
            _tokenize_public(
                tokenizer, f"\n\n[PREVIOUS]\n{previous}", add_special_tokens=False
            ),
            _tokenize_public(
                tokenizer, f"\n\n[NEXT]\n{following}", add_special_tokens=False
            ),
        ]
        target_indices = _head_tail_indices(
            len(raw_target["input_ids"]), content_budget
        )
        remaining = (
            max(0, content_budget - len(target_indices))
            if len(raw_target["input_ids"]) <= content_budget
            else 0
        )
        context_ids = [part["input_ids"] for part in context_parts]
        kept_parts = _allocate_context(context_ids, remaining)
        combined_raw = {
            "input_ids": [
                *raw_target["input_ids"],
                *(token for part in context_ids for token in part),
            ],
            "attention_mask": [
                *raw_target["attention_mask"],
                *(token for part in context_parts for token in part["attention_mask"]),
            ],
            "special_tokens_mask": [
                *raw_target["special_tokens_mask"],
                *(
                    token
                    for part in context_parts
                    for token in part["special_tokens_mask"]
                ),
            ],
        }
        if "token_type_ids" in raw_target:
            if not all("token_type_ids" in part for part in context_parts):
                raise RuntimeError("Tokenizer inconsistently returned token_type_ids")
            combined_raw["token_type_ids"] = [
                *raw_target["token_type_ids"],
                *(token for part in context_parts for token in part["token_type_ids"]),
            ]
        context_indices: list[int] = []
        offset = len(raw_target["input_ids"])
        for original, kept in zip(context_ids, kept_parts, strict=True):
            context_indices.extend(range(offset, offset + len(kept)))
            offset += len(original)
        selected_indices = [*target_indices, *context_indices]
        # Derive the special-token template from the public target encoding, then
        # place the label-blind, target-prioritized content inside that template.
        content_start, content_end, special = _content_bounds(
            formatted_target, raw_target
        )
        input_ids = [
            *formatted_target["input_ids"][:content_start],
            *(combined_raw["input_ids"][index] for index in selected_indices),
            *formatted_target["input_ids"][content_end:],
        ]
        attention_mask = [
            *formatted_target["attention_mask"][:content_start],
            *(combined_raw["attention_mask"][index] for index in selected_indices),
            *formatted_target["attention_mask"][content_end:],
        ]
        token_type_ids = None
        if "token_type_ids" in formatted_target:
            token_type_ids = [
                *formatted_target["token_type_ids"][:content_start],
                *(combined_raw["token_type_ids"][index] for index in selected_indices),
                *formatted_target["token_type_ids"][content_end:],
            ]
        target_count = len(raw_target["input_ids"])
        target_kept = len(target_indices)
        context_total = sum(map(len, context_ids))
        context_kept = sum(map(len, kept_parts))
    if len(input_ids) > max_length:
        raise RuntimeError("Target-preserving tokenizer policy exceeded max_length")
    if len(attention_mask) != len(input_ids) or (
        token_type_ids is not None and len(token_type_ids) != len(input_ids)
    ):
        raise RuntimeError("Tokenizer returned misaligned model inputs")
    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        target_token_count=target_count,
        target_tokens_kept=target_kept,
        context_token_count=context_total,
        context_tokens_kept=context_kept,
        special_token_count=special,
    )


def audit_lengths(
    tokenizer: TokenizerLike, frame: pd.DataFrame, *, max_length: int, input_mode: str
) -> dict[str, Any]:
    total: list[int] = []
    target: list[int] = []
    target_truncated = context_truncated = 0
    for row in frame.itertuples(index=False):
        encoded = encode_example(
            tokenizer,
            target=str(row.target_clause),
            section=str(row.section_heading),
            previous=str(row.previous_clause),
            following=str(row.next_clause),
            input_mode=input_mode,
            max_length=max_length,
        )
        total.append(len(encoded.input_ids))
        target.append(encoded.target_token_count + encoded.special_token_count)
        target_truncated += int(encoded.target_truncated)
        context_truncated += int(encoded.context_truncated)

    def percentiles(values: list[int]) -> dict[str, float]:
        return {
            name: round(float(np.percentile(values, value)), 3)
            for name, value in (
                ("min", 0),
                ("p50", 50),
                ("p90", 90),
                ("p95", 95),
                ("p99", 99),
                ("max", 100),
            )
        }

    return {
        "rows": len(frame),
        "input_mode": input_mode,
        "max_length": max_length,
        "target_with_special_tokens": percentiles(target),
        "encoded_length": percentiles(total),
        "target_truncated_rows": target_truncated,
        "context_truncated_rows": context_truncated,
        "overflow_policy": "deterministic head+tail target truncation; all context is dropped first",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-mode", choices=("target_only", "local_context"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    frame, manifest, _ = load_effective_dataset(config)
    model_path = resolve_path(args.model_path or config["model"]["default_local_path"])
    require_local_model_files(model_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=config["model"].get("revision"),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
        fix_mistral_regex=False,
    )
    mode = args.input_mode or config["tokenization"]["input_mode"]
    max_length = args.max_length or int(config["tokenization"]["max_length"])
    report = {
        "schema_version": 1,
        "loads_model_weights": False,
        "offline_only": True,
        "effective_dataset_sha256": manifest["effective_dataset_sha256"],
        "tokenizer_source": str(model_path),
        **audit_lengths(tokenizer, frame, max_length=max_length, input_mode=mode),
    }
    output = args.output.resolve()
    with immutable_directory(output) as building:
        write_json(building / "tokenizer_length_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
