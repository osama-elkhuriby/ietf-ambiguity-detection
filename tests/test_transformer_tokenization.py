from __future__ import annotations

import pandas as pd
import pytest

from models.transformer.tokenization import audit_token_lengths, encode_example


class FakeTokenizer:
    sep_token_id = 99
    cls_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(10, 10 + len(text.split())))

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return 3 if pair else 2

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return [1, *token_ids_0, 2]
        return [1, *token_ids_0, 2, *token_ids_1, 2]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))


def test_target_only_uses_deterministic_head_and_tail() -> None:
    encoded = encode_example(
        FakeTokenizer(),
        target="one two three four five six seven eight",
        variant="target_only",
        max_length=8,
    )

    assert encoded.target_token_count == 8
    assert encoded.target_tokens_kept == 6
    assert encoded.target_truncated is True
    assert encoded.input_ids == [1, 10, 11, 12, 15, 16, 17, 2]


def test_context_is_trimmed_before_target_and_budget_is_redistributed() -> None:
    encoded = encode_example(
        FakeTokenizer(),
        target="target has four tokens",
        previous="previous has two",
        following="next context has many many tokens here",
        variant="local_context",
        max_length=12,
    )

    assert encoded.target_tokens_kept == 4
    assert encoded.previous_tokens_kept == 2
    assert encoded.next_tokens_kept == 2
    assert encoded.context_truncated is True
    assert encoded.target_truncated is False
    assert len(encoded.input_ids) == 12


def test_invalid_variant_and_too_small_budget_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        encode_example(FakeTokenizer(), target="text", variant="future")
    with pytest.raises(ValueError, match="no room"):
        encode_example(
            FakeTokenizer(), target="text", variant="local_context", max_length=3
        )


def test_length_audit_reports_target_and_context_truncation() -> None:
    frame = pd.DataFrame(
        {
            "target_clause": ["short target", "one two three four five six"],
            "previous_clause": ["", "previous context has words"],
            "next_clause": ["", "next context has words"],
        }
    )

    report = audit_token_lengths(FakeTokenizer(), frame, max_length=8)

    assert report["rows"] == 2
    assert report["target_truncated_rows"] == 0
    assert report["context_truncated_rows"] == 1
    assert report["target_with_special_tokens"]["max"] == 8
