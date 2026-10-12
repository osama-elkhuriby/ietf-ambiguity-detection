from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from corpus.build_model_dataset import (
    apply_split_policy,
    build_model_dataset,
    load_split_policy,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "a-1",
                "family_id": "family-a",
                "ambiguity_label": "no",
                "label": 0,
                "split": "test",
            },
            {
                "record_id": "b-1",
                "family_id": "family-b",
                "ambiguity_label": "yes",
                "label": 1,
                "split": "train",
            },
            {
                "record_id": "c-1",
                "family_id": "family-c",
                "ambiguity_label": "no",
                "label": 0,
                "split": "validation",
            },
        ]
    )


def _policy(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "test-policy",
                "rationale": "exercise an explicit family swap",
                "assignments": {
                    "train": ["family-a"],
                    "validation": ["family-c"],
                    "test": ["family-b"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_policy_application_is_family_level_and_preserves_rows(tmp_path: Path) -> None:
    policy, _ = load_split_policy(_policy(tmp_path / "policy.json"))
    result = apply_split_policy(_source(), policy)

    assert result["record_id"].tolist() == ["a-1", "b-1", "c-1"]
    assert result["split"].tolist() == ["train", "test", "validation"]
    assert result.groupby("family_id")["split"].nunique().max() == 1


def test_policy_rejects_missing_and_duplicate_family_assignments(
    tmp_path: Path,
) -> None:
    incomplete = {
        "family-a": "train",
        "family-b": "test",
    }
    with pytest.raises(ValueError, match="missing from split policy"):
        apply_split_policy(_source(), incomplete)

    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "assignments": {
                    "train": ["family-a"],
                    "validation": ["family-a"],
                    "test": ["family-b", "family-c"],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_split_policy(path)


def test_builder_is_immutable_hashed_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _source().to_csv(source, index=False)
    before = _hash(source)
    output = tmp_path / "model-v2"

    manifest = build_model_dataset(
        source_path=source,
        split_config=_policy(tmp_path / "policy.json"),
        output_dir=output,
    )

    assert _hash(source) == before
    assert manifest["source_preserved"] is True
    assert manifest["split_summary"]["train"]["families"] == ["family-a"]
    assert manifest["split_summary"]["test"]["labels"] == {"1": 1}
    assert _hash(output / "training_dataset.csv") == manifest["dataset_sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        build_model_dataset(
            source_path=source,
            split_config=tmp_path / "policy.json",
            output_dir=output,
        )
