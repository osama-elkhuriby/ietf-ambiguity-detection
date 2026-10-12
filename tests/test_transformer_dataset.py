from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from models.transformer.dataset import (
    build_transformer_dataset,
    leakage_audit,
    normalize_for_audit,
    pair_id,
    require_training_ready,
    token_ngrams,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(*, duplicate_test: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "train-1",
                "family_id": "family-train",
                "document_id": "draft-train-01",
                "revision": "01",
                "clause_ids": "train-c2",
                "clause_text": "An endpoint MUST send a request.",
                "section_ids": "3",
                "lineage_ids": "lineage-train",
                "ambiguity_label": "yes",
                "label": 1,
                "llm_label_tier": "HIGH",
                "training_weight": 1.0,
                "llm_ambiguity_subtype": "U-2",
                "split": "train",
            },
            {
                "record_id": "validation-1",
                "family_id": "family-validation",
                "document_id": "draft-validation-01",
                "revision": "01",
                "clause_ids": "validation-c2",
                "clause_text": "A server SHOULD process the response.",
                "section_ids": "4",
                "lineage_ids": "lineage-validation",
                "ambiguity_label": "no",
                "label": 0,
                "llm_label_tier": "MEDIUM",
                "training_weight": 0.5,
                "llm_ambiguity_subtype": "none",
                "split": "validation",
            },
            {
                "record_id": "test-1",
                "family_id": "family-test",
                "document_id": "draft-test-01",
                "revision": "01",
                "clause_ids": "test-c2",
                "clause_text": (
                    "An endpoint MUST send a request."
                    if duplicate_test
                    else "A client MAY close the connection."
                ),
                "section_ids": "5",
                "lineage_ids": "lineage-test",
                "ambiguity_label": "no",
                "label": 0,
                "llm_label_tier": "MEDIUM",
                "training_weight": 0.5,
                "llm_ambiguity_subtype": "none",
                "split": "test",
            },
            {
                "record_id": "uncertain-1",
                "family_id": "family-train",
                "document_id": "draft-train-01",
                "revision": "01",
                "clause_ids": "train-c3",
                "clause_text": "An optional uncertain clause.",
                "section_ids": "3",
                "lineage_ids": "lineage-uncertain",
                "ambiguity_label": "uncertain",
                "label": "",
                "llm_label_tier": "UNCERTAIN",
                "training_weight": 0.0,
                "llm_ambiguity_subtype": "none",
                "split": "train",
            },
        ]
    )


def _write_alignment(
    root: Path, family: str, document: str, target_id: str
) -> Path:
    path = root / family / "run-v1" / "inputs" / f"{document}.json"
    path.parent.mkdir(parents=True)
    section = {"family-train": "3", "family-validation": "4"}.get(family, "5")
    clauses = [
        {
            "index": 1,
            "clause_id": f"{document}-previous",
            "section_id": section,
            "section_heading": "Operation",
            "raw_text": f"Previous context for {document}.",
        },
        {
            "index": 2,
            "clause_id": target_id,
            "section_id": section,
            "section_heading": "Operation",
            "raw_text": f"Target text for {document}.",
        },
        {
            "index": 3,
            "clause_id": f"{document}-next",
            "section_id": section,
            "section_heading": "Operation",
            "raw_text": f"Next context for {document}.",
        },
    ]
    path.write_text(json.dumps({"clauses": clauses}), encoding="utf-8")
    return path


def _write_all_alignments(root: Path) -> None:
    _write_alignment(root, "family-train", "draft-train-01", "train-c2")
    _write_alignment(
        root, "family-validation", "draft-validation-01", "validation-c2"
    )
    _write_alignment(root, "family-test", "draft-test-01", "test-c2")


def test_normalization_and_ngrams_are_deterministic() -> None:
    assert normalize_for_audit("  The\u00a0Endpoint  MUST Send. ") == (
        "the endpoint must send."
    )
    assert token_ngrams("one two three four") == {
        ("one", "two", "three"),
        ("two", "three", "four"),
    }


def test_builder_materializes_context_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "training.csv"
    _source_rows().to_csv(source, index=False)
    before = _hash(source)
    alignments = tmp_path / "alignments"
    _write_all_alignments(alignments)
    output = tmp_path / "transformer-v1"

    manifest = build_transformer_dataset(
        input_path=source,
        alignment_root=alignments,
        output_dir=output,
    )

    assert _hash(source) == before
    assert manifest["binary_rows"] == 3
    assert manifest["excluded_uncertain_rows"] == 1
    assert manifest["training_blocked"] is False
    assert manifest["context_resolution_counts"] == {"both": 3}
    dataset = pd.read_csv(output / "dataset.csv")
    train = dataset.loc[dataset["record_id"] == "train-1"].iloc[0]
    assert train["target_clause"] == "An endpoint MUST send a request."
    assert train["previous_clause"] == "Previous context for draft-train-01."
    assert train["next_clause"] == "Next context for draft-train-01."
    assert json.loads(train["context_source_ids"])["target"] == ["train-c2"]
    assert json.loads(train["source_hashes"])["annotation_source"] == before
    assert (output / "manifest.json").is_file()
    assert (output / "split_assignments.csv").is_file()
    assert (output / "leakage_audit.json").is_file()
    assert len(pd.read_csv(output / "splits/train.csv")) == 1
    assert len(pd.read_csv(output / "splits/validation.csv")) == 1
    assert len(pd.read_csv(output / "splits/test.csv")) == 1
    assert manifest["split_files"]["test"]["rows"] == 1


def test_unreviewed_exact_cross_split_duplicate_blocks_training(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training.csv"
    _source_rows(duplicate_test=True).to_csv(source, index=False)
    alignments = tmp_path / "alignments"
    _write_all_alignments(alignments)
    output = tmp_path / "blocked"

    manifest = build_transformer_dataset(
        input_path=source,
        alignment_root=alignments,
        output_dir=output,
    )

    assert manifest["training_blocked"] is True
    audit = json.loads((output / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["counts"]["unreviewed_exact_cross_split_duplicates"] == 1
    with pytest.raises(RuntimeError, match="blocked"):
        require_training_ready(manifest)


def test_reviewed_exact_duplicate_allowlist_unblocks_training(tmp_path: Path) -> None:
    source = tmp_path / "training.csv"
    _source_rows(duplicate_test=True).to_csv(source, index=False)
    alignments = tmp_path / "alignments"
    _write_all_alignments(alignments)
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps({"allowed_pairs": [pair_id("train-1", "test-1")]}),
        encoding="utf-8",
    )

    manifest = build_transformer_dataset(
        input_path=source,
        alignment_root=alignments,
        output_dir=tmp_path / "allowed",
        duplicate_allowlist=allowlist,
    )

    assert manifest["training_blocked"] is False
    require_training_ready(manifest)


def test_leakage_audit_reports_near_duplicates_and_context_overlap() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["train", "test"],
            "family_id": ["a", "b"],
            "split": ["train", "test"],
            "target_clause": [
                "the endpoint must send the final protocol value now",
                "the endpoint must send the final protocol value today",
            ],
            "previous_clause": [
                "",
                "the endpoint must send the final protocol value now",
            ],
            "next_clause": ["", ""],
        }
    )

    audit = leakage_audit(frame, near_duplicate_threshold=0.7)

    assert audit["counts"]["near_cross_split_duplicates"] == 1
    assert audit["counts"]["target_as_cross_split_context"] == 1


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_transformer_dataset(
            input_path=tmp_path / "missing.csv",
            alignment_root=tmp_path,
            output_dir=output,
        )
