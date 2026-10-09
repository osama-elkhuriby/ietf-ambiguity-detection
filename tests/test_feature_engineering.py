from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from scipy import sparse

from models.classical.clause_features import (
    FEATURE_COLUMNS,
    extract_clause_features,
    extract_dataframe_features,
)
from models.classical.feature_pipeline import (
    FeaturePreprocessor,
    build_feature_artifacts,
    load_binary_dataset,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_clause_features_cover_requested_language_groups() -> None:
    text = (
        "If it is generally appropriate, the endpoint SHOULD NOT wait until "
        "Section 3; otherwise, it MAY continue."
    )
    features = extract_clause_features(
        text,
        previous_text="The peer sends a request.",
        next_text="Processing then resumes.",
        section_id="4.2.1",
    )

    assert tuple(features) == FEATURE_COLUMNS
    assert features["basic_word_count"] > 0
    assert features["basic_sentence_count"] == 1
    assert features["vague_term_count"] >= 2
    assert features["vague_term_present"] == 1
    assert features["anaphora_count"] >= 2
    assert features["cross_reference_count"] == 1
    assert features["conditional_count"] == 1
    assert features["exception_count"] == 1
    assert features["temporal_count"] == 1
    assert features["normative_should_not_count"] == 1
    assert features["normative_should_count"] == 0
    assert features["normative_may_count"] == 1
    assert features["context_previous_present"] == 1
    assert features["context_next_present"] == 1
    assert features["context_section_depth"] == 3


def test_missing_values_are_safe_and_schema_is_stable() -> None:
    missing = extract_clause_features(None, previous_text=float("nan"), section_id=None)
    pandas_missing = extract_clause_features(pd.NA, next_text=pd.NA)
    empty = extract_clause_features("")
    assert tuple(missing) == FEATURE_COLUMNS
    assert missing == empty
    assert pandas_missing == empty
    assert all(value == 0 for value in missing.values())


def test_optional_syntax_features_use_dependency_parse() -> None:
    class Token:
        def __init__(self, index: int, dependency: str) -> None:
            self.i = index
            self.dep_ = dependency
            self.is_space = False
            self.head = self
            self.children: list[Token] = []

    condition = Token(0, "advcl")
    root = Token(1, "ROOT")
    conjunction = Token(2, "cc")
    condition.head = root
    conjunction.head = root
    root.children = [condition, conjunction]

    features = extract_clause_features(
        "If ready, send it.", nlp=lambda _: [condition, root, conjunction]
    )
    assert features["syntax_available"] == 1
    assert features["syntax_mean_dependency_distance"] == 1.0
    assert features["syntax_max_dependency_depth"] == 2
    assert features["syntax_subordinate_clause_count"] == 1
    assert features["syntax_coordinate_conjunction_count"] == 1


def test_dataframe_extraction_is_deterministic_and_ignores_metadata() -> None:
    first = pd.DataFrame(
        {
            "clause_text": ["An endpoint MUST send the value."],
            "previous_clause": ["Before."],
            "next_clause": ["After."],
            "section_ids": ["3.1"],
            "record_id": ["record-a"],
            "label": [1],
            "llm_label_tier": ["HIGH"],
            "training_weight": [1.0],
            "split": ["train"],
            "justification": ["leaking rationale"],
        }
    )
    second = first.assign(
        record_id="different-id",
        label=0,
        llm_label_tier="MEDIUM",
        training_weight=0.5,
        split="test",
        justification="different rationale",
    )

    features_a = extract_dataframe_features(first)
    features_b = extract_dataframe_features(second)
    pd.testing.assert_frame_equal(features_a, features_b)
    pd.testing.assert_frame_equal(features_a, extract_dataframe_features(first))
    assert tuple(features_a.columns) == FEATURE_COLUMNS
    assert not any(
        forbidden in column
        for column in features_a.columns
        for forbidden in ("label", "tier", "weight", "record_id", "split", "rationale")
    )


def test_binary_loading_excludes_uncertain_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "annotations.csv"
    frame = sample_dataset(include_uncertain=True)
    frame.to_csv(source, index=False)
    before = file_hash(source)

    binary, excluded = load_binary_dataset(source)

    assert excluded == 1
    assert set(binary["ambiguity_label"]) == {"yes", "no"}
    assert binary["split"].tolist() == ["train", "validation", "test"]
    assert binary["training_weight"].tolist() == [1.0, 0.5, 0.5]
    assert file_hash(source) == before
    assert len(pd.read_csv(source)) == 4


def test_binary_loading_rejects_family_split_leakage(tmp_path: Path) -> None:
    source = tmp_path / "leaking.csv"
    frame = sample_dataset(include_uncertain=False)
    frame.loc[1, "family_id"] = "family-train"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="families span multiple"):
        load_binary_dataset(source)


def test_tfidf_vocabulary_is_fit_only_on_training_text() -> None:
    train_text = pd.Series(["alpha beta", "alpha gamma"])
    validation_text = pd.Series(["alpha validationsecret"])
    train_features = extract_dataframe_features(
        pd.DataFrame({"clause_text": train_text})
    )
    validation_features = extract_dataframe_features(
        pd.DataFrame({"clause_text": validation_text})
    )
    preprocessor = FeaturePreprocessor.create(max_tfidf_features=100, min_df=1)
    preprocessor.fit(train_text, train_features)

    vocabulary = set(preprocessor.vectorizer.get_feature_names_out())
    assert "validationsecret" not in vocabulary
    tfidf, engineered, combined = preprocessor.transform(
        validation_text, validation_features
    )
    assert tfidf.shape[1] == len(vocabulary)
    assert engineered.shape[1] == len(FEATURE_COLUMNS)
    assert combined.shape[1] == len(vocabulary) + len(FEATURE_COLUMNS)


def test_pipeline_preserves_splits_and_exports_separate_weights(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "training.csv"
    sample_dataset(include_uncertain=True).to_csv(source, index=False)
    source_hash = file_hash(source)
    output = tmp_path / "features-v1"

    def no_context(frame: pd.DataFrame, alignment_root: Path):
        enriched = frame.copy()
        enriched["previous_clause"] = ""
        enriched["next_clause"] = ""
        enriched["section_heading"] = ""
        return enriched, 0

    monkeypatch.setattr(
        "models.classical.feature_pipeline.add_frozen_context", no_context
    )
    manifest = build_feature_artifacts(
        input_path=source,
        alignment_root=tmp_path,
        output_dir=output,
        max_tfidf_features=100,
        min_df=1,
        syntax_mode="disabled",
    )

    assert file_hash(source) == source_hash
    assert manifest["excluded_uncertain_rows"] == 1
    assert manifest["source_preserved"] is True
    assert manifest["split_summary"]["train"]["families"] == ["family-train"]
    assert manifest["split_summary"]["validation"]["families"] == ["family-validation"]
    assert manifest["split_summary"]["test"]["families"] == ["family-test"]

    for split in ("train", "validation", "test"):
        labels = pd.read_csv(output / f"targets/{split}_labels.csv")
        weights = pd.read_csv(output / f"weights/{split}_sample_weights.csv")
        rows = pd.read_csv(output / f"rows/{split}_rows.csv")
        engineered = sparse.load_npz(output / f"matrices/{split}_engineered.npz")
        tfidf = sparse.load_npz(output / f"matrices/{split}_tfidf.npz")
        combined = sparse.load_npz(output / f"matrices/{split}_combined.npz")
        assert len(labels) == len(weights) == len(rows) == engineered.shape[0]
        assert combined.shape[1] == tfidf.shape[1] + engineered.shape[1]
        assert "training_weight" not in labels.columns
        assert "label" not in weights.columns


def sample_dataset(*, include_uncertain: bool) -> pd.DataFrame:
    rows = [
        {
            "record_id": "train-1",
            "family_id": "family-train",
            "document_id": "draft-train-01",
            "revision": "01",
            "clause_ids": "clause-train-1",
            "clause_text": "An endpoint MUST send alpha.",
            "section_ids": "3.1",
            "lineage_ids": "lineage-train-1",
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
            "clause_ids": "clause-validation-1",
            "clause_text": "A peer SHOULD process validationsecret.",
            "section_ids": "4",
            "lineage_ids": "lineage-validation-1",
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
            "clause_ids": "clause-test-1",
            "clause_text": "The server MAY send testsecret.",
            "section_ids": "5",
            "lineage_ids": "lineage-test-1",
            "ambiguity_label": "no",
            "label": 0,
            "llm_label_tier": "MEDIUM",
            "training_weight": 0.5,
            "llm_ambiguity_subtype": "none",
            "split": "test",
        },
    ]
    if include_uncertain:
        rows.append(
            {
                "record_id": "uncertain-1",
                "family_id": "family-train",
                "document_id": "draft-train-01",
                "revision": "01",
                "clause_ids": "clause-uncertain-1",
                "clause_text": "The endpoint processes the value.",
                "section_ids": "6",
                "lineage_ids": "lineage-uncertain-1",
                "ambiguity_label": "uncertain",
                "label": -1,
                "llm_label_tier": "UNCERTAIN",
                "training_weight": 0.0,
                "llm_ambiguity_subtype": "none",
                "split": "train",
            }
        )
    return pd.DataFrame(rows)
