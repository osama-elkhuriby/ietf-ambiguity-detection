from __future__ import annotations

from corpus.annotate_with_ollama import (
    consensus_decision,
    prompt_payload,
    stratified_limit,
    training_rows,
    validate_annotation,
)


def annotation(label: str, subtype: str = "none") -> dict[str, str]:
    if label == "yes":
        return {
            "is_ambiguous": "yes",
            "ambiguity_subtype": subtype,
            "evidence_span": "MUST process",
            "interpretation_a": "Process immediately.",
            "interpretation_b": "Process after validation.",
            "missing_information": "",
            "justification": "The timing constraint has two operational readings.",
        }
    return {
        "is_ambiguous": label,
        "ambiguity_subtype": "none",
        "evidence_span": "",
        "interpretation_a": "",
        "interpretation_b": "",
        "missing_information": "context" if label == "uncertain" else "",
        "justification": "The clause has one operational reading.",
    }


def prediction(model: str, label: str, status: str = "valid") -> dict[str, object]:
    return {"model": model, "status": status, "annotation": annotation(label, "U-2")}


def test_prompt_payload_does_not_expose_candidate_or_human_labels() -> None:
    row = {
        "document_id": "draft-example-01",
        "section_ids": "3",
        "clause_text": "An endpoint MUST process the value.",
        "weak_label": "SILVER_CHANGE_CANDIDATE",
        "ambiguity_label": "UNCERTAIN",
        "human_is_ambiguous": "yes",
        "errata_id": "1234",
    }
    payload = prompt_payload(row, {"previous_clause": "Before.", "next_clause": "After."})
    serialized = str(payload)
    assert payload["target_clause"] == row["clause_text"]
    assert "SILVER_CHANGE_CANDIDATE" not in serialized
    assert "errata" not in serialized.lower()
    assert "human" not in serialized.lower()


def test_validation_requires_exact_evidence_for_positive_label() -> None:
    valid = annotation("yes", "U-2")
    assert validate_annotation(valid, "An endpoint MUST process the value.") == []
    invalid = dict(valid, evidence_span="must process")
    assert "exact target-clause substring" in " ".join(
        validate_annotation(invalid, "An endpoint MUST process the value.")
    )


def test_validation_rejects_positive_without_demonstrated_ambiguity() -> None:
    value = annotation("yes", "U-2")
    value["interpretation_a"] = ""
    value["interpretation_b"] = ""
    assert "two distinct interpretations" in " ".join(
        validate_annotation(value, "An endpoint MUST process the value.")
    )


def test_consensus_rules_use_two_of_three_majority() -> None:
    unanimous = consensus_decision(
        [prediction("a", "yes"), prediction("b", "yes"), prediction("c", "yes")]
    )
    assert unanimous["llm_is_ambiguous"] == "yes"
    assert unanimous["llm_label_tier"] == "HIGH"

    abstention = consensus_decision(
        [
            prediction("a", "no"),
            prediction("b", "no"),
            prediction("c", "uncertain"),
        ]
    )
    assert abstention["llm_is_ambiguous"] == "no"
    assert abstention["llm_label_tier"] == "MEDIUM"

    majority_yes = consensus_decision(
        [prediction("a", "yes"), prediction("b", "no"), prediction("c", "yes")]
    )
    assert majority_yes["llm_is_ambiguous"] == "yes"
    assert majority_yes["llm_label_tier"] == "MEDIUM"
    assert majority_yes["consensus_reason"] == "majority_yes"

    majority_no = consensus_decision(
        [prediction("a", "no"), prediction("b", "yes"), prediction("c", "no")]
    )
    assert majority_no["llm_is_ambiguous"] == "no"
    assert majority_no["llm_label_tier"] == "MEDIUM"
    assert majority_no["consensus_reason"] == "majority_no"

    invalid = consensus_decision(
        [
            prediction("a", "yes"),
            prediction("b", "yes"),
            prediction("c", "yes", status="invalid"),
        ]
    )
    assert invalid["llm_is_ambiguous"] == "yes"
    assert invalid["llm_label_tier"] == "MEDIUM"

    insufficient_valid_votes = consensus_decision(
        [
            prediction("a", "yes"),
            prediction("b", "yes", status="invalid"),
            prediction("c", "no", status="invalid"),
        ]
    )
    assert insufficient_valid_votes["llm_is_ambiguous"] == "uncertain"
    assert insufficient_valid_votes["llm_label_tier"] == "UNCERTAIN"


def test_training_export_excludes_uncertain_and_splits_by_family() -> None:
    base = {
        "record_id": "r1",
        "document_id": "draft-01",
        "revision": "01",
        "clause_ids": "c1",
        "clause_text": "An endpoint MUST process the value.",
        "section_ids": "3",
        "lineage_ids": "l1",
        "llm_ambiguity_subtype": "none",
    }
    consensus = [
        dict(
            base,
            family_id="train-family",
            llm_is_ambiguous="no",
            llm_label_tier="HIGH",
        ),
        dict(
            base,
            record_id="r2",
            family_id="validation-family",
            llm_is_ambiguous="yes",
            llm_label_tier="MEDIUM",
        ),
        dict(
            base,
            record_id="r3",
            family_id="test-family",
            llm_is_ambiguous="uncertain",
            llm_label_tier="UNCERTAIN",
        ),
    ]
    rows = training_rows(consensus, "validation-family", "test-family")
    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert [row["split"] for row in rows] == ["train", "validation"]
    assert [row["training_weight"] for row in rows] == [1.0, 0.5]


def test_pilot_limit_interleaves_family_and_candidate_groups() -> None:
    rows = [
        {
            "record_id": f"{family}-{label}-{number}",
            "family_id": family,
            "weak_label": label,
        }
        for family in ("a", "b")
        for label in ("change", "clear")
        for number in range(3)
    ]
    selected = stratified_limit(rows, 4)
    assert len(selected) == 4
    assert {(row["family_id"], row["weak_label"]) for row in selected} == {
        ("a", "change"),
        ("a", "clear"),
        ("b", "change"),
        ("b", "clear"),
    }
