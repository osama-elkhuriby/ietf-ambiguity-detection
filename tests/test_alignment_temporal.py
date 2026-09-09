from datetime import date

import pytest
from test_alignment_matching import align, document

from alignment.version_alignment.features import features
from alignment.version_alignment.lineage import build_lineage
from alignment.version_alignment.models import Alignment, Evidence
from alignment.version_alignment.temporal import (
    guard,
    prediction_view,
    resolve_reference,
)
from eval.alignment_metrics import evaluate


def test_guard_future_roles_unknown_transitive_and_same_day():
    target = document(1, ["Text."])
    for role, day in [
        ("LABEL_SUPPORT", date(2020, 1, 1)),
        ("MODEL_FEATURE", None),
        ("MODEL_FEATURE", date(2020, 1, 3)),
        ("MODEL_FEATURE", date(2020, 1, 2)),
    ]:
        with pytest.raises(ValueError):
            guard(Evidence(evidence_id="bad", role=role, available_at=day), target)
    dependency = Evidence(
        evidence_id="future", role="MODEL_FEATURE", available_at=date(2020, 2, 1)
    )
    with pytest.raises(ValueError, match="parent -> future"):
        guard(
            Evidence(
                evidence_id="parent",
                role="MODEL_FEATURE",
                available_at=date(2020, 1, 1),
                dependencies=[dependency],
            ),
            target,
        )


def test_reference_resolution_never_latest_future():
    docs = [document(i, ["Text."]) for i in range(3)]
    assert (
        resolve_reference(docs[0].family_id, docs, docs[1]).document_id
        == docs[1].document_id
    )
    assert resolve_reference(docs[2].document_id, docs, docs[1]) is None
    assert resolve_reference("missing", docs, docs[1]) is None


def test_prefix_features_and_lineage_unchanged_by_future():
    text = "The client MUST verify the entire certificate chain before transmission."
    docs = [document(i, [text]) for i in range(3)]
    first, _ = align(docs[0], docs[1])
    second, _ = align(docs[1], docs[2])
    before = features(docs[:2], first, docs[1].document_id)
    after = features(docs, first + second, docs[1].document_id)
    assert before == after
    early = build_lineage(docs[:2], first)
    full = build_lineage(docs, first + second)
    assert early == full[: len(early)]
    assert before[0]["prior_observed_edits"] == 0
    assert len({n["lineage_id"] for n in full}) == 1


def test_serialization_and_gold_metrics():
    docs = [
        document(i, ["The client MUST verify the entire chain before transmission."])
        for i in range(2)
    ]
    rows, _candidates = align(*docs)
    restored = Alignment.model_validate_json(rows[0].model_dump_json())
    assert restored == rows[0]
    payload = [r.model_dump(mode="json") for r in rows]
    assert evaluate(payload)["link_precision"] is None
    gold = [dict(r, reviewer="synthetic-test", scope_complete=True) for r in payload]
    result = evaluate(payload, payload, gold)
    assert result["link_f1"] == result["candidate_recall"] == 1
    with pytest.raises(ValueError):
        evaluate(payload, gold=payload)


def test_tls00_01_same_submission_day_allows_only_predecessors():
    docs = [document(i, ["The client MUST verify the certificate."]) for i in range(3)]
    for doc in docs:
        doc.publication_date = doc.available_at = date(2014, 4, 17)
    prefix, _ = prediction_view(docs, [], docs[1].document_id)
    assert [d.revision for d in prefix] == ["00", "01"]
    assert resolve_reference(docs[0].document_id, docs, docs[1]) == docs[0]
    assert resolve_reference(docs[2].document_id, docs, docs[1]) is None
    with pytest.raises(ValueError):
        guard(
            Evidence(
                evidence_id="same-day-discussion",
                role="MODEL_FEATURE",
                available_at=date(2014, 4, 17),
                document_id="external-discussion",
            ),
            docs[1],
        )
    assert features(docs, [], docs[1].document_id)[0]["revisions_observed"] == 2
