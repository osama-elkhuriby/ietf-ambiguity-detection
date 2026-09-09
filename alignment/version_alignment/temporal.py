"""Fail-closed prediction evidence and immutable target-prefix views."""

from __future__ import annotations

from .models import Alignment, Document, Evidence


def guard(
    evidence: Evidence,
    target: Document,
    chain: tuple[str, ...] = (),
    *,
    same_day_predecessors: frozenset[str] = frozenset(),
) -> None:
    path = (*chain, evidence.evidence_id)
    if evidence.role != "MODEL_FEATURE":
        raise ValueError(
            f"Non-feature evidence in prediction pipeline: {' -> '.join(path)} ({evidence.role})"
        )
    # Unknown intra-day ordering for external evidence is conservatively rejected.
    same_day_external = (
        evidence.available_at == target.publication_date
        and evidence.document_id != target.document_id
        and evidence.document_id not in same_day_predecessors
    )
    if (
        evidence.available_at is None
        or evidence.available_at > target.publication_date
        or same_day_external
    ):
        raise ValueError(
            f"Unavailable/future evidence {' -> '.join(path)}: {evidence.available_at}, target {target.document_id} at {target.publication_date}"
        )
    for dependency in evidence.dependencies:
        guard(dependency, target, path, same_day_predecessors=same_day_predecessors)


def predecessors(documents: list[Document], target: Document) -> frozenset[str]:
    """Same-family revision order resolves same-day draft submission granularity."""
    if target.revision is None:
        return frozenset()
    return frozenset(
        d.document_id
        for d in documents
        if d.family_id == target.family_id
        and d.revision is not None
        and int(d.revision) < int(target.revision)
        and d.available_at == target.publication_date
    )


def prediction_view(
    documents: list[Document], rows: list[Alignment], target_id: str
) -> tuple[list[Document], list[Alignment]]:
    target = next((d for d in documents if d.document_id == target_id), None)
    if target is None:
        raise ValueError(f"Unknown target: {target_id}")
    if target.revision is None:
        raise ValueError(
            "Prediction-time MVP features are for draft revisions, not final RFCs"
        )
    prefix = [
        d
        for d in documents
        if d.family_id == target.family_id
        and d.revision is not None
        and int(d.revision) <= int(target.revision)
    ]
    same_day = predecessors(prefix, target)
    for document in prefix:
        guard(
            Evidence(
                evidence_id=document.source_sha256,
                role="MODEL_FEATURE",
                available_at=document.available_at,
                document_id=document.document_id,
            ),
            target,
            same_day_predecessors=same_day,
        )
    ids = {d.document_id for d in prefix}
    selected = [
        r for r in rows if r.from_document_id in ids and r.to_document_id in ids
    ]
    if any(r.review_status != "NOT_REVIEWED" for r in selected):
        raise ValueError(
            "Reviewed offline alignment cannot enter automatic prediction history without evidence validation"
        )
    return prefix, selected


def resolve_reference(
    reference: str, catalog: list[Document], target: Document
) -> Document | None:
    choices = []
    for document in catalog:
        if document.document_id != reference and document.family_id != reference:
            continue
        try:
            guard(
                Evidence(
                    evidence_id=document.source_sha256,
                    role="MODEL_FEATURE",
                    available_at=document.available_at,
                    document_id=document.document_id,
                ),
                target,
                same_day_predecessors=predecessors(catalog, target),
            )
        except ValueError:
            continue
        choices.append(document)
    return max(
        choices, key=lambda d: (d.available_at, int(d.revision or -1)), default=None
    )
