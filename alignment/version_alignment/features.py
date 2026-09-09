"""Basic retrospective draft features with target-prefix evidence admission."""

from __future__ import annotations

from .models import Alignment, Document
from .temporal import prediction_view


def features(
    documents: list[Document], rows: list[Alignment], target_id: str
) -> list[dict]:
    prefix, admissible = prediction_view(documents, rows, target_id)
    target = next(d for d in prefix if d.document_id == target_id)
    incoming = {c: r for r in admissible for c in r.target_clause_ids}
    histories = {}
    for document in prefix:
        for clause in document.clauses:
            row = incoming.get(clause.clause_id)
            parents = [histories.get(c) for c in row.source_clause_ids] if row else []
            uncertain = bool(row and row.relation == "UNCERTAIN") or any(
                p is None or p[1] for p in parents
            )
            events = set().union(*(p[0] for p in parents if p)) if parents else set()
            if row and row.relation not in {"UNCHANGED", "INSERTED", "UNCERTAIN"}:
                events.add(row.record_id)
            histories[clause.clause_id] = (events, uncertain)
    return [
        {
            "schema_version": "1.0",
            "family_id": target.family_id,
            "document_id": target_id,
            "clause_id": c.clause_id,
            "evidence_role": "MODEL_FEATURE",
            "available_at": target.publication_date.isoformat(),
            "prior_observed_edits": None
            if histories[c.clause_id][1]
            else len(histories[c.clause_id][0]),
            "history_uncertain": histories[c.clause_id][1],
            "history_complete": int(prefix[0].revision) == 0
            and len(prefix) == int(target.revision) + 1,
            "revisions_observed": len(prefix),
            "elapsed_document_days": (
                target.publication_date - prefix[0].publication_date
            ).days,
            "previous_lexical_distance": 1 - incoming[c.clause_id].scores["lexical"]
            if c.clause_id in incoming
            and incoming[c.clause_id].relation != "UNCERTAIN"
            and incoming[c.clause_id].scores.get("lexical") is not None
            else None,
            "previous_semantic_distance": 1 - incoming[c.clause_id].scores["semantic"]
            if c.clause_id in incoming
            and incoming[c.clause_id].relation != "UNCERTAIN"
            and incoming[c.clause_id].scores.get("semantic") is not None
            else None,
            "change_flags": incoming[c.clause_id].changes
            if c.clause_id in incoming and incoming[c.clause_id].relation != "UNCERTAIN"
            else {},
            "dependency_document_ids": [d.document_id for d in prefix],
            "dependency_source_hashes": [d.source_sha256 for d in prefix],
        }
        for c in target.clauses
    ]
