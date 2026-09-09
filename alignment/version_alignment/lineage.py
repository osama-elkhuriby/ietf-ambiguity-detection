"""Forward-only lineage IDs; appending RFC outcomes cannot rename draft history."""

from __future__ import annotations

from .io import digest
from .models import Alignment, Document


def build_lineage(documents: list[Document], rows: list[Alignment]) -> list[dict]:
    known: dict[str, str] = {}
    nodes = []
    by_target = {}
    for row in rows:
        for target in row.target_clause_ids:
            by_target[target] = row
    for document in documents:
        for clause in document.clauses:
            row = by_target.get(clause.clause_id)
            parents = (
                sorted({known[s] for s in row.source_clause_ids if s in known})
                if row and row.relation != "UNCERTAIN"
                else []
            )
            continuation = (
                row
                and row.relation in {"UNCHANGED", "MODIFIED", "MOVED"}
                and len(parents) == 1
            )
            identity = (
                parents[0]
                if continuation
                else "lineage:" + digest([document.family_id, clause.clause_id])[:24]
            )
            known[clause.clause_id] = identity
            nodes.append(
                {
                    "schema_version": "1.0",
                    "lineage_id": identity,
                    "clause_id": clause.clause_id,
                    "document_id": document.document_id,
                    "revision": document.revision,
                    "available_at": document.available_at.isoformat(),
                    "parent_lineage_ids": parents if not continuation else [],
                    "parent_clause_ids": row.source_clause_ids
                    if row and row.relation != "UNCERTAIN"
                    else [],
                    "alignment_record_id": row.record_id if row else None,
                    "history_uncertain": bool(row and row.relation == "UNCERTAIN"),
                    "source_sha256": document.source_sha256,
                }
            )
    for row in rows:
        row.lineage_ids = sorted(
            {known[c] for c in row.source_clause_ids + row.target_clause_ids}
        )
    return nodes
