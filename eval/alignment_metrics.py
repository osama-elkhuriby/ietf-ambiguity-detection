"""Alignment workload and optional explicitly reviewed gold metrics."""

from __future__ import annotations

from collections import Counter


def links(row):
    if row["relation"] == "UNCERTAIN":
        return set()
    return {
        (row["from_document_id"], row["to_document_id"], s, t)
        for s in row["source_clause_ids"]
        for t in row["target_clause_ids"]
    }


def evaluate(
    rows: list[dict],
    candidates: list[dict] | None = None,
    gold: list[dict] | None = None,
) -> dict:
    report = {
        "relationships": len(rows),
        "relationship_counts": dict(Counter(r["relation"] for r in rows)),
        "confidence_distribution": dict(Counter(r["confidence"] for r in rows)),
        "manual_review_count": sum(r["review_required"] for r in rows),
        "manual_review_rate": sum(r["review_required"] for r in rows) / len(rows)
        if rows
        else 0,
        "accepted_link_relationship_rate": sum(
            r["relation"] not in {"INSERTED", "DELETED", "UNCERTAIN"} for r in rows
        )
        / len(rows)
        if rows
        else 0,
        "candidate_recall": None,
        "link_precision": None,
        "link_recall": None,
        "link_f1": None,
        "gold_status": "No complete human-reviewed gold scope supplied; accuracy is not measured",
        "evaluation_scope": (
            "Single-family alignment run; no cross-family generalization or "
            "ambiguity evaluation"
        ),
    }
    if gold is not None:
        if not gold or any(
            not r.get("reviewer") or not r.get("scope_complete") for r in gold
        ):
            raise ValueError(
                "Gold needs reviewer identity and scope_complete=true for complete reviewed document-pair scopes"
            )
        pairs = {(r["from_document_id"], r["to_document_id"]) for r in gold}
        predicted = set().union(
            *(
                links(r)
                for r in rows
                if (r["from_document_id"], r["to_document_id"]) in pairs
            )
        )
        truth = set().union(*(links(r) for r in gold))
        tp = len(predicted & truth)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(truth) if truth else 0.0
        candidate_links = {
            (r["from_document_id"], r["to_document_id"], s, t)
            for r in candidates or []
            for s in r["source_clause_ids"]
            for t in r["target_clause_ids"]
        }
        report.update(
            link_precision=precision,
            link_recall=recall,
            link_f1=2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            candidate_recall=len(truth & candidate_links) / len(truth)
            if truth and candidates is not None
            else None,
            gold_status="Reviewed complete scopes",
            gold_links=len(truth),
        )
    return report
