"""Uncalibrated confidence, uncertainty preservation and reproducible review sampling."""

from __future__ import annotations

from collections import defaultdict

from . import ALGORITHM_VERSION
from .candidates import Candidate, joined
from .config import Config
from .io import digest
from .models import Alignment, Document
from .scoring import change_summary


def records(
    a: Document,
    b: Document,
    candidates: list[Candidate],
    selected: list[Candidate],
    config: Config,
    created_at: str,
    model_id: str,
) -> list[Alignment]:
    ac, bc = a.clauses, b.clauses
    by_a, by_b = defaultdict(list), defaultdict(list)
    for edge in candidates:
        for i in edge.source:
            by_a[i].append(edge)
        for j in edge.target:
            by_b[j].append(edge)
    output, covered_a, covered_b = [], set(), set()

    def emit(
        source,
        target,
        relation,
        edge=None,
        reasons=None,
        margin=None,
        moved=False,
        alternatives=None,
    ):
        source_ids, target_ids = (
            [ac[i].clause_id for i in source],
            [bc[j].clause_id for j in target],
        )
        left = joined(ac, source).normalized_text if source else ""
        right = joined(bc, target).normalized_text if target else ""
        scores = (
            dict(edge.scores)
            if edge
            else {"semantic": None, "lexical": None, "combined": None}
        )
        scores["candidate_margin"] = margin
        reasons = list(reasons or [])
        if (
            source
            and target
            and {ac[i].classification for i in source}
            != {bc[j].classification for j in target}
        ):
            reasons.append("CLASS_TRANSITION")
        if any(
            c.classification == "MANUAL_REVIEW"
            for c in [*(ac[i] for i in source), *(bc[j] for j in target)]
        ):
            reasons.append("CLAUSE_CLASS_UNRESOLVED")
        changes = change_summary(left, right) if source and target else {}
        if any(k in changes for k in ("actor", "modal", "negation", "number")):
            reasons.append("PROTECTED_TOKEN_CHANGE")
        if relation in {"SPLIT", "MERGED", "MOVED", "UNCERTAIN"}:
            reasons.append(relation)
        high = (
            edge is not None
            and relation != "UNCERTAIN"
            and edge.combined >= config.high_score
            and (edge.locked or (margin is not None and margin >= config.high_margin))
        )
        confidence = (
            "HIGH"
            if high
            else ("MEDIUM" if edge and relation != "UNCERTAIN" else "LOW")
        )
        identity = digest(
            [a.document_id, b.document_id, source_ids, target_ids, relation]
        )
        if (
            high
            and int(digest([config.seed, identity])[:8], 16) / 0xFFFFFFFF
            < config.high_audit_fraction
        ):
            reasons.append("HIGH_CONFIDENCE_AUDIT")
        if not high and not reasons:
            reasons.append("UNVALIDATED_CONFIDENCE")
        output.append(
            Alignment(
                record_id=identity,
                family_id=a.family_id,
                from_document_id=a.document_id,
                to_document_id=b.document_id,
                from_revision=a.revision,
                to_revision=b.revision,
                from_publication_date=a.publication_date,
                to_publication_date=b.publication_date,
                source_clause_ids=source_ids,
                target_clause_ids=target_ids,
                relation=relation,
                text_changed=left != right,
                movement={
                    "detected": moved,
                    "criterion": "section title correspondence or anchored order inversion",
                },
                scores=scores,
                confidence=confidence,
                review_required=bool(reasons),
                review_reasons=sorted(set(reasons)),
                alternatives=alternatives or [],
                changes=changes,
                provenance={
                    "source_sha256": a.source_sha256,
                    "target_sha256": b.source_sha256,
                    "source_input_sha256": a.input_sha256,
                    "target_input_sha256": b.input_sha256,
                    "config_hash": digest(config),
                    "model_artifact_id": model_id,
                },
                algorithm_version=ALGORITHM_VERSION,
                created_at=created_at,
            )
        )

    anchor_pairs = sorted((e.source[0], e.target[0]) for e in selected if e.locked)
    for edge in selected:
        competitors = {id(e): e for i in edge.source for e in by_a[i] if e is not edge}
        competitors.update(
            {id(e): e for j in edge.target for e in by_b[j] if e is not edge}
        )
        alternatives = sorted(
            competitors.values(), key=lambda e: (-e.combined, e.source, e.target)
        )
        margin = edge.combined - max([e.combined for e in alternatives], default=0.0)
        alt_records = [
            {
                "source_clause_ids": [ac[i].clause_id for i in e.source],
                "target_clause_ids": [bc[j].clause_id for j in e.target],
                "score": e.combined,
            }
            for e in alternatives[:3]
        ]
        x, y = ac[edge.source[0]], bc[edge.target[0]]
        same_title = x.section_heading.casefold() == y.section_heading.casefold()
        # A changed title alone is not a move. Require an established title elsewhere,
        # or crossing reliable anchors rather than absolute index displacement.
        moved_section = not same_title and any(
            c.section_heading.casefold() == x.section_heading.casefold() for c in bc
        )
        inverted = (
            sum(
                (i < edge.source[0] and j > edge.target[0])
                or (i > edge.source[0] and j < edge.target[0])
                for i, j in anchor_pairs
            )
            >= 2
        )
        moved = moved_section or inverted
        relation = (
            "SPLIT"
            if len(edge.target) > 1
            else "MERGED"
            if len(edge.source) > 1
            else "MOVED"
            if moved
            else "UNCHANGED"
            if edge.exact
            else "MODIFIED"
        )
        if not edge.locked and margin < config.uncertain_margin:
            relation = "UNCERTAIN"
        emit(
            edge.source,
            edge.target,
            relation,
            edge,
            margin=margin,
            moved=moved,
            alternatives=alt_records,
        )
        covered_a.update(edge.source)
        covered_b.update(edge.target)
    for side, clauses, covered, alternatives_by_id in [
        (0, ac, covered_a, by_a),
        (1, bc, covered_b, by_b),
    ]:
        for i, c in enumerate(clauses):
            if i in covered:
                continue
            alternatives = sorted(
                alternatives_by_id[i], key=lambda e: (-e.combined, e.source, e.target)
            )[:3]
            uncertain = bool(
                alternatives and alternatives[0].combined >= config.match_floor
            )
            emit(
                (i,) if side == 0 else (),
                (i,) if side == 1 else (),
                "UNCERTAIN" if uncertain else "DELETED" if side == 0 else "INSERTED",
                reasons=[
                    "COMPETING_ASSIGNMENT" if uncertain else "NO_ACCEPTED_CANDIDATE"
                ],
                alternatives=[
                    {
                        "source_clause_ids": [ac[k].clause_id for k in e.source],
                        "target_clause_ids": [bc[k].clause_id for k in e.target],
                        "score": e.combined,
                    }
                    for e in alternatives
                ],
            )
    return sorted(output, key=lambda r: (r.source_clause_ids, r.target_clause_ids))


def stratified_sample(rows: list[Alignment], size: int, seed: int) -> list[Alignment]:
    strata = defaultdict(list)
    for row in rows:
        strata[(row.relation, row.confidence)].append(row)
    for key in strata:
        strata[key].sort(key=lambda r: digest([seed, r.record_id]))
    result = []
    while len(result) < size and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(result) < size:
                result.append(strata[key].pop())
    return result
