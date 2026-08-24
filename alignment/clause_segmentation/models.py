"""Canonical clause-level representation built from segmented RFC documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from alignment.segmentation.models import BlockType, SourceLocation

ClauseLabel = Literal["NORMATIVE", "CONTEXTUAL", "EXCLUDED", "MANUAL_REVIEW"]
SectionPrior = Literal["normative", "contextual", "mixed", "neutral"]


@dataclass(slots=True)
class Clause:
    """One traceable prose sentence, list item, or excluded structural block."""

    clause_id: str
    index: int
    document_id: str
    revision_id: str | None
    block_id: str
    section_id: str
    section_heading: str
    section_path: list[str]
    paragraph_index: int
    sentence_or_item_index: int
    raw_text: str
    block_type: BlockType
    classification: ClauseLabel
    classification_reason: str
    section_prior: SectionPrior
    source_location: SourceLocation
    text_start: int
    text_end: int
    bcp14_keywords: list[str] = field(default_factory=list)
    reference_targets: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    mixed_clause: bool = False
    manual_review: bool = False


@dataclass(slots=True)
class ClauseDocument:
    """Clause-level view of one parsed Internet-Draft revision or RFC."""

    document_id: str
    revision_id: str | None
    source_path: str
    source_sha256: str
    clauses: list[Clause]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, int]:
        counts = Counter(clause.classification for clause in self.clauses)
        return {
            "clauses": len(self.clauses),
            "normative": counts["NORMATIVE"],
            "contextual": counts["CONTEXTUAL"],
            "excluded": counts["EXCLUDED"],
            "manual_review": counts["MANUAL_REVIEW"],
            "mixed": sum(clause.mixed_clause for clause in self.clauses),
            "warnings": len(self.warnings),
        }
