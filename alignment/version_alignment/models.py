"""Versioned boundary contracts. Existing segmentation extras are preserved."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["NORMATIVE", "CONTEXTUAL", "EXCLUDED", "MANUAL_REVIEW"]
Relation = Literal[
    "UNCHANGED",
    "MODIFIED",
    "INSERTED",
    "DELETED",
    "SPLIT",
    "MERGED",
    "MOVED",
    "UNCERTAIN",
]


class Clause(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    clause_id: str
    document_id: str
    revision_id: str | None
    index: int = Field(ge=1)
    block_id: str
    block_type: str
    section_id: str
    section_heading: str
    section_path: list[str]
    raw_text: str
    classification: Label
    source_location: dict[str, Any]
    text_start: int = Field(ge=0)
    text_end: int = Field(ge=0)
    reference_targets: list[str] = Field(default_factory=list)
    bcp14_keywords: list[str] = Field(default_factory=list)
    normalized_text: str = ""
    normalization_steps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_span(self) -> Clause:
        if self.text_end < self.text_start or self.text_end - self.text_start != len(
            self.raw_text
        ):
            raise ValueError(f"Invalid block-relative span: {self.clause_id}")
        if not self.clause_id or not self.raw_text.strip():
            raise ValueError("Empty clause ID/text is not an alignable clause")
        return self


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family_id: str
    document_id: str
    revision: str | None
    publication_date: date
    available_at: date
    date_precision: Literal["day", "month"] = "day"
    source_path: str
    source_sha256: str
    input_sha256: str
    input_path: str
    prepared: bool = False
    clauses: list[Clause]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_clauses(self) -> Document:
        ids = [c.clause_id for c in self.clauses]
        positions = [c.index for c in self.clauses]
        if len(ids) != len(set(ids)) or len(positions) != len(set(positions)):
            raise ValueError(f"Duplicate clause ID or position in {self.document_id}")
        if positions != sorted(positions):
            raise ValueError(f"Incorrect clause order in {self.document_id}")
        if any(
            c.document_id != self.document_id or c.revision_id != self.revision
            for c in self.clauses
        ):
            raise ValueError(f"Clause/document identity mismatch in {self.document_id}")
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    role: Literal["MODEL_FEATURE", "LABEL_SUPPORT", "ANNOTATION_METADATA"]
    available_at: date | None
    document_id: str | None = None
    dependencies: list[Evidence] = Field(default_factory=list)


class Alignment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"
    record_id: str
    family_id: str
    from_document_id: str
    to_document_id: str
    from_revision: str | None
    to_revision: str | None
    from_publication_date: date
    to_publication_date: date
    source_clause_ids: list[str]
    target_clause_ids: list[str]
    relation: Relation
    text_changed: bool
    movement: dict[str, Any]
    scores: dict[str, float | None]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_method: str = "uncalibrated_rule_bands_v1"
    calibrated_probability: float | None = None
    review_required: bool
    review_status: Literal["NOT_REVIEWED", "ACCEPTED", "CORRECTED", "REJECTED"] = (
        "NOT_REVIEWED"
    )
    review_reasons: list[str]
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    lineage_ids: list[str] = Field(default_factory=list)
    changes: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any]
    algorithm_version: str
    created_at: str

    @model_validator(mode="after")
    def cardinality(self) -> Alignment:
        a, b = len(self.source_clause_ids), len(self.target_clause_ids)
        valid = {
            "UNCHANGED": a == b == 1,
            "MODIFIED": a == b == 1,
            "MOVED": a == b == 1,
            "INSERTED": a == 0 and b == 1,
            "DELETED": a == 1 and b == 0,
            "SPLIT": a == 1 and b >= 2,
            "MERGED": a >= 2 and b == 1,
            "UNCERTAIN": a + b > 0,
        }
        if not valid[self.relation]:
            raise ValueError(f"Invalid {self.relation} cardinality: {a}:{b}")
        if a != len(set(self.source_clause_ids)) or b != len(
            set(self.target_clause_ids)
        ):
            raise ValueError("Repeated clause in relationship")
        return self
