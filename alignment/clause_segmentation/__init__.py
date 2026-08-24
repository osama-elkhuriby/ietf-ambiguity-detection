"""Clause refinement and explainable normative/contextual classification."""

from .classifier import ClassificationDecision, classify_clause, extract_bcp14_keywords
from .models import Clause, ClauseDocument, ClauseLabel, SectionPrior
from .segmenter import segment_document, split_sentences

__all__ = [
    "ClassificationDecision",
    "Clause",
    "ClauseDocument",
    "ClauseLabel",
    "SectionPrior",
    "classify_clause",
    "extract_bcp14_keywords",
    "segment_document",
    "split_sentences",
]
