"""Classical ambiguity-detection data and modeling utilities."""

from .clause_features import (
    FEATURE_COLUMNS,
    extract_clause_features,
    extract_dataframe_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "extract_clause_features",
    "extract_dataframe_features",
]
