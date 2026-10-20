"""Leakage-safe clause revision features for final-annotation experiments.

The predecessor cache is produced by ``run_signal_check.py``.  It matches a
clause only against numbered drafts that existed at the prediction time:
H0 compares t-1 -> t and H1 compares t-2 -> t-1 after following the H0
predecessor.  This module deliberately reads only identity columns from the
private annotation key and only match outputs from the cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_PATH = ROOT / "results/annotation/key_PRIVATE.csv"
DEFAULT_CACHE_DIR = ROOT / "results/revision_features/matcher_cache"
EXPECTED_MATCHER = "section_constrained_lexical_minilm_v3"

REVISION_FEATURE_COLUMNS = tuple(
    f"{tier}_{feature}"
    for tier in ("h0", "h1")
    for feature in (
        "has_predecessor",
        "unchanged",
        "modified",
        "moved",
        "match_change_distance",
        "lexical_change_distance",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_from_document(documents: pd.Series) -> pd.Series:
    return documents.astype(str).str.replace(r"-\d+$", "", regex=True)


def load_revision_features(
    frame: pd.DataFrame,
    *,
    key_path: Path = DEFAULT_KEY_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Return label-free, past-only revision features aligned to ``frame``.

    ``frame`` must contain unique ``item_id`` and ``document`` values.  The
    annotation key is restricted at read time to identity columns, preventing
    candidate-source, model-vote, owner, or label fields from entering the
    feature table.
    """
    required = {"item_id", "document"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Revision features require columns: {sorted(missing)}")
    if frame["item_id"].astype(str).duplicated().any():
        raise ValueError("Revision features require unique item_id values")
    if not key_path.is_file():
        raise FileNotFoundError(f"Annotation identity key not found: {key_path}")
    cache_paths = sorted(cache_dir.glob("*.parquet"))
    if not cache_paths:
        raise FileNotFoundError(f"No revision matcher caches found in: {cache_dir}")

    identity = pd.read_csv(
        key_path,
        usecols=["item_id", "record_id", "family_id"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if identity["item_id"].duplicated().any():
        raise ValueError("Annotation identity key contains duplicate item_id values")

    cache_columns = [
        "record_id",
        "family_id",
        "document_id",
        "H0_has_predecessor",
        "H0_relation_type",
        "H0_similarity",
        "H0_lexical_similarity",
        "H1_has_predecessor",
        "H1_relation_type",
        "H1_similarity",
        "H1_lexical_similarity",
        "matcher",
    ]
    caches = pd.concat(
        [pd.read_parquet(path, columns=cache_columns) for path in cache_paths],
        ignore_index=True,
    )
    if caches["record_id"].duplicated().any():
        raise ValueError("Revision matcher caches contain duplicate record_id values")
    if not caches["matcher"].eq(EXPECTED_MATCHER).all():
        unexpected = sorted(set(caches["matcher"].astype(str)) - {EXPECTED_MATCHER})
        raise ValueError(f"Unexpected revision matcher version(s): {unexpected}")

    rows = frame.loc[:, ["item_id", "document"]].copy()
    rows["item_id"] = rows["item_id"].astype(str)
    rows["family_id"] = _family_from_document(rows["document"])
    joined = rows.merge(
        identity,
        on=["item_id", "family_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        caches,
        on=["record_id", "family_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["record_id"].isna().any():
        missing_items = joined.loc[joined["record_id"].isna(), "item_id"].tolist()
        raise ValueError(f"Revision identity is missing for item_id values: {missing_items[:5]}")
    if joined["matcher"].isna().any():
        missing_items = joined.loc[joined["matcher"].isna(), "item_id"].tolist()
        raise ValueError(f"Revision cache is missing item_id values: {missing_items[:5]}")
    if not joined["document"].eq(joined["document_id"]).all():
        raise ValueError("Revision cache document identity does not match annotation rows")

    output: dict[str, Any] = {}
    valid_relations = {"NONE", "UNCHANGED", "MODIFIED", "MOVED"}
    for source_tier, prefix in (("H0", "h0"), ("H1", "h1")):
        has_predecessor = pd.to_numeric(
            joined[f"{source_tier}_has_predecessor"], errors="raise"
        ).astype(np.float64)
        if not has_predecessor.isin({0.0, 1.0}).all():
            raise ValueError(f"{source_tier} predecessor flag must be binary")
        relations = joined[f"{source_tier}_relation_type"].astype(str)
        unexpected_relations = sorted(set(relations) - valid_relations)
        if unexpected_relations:
            raise ValueError(
                f"Unexpected {source_tier} relation type(s): {unexpected_relations}"
            )
        output[f"{prefix}_has_predecessor"] = has_predecessor
        for relation in ("UNCHANGED", "MODIFIED", "MOVED"):
            output[f"{prefix}_{relation.casefold()}"] = relations.eq(relation).astype(
                np.float64
            )
        for source_name, output_name in (
            ("similarity", "match_change_distance"),
            ("lexical_similarity", "lexical_change_distance"),
        ):
            similarity = pd.to_numeric(
                joined[f"{source_tier}_{source_name}"], errors="coerce"
            )
            distance = (1.0 - similarity).clip(lower=0.0, upper=1.0)
            output[f"{prefix}_{output_name}"] = distance.where(
                has_predecessor.eq(1.0), 0.0
            ).fillna(0.0)

    features = pd.DataFrame(output, columns=REVISION_FEATURE_COLUMNS)
    features.index = frame.index
    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Revision features contain non-finite values")
    return features.astype(np.float64)


def revision_feature_provenance(
    *,
    key_path: Path = DEFAULT_KEY_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Describe the exact past-only inputs used to build revision features."""
    cache_paths = sorted(cache_dir.glob("*.parquet"))
    return {
        "matcher": EXPECTED_MATCHER,
        "history": {
            "h0": "t-1 -> t",
            "h1": "t-2 -> t-1, following the H0 predecessor",
        },
        "leakage_control": (
            "Only item_id/record_id/family_id identity fields and cached matches "
            "from numbered drafts at or before t are read; labels, candidate source, "
            "votes, rationales, later drafts, RFC endpoints, and errata are excluded."
        ),
        "identity_key_sha256": _sha256(key_path),
        "cache_sha256": {path.name: _sha256(path) for path in cache_paths},
        "feature_columns": list(REVISION_FEATURE_COLUMNS),
    }
