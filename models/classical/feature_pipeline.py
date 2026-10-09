"""Leakage-safe feature preparation for the clause-level ambiguity dataset.

This module does not train a classifier.  It produces three aligned matrix
families for later experiments: TF-IDF only, engineered features only, and a
combined sparse matrix.  All learned preprocessing is fitted on the existing
training split and reused unchanged for validation and test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from corpus.annotate_with_ollama import ContextResolver
from models.classical.clause_features import (
    FEATURE_COLUMNS,
    extract_dataframe_features,
    safe_text,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "corpus/processed/llm_annotations/v1/training_dataset.csv"
DEFAULT_ALIGNMENT_ROOT = ROOT / "corpus/processed/alignments"
DEFAULT_OUTPUT_DIR = ROOT / "corpus/processed/features/v1"

REQUIRED_COLUMNS = frozenset(
    {
        "record_id",
        "family_id",
        "document_id",
        "clause_ids",
        "clause_text",
        "section_ids",
        "ambiguity_label",
        "label",
        "training_weight",
        "split",
    }
)
VALID_SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _write_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    sparse.save_npz(temporary, matrix.tocsr(), compressed=True)
    temporary.replace(path)


def _write_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def load_binary_dataset(path: Path) -> tuple[pd.DataFrame, int]:
    """Read a labeled CSV and return a validated binary copy.

    Uncertain rows are filtered from the returned copy only.  The source file
    is never modified, and existing split and sample-weight columns are kept.
    """
    frame = pd.read_csv(
        path,
        dtype={
            "record_id": "string",
            "family_id": "string",
            "document_id": "string",
            "revision": "string",
            "clause_ids": "string",
            "clause_text": "string",
            "section_ids": "string",
            "lineage_ids": "string",
            "ambiguity_label": "string",
            "llm_label_tier": "string",
            "llm_ambiguity_subtype": "string",
            "split": "string",
        },
        keep_default_na=True,
    )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Input dataset is missing required columns: {sorted(missing)}"
        )
    if frame["record_id"].duplicated().any():
        duplicates = frame.loc[frame["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id values: {duplicates[:5]}")

    labels = frame["ambiguity_label"].fillna("").str.casefold()
    binary_mask = labels.isin({"yes", "no"})
    excluded = int((~binary_mask).sum())
    binary = frame.loc[binary_mask].copy()
    binary["ambiguity_label"] = labels.loc[binary_mask]
    expected = binary["ambiguity_label"].map({"no": 0, "yes": 1}).astype(int)
    supplied = pd.to_numeric(binary["label"], errors="raise").astype(int)
    if not supplied.equals(expected):
        raise ValueError("Numeric label disagrees with ambiguity_label")
    binary["label"] = supplied
    binary["training_weight"] = pd.to_numeric(
        binary["training_weight"], errors="raise"
    ).astype(float)
    if not np.isfinite(binary["training_weight"]).all():
        raise ValueError("training_weight contains a non-finite value")

    split_values = set(binary["split"].dropna().astype(str))
    invalid_splits = split_values - set(VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unexpected split values: {sorted(invalid_splits)}")
    if "train" not in split_values:
        raise ValueError("The existing dataset does not contain a training split")
    if binary["split"].isna().any():
        raise ValueError("Binary rows must retain an existing split assignment")
    family_split_counts = binary.groupby("family_id", dropna=False)["split"].nunique()
    leaking_families = (
        family_split_counts[family_split_counts > 1].index.astype(str).tolist()
    )
    if leaking_families:
        raise ValueError(
            f"Document families span multiple existing splits: {leaking_families[:5]}"
        )
    return binary.reset_index(drop=True), excluded


def add_frozen_context(
    frame: pd.DataFrame, alignment_root: Path
) -> tuple[pd.DataFrame, int]:
    """Attach same-version neighboring text from frozen alignment inputs."""
    rows = [
        {key: safe_text(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    resolver = ContextResolver(alignment_root)
    resolver.prepare(rows)
    contexts = [resolver.resolve(row) for row in rows]
    enriched = frame.copy()
    enriched["previous_clause"] = [
        context.get("previous_clause", "") for context in contexts
    ]
    enriched["next_clause"] = [context.get("next_clause", "") for context in contexts]
    enriched["section_heading"] = [
        context.get("section_heading", "") for context in contexts
    ]
    return enriched, sum(bool(context) for context in contexts)


@dataclass
class FeaturePreprocessor:
    """Train-only fitted TF-IDF and numerical scaling components."""

    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    fitted: bool = False

    @classmethod
    def create(cls, max_tfidf_features: int, min_df: int) -> FeaturePreprocessor:
        return cls(
            vectorizer=TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=max_tfidf_features,
                sublinear_tf=True,
                dtype=np.float64,
                token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b",
            ),
            scaler=StandardScaler(),
        )

    def fit(self, texts: pd.Series, engineered: pd.DataFrame) -> FeaturePreprocessor:
        if list(engineered.columns) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Engineered feature columns do not match the stable schema"
            )
        clean_texts = texts.fillna("").astype(str).tolist()
        self.vectorizer.fit(clean_texts)
        self.scaler.fit(engineered.to_numpy(dtype=np.float64))
        self.fitted = True
        return self

    def transform(
        self, texts: pd.Series, engineered: pd.DataFrame
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if not self.fitted:
            raise RuntimeError(
                "FeaturePreprocessor must be fitted on training data first"
            )
        if list(engineered.columns) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Engineered feature columns do not match the stable schema"
            )
        tfidf = self.vectorizer.transform(texts.fillna("").astype(str).tolist()).tocsr()
        scaled = self.scaler.transform(engineered.to_numpy(dtype=np.float64))
        engineered_matrix = sparse.csr_matrix(scaled)
        combined = sparse.hstack((tfidf, engineered_matrix), format="csr")
        return tfidf, engineered_matrix, combined

    def tfidf_feature_names(self) -> list[str]:
        if not self.fitted:
            raise RuntimeError("FeaturePreprocessor has not been fitted")
        return [f"tfidf__{name}" for name in self.vectorizer.get_feature_names_out()]


def load_spacy_pipeline(mode: str, model_name: str) -> tuple[Any | None, str]:
    """Load the configured local spaCy model without downloading anything."""
    if mode == "disabled":
        return None, "disabled"
    try:
        import spacy

        return spacy.load(model_name), f"enabled:{model_name}"
    except (ImportError, OSError) as exc:
        if mode == "required":
            raise RuntimeError(
                f"spaCy syntax features requested but {model_name!r} is unavailable"
            ) from exc
        print(
            f"Warning: syntax features disabled because {model_name!r} is unavailable.",
            file=sys.stderr,
        )
        return None, f"unavailable:{model_name}"


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in VALID_SPLITS:
        subset = frame.loc[frame["split"] == split]
        result[split] = {
            "rows": len(subset),
            "labels": {
                str(label): int(count)
                for label, count in sorted(Counter(subset["label"]).items())
            },
            "families": sorted(set(subset["family_id"].astype(str))),
        }
    return result


def build_feature_artifacts(
    *,
    input_path: Path,
    alignment_root: Path,
    output_dir: Path,
    max_tfidf_features: int = 20_000,
    min_df: int = 2,
    syntax_mode: str = "auto",
    spacy_model: str = "en_core_web_sm",
) -> dict[str, Any]:
    """Create reproducible feature artifacts without modifying source data."""
    input_path = input_path.resolve()
    alignment_root = alignment_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new versioned path."
        )
    if max_tfidf_features <= 0:
        raise ValueError("max_tfidf_features must be positive")
    if min_df <= 0:
        raise ValueError("min_df must be positive")

    source_hash_before = sha256_file(input_path)
    frame, excluded_uncertain = load_binary_dataset(input_path)
    enriched, context_resolved = add_frozen_context(frame, alignment_root)
    nlp, syntax_status = load_spacy_pipeline(syntax_mode, spacy_model)
    engineered = extract_dataframe_features(enriched, nlp=nlp)
    if engineered.isna().any().any():
        raise ValueError("Engineered features contain missing values")

    train_mask = enriched["split"] == "train"
    preprocessor = FeaturePreprocessor.create(max_tfidf_features, min_df)
    preprocessor.fit(
        enriched.loc[train_mask, "clause_text"], engineered.loc[train_mask]
    )

    output_dir.mkdir(parents=True)
    audit_frame = pd.concat(
        [
            enriched[["record_id", "family_id", "split"]].reset_index(drop=True),
            engineered.reset_index(drop=True),
        ],
        axis=1,
    )
    _write_csv(output_dir / "audit/engineered_features.csv", audit_frame)

    matrix_shapes: dict[str, dict[str, list[int]]] = {}
    for split in VALID_SPLITS:
        mask = enriched["split"] == split
        split_frame = enriched.loc[mask]
        split_features = engineered.loc[mask]
        tfidf, numerical, combined = preprocessor.transform(
            split_frame["clause_text"], split_features
        )
        matrices = {
            "tfidf": tfidf,
            "engineered": numerical,
            "combined": combined,
        }
        matrix_shapes[split] = {}
        for name, matrix in matrices.items():
            _write_sparse(output_dir / f"matrices/{split}_{name}.npz", matrix)
            matrix_shapes[split][name] = list(matrix.shape)

        labels = split_frame[["record_id", "label"]].reset_index(drop=True)
        weights = split_frame[["record_id", "training_weight"]].reset_index(drop=True)
        rows = split_frame[
            ["record_id", "family_id", "document_id", "clause_ids", "split"]
        ].reset_index(drop=True)
        _write_csv(output_dir / f"targets/{split}_labels.csv", labels)
        _write_csv(output_dir / f"weights/{split}_sample_weights.csv", weights)
        _write_csv(output_dir / f"rows/{split}_rows.csv", rows)

    tfidf_names = preprocessor.tfidf_feature_names()
    engineered_names = [f"engineered__{name}" for name in FEATURE_COLUMNS]
    _write_json(
        output_dir / "feature_names.json",
        {
            "tfidf": tfidf_names,
            "engineered": engineered_names,
            "combined": [*tfidf_names, *engineered_names],
        },
    )
    _write_joblib(
        output_dir / "preprocessing/tfidf_vectorizer.joblib", preprocessor.vectorizer
    )
    _write_joblib(
        output_dir / "preprocessing/engineered_scaler.joblib", preprocessor.scaler
    )

    source_hash_after = sha256_file(input_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source dataset changed while building feature artifacts")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(input_path),
        "source_sha256": source_hash_before,
        "source_rows": len(pd.read_csv(input_path, usecols=["record_id"])),
        "binary_rows": len(enriched),
        "excluded_uncertain_rows": excluded_uncertain,
        "source_preserved": True,
        "alignment_root": str(alignment_root),
        "context_resolved_rows": context_resolved,
        "syntax_status": syntax_status,
        "split_summary": _split_summary(enriched),
        "matrix_shapes": matrix_shapes,
        "engineered_feature_count": len(FEATURE_COLUMNS),
        "tfidf_feature_count": len(tfidf_names),
        "combined_feature_count": len(tfidf_names) + len(FEATURE_COLUMNS),
        "tfidf_config": {
            "ngram_range": [1, 2],
            "min_df": min_df,
            "max_features": max_tfidf_features,
            "sublinear_tf": True,
            "fit_split": "train",
        },
        "numerical_scaler": {
            "type": "StandardScaler",
            "fit_split": "train",
        },
        "sample_weights_location": "weights/<split>_sample_weights.csv",
        "labels_location": "targets/<split>_labels.csv",
        "metadata_location": "rows/<split>_rows.csv",
        "leakage_controls": [
            "only clause text and same-version context feed engineered features",
            "labels, rationales, tiers, weights, IDs, and split names are excluded from matrices",
            "TF-IDF vocabulary and IDF are fitted on train only",
            "the engineered-feature scaler is fitted on train only",
            "existing family-level split assignments are preserved",
            "sample weights are exported separately from feature matrices",
        ],
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe engineered and TF-IDF clause features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tfidf-features", type=int, default=20_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument(
        "--syntax",
        choices=("auto", "disabled", "required"),
        default="auto",
        help="Use a local spaCy parser, disable syntax, or require the model.",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_feature_artifacts(
        input_path=args.input,
        alignment_root=args.alignment_root,
        output_dir=args.output_dir,
        max_tfidf_features=args.max_tfidf_features,
        min_df=args.min_df,
        syntax_mode=args.syntax,
        spacy_model=args.spacy_model,
    )
    print(f"Feature artifacts: {args.output_dir.resolve()}")
    print(f"Binary rows: {manifest['binary_rows']}")
    print(f"Engineered features: {manifest['engineered_feature_count']}")
    print(f"TF-IDF features: {manifest['tfidf_feature_count']}")
    print(f"Syntax: {manifest['syntax_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
