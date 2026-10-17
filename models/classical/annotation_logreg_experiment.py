"""Shared features and grouped validation for annotation logistic-regression runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from models.classical.clause_features import (
    BCP14_RE,
    extract_dataframe_features,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/annotation/final_annotation.csv"
FeatureMode = Literal[
    "engineered_tfidf",
    "tfidf",
    "embeddings",
    "tfidf_embeddings",
]
DEFAULT_OUTPUTS = {
    "engineered_tfidf": ROOT / "results/annotation/v2_engineered_tfidf",
    "tfidf": ROOT / "results/annotation/v2_tfidf_only",
    "embeddings": ROOT / "results/annotation/v2_embeddings_only",
    "tfidf_embeddings": ROOT / "results/annotation/v2_tfidf_embeddings",
}
RANDOM_STATE = 42
N_SPLITS = 3
PREDICTION_THRESHOLD = 0.5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

CORE_COLUMNS = (
    "basic_word_count",
    "vague_term_count",
    "anaphora_count",
    "cross_reference_count",
    "conditional_count",
    "exception_count",
    "temporal_count",
    "ordering_count",
    "syntax_mean_dependency_distance",
    "syntax_subordinate_clause_count",
)
V2_COLUMNS = (
    *CORE_COLUMNS,
    "has_strong_modal",
    "has_should_modal",
    "has_may_modal",
    "modal_level_count",
    "has_no_modal",
    "has_numeric_value",
    "numeric_count",
    "hedge_term_count",
    "has_if_without_otherwise",
    "target_context_overlap",
    "pronoun_without_noun_before",
    "bracket_reference_count",
    "section_reference_count",
)
LOG_COUNT_COLUMNS = (
    "basic_word_count",
    "vague_term_count",
    "anaphora_count",
    "cross_reference_count",
    "conditional_count",
    "exception_count",
    "temporal_count",
    "ordering_count",
    "syntax_subordinate_clause_count",
    "modal_level_count",
    "numeric_count",
    "hedge_term_count",
    "bracket_reference_count",
    "section_reference_count",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
NUMERIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|milliseconds?|msecs?|ms|seconds?|secs?|s|"
    r"minutes?|mins?|hours?|hrs?|bytes?|octets?|packets?|times|x))?\b"
    r"|\b(?:twice|thrice)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:times|milliseconds?|msecs?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|"
    r"bytes?|octets?|packets?|percent)\b",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(
    r"\b(?:likely|generally|typically|possible|where|such\s+as|etc|sufficient|"
    r"appropriate|reasonable|excessive|timely)\b",
    re.IGNORECASE,
)
EXPLICIT_ALTERNATIVE_RE = re.compile(r"\b(?:else|otherwise|unless)\b", re.IGNORECASE)
ANAPHORIC_PRONOUN_START_RE = re.compile(
    r"^\s*(?:(?:and|but|then|otherwise)\s+)?"
    r"(?P<pronoun>it|this|that|these|they)\s+"
    r"(?=(?:is|are|was|were|be|been|being|has|have|had|does|do|did|"
    r"will|would|shall|should|can|could|may|might|must)\b)",
    re.IGNORECASE,
)
BRACKET_REFERENCE_RE = re.compile(r"\[[^\]\r\n]+\]")
SECTION_REFERENCE_RE = re.compile(
    r"\bsections?\s+[A-Za-z0-9][A-Za-z0-9.()\-]*"
    r"(?:\s*(?:and|,|-)\s*[A-Za-z0-9][A-Za-z0-9.()\-]*)*",
    re.IGNORECASE,
)
MODAL_LEVELS = {
    "strong": {"must", "must not", "shall", "shall not", "required"},
    "should": {"should", "should not", "recommended", "not recommended"},
    "may": {"may", "optional"},
}


def _new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
        dtype=np.float64,
        token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b",
    )


def _new_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        class_weight="balanced",
        l1_ratio=1.0,
        max_iter=3_000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= PREDICTION_THRESHOLD
    pr_auc = float(average_precision_score(labels, probabilities))
    return {
        "pr_auc": pr_auc,
        "precision_at_0.5": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_at_0.5": float(recall_score(labels, predictions, zero_division=0)),
        "f1_at_0.5": float(f1_score(labels, predictions, zero_division=0)),
        "f2_at_0.5": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
    }


def _encode_target_embeddings(texts: pd.Series) -> np.ndarray:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        device="cpu",
        local_files_only=True,
    )
    encoder.eval()
    embeddings = encoder.encode(
        texts.astype(str).tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64, copy=False)
    if embeddings.ndim != 2 or embeddings.shape != (len(texts), 384):
        raise ValueError(f"Unexpected clause-embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Clause embeddings contain non-finite values")
    return embeddings


def _jaccard(left: str, right: str) -> float:
    left_words = {word.casefold() for word in WORD_RE.findall(left)}
    right_words = {word.casefold() for word in WORD_RE.findall(right)}
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0


def extract_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed, label-free numerical feature schema for v2."""
    base = extract_dataframe_features(
        frame,
        text_column="target_clause",
        previous_column="previous_clause",
        next_column="next_clause",
        section_column="__section_id_unavailable__",
    )
    rows: list[dict[str, int | float]] = []
    for row, values in zip(frame.to_dict(orient="records"), base.to_dict(orient="records")):
        target = str(row["target_clause"] or "").strip()
        previous = str(row.get("previous_clause", "") or "").strip()
        following = str(row.get("next_clause", "") or "").strip()
        modal_terms = {
            match.group(0).casefold()
            for match in BCP14_RE.finditer(target)
        }
        levels = {
            level
            for level, terms in MODAL_LEVELS.items()
            if modal_terms & terms
        }
        numeric_matches = list(NUMERIC_RE.finditer(target))
        explicit_context = f"{target} {following}"
        pronoun_match = ANAPHORIC_PRONOUN_START_RE.search(target)
        rows.append(
            {
                **{name: values[name] for name in CORE_COLUMNS},
                "has_strong_modal": int(bool(levels & {"strong"})),
                "has_should_modal": int(bool(levels & {"should"})),
                "has_may_modal": int(bool(levels & {"may"})),
                "modal_level_count": len(levels),
                "has_no_modal": int(not modal_terms),
                "has_numeric_value": int(bool(numeric_matches)),
                "numeric_count": len(numeric_matches),
                "hedge_term_count": len(HEDGE_RE.findall(target)),
                "has_if_without_otherwise": int(
                    values["conditional_count"] > 0
                    and not EXPLICIT_ALTERNATIVE_RE.search(explicit_context)
                ),
                "target_context_overlap": max(
                    _jaccard(target, previous),
                    _jaccard(target, following),
                ),
                "pronoun_without_noun_before": int(pronoun_match is not None),
                "bracket_reference_count": len(BRACKET_REFERENCE_RE.findall(target)),
                "section_reference_count": len(SECTION_REFERENCE_RE.findall(target)),
            }
        )
    return pd.DataFrame(rows, columns=V2_COLUMNS, index=frame.index).astype(np.float64)


def build_validation_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group all revisions of the same draft family together."""
    family_ids = (
        frame["document"]
        .str.replace(r"-\d+$", "", regex=True)
        .to_numpy()
    )

    _, groups = np.unique(family_ids, return_inverse=True)

    return groups.astype(np.int64)


def _log_counts(features: pd.DataFrame) -> pd.DataFrame:
    transformed = features.copy()
    transformed.loc[:, LOG_COUNT_COLUMNS] = np.log1p(
        transformed.loc[:, LOG_COUNT_COLUMNS].to_numpy(dtype=np.float64)
    )
    return transformed


def _fit_components(
    feature_mode: FeatureMode,
    train_texts: pd.Series,
    train_features: pd.DataFrame | None,
    train_embeddings: np.ndarray | None,
) -> tuple[dict[str, Any], sparse.csr_matrix]:
    components: dict[str, Any] = {}
    matrices: list[sparse.csr_matrix] = []
    if "tfidf" in feature_mode:
        vectorizer = _new_vectorizer()
        matrices.append(
            vectorizer.fit_transform(train_texts.astype(str).tolist()).tocsr()
        )
        components["vectorizer"] = vectorizer
    if "engineered" in feature_mode:
        if train_features is None:
            raise ValueError("Engineered features are required for this model")
        scaler = StandardScaler()
        matrix = sparse.csr_matrix(
            scaler.fit_transform(_log_counts(train_features).to_numpy())
        )
        matrices.append(matrix)
        components["numerical_scaler"] = scaler
    if "embeddings" in feature_mode:
        if train_embeddings is None:
            raise ValueError("Clause embeddings are required for this model")
        scaler = StandardScaler()
        matrix = sparse.csr_matrix(scaler.fit_transform(train_embeddings))
        matrices.append(matrix)
        components["embedding_scaler"] = scaler
    if not matrices:
        raise ValueError(f"No feature components configured for {feature_mode!r}")
    return components, sparse.hstack(matrices, format="csr")


def _transform_components(
    feature_mode: FeatureMode,
    components: dict[str, Any],
    texts: pd.Series,
    features: pd.DataFrame | None,
    embeddings: np.ndarray | None,
) -> sparse.csr_matrix:
    matrices: list[sparse.csr_matrix] = []
    if "tfidf" in feature_mode:
        vectorizer: TfidfVectorizer = components["vectorizer"]
        matrices.append(vectorizer.transform(texts.astype(str).tolist()).tocsr())
    if "engineered" in feature_mode:
        if features is None:
            raise ValueError("Engineered features are required for this model")
        scaler: StandardScaler = components["numerical_scaler"]
        matrices.append(
            sparse.csr_matrix(scaler.transform(_log_counts(features).to_numpy()))
        )
    if "embeddings" in feature_mode:
        if embeddings is None:
            raise ValueError("Clause embeddings are required for this model")
        scaler = components["embedding_scaler"]
        matrices.append(sparse.csr_matrix(scaler.transform(embeddings)))
    if not matrices:
        raise ValueError(f"No feature components configured for {feature_mode!r}")
    return sparse.hstack(matrices, format="csr")


def train_and_validate(
    input_path: Path,
    output_dir: Path,
    *,
    feature_mode: FeatureMode,
) -> dict[str, Any]:
    if feature_mode not in DEFAULT_OUTPUTS:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {
        "item_id",
        "document",
        "target_clause",
        "previous_clause",
        "next_clause",
        "final_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input annotation file contains no rows")
    if frame["item_id"].duplicated().any():
        raise ValueError("Input contains duplicate item_id values")
    if frame["document"].str.strip().eq("").any():
        raise ValueError("Every row must have a document value for grouped validation")
    if frame["target_clause"].str.strip().eq("").any():
        raise ValueError("Every row must have a target_clause")
    labels = frame["final_label"].str.strip().str.casefold()
    if not labels.isin({"yes", "no"}).all():
        raise ValueError("final_label must contain only yes/no")
    y = labels.eq("yes").astype(np.int8).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("Training requires both yes and no labels")

    use_engineered = "engineered" in feature_mode
    use_embeddings = "embeddings" in feature_mode
    features = extract_v2_features(frame) if use_engineered else None
    embeddings = _encode_target_embeddings(frame["target_clause"]) if use_embeddings else None
    groups = build_validation_groups(frame)
    group_count = len(np.unique(groups))
    positive_group_count = len(np.unique(groups[y == 1]))
    if group_count < N_SPLITS or positive_group_count < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold grouped validation requires at least "
            f"{N_SPLITS} total and positive groups; found {group_count} and "
            f"{positive_group_count}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_scores = np.full(len(frame), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, int | float]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y, groups), start=1
    ):
        if np.unique(y[train_idx]).size != 2 or np.unique(y[valid_idx]).size != 2:
            raise ValueError(f"Fold {fold} does not contain both label classes")
        train_features = (
            features.iloc[train_idx] if features is not None else None
        )
        valid_features = (
            features.iloc[valid_idx] if features is not None else None
        )
        train_embeddings = embeddings[train_idx] if embeddings is not None else None
        valid_embeddings = embeddings[valid_idx] if embeddings is not None else None
        components, train_matrix = _fit_components(
            feature_mode,
            frame["target_clause"].iloc[train_idx],
            train_features,
            train_embeddings,
        )
        valid_matrix = _transform_components(
            feature_mode,
            components,
            frame["target_clause"].iloc[valid_idx],
            valid_features,
            valid_embeddings,
        )
        classifier = _new_classifier().fit(train_matrix, y[train_idx])
        probabilities = classifier.predict_proba(valid_matrix)[:, 1]
        oof_scores[valid_idx] = probabilities
        validation_metrics = _binary_metrics(y[valid_idx], probabilities)
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "validation_rows": len(valid_idx),
                "train_positives": int(y[train_idx].sum()),
                "validation_positives": int(y[valid_idx].sum()),
                "validation_groups": len(np.unique(groups[valid_idx])),
                "selected_features": int(np.count_nonzero(classifier.coef_)),
                "average_precision": float(
                    average_precision_score(y[valid_idx], probabilities)
                ),
                "roc_auc": float(roc_auc_score(y[valid_idx], probabilities)),
                **validation_metrics,
            }
        )
    if np.isnan(oof_scores).any():
        raise RuntimeError("Cross-validation did not score every row")

    fold_frame = pd.DataFrame(fold_metrics)
    pooled_oof_metrics = _binary_metrics(y, oof_scores)
    model_metrics: dict[str, Any] = {
        "type": "LogisticRegression",
        "penalty": "L1",
        "class_weight": "balanced",
        "C": 0.5,
        "feature_selection": "L1 regularization fit inside each training fold",
        "feature_set": feature_mode,
    }
    feature_notes: dict[str, str] = {}
    if use_engineered:
        model_metrics.update(
            {
                "numerical_feature_count": len(V2_COLUMNS),
                "count_transform": (
                    "log1p, fit-free and applied before fold-local StandardScaler"
                ),
            }
        )
        feature_notes.update(
            {
                "syntax_features": (
                    "Syntax parser unavailable; syntax features are zero."
                ),
                "pronoun_without_noun_before": (
                    "Conservative start-of-clause subject-pronoun heuristic."
                ),
                "section_reference_count": (
                    "Counts explicit Section/Sections references; section depth omitted."
                ),
                "target_context_overlap": (
                    "Maximum word-set Jaccard overlap with previous or next clause."
                ),
            }
        )
    if "tfidf" in feature_mode:
        model_metrics["tfidf"] = {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 20_000,
            "sublinear_tf": True,
        }
    if use_embeddings:
        assert embeddings is not None
        model_metrics["clause_embeddings"] = {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "dimensions": int(embeddings.shape[1]),
            "normalized": True,
            "scaler": "fold-local StandardScaler",
        }
        feature_notes["embeddings"] = (
            "Frozen sentence encoder run locally without downloading model files."
        )
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "positive_rows": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "draft_families": int(frame["document"].str.replace(r"-\d+$", "", regex=True).nunique()),
        "validation_group_count": group_count,
        "positive_validation_group_count": positive_group_count,
        "validation": {
            "strategy": "3-fold StratifiedGroupKFold; draft revisions grouped by family",
            "random_state": RANDOM_STATE,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "mean_fold_average_precision": float(fold_frame["average_precision"].mean()),
            "std_fold_average_precision": float(fold_frame["average_precision"].std(ddof=0)),
            "pooled_oof_average_precision": float(average_precision_score(y, oof_scores)),
            "pooled_oof_roc_auc": float(roc_auc_score(y, oof_scores)),
            "pooled_oof_metrics": {
                **pooled_oof_metrics,
                "roc_auc": float(roc_auc_score(y, oof_scores)),
            },
            "folds": fold_metrics,
        },
        "model": model_metrics,
        "feature_notes": feature_notes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(exist_ok=True)
    if features is not None:
        features.to_csv(
            output_dir / "numerical_features.csv",
            index=False,
            lineterminator="\n",
        )
    frame[["item_id", "document", "final_label"]].assign(
        validation_group=groups
    ).to_csv(output_dir / "rows_and_labels.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "document": frame["document"],
            "final_label": frame["final_label"],
            "validation_group": groups,
            "oof_probability_yes": oof_scores,
        }
    ).to_csv(output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False, lineterminator="\n")

    final_components, final_matrix = _fit_components(
        feature_mode,
        frame["target_clause"],
        features,
        embeddings,
    )
    final_classifier = _new_classifier().fit(final_matrix, y)
    final_components["classifier"] = final_classifier
    final_components["numerical_columns"] = (
        list(V2_COLUMNS) if use_engineered else []
    )
    final_components["log_count_columns"] = (
        list(LOG_COUNT_COLUMNS) if use_engineered else []
    )
    final_components["embedding_model"] = (
        EMBEDDING_MODEL if use_embeddings else None
    )
    final_components["embedding_revision"] = (
        EMBEDDING_REVISION if use_embeddings else None
    )
    final_components["feature_mode"] = feature_mode
    sparse.save_npz(matrices_dir / "combined.npz", final_matrix, compressed=True)
    matrix_shapes = {"combined": list(final_matrix.shape)}
    if "tfidf" in feature_mode:
        vectorizer = final_components["vectorizer"]
        tfidf_matrix = vectorizer.transform(frame["target_clause"].tolist()).tocsr()
        sparse.save_npz(matrices_dir / "tfidf.npz", tfidf_matrix, compressed=True)
        matrix_shapes["tfidf"] = list(tfidf_matrix.shape)
    if use_engineered:
        assert features is not None
        numerical_scaler = final_components["numerical_scaler"]
        numerical_matrix = sparse.csr_matrix(
            numerical_scaler.transform(_log_counts(features).to_numpy())
        )
        sparse.save_npz(
            matrices_dir / "numerical_scaled.npz",
            numerical_matrix,
            compressed=True,
        )
        matrix_shapes["numerical_scaled"] = list(numerical_matrix.shape)
    if use_embeddings:
        assert embeddings is not None
        embedding_scaler = final_components["embedding_scaler"]
        embedding_matrix = sparse.csr_matrix(
            embedding_scaler.transform(embeddings)
        )
        sparse.save_npz(
            matrices_dir / "embeddings_scaled.npz",
            embedding_matrix,
            compressed=True,
        )
        matrix_shapes["embeddings_scaled"] = list(embedding_matrix.shape)
    joblib.dump(final_components, output_dir / "final_model.joblib")
    tfidf_feature_count = (
        len(final_components["vectorizer"].get_feature_names_out())
        if "tfidf" in feature_mode
        else 0
    )
    embeddings_dimension = embeddings.shape[1] if embeddings is not None else 0
    manifest = {
        "source_path": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "feature_columns": list(V2_COLUMNS) if use_engineered else [],
        "log_count_columns": list(LOG_COUNT_COLUMNS) if use_engineered else [],
        "feature_set": feature_mode,
        "matrix_shapes": matrix_shapes,
        "tfidf_feature_count": tfidf_feature_count,
        "embedding_dimensions": embeddings_dimension,
        "validation_groups": group_count,
        "metrics_file": "metrics.json",
        "validation_note": (
            "Evaluation is out-of-fold; final matrices and final_model.joblib "
            "are fitted on all annotation rows."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def run_cli(feature_mode: FeatureMode) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train and validate annotation logistic regression ({feature_mode})."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS[feature_mode])
    args = parser.parse_args()
    train_and_validate(args.input, args.output_dir, feature_mode=feature_mode)
