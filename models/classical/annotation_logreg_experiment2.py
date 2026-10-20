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

from models.classical.revision_features import (
    REVISION_FEATURE_COLUMNS,
    load_revision_features,
    revision_feature_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/annotation/final_annotation_ready.csv"
FeatureMode = Literal[
    "engineered_tfidf",
    "tfidf",
    "embeddings",
    "tfidf_embeddings",
    "engineered_tfidf_revision",
]
DEFAULT_OUTPUTS = {
    "engineered_tfidf": ROOT / "results/ML/v2_engineered_tfidf",
    "tfidf": ROOT / "results/ML/v2_tfidf_only",
    "embeddings": ROOT / "results/ML/v2_embeddings_only",
    "tfidf_embeddings": ROOT / "results/ML/v2_tfidf_embeddings",
    "engineered_tfidf_revision": ROOT / "results/ML/v2_engineered_tfidf_revision",
}
RANDOM_STATE = 42
N_SPLITS = 3
PREDICTION_THRESHOLD = 0.5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Fixed order: 22 base counts followed by four interactions. These are cues,
# not ambiguity decisions; only the three clause-text columns are inspected.
V2_COLUMNS = (
    "target_word_count",
    "target_sentence_count",
    "subjective_term_count",
    "vague_time_count",
    "vague_quantity_count",
    "vague_quality_count",
    "numeric_bound_count",
    "measurement_unit_count",
    "named_threshold_count",
    "formula_count",
    "reference_count_target",
    "reference_count_context",
    "definition_cue_count_target",
    "definition_cue_count_context",
    "condition_marker_count",
    "exception_marker_count",
    "negation_count",
    "explicit_alternative_count",
    "must_count",
    "should_count",
    "may_count",
    "deictic_expression_count",
    "subjective_near_bound_count",
    "subjective_near_reference_count",
    "subjective_with_context_definition",
    "exception_negation_interaction",
)
# Keep exception_negation_interaction available from the extractor and in the
# raw feature artifact, but exclude it from every trained numerical matrix.
TRAINING_FEATURE_COLUMNS = tuple(
    column for column in V2_COLUMNS if column != "exception_negation_interaction"
)
# Training features are nonnegative counts. Apply log1p and the train-fitted
# StandardScaler to this training-only schema.
LOG_COUNT_COLUMNS = TRAINING_FEATURE_COLUMNS
PROXIMITY_WINDOW_TOKENS = 12

FEATURE_LEXICONS = {
    "time": ("recently", "short time", "long period", "periodically", "timely",
             "quickly", "promptly"),
    "quantity": ("excessive", "limited number", "many", "few", "significantly larger"),
    "quality": ("reasonable", "adequate", "appropriate", "suitable", "sufficient",
                "reliable", "feasible", "robust", "safe"),
    "condition": ("if", "when", "whenever", "provided that"),
    "exception": ("unless", "except", "otherwise", "however"),
    "negation": ("not", "never", "neither", "without"),
    "deictic": ("this", "these", "such", "the former", "the latter", "at that time"),
    "definition": ("means", "is defined as", "that is", "as defined in",
                   "as described in", "the following", "listed below", "described above"),
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[_'-][A-Za-z0-9]+)*")


def _phrase_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    # Prefer the longest phrase at a given position; no substring matches.
    alternatives = [r"\s+".join(map(re.escape, phrase.split()))
                    for phrase in sorted(set(phrases), key=lambda x: (-len(x), x))]
    return re.compile(r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])",
                      re.IGNORECASE)


LEXICON_PATTERNS = {name: _phrase_pattern(phrases)
                    for name, phrases in FEATURE_LEXICONS.items()}
SUBJECTIVE_RE = _phrase_pattern(tuple(
    phrase for name in ("time", "quantity", "quality")
    for phrase in FEATURE_LEXICONS[name]
))
REFERENCE_RE = re.compile(
    r"\bsections?\s+(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)(?!\w)"
    r"|\bRFC\s*\d+\b|\[[A-Za-z][A-Za-z0-9_.:-]*\]",
    re.IGNORECASE,
)
UNIT_PATTERN = (
    r"(?:milliseconds?|msecs?|ms|microseconds?|us|nanoseconds?|ns|seconds?|secs?|s|"
    r"minutes?|mins?|hours?|hrs?|bytes?|octets?|bits?|packets?|percent|KiB|MiB|KB|MB)\b"
    r"|%"
)
UNIT_RE = re.compile(
    r"(?<![A-Za-z_])(?:" + UNIT_PATTERN + r")(?!\w)", re.IGNORECASE,
)
NUMBER_PATTERN = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|hundred|thousand)"
)
IDENTIFIER_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
COMPARATOR_PATTERN = (
    r"(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|"
    r"not\s+(?:less|greater|more)\s+than|(?:less|greater|more|fewer)\s+than|"
    r"(?:equal\s+to|equals)|minimum\s+(?:of\s+)?|maximum\s+(?:of\s+)?|[<>]=?|[≤≥])"
)
NUMERIC_BOUND_RE = re.compile(
    r"(?<![\w.])" + COMPARATOR_PATTERN + r"\s*" + NUMBER_PATTERN
    + r"(?:\s*(?:" + UNIT_PATTERN + r"))?(?!\w|\.\d)"
    r"|\bbetween\s+" + NUMBER_PATTERN + r"\s+and\s+" + NUMBER_PATTERN + r"\b"
    r"|\b" + NUMBER_PATTERN + r"\s*(?:times|x|\*)\s*(?:the\s+)?"
    r"(?:" + IDENTIFIER_PATTERN + r")\b"
    r"|\b(?:twice|thrice)\s+(?:the\s+)?" + IDENTIFIER_PATTERN + r"\b",
    re.IGNORECASE,
)
# Explicit threshold names, plus identifiers directly following a comparison.
# A protocol field or bare digit is not automatically a threshold or bound.
NAMED_THRESHOLD_RE = re.compile(
    r"\bk[A-Z][A-Za-z0-9]*\b|\b(?:PTO|RTT)\b"
    r"|\b(?i:(?:[a-z][a-z0-9]*_)+(?:threshold|limit|timeout|granularity|pto|rtt)"
    r"(?:_[a-z0-9]+)*|(?:threshold|limit|timeout|granularity|pto|rtt)"
    r"(?:_[a-z0-9]+)+)\b",
)
COMPARISON_OPERAND_RE = re.compile(
    COMPARATOR_PATTERN + r"\s*(?P<operand>" + IDENTIFIER_PATTERN + r")\b",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?|" + IDENTIFIER_PATTERN + r")"
    r"(?:\s*(?:\*\*|[*/+^×÷]|\s-\s)\s*"
    r"(?:\d+(?:\.\d+)?|" + IDENTIFIER_PATTERN + r"))+(?!\w)"
    r"|\b(?:min|max)\(\s*[^()\n]+\)",
)
ALTERNATIVE_RE = re.compile(
    r"(?i:\beither\b[^.;!?]*?\bor\b|\balternatively\b)"
    r"|\bMAY\b[^.;!?]*?(?i:\bor\b)"
)
MODAL_PATTERNS = {modal: re.compile(r"\b" + modal + r"\b")
                  for modal in ("MUST", "SHOULD", "MAY")}



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


def _clause_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [match.span() for match in pattern.finditer(text)]


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    # Retain character offsets for proximity calculations.
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _threshold_spans(text: str) -> list[tuple[int, int]]:
    spans = _spans(NAMED_THRESHOLD_RE, text)
    for match in COMPARISON_OPERAND_RE.finditer(text):
        operand = match.group("operand")
        # Avoid treating number words and ordinary prose as variable names.
        if "_" in operand or re.fullmatch(r"[A-Z][A-Z0-9]+", operand) or re.fullmatch(
            r"[a-z]+[A-Z][A-Za-z0-9]*", operand
        ):
            span = match.span("operand")
            if not any(start <= span[0] and span[1] <= end for start, end in spans):
                spans.append(span)
    return sorted(set(spans))


def _nearby_count(
    text: str,
    subjective: list[tuple[int, int]],
    cues: list[tuple[int, int]],
) -> int:
    """Count each subjective span once if <=12 word tokens separate it from a cue."""
    tokens = _spans(WORD_RE, text)

    def near(left: tuple[int, int], right: tuple[int, int]) -> bool:
        if left[0] < right[1] and right[0] < left[1]:
            return True
        gap_start, gap_end = (left[1], right[0]) if left[1] <= right[0] else (
            right[1], left[0]
        )
        return sum(start >= gap_start and end <= gap_end
                   for start, end in tokens) <= PROXIMITY_WINDOW_TOKENS

    return sum(any(near(span, cue) for cue in cues) for span in subjective)


def extract_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract the 22 base cues and four interactions without label metadata.

    Context means previous_clause and next_clause, processed separately so
    phrases cannot bridge their boundary. References and definition pointers
    are observable cues; their presence does not establish that they resolve
    ambiguity. Numeric bounds, formulas and sentence boundaries are deterministic
    heuristics, not a full RFC grammar or mathematical parser.
    """
    rows: list[dict[str, int | float]] = []
    for row in frame.to_dict(orient="records"):
        target = _clause_text(row.get("target_clause", ""))
        context = [_clause_text(row.get(name, ""))
                   for name in ("previous_clause", "next_clause")]
        references = _spans(REFERENCE_RE, target)
        implementation_text = _mask_spans(target, references)
        subjective = _spans(SUBJECTIVE_RE, target)
        bounds = _spans(NUMERIC_BOUND_RE, implementation_text)
        thresholds = _threshold_spans(implementation_text)
        # HTTP version identifiers use a slash but are not arithmetic formulas.
        formula_text = _mask_spans(implementation_text, _spans(
            re.compile(r"\bHTTPS?/\d+(?:\.\d+)*\b", re.IGNORECASE),
            implementation_text,
        ))
        formulas = _spans(FORMULA_RE, formula_text)
        # Mask citations and decimal points before the simple sentence split.
        sentence_text = REFERENCE_RE.sub(lambda match: match.group().replace(".", " "), target)
        sentence_text = re.sub(r"(?<=\d)\.(?=\d)", " ", sentence_text)
        sentence_count = sum(bool(part.strip()) for part in re.split(
            r'[.!?]+(?:["\')\]]*)(?:\s+|$)', sentence_text
        ))
        definitions_context = sum(len(_spans(LEXICON_PATTERNS["definition"], text))
                                  for text in context)
        exception_count = len(_spans(LEXICON_PATTERNS["exception"], target))
        negation_count = len(_spans(LEXICON_PATTERNS["negation"], target))
        rows.append({
            "target_word_count": len(WORD_RE.findall(target)),
            "target_sentence_count": sentence_count,
            "subjective_term_count": len(subjective),
            "vague_time_count": len(_spans(LEXICON_PATTERNS["time"], target)),
            "vague_quantity_count": len(_spans(LEXICON_PATTERNS["quantity"], target)),
            "vague_quality_count": len(_spans(LEXICON_PATTERNS["quality"], target)),
            "numeric_bound_count": len(bounds),
            "measurement_unit_count": len(_spans(UNIT_RE, implementation_text)),
            "named_threshold_count": len(thresholds),
            "formula_count": len(formulas),
            "reference_count_target": len(references),
            "reference_count_context": sum(len(_spans(REFERENCE_RE, text))
                                           for text in context),
            "definition_cue_count_target": len(_spans(LEXICON_PATTERNS["definition"], target)),
            "definition_cue_count_context": definitions_context,
            "condition_marker_count": len(_spans(LEXICON_PATTERNS["condition"], target)),
            "exception_marker_count": exception_count,
            "negation_count": negation_count,
            "explicit_alternative_count": len(_spans(ALTERNATIVE_RE, target)),
            "must_count": len(_spans(MODAL_PATTERNS["MUST"], target)),
            "should_count": len(_spans(MODAL_PATTERNS["SHOULD"], target)),
            "may_count": len(_spans(MODAL_PATTERNS["MAY"], target)),
            "deictic_expression_count": len(_spans(LEXICON_PATTERNS["deictic"], target)),
            "subjective_near_bound_count": _nearby_count(
                target, subjective, bounds + thresholds + formulas
            ),
            "subjective_near_reference_count": _nearby_count(target, subjective, references),
            "subjective_with_context_definition": len(subjective) * int(definitions_context > 0),
            "exception_negation_interaction": exception_count * negation_count,
        })
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
    transformed = features.loc[:, TRAINING_FEATURE_COLUMNS].copy()
    transformed.loc[:, LOG_COUNT_COLUMNS] = np.log1p(
        transformed.loc[:, LOG_COUNT_COLUMNS].to_numpy(dtype=np.float64)
    )
    return transformed


def _fit_components(
    feature_mode: FeatureMode,
    train_texts: pd.Series,
    train_features: pd.DataFrame | None,
    train_embeddings: np.ndarray | None,
    train_revision_features: pd.DataFrame | None = None,
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
    if "revision" in feature_mode:
        if train_revision_features is None:
            raise ValueError("Revision features are required for this model")
        scaler = StandardScaler()
        matrix = sparse.csr_matrix(
            scaler.fit_transform(
                train_revision_features.loc[:, REVISION_FEATURE_COLUMNS].to_numpy()
            )
        )
        matrices.append(matrix)
        components["revision_scaler"] = scaler
    if not matrices:
        raise ValueError(f"No feature components configured for {feature_mode!r}")
    return components, sparse.hstack(matrices, format="csr")


def _transform_components(
    feature_mode: FeatureMode,
    components: dict[str, Any],
    texts: pd.Series,
    features: pd.DataFrame | None,
    embeddings: np.ndarray | None,
    revision_features: pd.DataFrame | None = None,
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
    if "revision" in feature_mode:
        if revision_features is None:
            raise ValueError("Revision features are required for this model")
        scaler = components["revision_scaler"]
        matrices.append(
            sparse.csr_matrix(
                scaler.transform(
                    revision_features.loc[:, REVISION_FEATURE_COLUMNS].to_numpy()
                )
            )
        )
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
    use_revision = "revision" in feature_mode
    features = extract_v2_features(frame) if use_engineered else None
    embeddings = _encode_target_embeddings(frame["target_clause"]) if use_embeddings else None
    revision_features = load_revision_features(frame) if use_revision else None
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
        if np.intersect1d(groups[train_idx], groups[valid_idx]).size:
            raise RuntimeError(f"Fold {fold} leaks draft families across partitions")
        train_features = (
            features.iloc[train_idx] if features is not None else None
        )
        valid_features = (
            features.iloc[valid_idx] if features is not None else None
        )
        train_embeddings = embeddings[train_idx] if embeddings is not None else None
        valid_embeddings = embeddings[valid_idx] if embeddings is not None else None
        train_revision_features = (
            revision_features.iloc[train_idx] if revision_features is not None else None
        )
        valid_revision_features = (
            revision_features.iloc[valid_idx] if revision_features is not None else None
        )
        components, train_matrix = _fit_components(
            feature_mode,
            frame["target_clause"].iloc[train_idx],
            train_features,
            train_embeddings,
            train_revision_features,
        )
        valid_matrix = _transform_components(
            feature_mode,
            components,
            frame["target_clause"].iloc[valid_idx],
            valid_features,
            valid_embeddings,
            valid_revision_features,
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
                "numerical_feature_count": len(TRAINING_FEATURE_COLUMNS),
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
    if use_revision:
        model_metrics["revision_features"] = {
            "count": len(REVISION_FEATURE_COLUMNS),
            "columns": list(REVISION_FEATURE_COLUMNS),
            "scaler": "fold-local StandardScaler",
            "history": "H0: t-1 -> t; H1: t-2 -> t-1 via the H0 predecessor",
        }
        feature_notes["revision_leakage_control"] = (
            "Label-free predecessor matches use only numbered drafts at or before "
            "the row revision; labels, candidate source, later drafts, RFC endpoints, "
            "errata, votes, and rationales are excluded."
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
    if revision_features is not None:
        revision_features.to_csv(
            output_dir / "revision_features.csv",
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
        revision_features,
    )
    final_classifier = _new_classifier().fit(final_matrix, y)
    final_components["classifier"] = final_classifier
    final_components["numerical_columns"] = (
        list(TRAINING_FEATURE_COLUMNS) if use_engineered else []
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
    final_components["revision_columns"] = (
        list(REVISION_FEATURE_COLUMNS) if use_revision else []
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
    if use_revision:
        assert revision_features is not None
        revision_scaler = final_components["revision_scaler"]
        revision_matrix = sparse.csr_matrix(
            revision_scaler.transform(
                revision_features.loc[:, REVISION_FEATURE_COLUMNS].to_numpy()
            )
        )
        sparse.save_npz(
            matrices_dir / "revision_scaled.npz",
            revision_matrix,
            compressed=True,
        )
        matrix_shapes["revision_scaled"] = list(revision_matrix.shape)
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
        "feature_columns": list(TRAINING_FEATURE_COLUMNS) if use_engineered else [],
        "log_count_columns": list(LOG_COUNT_COLUMNS) if use_engineered else [],
        "feature_set": feature_mode,
        "matrix_shapes": matrix_shapes,
        "tfidf_feature_count": tfidf_feature_count,
        "embedding_dimensions": embeddings_dimension,
        "revision_feature_columns": (
            list(REVISION_FEATURE_COLUMNS) if use_revision else []
        ),
        "revision_provenance": revision_feature_provenance() if use_revision else None,
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
