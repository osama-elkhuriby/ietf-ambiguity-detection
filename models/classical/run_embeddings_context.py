from __future__ import annotations

import hashlib
import json
import os
import sys
import datetime
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "corpus/processed/features/v3"
REVIEW_ROOT = ROOT / "corpus/processed/ground_truth"
ALIGNMENT_ROOT = ROOT / "corpus/processed/alignments"
OUTPUT_ROOT = ROOT / "results/embeddings_context"
CACHE_ROOT = OUTPUT_ROOT / "cache"
RANDOM_SEED = 20261012
BOOTSTRAP_REPLICATES = 1000
MIN_HELDOUT_FAMILY_ROWS = 30
SMALL_FAMILIES = {"draft-ietf-6man-rfc2460bis", "draft-ietf-httpbis-cache-header"}
C_GRID = (0.01, 0.1, 1.0, 10.0)
ENCODERS = {
    "MiniLM": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "BGE-small": (
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    ),
}


def digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, list[str], np.ndarray, list[str]]:
    parts = []
    for split in ("train", "validation", "test"):
        rows = pd.read_csv(
            FEATURE_ROOT / "rows" / f"{split}_rows.csv",
            dtype={
                "record_id": str,
                "family_id": str,
                "document_id": str,
                "clause_ids": str,
            },
        )
        labels = pd.read_csv(
            FEATURE_ROOT / "targets" / f"{split}_labels.csv",
            dtype={"record_id": str},
        )
        merged = rows.merge(
            labels[["record_id", "label"]],
            on="record_id",
            validate="one_to_one",
        )
        merged["split"] = split
        parts.append(merged)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != 2546 or frame["record_id"].duplicated().any():
        raise ValueError("Unexpected v3 row count or duplicate record IDs")
    if frame["label"].astype(int).sum() != 455:
        raise ValueError("Unexpected full-cohort positive count")

    target_by_id: dict[str, str] = {}
    for path in sorted(REVIEW_ROOT.glob("*/review/all_candidates_v1/reviewer_a.csv")):
        selected = pd.read_csv(
            path,
            usecols=["record_id", "clause_text", "section_ids"],
            dtype=str,
            keep_default_na=False,
        )
        for row in selected.itertuples(index=False):
            if row.record_id in target_by_id:
                raise ValueError(f"Duplicate target text for {row.record_id}")
            target_by_id[row.record_id] = row.clause_text
    missing_targets = set(frame["record_id"]) - target_by_id.keys()
    if missing_targets:
        raise ValueError(f"Target clause text missing for {len(missing_targets)} rows")
    frame["clause_text"] = frame["record_id"].map(target_by_id)
    frame["section_ids"] = frame["record_id"].map(
        {
            row.record_id: row.section_ids
            for path in sorted(REVIEW_ROOT.glob("*/review/all_candidates_v1/reviewer_a.csv"))
            for row in pd.read_csv(
                path,
                usecols=["record_id", "section_ids"],
                dtype=str,
                keep_default_na=False,
            ).itertuples(index=False)
        }
    )

    engineered = pd.read_csv(
        FEATURE_ROOT / "audit" / "engineered_features.csv",
        dtype={"record_id": str},
    )
    excluded = {"record_id", "family_id", "split", "label"}
    feature_columns = [
        column
        for column in engineered.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(engineered[column])
    ]
    if len(feature_columns) != 62:
        raise ValueError(f"Expected 62 engineered features, found {len(feature_columns)}")
    engineered = frame[["record_id"]].merge(
        engineered[["record_id", *feature_columns]],
        on="record_id",
        validate="one_to_one",
        sort=False,
    )
    numeric = engineered[feature_columns].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=int)
    return frame, frame["clause_text"].fillna("").astype(str).tolist(), numeric, feature_columns


def resolve_context(frame: pd.DataFrame) -> tuple[list[str], list[str], list[dict[str, str]]]:
    corpus_root = str(ROOT / "corpus")
    if corpus_root not in sys.path:
        sys.path.insert(0, corpus_root)
    if not hasattr(datetime, "UTC"):
        setattr(datetime, "UTC", datetime.timezone.utc)
    from annotate_with_ollama import ContextResolver, prompt_payload

    rows = frame[
        ["record_id", "family_id", "document_id", "clause_ids", "clause_text", "section_ids"]
    ].fillna("").to_dict(orient="records")
    resolver = ContextResolver(ALIGNMENT_ROOT)
    resolver.prepare(rows)
    contexts = [resolver.resolve(row) for row in rows]
    if any(not context for context in contexts):
        raise ValueError("ContextResolver could not locate one or more target clauses")
    previous = [context.get("previous_clause", "") for context in contexts]
    following = [context.get("next_clause", "") for context in contexts]

    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = np.sort(rng.choice(len(rows), size=20, replace=False))
    examples: list[dict[str, str]] = []
    for index in sample_indices:
        row = rows[int(index)]
        payload = prompt_payload(row, contexts[int(index)])
        if (
            payload["target_clause"] != row["clause_text"]
            or payload["previous_clause_context"] != previous[int(index)]
            or payload["next_clause_context"] != following[int(index)]
        ):
            raise AssertionError("Resolved sample differs from annotation prompt payload")
        if len(examples) < 3:
            examples.append(
                {
                    "record_id": row["record_id"],
                    "target": row["clause_text"],
                    "previous": previous[int(index)],
                    "next": following[int(index)],
                }
            )
    print("Annotation-payload equivalence: 20/20 sampled rows matched.")
    for example in examples:
        print(
            "SAMPLE",
            example["record_id"],
            json.dumps(
                {key: excerpt(value) for key, value in example.items() if key != "record_id"},
                ensure_ascii=False,
            ),
        )
    return previous, following, examples


def excerpt(value: str, limit: int = 180) -> str:
    normalized = " ".join(value.split())
    if len(normalized) > limit:
        return normalized[: limit - 1] + "…"
    return normalized


def context_coverage(frame: pd.DataFrame, previous: list[str], following: list[str]) -> pd.DataFrame:
    data = frame[["split", "family_id"]].copy()
    data["previous_present"] = [bool(value.strip()) for value in previous]
    data["next_present"] = [bool(value.strip()) for value in following]
    output = (
        data.groupby(["split", "family_id"], sort=True)
        .agg(
            n_rows=("family_id", "size"),
            previous_n=("previous_present", "sum"),
            next_n=("next_present", "sum"),
        )
        .reset_index()
    )
    output["previous_pct"] = output["previous_n"] / output["n_rows"] * 100
    output["next_pct"] = output["next_n"] / output["n_rows"] * 100
    return output


def encode_inputs(
    frame: pd.DataFrame,
    target: list[str],
    previous: list[str],
    following: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    text_fingerprint = digest(
        {
            "record_ids": frame["record_id"].tolist(),
            "target": target,
            "previous": previous,
            "next": following,
        }
    )
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for key, (model_name, revision) in ENCODERS.items():
        fingerprint = digest(
            {"text_fingerprint": text_fingerprint, "model": model_name, "revision": revision}
        )
        cache_path = CACHE_ROOT / f"{key.lower()}-{fingerprint}.npz"
        expected = (len(frame), 384)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            arrays = {part: cached[part] for part in ("target", "previous", "next")}
            if any(array.shape != expected or not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError(f"Invalid embedding cache: {cache_path}")
            embeddings[key] = arrays
            continue

        model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
            local_files_only=True,
        )
        model.eval()
        arrays: dict[str, np.ndarray] = {}
        for part, texts in (
            ("target", target),
            ("previous", previous),
            ("next", following),
        ):
            result = np.zeros(expected, dtype=np.float32)
            indexes = [index for index, value in enumerate(texts) if value.strip()]
            if indexes:
                vectors = model.encode(
                    [texts[index] for index in indexes],
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32, copy=False)
                if vectors.shape != (len(indexes), expected[1]):
                    raise ValueError(f"Unexpected {key} {part} embedding shape: {vectors.shape}")
                result[indexes] = vectors
            arrays[part] = result
        np.savez_compressed(cache_path, **arrays)
        embeddings[key] = arrays
        del model
    return embeddings


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b",
    )


def transform_features(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target_text: list[str],
    engineered: np.ndarray,
    embeddings: dict[str, dict[str, np.ndarray]],
) -> dict[str, sparse.csr_matrix]:
    train_eng = engineered[train_indices]
    test_eng = engineered[test_indices]
    imputer = SimpleImputer(strategy="median")
    train_eng = imputer.fit_transform(train_eng)
    test_eng = imputer.transform(test_eng)
    engineered_scaler = StandardScaler()
    train_eng = engineered_scaler.fit_transform(train_eng)
    test_eng = engineered_scaler.transform(test_eng)
    train_eng_sparse = sparse.csr_matrix(train_eng)
    test_eng_sparse = sparse.csr_matrix(test_eng)

    vectorizer = make_vectorizer()
    train_tfidf = vectorizer.fit_transform([target_text[i] for i in train_indices])
    test_tfidf = vectorizer.transform([target_text[i] for i in test_indices])
    features = {
        "E0_combined_reference": sparse.hstack(
            [train_eng_sparse, train_tfidf], format="csr"
        ),
    }
    test_features: dict[str, sparse.csr_matrix] = {
        "E0_combined_reference": sparse.hstack(
            [test_eng_sparse, test_tfidf], format="csr"
        ),
    }

    for encoder, arrays in embeddings.items():
        train_target = arrays["target"][train_indices]
        test_target = arrays["target"][test_indices]
        target_scaler = StandardScaler()
        train_target_scaled = target_scaler.fit_transform(train_target)
        test_target_scaled = target_scaler.transform(test_target)
        train_context = np.hstack(
            [arrays[part][train_indices] for part in ("target", "previous", "next")]
        )
        test_context = np.hstack(
            [arrays[part][test_indices] for part in ("target", "previous", "next")]
        )
        context_scaler = StandardScaler()
        train_context_scaled = context_scaler.fit_transform(train_context)
        test_context_scaled = context_scaler.transform(test_context)
        target_train_sparse = sparse.csr_matrix(train_target_scaled)
        target_test_sparse = sparse.csr_matrix(test_target_scaled)
        context_train_sparse = sparse.csr_matrix(train_context_scaled)
        context_test_sparse = sparse.csr_matrix(test_context_scaled)

        name = f"{encoder}_E1_target_only"
        features[name] = target_train_sparse
        test_features[name] = target_test_sparse
        name = f"{encoder}_E2_target_plus_engineered"
        features[name] = sparse.hstack(
            [target_train_sparse, train_eng_sparse], format="csr"
        )
        test_features[name] = sparse.hstack(
            [target_test_sparse, test_eng_sparse], format="csr"
        )
        name = f"{encoder}_E3_target_prev_next"
        features[name] = context_train_sparse
        test_features[name] = context_test_sparse
        name = f"{encoder}_E4_context_plus_engineered"
        features[name] = sparse.hstack(
            [context_train_sparse, train_eng_sparse], format="csr"
        )
        test_features[name] = sparse.hstack(
            [context_test_sparse, test_eng_sparse], format="csr"
        )
        name = f"{encoder}_E5_full_stack"
        features[name] = sparse.hstack(
            [context_train_sparse, train_eng_sparse, train_tfidf], format="csr"
        )
        test_features[name] = sparse.hstack(
            [context_test_sparse, test_eng_sparse, test_tfidf], format="csr"
        )

    return {
        key: (features[key], test_features[key])
        for key in features
    }


def eligible_families(frame: pd.DataFrame) -> list[str]:
    counts = frame["family_id"].value_counts()
    small = set(counts[counts < MIN_HELDOUT_FAMILY_ROWS].index)
    if small != SMALL_FAMILIES:
        raise ValueError(f"Unexpected small-family set: {sorted(small)}")
    return sorted(counts[counts >= MIN_HELDOUT_FAMILY_ROWS].index.tolist())


def fit_lr(matrix: sparse.csr_matrix, labels: np.ndarray, c_value: float = 1.0) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=c_value,
        random_state=RANDOM_SEED,
    )
    return model.fit(matrix, labels)


def nested_selected_model(
    frame: pd.DataFrame,
    target_text: list[str],
    engineered: np.ndarray,
    embeddings: dict[str, dict[str, np.ndarray]],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_families = frame.iloc[outer_train]["family_id"].to_numpy()
    inner_validation_pool = outer_train[
        ~np.isin(train_families, list(SMALL_FAMILIES))
    ]
    groups = frame.iloc[inner_validation_pool]["family_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    inner_oof: dict[str, np.ndarray] = {}
    cached_folds = []
    for fit_local, valid_local in splitter.split(inner_validation_pool, groups=groups):
        valid_indices = inner_validation_pool[valid_local]
        fit_indices = outer_train[~np.isin(outer_train, valid_indices)]
        x_pairs = transform_features(
            fit_indices, valid_indices, target_text, engineered, embeddings
        )
        cached_folds.append((fit_indices, valid_indices, x_pairs))
        for name, (x_fit, x_valid) in x_pairs.items():
            inner_oof.setdefault(name, np.full(len(frame), np.nan))
            classifier = fit_lr(x_fit, frame.iloc[fit_indices]["label"].to_numpy(dtype=int))
            inner_oof[name][valid_indices] = classifier.predict_proba(x_valid)[:, 1]

    candidate_names = sorted(name for name in inner_oof if name != "E0_combined_reference")
    candidate_aps = {
        name: average_precision_score(
            frame.iloc[inner_validation_pool]["label"].to_numpy(dtype=int),
            inner_oof[name][inner_validation_pool],
        )
        for name in candidate_names
    }
    selected_name = max(candidate_names, key=lambda name: (candidate_aps[name], name))

    c_scores: dict[float, np.ndarray] = {
        c_value: np.full(len(frame), np.nan) for c_value in C_GRID
    }
    for fit_indices, valid_indices, x_pairs in cached_folds:
        x_fit, x_valid = x_pairs[selected_name]
        y_fit = frame.iloc[fit_indices]["label"].to_numpy(dtype=int)
        for c_value in C_GRID:
            classifier = fit_lr(x_fit, y_fit, c_value=c_value)
            c_scores[c_value][valid_indices] = classifier.predict_proba(x_valid)[:, 1]
    c_aps = {
        c_value: average_precision_score(
            frame.iloc[inner_validation_pool]["label"].to_numpy(dtype=int),
            c_scores[c_value][inner_validation_pool],
        )
        for c_value in C_GRID
    }
    selected_c = max(C_GRID, key=lambda value: (c_aps[value], -value))

    outer_pairs = transform_features(
        outer_train, outer_test, target_text, engineered, embeddings
    )
    x_fit, x_test = outer_pairs[selected_name]
    classifier = fit_lr(
        x_fit,
        frame.iloc[outer_train]["label"].to_numpy(dtype=int),
        c_value=selected_c,
    )
    probability = classifier.predict_proba(x_test)[:, 1]
    details = {
        "heldout_family": str(frame.iloc[outer_test[0]]["family_id"]),
        "selected_configuration": selected_name,
        "inner_ap_at_c1": float(candidate_aps[selected_name]),
        "selected_c": selected_c,
        "inner_ap_by_c": {str(key): float(value) for key, value in c_aps.items()},
    }
    return probability, details


def metrics_for(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    family_ids: np.ndarray,
    baseline_scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    ci_ap, diff_ci = bootstrap_intervals(labels, scores, baseline_scores, rng)
    family_aps = []
    for family in sorted(set(family_ids)):
        mask = family_ids == family
        family_aps.append(
            {
                "model": name,
                "family_id": family,
                "n_rows": int(mask.sum()),
                "n_positive": int(labels[mask].sum()),
                "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            }
        )
    aps = [row["average_precision"] for row in family_aps]
    top_count = max(1, int(np.ceil(len(labels) * 0.20)))
    top_indices = np.argsort(-scores, kind="stable")[:top_count]
    cm = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return (
        {
            "model": name,
            "n_rows": len(labels),
            "n_positive": int(labels.sum()),
            "pooled_oof_ap": float(average_precision_score(labels, scores)),
            "pooled_ap_ci95_low": ci_ap[0],
            "pooled_ap_ci95_high": ci_ap[1],
            "mean_family_ap": float(np.mean(aps)),
            "sd_family_ap": float(np.std(aps, ddof=1)),
            "precision_at_0.5": float(precision),
            "recall_at_0.5": float(recall),
            "f1_at_0.5": float(f1),
            "f2_at_0.5": float(fbeta_score(labels, prediction, beta=2, zero_division=0)),
            "recall_at_top20_reviewed": float(labels[top_indices].sum() / labels.sum()),
            "confusion_tn_fp_fn_tp": json.dumps([int(value) for value in cm]),
            "ap_difference_vs_e0": float(
                average_precision_score(labels, scores)
                - average_precision_score(labels, baseline_scores)
            ),
            "paired_ap_difference_ci95_low": diff_ci[0],
            "paired_ap_difference_ci95_high": diff_ci[1],
            "threshold": 0.5,
        },
        family_aps,
    )


def bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ap_values: list[float] = []
    diff_values: list[float] = []
    attempts = 0
    while len(ap_values) < BOOTSTRAP_REPLICATES and attempts < BOOTSTRAP_REPLICATES * 20:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size < 2:
            continue
        current_ap = average_precision_score(labels[indices], scores[indices])
        baseline_ap = average_precision_score(labels[indices], baseline_scores[indices])
        ap_values.append(float(current_ap))
        diff_values.append(float(current_ap - baseline_ap))
    if len(ap_values) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("Could not produce 1,000 valid bootstrap replicates")
    return (
        tuple(np.quantile(ap_values, [0.025, 0.975]).tolist()),
        tuple(np.quantile(diff_values, [0.025, 0.975]).tolist()),
    )


def run_evaluations(
    frame: pd.DataFrame,
    target_text: list[str],
    engineered: np.ndarray,
    embeddings: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    families = eligible_families(frame)
    evaluation_mask = frame["family_id"].isin(families).to_numpy()
    evaluation_indices = np.flatnonzero(evaluation_mask)
    evaluated = frame.iloc[evaluation_indices].reset_index(drop=True)
    if len(evaluated) != 2540 or evaluated["label"].sum() != 453:
        raise ValueError("Unexpected LOFO evaluation cohort")
    scores: dict[str, np.ndarray] = {}
    nested_details = []
    for fold_index, family in enumerate(families):
        test_indices = np.flatnonzero(frame["family_id"].to_numpy() == family)
        train_indices = np.flatnonzero(frame["family_id"].to_numpy() != family)
        x_pairs = transform_features(
            train_indices, test_indices, target_text, engineered, embeddings
        )
        for name, (x_train, x_test) in x_pairs.items():
            scores.setdefault(name, np.full(len(evaluated), np.nan))
            classifier = fit_lr(x_train, frame.iloc[train_indices]["label"].to_numpy(dtype=int))
            output_indices = np.searchsorted(evaluation_indices, test_indices)
            scores[name][output_indices] = classifier.predict_proba(x_test)[:, 1]
        nested_probability, details = nested_selected_model(
            frame,
            target_text,
            engineered,
            embeddings,
            train_indices,
            test_indices,
        )
        nested_name = "Nested_inner_CV_selected_embedding"
        scores.setdefault(nested_name, np.full(len(evaluated), np.nan))
        output_indices = np.searchsorted(evaluation_indices, test_indices)
        scores[nested_name][output_indices] = nested_probability
        nested_details.append(details)
        print(
            f"LOFO {fold_index+1}/{len(families)} {family}: "
            f"nested={details['selected_configuration']} C={details['selected_c']}"
        )
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("An evaluated row is missing an out-of-fold prediction")

    baseline = scores["E0_combined_reference"]
    rng = np.random.default_rng(RANDOM_SEED)
    result_rows = []
    family_rows = []
    for name, probability in scores.items():
        row, by_family = metrics_for(
            name,
            evaluated["label"].to_numpy(dtype=int),
            probability,
            evaluated["family_id"].to_numpy(),
            baseline,
            rng,
        )
        result_rows.append(row)
        family_rows.extend(by_family)
    return pd.DataFrame(result_rows), pd.DataFrame(family_rows), nested_details


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.columns:
        if pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_summary(
    results: pd.DataFrame,
    family_results: pd.DataFrame,
    coverage: pd.DataFrame,
    nested_details: list[dict[str, Any]],
    examples: list[dict[str, str]],
) -> None:
    output = [
        "# Contextual sentence-embedding experiment",
        "",
        "## Protocol and assumptions",
        "",
        "- Context text is resolved by the exact `ContextResolver` and `prompt_payload` implementation in `corpus/annotate_with_ollama.py`; the target text and same-revision previous/next clauses are passed exactly as in the annotation prompt. A fixed-seed random sample of 20 rows matched the generated prompt fields in all three text positions.",
        "- Target text is loaded using only `record_id`, `clause_text`, and `section_ids` from reviewer CSVs. The model receives target text, context text, and the existing engineered feature values only; labels are used only as outcomes.",
        "- No votes, rationales, errata, final RFC text, candidate-source values, sample weights, or revision/future information are used. No per-LLM vote columns or vote artifacts were found in the processed CSV/JSON outputs, so unanimous/2-1 subset metrics are unavailable.",
        "- MiniLM and BGE-small are frozen local encoders, used offline and without fine-tuning. Each target/previous/next text is encoded separately. Empty neighbor strings map to all-zero vectors rather than an embedding of a special empty-text token.",
        "- BGE-small is used without retrieval/query prefixes because these are clause representations for classification rather than query-document retrieval.",
        "- Embeddings are computed once from text only and cached; all learned feature transforms (TF-IDF vocabulary/IDF, imputation, and standardization) are fit separately inside each outer LOFO training fold. Inner-C selection uses GroupKFold splits of outer-training families only.",
        "- Families with fewer than 30 rows (6man: 1; cache-header: 5) remain in every training fold and are never held out. This yields 9 scored families, 2,540 scored rows, and 453 positives.",
        "- Results use LogisticRegression(class_weight='balanced', max_iter=2000). Threshold metrics use the default 0.5 only; all other comparisons use threshold-free AP. Bootstrap intervals resample evaluated rows 1,000 times, and AP-difference intervals use paired row resamples against E0.",
        "- The nested selector first chooses an embedding configuration using inner grouped OOF AP at C=1, then chooses its C from {0.01, 0.1, 1, 10} on those same inner grouped folds. Outer held-out families are not used in either selection step; the nested selector is the leakage-safe tuned result. The fixed-configuration E1-E5 comparisons use C=1.",
        "",
        "## Prev/next context coverage",
        "",
        markdown_table(
            coverage,
            ["split", "family_id", "n_rows", "previous_n", "previous_pct", "next_n", "next_pct"],
        ),
        "",
        "## Prompt-text spot check",
        "",
        "All 20 random sample rows matched the annotation payload’s target, previous-clause, and next-clause strings exactly. Three sample excerpts (clipped for readability):",
        "",
    ]
    for index, example in enumerate(examples, start=1):
        output.extend(
            [
                f"{index}. `{example['record_id']}`",
                f"   - Target: {excerpt(example['target']) or '(empty)'}",
                f"   - Previous: {excerpt(example['previous']) or '(none)'}",
                f"   - Next: {excerpt(example['next']) or '(none)'}",
            ]
        )
    output.extend(
        [
            "",
            "## Encoders",
            "",
            "- `MiniLM`: `sentence-transformers/all-MiniLM-L6-v2` (cached revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions).",
            "- `BGE-small`: `BAAI/bge-small-en-v1.5` (cached revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions).",
            "",
            "## LOFO comparison",
            "",
            "E0 is the existing combined text+engineered baseline reconstructed from fold-local target TF-IDF and fold-local transforms of the raw engineered features. The notebook reference is pooled AP 0.291 on 9 scored families; 0.293 is the 11-family result before applying the <30-row held-out rule.",
            "",
            markdown_table(
                results,
                [
                    "model",
                    "pooled_oof_ap",
                    "pooled_ap_ci95_low",
                    "pooled_ap_ci95_high",
                    "mean_family_ap",
                    "sd_family_ap",
                    "precision_at_0.5",
                    "recall_at_0.5",
                    "f1_at_0.5",
                    "f2_at_0.5",
                    "recall_at_top20_reviewed",
                    "confusion_tn_fp_fn_tp",
                    "ap_difference_vs_e0",
                    "paired_ap_difference_ci95_low",
                    "paired_ap_difference_ci95_high",
                ],
            ),
            "",
            "Per-family AP for each model is in `per_family_ap.csv`; the complete metrics table is in `results.csv`.",
            "",
            "### Inner-C nested selection",
            "",
            markdown_table(
                pd.DataFrame(nested_details)[
                    ["heldout_family", "selected_configuration", "inner_ap_at_c1", "selected_c"]
                ],
                ["heldout_family", "selected_configuration", "inner_ap_at_c1", "selected_c"],
            ),
            "",
            "The selected configuration and C may vary by held-out family. This is a nested model-selection result, not a single globally fixed encoder/configuration.",
            "",
            "Labels are LLM-consensus labels on a candidate cohort selected using future information; AP and prevalence do not transfer to the full clause population.",
            "",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    frame, target_text, engineered, _ = load_inputs()
    previous, following, examples = resolve_context(frame)
    coverage = context_coverage(frame, previous, following)
    embeddings = encode_inputs(frame, target_text, previous, following)
    results, family_results, nested_details = run_evaluations(
        frame, target_text, engineered, embeddings
    )
    results.to_csv(OUTPUT_ROOT / "results.csv", index=False)
    family_results.to_csv(OUTPUT_ROOT / "per_family_ap.csv", index=False)
    write_summary(results, family_results, coverage, nested_details, examples)
    print(f"Wrote {OUTPUT_ROOT / 'results.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'per_family_ap.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'summary.md'}")
    print(results[["model", "pooled_oof_ap", "ap_difference_vs_e0"]].to_string(index=False))


if __name__ == "__main__":
    main()
