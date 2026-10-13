from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "corpus/processed/features/v3"
ALIGNMENT_ROOT = ROOT / "corpus/processed/alignments"
REVIEW_ROOT = ROOT / "corpus/processed/ground_truth"
OUTPUT_ROOT = ROOT / "results/revision_features"
MATCHER_ROOT = OUTPUT_ROOT / "matcher_cache"
EMBEDDING_ROOT = OUTPUT_ROOT / "embedding_cache"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MATCH_THRESHOLD = 0.62
MATCHER_ID = "section_constrained_lexical_minilm_v3"
MIN_FAMILY_SIZE = 30
RANDOM_SEED = 17
BOOTSTRAP_REPLICATES = 1000
TOKEN_RE = re.compile(r"0x[0-9A-Fa-f]+|[A-Za-z_][\w.-]*|\d+(?:\.\d+)*|[^\w\s]")


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def lexical_similarity(left: str, right: str) -> float:
    from collections import Counter

    a, b = Counter(tokens(left)), Counter(tokens(right))
    overlap = 2 * sum((a & b).values()) / max(1, sum(a.values()) + sum(b.values()))
    edit_ratio = ratio(left, right) / 100
    return 0.6 * overlap + 0.4 * edit_ratio


def load_rows() -> pd.DataFrame:
    pieces = []
    for split in ("train", "validation", "test"):
        rows = pd.read_csv(FEATURE_ROOT / "rows" / f"{split}_rows.csv", dtype=str)
        labels = pd.read_csv(
            FEATURE_ROOT / "targets" / f"{split}_labels.csv", dtype={"record_id": str}
        )
        merged = rows.merge(labels, on="record_id", validate="one_to_one")
        merged["split"] = split
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    if result["record_id"].duplicated().any() or len(result) != 2546:
        raise ValueError("Unexpected v3 row count or duplicate record IDs")
    return result


def load_candidate_sources() -> dict[str, str]:
    source_by_id: dict[str, str] = {}
    for path in sorted(REVIEW_ROOT.glob("*/review/all_candidates_v1/reviewer_a.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                record_id = row["record_id"]
                if record_id in source_by_id:
                    raise ValueError(f"Duplicate candidate-source record: {record_id}")
                source_by_id[record_id] = row["weak_label"]
    return source_by_id


def load_input_documents(family_id: str, max_revision: int) -> dict[int, dict[str, Any]]:
    run_name = (
        "pilot-semantic-v5"
        if family_id == "draft-ietf-tls-tls13"
        else "lexical-first-classifier-v1"
    )
    input_dir = ALIGNMENT_ROOT / family_id / run_name / "inputs"
    documents = {}
    for path in sorted(input_dir.glob("*.json")):
        match = re.search(r"-(\d+)$", path.stem)
        if match is None:
            continue
        revision = int(match.group(1))
        if revision > max_revision:
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["family_id"] != family_id or int(document["revision"]) != revision:
            raise ValueError(f"Input identity mismatch: {path}")
        documents[revision] = document
    return documents


def document_embeddings(
    model: SentenceTransformer,
    document: dict[str, Any],
    indices: list[int] | None = None,
) -> np.ndarray:
    family_id = document["family_id"]
    document_id = document["document_id"]
    all_texts = [clause["normalized_text"] for clause in document["clauses"]]
    selected = list(range(len(all_texts))) if indices is None else sorted(set(indices))
    texts = [all_texts[index] for index in selected]
    key_data = {
        "family_id": family_id,
        "document_id": document_id,
        "source_sha256": document["source_sha256"],
        "texts": texts,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
    }
    if len(selected) != len(all_texts):
        key_data["indices"] = selected
    key = digest(key_data)
    path = EMBEDDING_ROOT / family_id / f"{key}.npy"
    metadata_path = path.with_suffix(".json")
    if path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("key") != key or metadata.get("shape") != [len(texts), 384]:
            raise ValueError(f"Invalid embedding-cache metadata: {metadata_path}")
        vectors = np.load(path, allow_pickle=False)
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid embedding-cache vectors: {path}")
        return vectors

    EMBEDDING_ROOT.joinpath(family_id).mkdir(parents=True, exist_ok=True)
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"Unexpected embedding shape for {document_id}: {vectors.shape}")
    np.save(path, vectors, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(
            {"key": key, "shape": list(vectors.shape), "model": MODEL_NAME,
             "model_revision": MODEL_REVISION},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return vectors


def section_candidates(target: dict[str, Any], candidates: list[dict[str, Any]]) -> list[int]:
    target_heading = str(target.get("section_heading", "")).casefold()
    target_section = str(target.get("section_id", ""))
    exact_section = [
        i for i, clause in enumerate(candidates)
        if str(clause.get("section_id", "")) == target_section and target_section
    ]
    if exact_section:
        return exact_section

    heading_scores: dict[str, float] = {}
    for clause in candidates:
        heading = str(clause.get("section_heading", "")).casefold()
        if heading not in heading_scores:
            heading_scores[heading] = ratio(target_heading, heading) / 100
    if not heading_scores:
        return []
    best = max(heading_scores.values())
    if best < 0.70:
        return []
    accepted_headings = {
        heading for heading, score in heading_scores.items() if score >= max(0.70, best - 0.08)
    }
    return [
        i for i, clause in enumerate(candidates)
        if str(clause.get("section_heading", "")).casefold() in accepted_headings
    ]


def match_clause(
    target: dict[str, Any],
    target_vector: np.ndarray,
    previous: dict[str, Any],
    previous_vectors: np.ndarray,
    candidate_indices: list[int] | None = None,
) -> dict[str, Any]:
    candidates = previous["clauses"]
    indices = (
        section_candidates(target, candidates)
        if candidate_indices is None
        else candidate_indices
    )
    if not indices:
        return {"has_predecessor": 0, "relation_type": "NONE", "similarity": np.nan,
                "lexical_similarity": np.nan, "predecessor_clause_id": ""}
    if previous_vectors.shape[0] != len(indices):
        raise ValueError("Candidate vector count does not match section candidates")
    similarities = previous_vectors @ target_vector
    lexical_scores = np.fromiter(
        (lexical_similarity(target["normalized_text"], candidates[i]["normalized_text"])
         for i in indices),
        dtype=np.float32,
        count=len(indices),
    )
    combined = 0.5 * similarities + 0.5 * lexical_scores
    best_local = min(
        range(len(indices)),
        key=lambda j: (
            -float(combined[j]),
            abs(int(candidates[indices[j]]["index"]) - int(target["index"])),
            str(candidates[indices[j]]["clause_id"]),
        ),
    )
    best_index = indices[best_local]
    best_clause = candidates[best_index]
    score = float(combined[best_local])
    if score < MATCH_THRESHOLD:
        return {"has_predecessor": 0, "relation_type": "NONE", "similarity": score,
                "lexical_similarity": float(lexical_scores[best_local]),
                "predecessor_clause_id": ""}
    if target["normalized_text"] == best_clause["normalized_text"]:
        relation = "UNCHANGED"
    elif target.get("section_id") != best_clause.get("section_id"):
        relation = "MOVED"
    else:
        relation = "MODIFIED"
    return {
        "has_predecessor": 1,
        "relation_type": relation,
        "similarity": score,
        "lexical_similarity": float(lexical_scores[best_local]),
        "predecessor_clause_id": best_clause["clause_id"],
    }


def aggregate_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches or not all(match["has_predecessor"] for match in matches):
        return {"has_predecessor": 0, "relation_type": "NONE",
                "similarity": np.nan, "lexical_similarity": np.nan}
    relations = {match["relation_type"] for match in matches}
    relation = next(iter(relations)) if len(relations) == 1 else "MIXED"
    return {
        "has_predecessor": 1,
        "relation_type": relation,
        "similarity": float(np.median([match["similarity"] for match in matches])),
        "lexical_similarity": float(np.median([match["lexical_similarity"] for match in matches])),
    }


def build_family_cache(
    family_id: str,
    family_rows: pd.DataFrame,
    model: SentenceTransformer,
) -> pd.DataFrame:
    started = time.monotonic()
    revisions = family_rows["document_id"].str.extract(r"-(\d+)$")[0].astype(int)
    max_revision = int(revisions.max())
    docs = load_input_documents(family_id, max_revision)
    needed = set(revisions)
    for revision in revisions:
        needed.update(r for r in (revision - 1, revision - 2) if r >= 0)
    missing = needed - docs.keys()
    if missing:
        raise ValueError(f"Missing required draft input revisions for {family_id}: {sorted(missing)}")
    clause_maps = {
        revision: {clause["clause_id"]: (index, clause)
                   for index, clause in enumerate(document["clauses"])}
        for revision, document in docs.items()
    }
    no_match = {
        "has_predecessor": 0,
        "relation_type": "NONE",
        "similarity": np.nan,
        "lexical_similarity": np.nan,
        "predecessor_clause_id": "",
    }

    def pair_matches(
        tasks: list[tuple[int, int, str]],
    ) -> dict[tuple[int, int, str], dict[str, Any]]:
        unique_tasks = list(dict.fromkeys(tasks))
        target_requests: dict[int, set[int]] = {}
        source_requests: dict[int, set[int]] = {}
        candidates_by_task: dict[tuple[int, int, str], list[int]] = {}
        targets_by_task: dict[tuple[int, int, str], tuple[int, dict[str, Any]]] = {}
        results: dict[tuple[int, int, str], dict[str, Any]] = {}
        for task in unique_tasks:
            from_revision, to_revision, target_id = task
            if from_revision not in docs or to_revision not in docs:
                results[task] = no_match.copy()
                continue
            target_entry = clause_maps[to_revision].get(target_id)
            if target_entry is None:
                raise ValueError(f"Clause {target_id} missing from {docs[to_revision]['document_id']}")
            target_index, target = target_entry
            indices = section_candidates(target, docs[from_revision]["clauses"])
            if not indices:
                results[task] = no_match.copy()
                continue
            targets_by_task[task] = (target_index, target)
            candidates_by_task[task] = indices
            target_requests.setdefault(to_revision, set()).add(target_index)
            source_requests.setdefault(from_revision, set()).update(indices)

        target_vectors = {
            revision: {
                index: vector
                for index, vector in zip(
                    selected,
                    document_embeddings(model, docs[revision], selected),
                    strict=True,
                )
            }
            for revision, requested in target_requests.items()
            for selected in [sorted(requested)]
        }
        source_vectors = {
            revision: {
                index: vector
                for index, vector in zip(
                    selected,
                    document_embeddings(model, docs[revision], selected),
                    strict=True,
                )
            }
            for revision, requested in source_requests.items()
            for selected in [sorted(requested)]
        }
        for task, (target_index, target) in targets_by_task.items():
            from_revision = task[0]
            indices = candidates_by_task[task]
            result = match_clause(
                target,
                target_vectors[task[1]][target_index],
                docs[from_revision],
                np.stack([source_vectors[from_revision][index] for index in indices]),
                indices,
            )
            results[task] = result
        return results

    rows_and_targets = []
    h0_tasks = []
    for row in family_rows.itertuples(index=False):
        t = int(re.search(r"-(\d+)$", row.document_id).group(1))
        target_ids = [value for value in str(row.clause_ids).split("|") if value]
        rows_and_targets.append((row, t, target_ids))
        h0_tasks.extend((t - 1, t, clause_id) for clause_id in target_ids)
    h0_results = pair_matches(h0_tasks)

    h1_tasks = []
    for _, t, target_ids in rows_and_targets:
        h0_matches = [h0_results[(t - 1, t, clause_id)] for clause_id in target_ids]
        if t >= 2 and h0_matches and all(item["has_predecessor"] for item in h0_matches):
            h1_tasks.extend(
                (t - 2, t - 1, item["predecessor_clause_id"])
                for item in h0_matches
            )
    h1_results = pair_matches(h1_tasks)

    output = []
    for row, t, target_ids in rows_and_targets:
        h0_matches = [h0_results[(t - 1, t, clause_id)] for clause_id in target_ids]
        h0 = aggregate_matches(h0_matches)
        h1_matches = []
        if t >= 2:
            if h0_matches and all(item["has_predecessor"] for item in h0_matches):
                h1_matches = [
                    h1_results[(t - 2, t - 1, item["predecessor_clause_id"])]
                    for item in h0_matches
                ]
        h1 = aggregate_matches(h1_matches)
        output.append(
            {
                "record_id": row.record_id,
                "family_id": family_id,
                "document_id": row.document_id,
                "H1_has_predecessor": h1["has_predecessor"],
                "H1_relation_type": h1["relation_type"],
                "H1_similarity": h1["similarity"],
                "H1_lexical_similarity": h1["lexical_similarity"],
                "H0_has_predecessor": h0["has_predecessor"],
                "H0_relation_type": h0["relation_type"],
                "H0_similarity": h0["similarity"],
                "H0_lexical_similarity": h0["lexical_similarity"],
                "matcher": MATCHER_ID,
            }
        )
    result = pd.DataFrame(output)
    result["matcher_elapsed_seconds"] = time.monotonic() - started
    return result


def run_coverage(rows: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    if set(rows["record_id"]) - sources.keys():
        missing = len(set(rows["record_id"]) - sources.keys())
        raise ValueError(f"Candidate-source join missing {missing} rows")
    merged = rows.copy()
    records = []
    for family, group in merged.groupby("family_id", sort=True):
        cache_path = MATCHER_ROOT / f"{family}.parquet"
        if cache_path.exists():
            cache = pd.read_parquet(cache_path)
            valid_cache = (
                len(cache) == len(group)
                and not cache["record_id"].duplicated().any()
                and cache["matcher"].eq(MATCHER_ID).all()
            )
            if not valid_cache:
                started = time.monotonic()
                cache = build_family_cache(
                    family,
                    group[["record_id", "family_id", "document_id", "clause_ids"]],
                    MODEL,
                )
                cache.to_parquet(cache_path, index=False)
                print(
                    f"matcher_cache: {family}, rows={len(cache)}, "
                    f"elapsed={time.monotonic()-started:.1f}s, refreshed matcher",
                    flush=True,
                )
        else:
            started = time.monotonic()
            cache = build_family_cache(
                family,
                group[["record_id", "family_id", "document_id", "clause_ids"]],
                MODEL,
            )
            cache.to_parquet(cache_path, index=False)
            print(f"matcher_cache: {family}, rows={len(cache)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
        if len(cache) != len(group) or cache["record_id"].duplicated().any():
            raise ValueError(f"Invalid matcher checkpoint: {cache_path}")
        group = group.merge(cache, on=["record_id", "family_id", "document_id"], validate="one_to_one")
        group["candidate_source"] = group["record_id"].map(sources)
        if group["candidate_source"].isna().any():
            raise ValueError(f"Candidate source missing after matching for {family}")
        for tier in ("H1", "H0"):
            for (label, source), subset in group.groupby(["label", "candidate_source"], sort=True):
                sims = subset[f"{tier}_similarity"].dropna()
                records.append(
                    {
                        "family_id": family,
                        "label": int(label),
                        "candidate_source": source,
                        "tier": tier,
                        "n_rows": len(subset),
                        "n_resolvable": int(subset[f"{tier}_has_predecessor"].sum()),
                        "resolvable_pct": float(subset[f"{tier}_has_predecessor"].mean() * 100),
                        "median_match_similarity": float(sims.median()) if len(sims) else np.nan,
                    }
                )
    coverage = pd.DataFrame(records)
    coverage.to_csv(OUTPUT_ROOT / "coverage_gate.csv", index=False)
    result = merged.merge(
        pd.concat(
            [pd.read_parquet(MATCHER_ROOT / f"{family}.parquet")
             for family in sorted(merged["family_id"].unique())],
            ignore_index=True,
        ),
        on=["record_id", "family_id", "document_id"],
        validate="one_to_one",
    )
    result["candidate_source"] = result["record_id"].map(sources)
    if result["candidate_source"].isna().any():
        raise ValueError("Candidate source missing after matching")
    return result


def make_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tier: str,
    include_source: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if include_source:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x_train = encoder.fit_transform(train[["candidate_source"]])
        x_test = encoder.transform(test[["candidate_source"]])
        return sparse.csr_matrix(x_train), sparse.csr_matrix(x_test)

    numeric_columns = [f"{tier}_has_predecessor", f"{tier}_similarity"]
    category_column = f"{tier}_relation_type"
    numeric_train = train[numeric_columns].to_numpy(dtype=float)
    numeric_test = test[numeric_columns].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(imputer.fit_transform(numeric_train))
    numeric_test = scaler.transform(imputer.transform(numeric_test))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_train = encoder.fit_transform(train[[category_column]])
    cat_test = encoder.transform(test[[category_column]])
    return (
        sparse.hstack([sparse.csr_matrix(numeric_train), cat_train], format="csr"),
        sparse.hstack([sparse.csr_matrix(numeric_test), cat_test], format="csr"),
    )


def recall_at_top_fraction(y: np.ndarray, scores: np.ndarray, fraction: float = 0.20) -> float:
    positive_total = int(y.sum())
    if positive_total == 0:
        return np.nan
    count = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-scores, kind="stable")[:count]
    return float(y[selected].sum() / positive_total)


def bootstrap_ap_ci(y: np.ndarray, scores: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    while len(values) < BOOTSTRAP_REPLICATES and attempts < BOOTSTRAP_REPLICATES * 20:
        attempts += 1
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        values.append(average_precision_score(y[indices], scores[indices]))
    if len(values) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("Could not produce all bootstrap AP replicates")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def evaluate_models(rows: pd.DataFrame) -> pd.DataFrame:
    eligible = sorted(
        family for family, n in rows["family_id"].value_counts().items() if n >= MIN_FAMILY_SIZE
    )
    evaluated = rows[rows["family_id"].isin(eligible)].copy().reset_index(drop=True)
    models = ["N1_selection_artifact_diagnostic", "R1_H1_predecessor_only", "R2_H0_predecessor_only"]
    predictions: dict[str, np.ndarray] = {}
    output_rows = []
    y = evaluated["label"].to_numpy(dtype=int)

    for model_name in models:
        scores = np.full(len(evaluated), np.nan)
        for fold_index, family in enumerate(eligible):
            test_mask = evaluated["family_id"].to_numpy() == family
            train = evaluated.loc[~test_mask].copy()
            test = evaluated.loc[test_mask].copy()
            if model_name.startswith("N1_"):
                x_train, x_test = make_fold_features(train, test, "H1", include_source=True)
            else:
                tier = "H1" if model_name.startswith("R1_") else "H0"
                x_train, x_test = make_fold_features(train, test, tier)
            classifier = LogisticRegression(class_weight="balanced", max_iter=2000)
            classifier.fit(x_train, train["label"].to_numpy(dtype=int))
            scores[test_mask] = classifier.predict_proba(x_test)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError(f"Missing OOF scores for {model_name}")
        predictions[model_name] = scores
        family_aps = {
            family: average_precision_score(
                evaluated.loc[evaluated["family_id"] == family, "label"],
                scores[evaluated["family_id"].to_numpy() == family],
            )
            for family in eligible
        }
        ci_low, ci_high = bootstrap_ap_ci(y, scores, RANDOM_SEED)
        output_rows.append(
            {
                "model": model_name,
                "n_evaluated_rows": len(evaluated),
                "heldout_families": "|".join(eligible),
                "pooled_oof_ap": average_precision_score(y, scores),
                "pooled_ap_ci95_low": ci_low,
                "pooled_ap_ci95_high": ci_high,
                "mean_family_ap": float(np.mean(list(family_aps.values()))),
                "sd_family_ap": float(np.std(list(family_aps.values()), ddof=1)),
                "per_family_ap_json": json.dumps(family_aps, sort_keys=True),
                "recall_at_top20_reviewed": recall_at_top_fraction(y, scores),
                "prevalence": float(y.mean()),
                "threshold": 0.5,
                "note": "N1 is a selection-artifact diagnostic, not a candidate model"
                if model_name.startswith("N1_") else "candidate_source excluded",
            }
        )
    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT_ROOT / "signal_check.csv", index=False)
    return result


def write_summary(coverage: pd.DataFrame, signal: pd.DataFrame) -> None:
    family_runtimes = {
        path.stem: float(pd.read_parquet(path)["matcher_elapsed_seconds"].iloc[0])
        for path in sorted(MATCHER_ROOT.glob("*.parquet"))
    }
    tls_runtime = family_runtimes.get("draft-ietf-tls-tls13", np.nan)
    all_family_runtime = sum(family_runtimes.values())
    lines = [
        "# Revision-aware signal check",
        "",
        "## Assumptions and protocol",
        "",
        "- Only numbered draft inputs at or before each row revision are loaded; no RFC endpoint, errata, labels, votes, rationales, candidate-source values, or saved alignment/lineage outputs enter matching.",
        "- The deterministic predecessor matcher uses same section ID when available, otherwise matching section headings; it ranks candidates by the mean of the repository-style lexical score and pinned MiniLM cosine similarity, with a 0.62 minimum score and stable position/ID tie-breaks.",
        "- H0 is the pair `(t-1 -> t)`. H1 uses `(t-1 -> t)` only to locate the same clause at t-1, then measures the historical pair `(t-2 -> t-1)`; this identity lookup rule was explicitly approved.",
        "- Candidate source is joined only after matching by `record_id`; it is diagnostic-only except in the isolated N1 diagnostic. It is never included in R1/R2.",
        "- LOFO uses the fixed v3 family assignments. Families with fewer than 30 rows remain in every training fold but are never held out for scoring.",
        "- Imputation, scaling, and one-hot encoders are fitted within each training fold. Logistic regression uses `class_weight='balanced'`, `max_iter=2000`, with threshold 0.5.",
        "- Bootstrap intervals resample evaluated rows with replacement; one-class replicates are redrawn until 1,000 valid AP replicates are obtained.",
        "",
        "## TLS matcher validation",
        "",
        "Validation against existing row-level TLS alignment relationships could not be completed: the retained TLS run folders contain pair summary reports, but no `alignments.jsonl`, `lineage.jsonl`, or candidate link records. Aggregate relation counts do not identify per-clause predecessors. No alignment output was used by the matcher. The current matcher uses one-to-one predecessor links; split/merge recognition is outside this quick signal check.",
        f"TLS matcher runtime: {tls_runtime:.1f} seconds; all-family matching runtime: {all_family_runtime:.1f} seconds (sum of per-family matcher wall-clock times).",
        "",
        "## Coverage gate",
        "",
        f"Coverage summary CSV: `coverage_gate.csv` ({len(coverage)} family/label/source/tier rows).",
        "",
        "Rows flagged for a >10 percentage-point difference in resolvable-predecessor rate between label 1 and label 0:",
        "",
    ]
    family_flags = []
    source_flags = []
    for tier in ("H1", "H0"):
        subset = coverage[coverage["tier"] == tier]
        for family, grp in subset.groupby("family_id"):
            by_label = grp.groupby("label").apply(
                lambda x: np.average(x["resolvable_pct"], weights=x["n_rows"]), include_groups=False
            )
            if 0 in by_label.index and 1 in by_label.index and abs(by_label[1] - by_label[0]) > 10:
                family_flags.append(f"- {tier}, family `{family}`: {by_label[1]-by_label[0]:+.1f} pp")
        for source, grp in subset.groupby("candidate_source"):
            by_label = grp.groupby("label").apply(
                lambda x: np.average(x["resolvable_pct"], weights=x["n_rows"]), include_groups=False
            )
            if 0 in by_label.index and 1 in by_label.index and abs(by_label[1] - by_label[0]) > 10:
                source_flags.append(f"- {tier}, source `{source}`: {by_label[1]-by_label[0]:+.1f} pp")
    lines.extend(family_flags or ["- No family-level differences exceed 10 pp."])
    lines.extend(
        source_flags or ["- No candidate-source-level differences exceed 10 pp."]
    )
    lines.extend(
        [
            "",
            "## Quick LOFO signal check",
            "",
            "The existing combined-matrix reference is pooled AP 0.293 for 11 families; the 9-held-out-family references are 0.292/0.291. The dataset prevalence reference is 0.179.",
            "",
            "```csv",
            signal.to_csv(index=False).strip(),
            "```",
            "",
            "**N1_selection_artifact_diagnostic** is a diagnostic, not a candidate model. Candidate source is used only by this diagnostic and is not combined with any revision feature.",
            "",
            "Rows are an enriched candidate cohort selected using future information; AP and prevalence do not transfer to the full clause population. Revision features measure change/churn signal and may not reflect ambiguity.",
            "",
            "Discrepancy to resolve: this dataset contains only 5 errata rows, all label 0, all in TLS. The progress report stated 121 errata positives and 242 TLS records. These counts do not agree; the source/dataset definition needs reconciliation. As requested, the 5 errata rows form an unseen category in the fold where TLS is held out.",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MATCHER_ROOT.mkdir(parents=True, exist_ok=True)
    EMBEDDING_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    sources = load_candidate_sources()
    if len(set(rows["record_id"]) - sources.keys()):
        raise ValueError("Candidate source does not cover every row")

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=True,
    )
    model.eval()
    global MODEL
    MODEL = model
    coverage_rows = run_coverage(rows, sources)
    coverage = pd.read_csv(OUTPUT_ROOT / "coverage_gate.csv")
    signal = evaluate_models(coverage_rows)
    write_summary(coverage, signal)
    print(f"Wrote {OUTPUT_ROOT / 'coverage_gate.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'signal_check.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'summary.md'}")


MODEL: SentenceTransformer


if __name__ == "__main__":
    main()
