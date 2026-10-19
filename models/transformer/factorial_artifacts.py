"""Build versioned soft-target and fold-specific context-policy artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT, normalize_for_audit, token_ngrams
from models.transformer.experiment import (
    load_development_split,
    load_frozen_folds,
    load_nested_folds,
)

DEFAULT_CONFIG = ROOT / "experiments/transformer/configs/deberta_clause_only.yaml"
DEFAULT_OUTPUT = ROOT / "corpus/processed/transformer_audits/factorial-v1"
CONSENSUS_PATH = ROOT / "corpus/processed/llm_annotations/v1/consensus.csv"
MODEL_NAMES = ("gemma3:27b", "gpt-oss:20b", "qwen3:32b")
MIN_VALID_VOTES = 2
CONTEXT_SIMILARITY_THRESHOLD = 0.85


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_factorial_artifacts(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load Step 4 artifacts only after verifying their recorded hashes."""

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scope") != "development_train_only":
        raise RuntimeError("Factorial artifacts are not development-only")
    hashes = manifest.get("artifact_hashes", {})
    for name in ("soft_targets.csv", "context_policy.csv"):
        path = artifact_dir / name
        if hashes.get(name) != sha256_file(path):
            raise RuntimeError(f"Factorial artifact hash mismatch: {name}")
    return (
        pd.read_csv(artifact_dir / "soft_targets.csv"),
        pd.read_csv(artifact_dir / "context_policy.csv"),
        manifest,
    )


def _parse_votes(payload: str) -> tuple[int, int]:
    parsed = json.loads(payload)
    valid = 0
    positive = 0
    for model in MODEL_NAMES:
        item = parsed.get(model, {})
        vote = item.get("vote") if item.get("status") == "valid" else None
        if vote not in {"yes", "no"}:
            continue
        valid += 1
        positive += int(vote == "yes")
    return valid, positive


def build_soft_targets(train: pd.DataFrame, consensus_path: Path) -> pd.DataFrame:
    """Create a rationales-free, one-to-one vote-fraction mapping for train rows."""

    source = pd.read_csv(consensus_path, usecols=["record_id", "votes_json"])
    if source["record_id"].duplicated().any():
        raise RuntimeError("Consensus source contains duplicate record IDs")
    wanted = set(train["record_id"].astype(str))
    source = source[source["record_id"].astype(str).isin(wanted)].copy()
    observed = set(source["record_id"].astype(str))
    if observed != wanted:
        missing = sorted(wanted - observed)
        raise RuntimeError(f"Consensus votes missing for train rows: {missing[:3]}")
    counts = source["votes_json"].map(_parse_votes)
    source["valid_vote_count"] = counts.map(lambda value: value[0])
    source["positive_vote_count"] = counts.map(lambda value: value[1])
    source["included_for_soft_training"] = (
        source["valid_vote_count"] >= MIN_VALID_VOTES
    )
    source["soft_target"] = np.where(
        source["included_for_soft_training"],
        source["positive_vote_count"] / source["valid_vote_count"],
        np.nan,
    )
    output = train[["record_id", "label"]].merge(
        source.drop(columns="votes_json"), on="record_id", validate="one_to_one"
    )
    if not output["soft_target"].dropna().between(0.0, 1.0).all():
        raise RuntimeError("Soft targets must be finite probabilities")
    return output.sort_values("record_id").reset_index(drop=True)


def _jaccard(first: frozenset[Any], second: frozenset[Any]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _role_by_position(length: int, nested_fold: Any) -> np.ndarray:
    roles = np.full(length, "", dtype=object)
    roles[nested_fold.fitting_indices] = "fitting"
    roles[nested_fold.selection_indices] = "inner_selection"
    roles[nested_fold.outer.validation_indices] = "outer_validation"
    if set(roles) != {"fitting", "inner_selection", "outer_validation"}:
        raise RuntimeError("Nested roles do not partition the full training data")
    return roles


def _context_matches(
    text: str,
    *,
    row_role: str,
    targets: list[dict[str, Any]],
    exact_index: dict[str, list[int]],
    gram_index: dict[Any, set[int]],
    threshold: float,
) -> tuple[float, list[str], list[str]]:
    normalized = normalize_for_audit(text)
    if not normalized:
        return 0.0, [], []
    grams = token_ngrams(text)
    candidates = set(exact_index.get(normalized, []))
    for gram in grams:
        candidates.update(gram_index.get(gram, set()))
    matched: list[tuple[float, str, str]] = []
    for index in candidates:
        target = targets[index]
        if target["role"] == row_role:
            continue
        similarity = (
            1.0
            if target["normalized"] == normalized
            else _jaccard(grams, target["grams"])
        )
        if similarity >= threshold:
            matched.append((similarity, target["record_id"], target["role"]))
    if not matched:
        return 0.0, [], []
    matched.sort(key=lambda value: (-value[0], value[1]))
    return (
        matched[0][0],
        [value[1] for value in matched],
        sorted({value[2] for value in matched}),
    )


def build_context_policy(
    train: pd.DataFrame, nested_folds: list[Any], *, threshold: float
) -> pd.DataFrame:
    """Flag cross-role target leakage in previous/next local context."""

    records: list[dict[str, Any]] = []
    for nested_fold in nested_folds:
        roles = _role_by_position(len(train), nested_fold)
        targets: list[dict[str, Any]] = []
        exact_index: dict[str, list[int]] = defaultdict(list)
        gram_index: dict[Any, set[int]] = defaultdict(set)
        for position, row in enumerate(train.itertuples(index=False)):
            normalized = normalize_for_audit(row.target_clause)
            grams = token_ngrams(row.target_clause)
            target = {
                "record_id": str(row.record_id),
                "role": str(roles[position]),
                "normalized": normalized,
                "grams": grams,
            }
            target_index = len(targets)
            targets.append(target)
            exact_index[normalized].append(target_index)
            for gram in grams:
                gram_index[gram].add(target_index)
        for position, row in enumerate(train.itertuples(index=False)):
            record: dict[str, Any] = {
                "fold": nested_fold.outer.number,
                "record_id": str(row.record_id),
                "partition_role": str(roles[position]),
            }
            for field in ("previous", "next"):
                value = getattr(row, f"{field}_clause")
                similarity, matched_ids, matched_roles = _context_matches(
                    str(value),
                    row_role=str(roles[position]),
                    targets=targets,
                    exact_index=exact_index,
                    gram_index=gram_index,
                    threshold=threshold,
                )
                record[f"{field}_allowed"] = not matched_ids
                record[f"{field}_max_similarity"] = similarity
                record[f"{field}_matched_record_ids"] = json.dumps(matched_ids)
                record[f"{field}_matched_roles"] = json.dumps(matched_roles)
            records.append(record)
    output = pd.DataFrame(records)
    if output.duplicated(["fold", "record_id"]).any():
        raise RuntimeError("Context policy must map each fold and record exactly once")
    return output.sort_values(["fold", "record_id"]).reset_index(drop=True)


def apply_context_policy(
    frame: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    fold: int,
    expected_role: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank cross-role context spans according to a frozen fold policy."""

    selected = policy[policy["fold"] == fold]
    merged = frame.merge(selected, on="record_id", how="left", validate="one_to_one")
    if merged["partition_role"].isna().any():
        raise RuntimeError("Context policy is missing one or more requested rows")
    if expected_role is not None and set(merged["partition_role"]) != {expected_role}:
        raise RuntimeError("Context policy role does not match the requested partition")
    counts: dict[str, int] = {}
    for field in ("previous", "next"):
        allowed = merged[f"{field}_allowed"].astype(bool)
        counts[f"{field}_removed"] = int((~allowed).sum())
        merged.loc[~allowed, f"{field}_clause"] = ""
    keep = list(frame.columns)
    return merged[keep].copy(), counts


def build(config_path: Path, output_dir: Path) -> Path:
    """Materialize immutable Step 4 governance artifacts."""

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_dir = (ROOT / config["dataset_dir"]).resolve()
    train = load_development_split(dataset_dir, "train")
    train_path = dataset_dir / "train.csv"
    fold_path = (ROOT / config["selection"]["fold_assignment"]).resolve()
    inner_path = (ROOT / config["selection"]["inner_selection_assignment"]).resolve()
    folds = load_frozen_folds(
        train,
        fold_path,
        expected_sha256=config["selection"]["fold_assignment_sha256"],
        expected_train_sha256=sha256_file(train_path),
    )
    nested = load_nested_folds(
        train,
        folds,
        inner_path,
        expected_sha256=config["selection"]["inner_selection_assignment_sha256"],
        expected_train_sha256=sha256_file(train_path),
    )
    if output_dir.exists():
        raise FileExistsError(f"Artifact output is immutable: {output_dir}")
    output_dir.mkdir(parents=True)
    soft = build_soft_targets(train, CONSENSUS_PATH)
    context = build_context_policy(
        train, nested, threshold=CONTEXT_SIMILARITY_THRESHOLD
    )
    soft.to_csv(output_dir / "soft_targets.csv", index=False, lineterminator="\n")
    context.to_csv(output_dir / "context_policy.csv", index=False, lineterminator="\n")
    removals: dict[str, dict[str, int]] = {}
    for fold in range(1, 6):
        current = context[context["fold"] == fold]
        similarities = pd.concat(
            [
                current.loc[~current["previous_allowed"], "previous_max_similarity"],
                current.loc[~current["next_allowed"], "next_max_similarity"],
            ],
            ignore_index=True,
        )
        removals[f"fold_{fold}"] = {
            "previous": int((~current["previous_allowed"]).sum()),
            "next": int((~current["next_allowed"]).sum()),
            "exact": int((similarities == 1.0).sum()),
            "near": int(
                ((similarities >= CONTEXT_SIMILARITY_THRESHOLD) & (similarities < 1.0)).sum()
            ),
        }
    manifest = {
        "schema_version": 1,
        "scope": "development_train_only",
        "rows": len(train),
        "minimum_valid_votes": MIN_VALID_VOTES,
        "soft_target_rows_included": int(soft["included_for_soft_training"].sum()),
        "soft_target_rows_excluded": int((~soft["included_for_soft_training"]).sum()),
        "soft_target_denominator": "valid_votes_only",
        "context_similarity": "normalized_word_trigram_jaccard",
        "context_similarity_threshold": CONTEXT_SIMILARITY_THRESHOLD,
        "context_removals": removals,
        "source_hashes": {
            "train.csv": sha256_file(train_path),
            "consensus.csv": sha256_file(CONSENSUS_PATH),
            "folds_v1.json": sha256_file(fold_path),
            "inner_selection_v1.json": sha256_file(inner_path),
            "experiment_plan.md": sha256_file(
                ROOT / "experiments/transformer/EXPERIMENT_PLAN.md"
            ),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    manifest["artifact_hashes"] = {
        name: sha256_file(output_dir / name)
        for name in ("soft_targets.csv", "context_policy.csv")
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build(args.config.resolve(), args.output.resolve())
    print(f"Factorial artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
