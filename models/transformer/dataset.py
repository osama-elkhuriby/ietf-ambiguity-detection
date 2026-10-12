"""Build immutable, leakage-audited transformer input artifacts.

Same-version neighbours are materialized before training so model code never
performs an implicit join against alignment outputs. Labels and audit metadata
remain available to the trainer, but only target and context may become text.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from models.classical.feature_pipeline import load_binary_dataset, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "corpus/processed/model_datasets/v2/training_dataset.csv"
DEFAULT_ALIGNMENT_ROOT = ROOT / "corpus/processed/alignments"
DEFAULT_OUTPUT_DIR = ROOT / "corpus/processed/transformer_inputs/v3"
VALID_SPLITS = ("train", "validation", "test")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

DATASET_COLUMNS = (
    "record_id",
    "family_id",
    "document_id",
    "revision",
    "clause_ids",
    "section_ids",
    "lineage_ids",
    "split",
    "previous_clause",
    "target_clause",
    "next_clause",
    "section_heading",
    "context_source_ids",
    "context_resolution_status",
    "alignment_source_path",
    "source_hashes",
    "label",
    "label_tier",
    "sample_weight",
    "ambiguity_subtype",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def normalize_for_audit(value: Any) -> str:
    """Return a conservative canonical representation for leakage checks."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.casefold().split())


def token_ngrams(value: Any, size: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = TOKEN_RE.findall(normalize_for_audit(value))
    if not tokens:
        return frozenset()
    width = min(size, len(tokens))
    return frozenset(
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def pair_id(first: str, second: str) -> str:
    return "|".join(sorted((str(first), str(second))))


def load_duplicate_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("allowed_pairs", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Duplicate allowlist must contain a list of pair IDs")
    return set(values)


@dataclass(frozen=True)
class FrozenDocument:
    path: Path
    sha256: str
    clauses: tuple[dict[str, Any], ...]


class FrozenContextResolver:
    """Resolve neighbour text and provenance from frozen alignment inputs."""

    def __init__(self, alignment_root: Path) -> None:
        self.alignment_root = alignment_root.resolve()
        self.documents: dict[tuple[str, str], FrozenDocument | None] = {}

    def prepare(self, rows: Iterable[dict[str, str]]) -> None:
        wanted_by_document: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            key = (row["family_id"], row["document_id"])
            wanted_by_document[key].update(
                value for value in row["clause_ids"].split("|") if value
            )
        for key, wanted in wanted_by_document.items():
            self.documents[key] = self._select_document(*key, wanted)

    def _select_document(
        self, family_id: str, document_id: str, wanted: set[str]
    ) -> FrozenDocument | None:
        paths = sorted(
            (self.alignment_root / family_id).glob(f"*/inputs/{document_id}.json")
        )
        selected: FrozenDocument | None = None
        selected_score = -1
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            clauses = payload.get("clauses")
            if not isinstance(clauses, list):
                continue
            ids = {str(clause.get("clause_id", "")) for clause in clauses}
            score = len(wanted & ids)
            if score > selected_score:
                selected = FrozenDocument(
                    path=path.resolve(),
                    sha256=sha256_file(path),
                    clauses=tuple(clauses),
                )
                selected_score = score
        return selected if selected_score > 0 else None

    @staticmethod
    def _empty(status: str, document: FrozenDocument | None = None) -> dict[str, str]:
        return {
            "context_resolution_status": status,
            "previous_clause": "",
            "next_clause": "",
            "section_heading": "",
            "context_source_ids": "{}",
            "alignment_source_path": "" if document is None else _repo_path(document.path),
            "alignment_source_sha256": "" if document is None else document.sha256,
        }

    def resolve(self, row: dict[str, str]) -> dict[str, str]:
        document = self.documents.get((row["family_id"], row["document_id"]))
        if document is None:
            return self._empty("missing_document")

        wanted = {value for value in row["clause_ids"].split("|") if value}
        ordered = sorted(
            document.clauses, key=lambda clause: int(clause.get("index", 0))
        )
        positions = [
            index
            for index, clause in enumerate(ordered)
            if str(clause.get("clause_id", "")) in wanted
        ]
        if not positions:
            return self._empty("missing_target", document)

        first, last = min(positions), max(positions)
        target_section = str(ordered[first].get("section_id", ""))
        previous = None
        following = None
        if first > 0 and str(ordered[first - 1].get("section_id", "")) == target_section:
            previous = ordered[first - 1]
        if (
            last + 1 < len(ordered)
            and str(ordered[last + 1].get("section_id", "")) == target_section
        ):
            following = ordered[last + 1]

        if previous is not None and following is not None:
            status = "both"
        elif previous is not None:
            status = "previous_only"
        elif following is not None:
            status = "next_only"
        else:
            status = "target_only"
        source_ids = {
            "previous": "" if previous is None else str(previous.get("clause_id", "")),
            "target": sorted(wanted),
            "next": "" if following is None else str(following.get("clause_id", "")),
        }
        return {
            "context_resolution_status": status,
            "previous_clause": "" if previous is None else str(previous.get("raw_text", "")),
            "next_clause": "" if following is None else str(following.get("raw_text", "")),
            "section_heading": str(ordered[first].get("section_heading", "")),
            "context_source_ids": json.dumps(
                source_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "alignment_source_path": _repo_path(document.path),
            "alignment_source_sha256": document.sha256,
        }


def attach_context(
    frame: pd.DataFrame, alignment_root: Path, source_sha256: str
) -> pd.DataFrame:
    rows = [
        {key: "" if pd.isna(value) else str(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    resolver = FrozenContextResolver(alignment_root)
    resolver.prepare(rows)
    contexts = [resolver.resolve(row) for row in rows]
    result = frame.copy()
    for field in (
        "previous_clause",
        "next_clause",
        "section_heading",
        "context_source_ids",
        "context_resolution_status",
        "alignment_source_path",
    ):
        result[field] = [context[field] for context in contexts]
    result["target_clause"] = result["clause_text"].fillna("").astype(str)
    result["source_hashes"] = [
        json.dumps(
            {
                "alignment_input": context["alignment_source_sha256"],
                "annotation_source": source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for context in contexts
    ]
    result["label_tier"] = result["llm_label_tier"].fillna("").astype(str)
    result["sample_weight"] = result["training_weight"].astype(float)
    result["ambiguity_subtype"] = (
        result["llm_ambiguity_subtype"].fillna("").astype(str)
    )
    for optional in ("revision", "section_ids", "lineage_ids"):
        if optional not in result:
            result[optional] = ""
    return result.loc[:, DATASET_COLUMNS].copy()


def _cross_split_pairs(frame: pd.DataFrame) -> Iterable[tuple[int, int]]:
    indices = {
        split: frame.index[frame["split"] == split].tolist() for split in VALID_SPLITS
    }
    for left_split, right_split in itertools.combinations(VALID_SPLITS, 2):
        yield from itertools.product(indices[left_split], indices[right_split])


def leakage_audit(
    frame: pd.DataFrame,
    *,
    near_duplicate_threshold: float = 0.85,
    allowed_exact_pairs: set[str] | None = None,
) -> dict[str, Any]:
    """Return deterministic cross-split duplicate and context-overlap evidence."""
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    allowed = allowed_exact_pairs or set()
    normalized = [normalize_for_audit(value) for value in frame["target_clause"]]
    ngrams = [token_ngrams(value) for value in frame["target_clause"]]
    records = frame["record_id"].astype(str).tolist()
    families = frame["family_id"].astype(str).tolist()
    splits = frame["split"].astype(str).tolist()

    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for left, right in _cross_split_pairs(frame):
        left_text, right_text = normalized[left], normalized[right]
        if not left_text or not right_text:
            continue
        identifier = pair_id(records[left], records[right])
        if left_text == right_text:
            exact.append(
                {
                    "pair_id": identifier,
                    "allowed": identifier in allowed,
                    "left_record_id": records[left],
                    "right_record_id": records[right],
                    "left_family": families[left],
                    "right_family": families[right],
                    "left_split": splits[left],
                    "right_split": splits[right],
                    "text_sha256": hashlib.sha256(left_text.encode("utf-8")).hexdigest(),
                    "text_excerpt": left_text[:240],
                }
            )
            continue
        left_grams, right_grams = ngrams[left], ngrams[right]
        if not left_grams or not right_grams:
            continue
        upper_bound = min(len(left_grams), len(right_grams)) / max(
            len(left_grams), len(right_grams)
        )
        if upper_bound < near_duplicate_threshold:
            continue
        similarity = len(left_grams & right_grams) / len(left_grams | right_grams)
        if similarity >= near_duplicate_threshold:
            near.append(
                {
                    "pair_id": identifier,
                    "left_record_id": records[left],
                    "right_record_id": records[right],
                    "left_family": families[left],
                    "right_family": families[right],
                    "left_split": splits[left],
                    "right_split": splits[right],
                    "token_ngram_jaccard": round(similarity, 6),
                    "left_excerpt": left_text[:240],
                    "right_excerpt": right_text[:240],
                }
            )

    targets: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(normalized):
        if value:
            targets[value].append(index)
    context_overlaps: list[dict[str, Any]] = []
    for context_row, field in itertools.product(
        frame.index, ("previous_clause", "next_clause")
    ):
        value = normalize_for_audit(frame.at[context_row, field])
        if not value:
            continue
        for target_row in targets.get(value, []):
            if splits[target_row] == splits[context_row]:
                continue
            context_overlaps.append(
                {
                    "target_record_id": records[target_row],
                    "target_split": splits[target_row],
                    "context_record_id": records[context_row],
                    "context_split": splits[context_row],
                    "context_field": field,
                    "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
            )

    exact.sort(key=lambda item: item["pair_id"])
    near.sort(key=lambda item: (-item["token_ngram_jaccard"], item["pair_id"]))
    context_overlaps.sort(
        key=lambda item: (
            item["target_record_id"], item["context_record_id"], item["context_field"]
        )
    )
    unreviewed = [item for item in exact if not item["allowed"]]
    return {
        "schema_version": 2,
        "normalization": "Unicode NFC, case-folded, whitespace-collapsed",
        "near_duplicate_method": "word 3-gram Jaccard; shorter texts use all tokens",
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_cross_split_duplicates": exact,
        "unreviewed_exact_cross_split_duplicates": unreviewed,
        "near_cross_split_duplicates": near,
        "target_as_cross_split_context": context_overlaps,
        "counts": {
            "exact_cross_split_duplicates": len(exact),
            "unreviewed_exact_cross_split_duplicates": len(unreviewed),
            "near_cross_split_duplicates": len(near),
            "target_as_cross_split_context": len(context_overlaps),
        },
        "training_blocked": bool(unreviewed),
    }


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in VALID_SPLITS:
        subset = frame.loc[frame["split"] == split]
        result[split] = {
            "rows": len(subset),
            "families": sorted(set(subset["family_id"].astype(str))),
            "labels": {
                str(label): int(count)
                for label, count in sorted(Counter(subset["label"]).items())
            },
            "tiers": {
                str(tier): int(count)
                for tier, count in sorted(Counter(subset["label_tier"]).items())
            },
        }
    return result


def build_transformer_dataset(
    *,
    input_path: Path = DEFAULT_INPUT,
    alignment_root: Path = DEFAULT_ALIGNMENT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    duplicate_allowlist: Path | None = None,
    near_duplicate_threshold: float = 0.85,
) -> dict[str, Any]:
    """Build a new immutable dataset directory and return its manifest."""
    input_path = input_path.resolve()
    alignment_root = alignment_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new versioned path."
        )
    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")

    source_hash_before = sha256_file(input_path)
    source_rows = len(pd.read_csv(input_path, usecols=["record_id"]))
    frame, excluded_uncertain = load_binary_dataset(input_path)
    dataset = attach_context(frame, alignment_root, source_hash_before)
    allowlist = load_duplicate_allowlist(duplicate_allowlist)
    audit = leakage_audit(
        dataset,
        near_duplicate_threshold=near_duplicate_threshold,
        allowed_exact_pairs=allowlist,
    )
    split_assignments = dataset[["record_id", "family_id", "split"]].copy()
    context_counts = {
        str(status): int(count)
        for status, count in sorted(Counter(dataset["context_resolution_status"]).items())
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "source_path": _repo_path(input_path),
        "source_sha256": source_hash_before,
        "source_rows": source_rows,
        "binary_rows": len(dataset),
        "excluded_uncertain_rows": excluded_uncertain,
        "alignment_root": _repo_path(alignment_root),
        "dataset_columns": list(DATASET_COLUMNS),
        "split_summary": _split_summary(dataset),
        "context_resolution_counts": context_counts,
        "near_duplicate_threshold": near_duplicate_threshold,
        "duplicate_allowlist": (
            None if duplicate_allowlist is None else _repo_path(duplicate_allowlist)
        ),
        "leakage_audit_counts": audit["counts"],
        "training_blocked": audit["training_blocked"],
        "training_block_reasons": (
            ["unreviewed exact target-text duplicates cross split"]
            if audit["training_blocked"]
            else []
        ),
        "input_contract": {
            "model_text_fields": ["previous_clause", "target_clause", "next_clause"],
            "forbidden_model_fields": [
                "family_id",
                "label",
                "label_tier",
                "sample_weight",
                "ambiguity_subtype",
                "source_hashes",
                "split",
            ],
            "future_revision_evidence": False,
        },
    }

    try:
        temporary.mkdir(parents=True)
        _write_csv(temporary / "dataset.csv", dataset)
        _write_csv(temporary / "split_assignments.csv", split_assignments)
        split_files: dict[str, dict[str, Any]] = {}
        for split in VALID_SPLITS:
            split_path = temporary / "splits" / f"{split}.csv"
            _write_csv(split_path, dataset.loc[dataset["split"] == split])
            split_files[split] = {
                "path": f"splits/{split}.csv",
                "rows": int((dataset["split"] == split).sum()),
                "sha256": sha256_file(split_path),
            }
        manifest["split_files"] = split_files
        manifest["dataset_sha256"] = sha256_file(temporary / "dataset.csv")
        manifest["split_assignments_sha256"] = sha256_file(
            temporary / "split_assignments.csv"
        )
        _write_json(temporary / "leakage_audit.json", audit)
        manifest["leakage_audit_sha256"] = sha256_file(
            temporary / "leakage_audit.json"
        )
        if source_hash_before != sha256_file(input_path):
            raise RuntimeError("Source dataset changed while building artifacts")
        _write_json(temporary / "manifest.json", manifest)
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def require_training_ready(manifest: dict[str, Any]) -> None:
    """Reject a dataset whose audit contains unresolved training blockers."""
    if manifest.get("training_blocked"):
        reasons = manifest.get("training_block_reasons", ["unspecified blocker"])
        raise RuntimeError(f"Transformer dataset is blocked: {', '.join(reasons)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable, leakage-audited transformer inputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duplicate-allowlist", type=Path)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_transformer_dataset(
        input_path=args.input,
        alignment_root=args.alignment_root,
        output_dir=args.output_dir,
        duplicate_allowlist=args.duplicate_allowlist,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(f"Transformer inputs: {args.output_dir.resolve()}")
    print(f"Binary rows: {manifest['binary_rows']}")
    print(f"Context: {manifest['context_resolution_counts']}")
    print(f"Leakage audit: {manifest['leakage_audit_counts']}")
    print(f"Training blocked: {manifest['training_blocked']}")
    return 2 if manifest["training_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
