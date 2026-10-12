"""Create an immutable model dataset with a reviewed family-level split policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from corpus.paths import REPO_ROOT

DEFAULT_SOURCE = (
    REPO_ROOT / "corpus/processed/llm_annotations/v1/training_dataset.csv"
)
DEFAULT_SPLIT_CONFIG = (
    REPO_ROOT / "experiments/transformer/configs/split_v2.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "corpus/processed/model_datasets/v2"
VALID_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = frozenset(
    {"record_id", "family_id", "ambiguity_label", "label", "split"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_split_policy(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Split policy must be a JSON object")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise TypeError("Split policy requires an assignments object")

    family_to_split: dict[str, str] = {}
    for split in VALID_SPLITS:
        families = assignments.get(split)
        if not isinstance(families, list) or not all(
            isinstance(family, str) and family for family in families
        ):
            raise ValueError(f"Split policy {split!r} must be a list of family IDs")
        for family in families:
            if family in family_to_split:
                raise ValueError(f"Family appears in multiple splits: {family}")
            family_to_split[family] = split
    return family_to_split, payload


def apply_split_policy(
    frame: pd.DataFrame, family_to_split: dict[str, str]
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if frame["record_id"].duplicated().any():
        raise ValueError("Source dataset contains duplicate record_id values")
    source_families = set(frame["family_id"].astype(str))
    configured_families = set(family_to_split)
    missing_assignments = source_families - configured_families
    unknown_assignments = configured_families - source_families
    if missing_assignments:
        raise ValueError(
            f"Families missing from split policy: {sorted(missing_assignments)}"
        )
    if unknown_assignments:
        raise ValueError(
            f"Split policy contains unknown families: {sorted(unknown_assignments)}"
        )

    result = frame.copy()
    result["split"] = result["family_id"].astype(str).map(family_to_split)
    if result["split"].isna().any():
        raise RuntimeError("Split assignment unexpectedly produced missing values")
    family_counts = result.groupby("family_id")["split"].nunique()
    if (family_counts != 1).any():
        raise RuntimeError("A family spans multiple derived splits")
    return result


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
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
        }
    return result


def build_model_dataset(
    *,
    source_path: Path = DEFAULT_SOURCE,
    split_config: Path = DEFAULT_SPLIT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    split_config = split_config.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new version."
        )
    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")

    source_hash = sha256_file(source_path)
    config_hash = sha256_file(split_config)
    frame = pd.read_csv(source_path, keep_default_na=True)
    family_to_split, policy = load_split_policy(split_config)
    derived = apply_split_policy(frame, family_to_split)
    original_assignments = {
        str(family): sorted(set(group["split"].astype(str)))
        for family, group in frame.groupby("family_id")
    }
    assignments = derived[["record_id", "family_id", "split"]].copy()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "source_path": _repo_path(source_path),
        "source_sha256": source_hash,
        "split_config_path": _repo_path(split_config),
        "split_config_sha256": config_hash,
        "policy_name": policy.get("name", ""),
        "policy_rationale": policy.get("rationale", ""),
        "rows": len(derived),
        "source_preserved": True,
        "original_family_assignments": original_assignments,
        "split_summary": _summary(derived),
    }

    try:
        temporary.mkdir(parents=True)
        derived.to_csv(
            temporary / "training_dataset.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        assignments.to_csv(
            temporary / "split_assignments.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        manifest["dataset_sha256"] = sha256_file(
            temporary / "training_dataset.csv"
        )
        manifest["split_assignments_sha256"] = sha256_file(
            temporary / "split_assignments.csv"
        )
        if source_hash != sha256_file(source_path):
            raise RuntimeError("Source dataset changed during derivation")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen family split to the silver model dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_model_dataset(
        source_path=args.source,
        split_config=args.split_config,
        output_dir=args.output_dir,
    )
    print(f"Model dataset: {args.output_dir.resolve()}")
    print(json.dumps(manifest["split_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
