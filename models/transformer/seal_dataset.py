"""Repackage audited inputs into isolated development and sealed-test artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT, require_training_ready

DEFAULT_SOURCE = ROOT / "corpus/processed/transformer_inputs/v3"
DEFAULT_OUTPUT_ROOT = ROOT / "corpus/processed/transformer_partitioned/v1"
DEVELOPMENT_SPLITS = ("train", "validation")


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified_split(source_dir: Path, manifest: dict[str, Any], split: str) -> Path:
    try:
        info = manifest["split_files"][split]
    except KeyError as error:
        raise RuntimeError(f"Source manifest is missing the {split!r} split") from error
    path = source_dir / info["path"]
    if sha256_file(path) != info["sha256"]:
        raise RuntimeError(f"Source {split} hash does not match its manifest")
    frame = pd.read_csv(path, usecols=["record_id", "family_id", "split"])
    if len(frame) != int(info["rows"]) or set(frame["split"]) != {split}:
        raise RuntimeError(f"Source {split} file violates its manifest")
    return path


def build_partitioned_artifacts(
    *, source_dir: Path = DEFAULT_SOURCE, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    """Atomically create a test-free development tree and a sealed test tree."""

    source_dir = source_dir.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Partitioned artifact is immutable and already exists: {output_root}"
        )
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require_training_ready(source_manifest)
    source_manifest_hash = sha256_file(source_manifest_path)
    verified = {
        split: _verified_split(source_dir, source_manifest, split)
        for split in (*DEVELOPMENT_SPLITS, "test")
    }

    temporary = output_root.with_name(
        f".{output_root.name}.building-{uuid.uuid4().hex}"
    )
    development_dir = temporary / "development"
    sealed_dir = temporary / "sealed_test"
    try:
        development_dir.mkdir(parents=True)
        sealed_dir.mkdir(parents=True)
        development_files: dict[str, dict[str, Any]] = {}
        for split in DEVELOPMENT_SPLITS:
            destination = development_dir / f"{split}.csv"
            shutil.copyfile(verified[split], destination)
            source_info = source_manifest["split_files"][split]
            development_files[split] = {
                "path": destination.name,
                "rows": int(source_info["rows"]),
                "sha256": sha256_file(destination),
            }
        development_manifest = {
            "schema_version": 1,
            "artifact_role": "development_only",
            "created_at": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": source_manifest_hash,
            "training_blocked": False,
            "leakage_audit_counts": source_manifest["leakage_audit_counts"],
            "split_files": development_files,
            "permitted_splits": list(DEVELOPMENT_SPLITS),
        }
        _json(development_dir / "manifest.json", development_manifest)

        sealed_path = sealed_dir / "test.csv"
        shutil.copyfile(verified["test"], sealed_path)
        test_info = source_manifest["split_files"]["test"]
        sealed_manifest = {
            "schema_version": 1,
            "artifact_role": "sealed_final_test_only",
            "created_at": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": source_manifest_hash,
            "family_id": "draft-ietf-dnsop-svcb-https",
            "sealed_file": {
                "path": sealed_path.name,
                "rows": int(test_info["rows"]),
                "sha256": sha256_file(sealed_path),
            },
        }
        _json(sealed_dir / "manifest.json", sealed_manifest)

        partition_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": source_manifest_hash,
            "development_manifest_sha256": sha256_file(
                development_dir / "manifest.json"
            ),
            "sealed_manifest_sha256": sha256_file(sealed_dir / "manifest.json"),
            "split_assignment_sha256": source_manifest["split_assignments_sha256"],
        }
        _json(temporary / "manifest.json", partition_manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return partition_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_partitioned_artifacts(
        source_dir=args.source_dir, output_root=args.output_root
    )
    print(f"Partitioned transformer artifact: {args.output_root.resolve()}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
