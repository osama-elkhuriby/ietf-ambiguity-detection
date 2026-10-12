from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from models.transformer.experiment import load_development_split
from models.transformer.seal_dataset import build_partitioned_artifacts


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "splits").mkdir(parents=True)
    split_files = {}
    for split in ("train", "validation", "test"):
        frame = pd.DataFrame(
            {
                "record_id": [f"{split}-1"],
                "family_id": [f"family-{split}"],
                "split": [split],
                "target_clause": [f"{split} text"],
                "previous_clause": [""],
                "next_clause": [""],
                "label": [int(split == "test")],
                "label_tier": ["HIGH"],
                "sample_weight": [1.0],
            }
        )
        path = source / "splits" / f"{split}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        split_files[split] = {
            "path": f"splits/{split}.csv",
            "rows": 1,
            "sha256": _hash(path),
        }
    manifest = {
        "training_blocked": False,
        "split_files": split_files,
        "split_assignments_sha256": "assignment-hash",
        "leakage_audit_counts": {
            "exact_cross_split_duplicates": 0,
            "near_cross_split_duplicates": 0,
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_partitioned_artifact_has_no_test_path_in_development_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partitioned"
    build_partitioned_artifacts(source_dir=_source(tmp_path), output_root=output)

    development = output / "development"
    manifest_text = (development / "manifest.json").read_text(encoding="utf-8")
    assert '"test"' not in manifest_text
    assert not (development / "test.csv").exists()
    assert len(load_development_split(development, "train")) == 1
    assert (output / "sealed_test/test.csv").is_file()
    assert _hash(output / "sealed_test/test.csv") == json.loads(
        (output / "sealed_test/manifest.json").read_text(encoding="utf-8")
    )["sealed_file"]["sha256"]


def test_partitioned_artifact_is_immutable(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "partitioned"
    build_partitioned_artifacts(source_dir=source, output_root=output)
    with pytest.raises(FileExistsError, match="immutable"):
        build_partitioned_artifacts(source_dir=source, output_root=output)
