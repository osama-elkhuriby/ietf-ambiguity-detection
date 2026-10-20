"""CPU-only regression tests for V1.2 recovery; never import model libraries."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.transformer_final.final_annotation_661_v1_2 import analysis, core
from experiments.transformer_final.final_annotation_661_v1_2.runner import build_training_inputs
from models.transformer_final.common import sha256_file


@pytest.fixture(scope="session")
def frozen() -> dict:
    return core.verify_frozen()


@pytest.fixture(scope="session")
def preflight_result() -> dict:
    return core.preflight()


def test_01_authoritative_dataset_identity() -> None:
    source = core.ROOT / "results" / "annotation" / "final_annotation.csv"
    assert sha256_file(source) == core.EXPECTED_DATASET_SHA256
    frame = pd.read_csv(core.DATASET_DIR / "dataset.csv")
    assert (len(frame), int(frame.label.sum())) == (2125, 136)


def test_02_historical_manifest_schema_and_authority() -> None:
    provenance = core._read_json(core.SOURCE_PROVENANCE_PATH)
    producer = provenance["authoritative_manifest_producer"]
    assert producer["path"] == "models/transformer_final/dataset.py"
    assert producer["function"] == "prepare_dataset"
    assert set(producer["fields"]) == core.MANIFEST_SCHEMA_FIELDS
    for outer in range(1, 6):
        assert set(core._read_json(core.INNER_DIR / f"outer_{outer}" / "manifest.json")) == core.MANIFEST_SCHEMA_FIELDS


def test_03_all_inner_manifests_exist_and_hash(frozen: dict) -> None:
    assert len(frozen["inner_datasets"]) == 5
    for item in frozen["inner_datasets"]:
        path = core.INNER_DIR / f"outer_{item['outer_fold']}" / "manifest.json"
        assert path.is_file()
        assert sha256_file(path) == item["manifest_sha256"]


def test_04_inner_datasets_equal_outer_effective_fitting_partitions() -> None:
    for outer in range(1, 6):
        root = core.INNER_DIR / f"outer_{outer}"
        frame = pd.read_csv(root / "dataset.csv", dtype={"record_id": str})
        parent = core._read_json(core.DATASET_DIR / "folds" / f"fold_{outer}.json")
        assert set(frame.record_id) == set(parent["fitting_record_ids"])
        report = core._read_json(root / "validation_report.json")
        assert report["effective_rows"] == len(parent["fitting_record_ids"])
        assert report["parent_candidate_rows"] - report["effective_rows"] == len(report["removed_record_ids"])


def test_05_all_15_inner_partitions() -> None:
    seen = 0
    for outer in range(1, 6):
        result = core.validate_inner_dataset(outer)
        assert set(result["fold_sha256"]) == {"1", "2", "3"}
        seen += len(result["fold_sha256"])
    assert seen == 15


def test_06_all_30_method_specific_inner_specs() -> None:
    specs = [spec for _, spec in core.iter_specs("inner")]
    assert len(specs) == 30
    assert {(s["method"], s["outer_fold"], s["inner_fold"]) for s in specs} == {
        (method, outer, inner)
        for method in core.METHODS for outer in range(1, 6) for inner in range(1, 4)
    }
    assert all(s["reuse_by_reference"] is False and s["automatic_retry"] is False for s in specs)


def test_07_historical_training_dataset_root_interface() -> None:
    _, spec = core.iter_specs("inner")[0]
    frame, manifest, split, dataset_root = build_training_inputs(spec)
    assert dataset_root == Path(spec["dataset_manifest_path"]).parent
    assert (dataset_root / "manifest.json").is_file()
    assert (dataset_root / "folds" / f"fold_{spec['train_fold']}.json").is_file()
    assert manifest["effective_dataset_sha256"] == sha256_file(dataset_root / "dataset.csv")
    assert set(split["fitting_record_ids"]) <= set(frame.record_id)
    assert set(split["scoring_record_ids"]) <= set(frame.record_id)


def test_08_preflight_covers_every_inner_cell(preflight_result: dict) -> None:
    assert preflight_result["status"] == "pass"
    assert preflight_result["inner_partitions"] == 15
    assert preflight_result["inner_execution_specs"] == 30
    assert preflight_result["training_executed"] is False


def test_09_outer_reuse_manifest_and_references(preflight_result: dict) -> None:
    reuse = core._read_json(core.REUSE_PATH)
    assert preflight_result["valid_reused_outer_cells"] == 10
    assert reuse["compatible_cells"] == 10
    assert len(reuse["cells"]) == 10
    for _, spec in core.iter_specs("outer"):
        assert spec["reuse_by_reference"] is True
        assert core.V11_ROOT in Path(spec["run_path"]).parents


def test_10_outer_predictions_and_completion_manifests_are_valid() -> None:
    results = core.validate_cells("outer", require_all=True)
    assert len(results) == 10
    assert {item["status"] for item in results} == {"valid_reused"}


def test_11_protocol_and_source_fingerprint_integrity(frozen: dict) -> None:
    record = core._read_json(core.FROZEN_PATH)
    assert frozen["protocol_sha256"] == record["protocol_sha256"] == sha256_file(core.PROTOCOL_PATH)
    assert frozen["source_snapshot_sha256"] == record["source_snapshot_sha256"]
    assert sha256_file(core.SOURCE_PROVENANCE_PATH) == record["source_provenance_sha256"]


def test_12_failed_attempt_is_immutable_and_incomplete() -> None:
    immutable = core._read_json(core.IMMUTABILITY_PATH)
    assert immutable["failed_attempt_file_count"] == 11
    core._assert_hashes(immutable["failed_attempt_files"], "failure")
    assert not (core.V11_ROOT / "completion_manifests" / f"{core.FAILED_CELL_ID}.json").exists()
    failed_model = [path for path in immutable["failed_attempt_files"] if path.endswith("model.safetensors")]
    assert len(failed_model) == 1 and (core.ROOT / failed_model[0]).stat().st_size > 700_000_000


def test_13_no_training_or_inference_in_cpu_control_modules() -> None:
    for name in ("core.py", "analysis.py"):
        source = (core.EXPERIMENT_ROOT / name).read_text(encoding="utf-8")
        imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = " ".join(ast.unparse(node) for node in imports)
        assert "torch" not in imported and "transformers" not in imported
    cli_source = (core.EXPERIMENT_ROOT / "cli.py").read_text(encoding="utf-8")
    assert '"train-outer"' not in cli_source


def test_14_inner_cells_remain_pending_and_never_reused() -> None:
    results = core.validate_cells("inner", require_all=False)
    assert len(results) == 30
    assert {item["status"] for item in results} == {"pending"}
    assert all(not Path(spec["completion_path"]).exists() for _, spec in core.iter_specs("inner"))


def test_15_analysis_pipeline_accepts_reused_outer_predictions() -> None:
    loaded = analysis._load_predictions("outer", "focal")
    assert set(loaded) == {1, 2, 3, 4, 5}
    dataset = analysis._load_dataset()
    assert len(dataset) == 2125
    assert sum(len(frame) for frame in loaded.values()) == 2125


def test_16_top20_error_report_helpers_are_deterministic() -> None:
    frame = pd.DataFrame({"item_id": ["3", "1", "2"], "record_id": ["c", "a", "b"], "probability_yes": [0.5, 0.9, 0.9]})
    ranked = analysis.stable_top_k(frame, "probability_yes", 2)
    assert ranked.item_id.tolist() == ["1", "2"]
    cdf = analysis.empirical_cdf([0.1, 0.2, 0.3], [0.05, 0.2, 0.4])
    assert np.allclose(cdf, [0.0, 2 / 3, 1.0])

