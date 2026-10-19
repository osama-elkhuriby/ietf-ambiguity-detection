from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from models.transformer_final.common import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = (
    ROOT
    / "experiments"
    / "transformer_final"
    / "recall_improvement_study_v3_optimizer_fix"
)
PACKAGE_ROOT = V3_ROOT / "manual_execution_v1"
EXPECTED_PROTOCOL = "9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51"
EXPECTED_SOURCE = "22af9f00506241a4d78918b3b1fc65f1a787281a685c966fc881df99671ee8c4"
EXPECTED_FAILED_LOG = "bf3dd00e4d6e39ce920862e56c04b5c0952e807cd8cc82b6b1a786865d6f8f0b"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_package_has_unique_one_cell_specs_and_outputs() -> None:
    package = read_json(PACKAGE_ROOT / "PACKAGE_MANIFEST.json")
    inventory = read_json(V3_ROOT / "EXPERIMENT_INVENTORY.json")
    assert package["protocol_sha256"] == EXPECTED_PROTOCOL
    assert package["source_snapshot_sha256"] == EXPECTED_SOURCE
    assert package["cell_count"] == 15
    assert package["one_cell_per_process"] is True
    assert package["automatic_retry"] is False
    assert package["automatic_multi_cell_execution"] is False
    assert len(package["execution_order"]) == len(set(package["execution_order"])) == 15

    attempts: set[str] = set()
    specs: set[Path] = set()
    runs: set[Path] = set()
    logs: set[Path] = set()
    authorizations: set[Path] = set()
    for cell_id in package["execution_order"]:
        record = inventory["attempts"][cell_id]
        spec_path = Path(record["spec_path"])
        cell = read_json(spec_path)
        auth_path = Path(record["authorization_path"])
        authorization = read_json(auth_path)
        assert sha256_file(spec_path) == package["specification_sha256"][cell_id]
        assert sha256_file(auth_path) == package["authorization_sha256"][cell_id]
        assert authorization["one_cell_only"] is True
        assert authorization["automatic_retry"] is False
        assert authorization["spec_sha256"] == sha256_file(spec_path)
        assert cell["protocol_sha256"] == EXPECTED_PROTOCOL
        assert cell["seed"] == 2718
        attempts.add(cell["attempt_id"])
        specs.add(spec_path.resolve())
        runs.add(Path(cell["run_path"]).resolve())
        logs.add(Path(cell["log_path"]).resolve())
        authorizations.add(auth_path.resolve())
    assert len(attempts) == len(specs) == len(runs) == len(logs) == 15
    assert len(authorizations) == 15


def test_manual_cpu_validator_preserves_frozen_protocol_and_failure() -> None:
    launcher = _load_module(
        "v3_manual_launcher_test", PACKAGE_ROOT / "manual_cell_launcher.py"
    )
    result = launcher.validate_package()
    assert result["protocol_sha256"] == EXPECTED_PROTOCOL
    assert result["source_snapshot_sha256"] == EXPECTED_SOURCE
    assert result["cuda_used"] is False
    assert result["cells"] == 15
    failed_spec = read_json(
        V3_ROOT / "cell_specs" / "partial4_fold1_optimizer_fix_attempt.json"
    )
    failed_log = Path(failed_spec["replaces_attempt"]["stderr_log_path"])
    assert sha256_file(failed_log) == EXPECTED_FAILED_LOG


def test_manual_execution_requires_environment_gate_without_mutating_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_module(
        "v3_manual_launcher_gate_test", PACKAGE_ROOT / "manual_cell_launcher.py"
    )
    monkeypatch.delenv("RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED", raising=False)
    spec_path = V3_ROOT / "cell_specs" / "manual-v1-partial4-fold2-seed2718.json"
    spec = read_json(spec_path)
    evidence = [
        Path(spec["run_path"]) / "run_manifest.json",
        Path(spec["log_path"]).with_suffix(".stdout.log"),
        Path(spec["log_path"]).with_suffix(".stderr.log"),
    ]
    before = {
        str(path): sha256_file(path) if path.is_file() else None for path in evidence
    }
    with pytest.raises(PermissionError, match="RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED"):
        launcher.run_spec(spec_path)
    after = {
        str(path): sha256_file(path) if path.is_file() else None for path in evidence
    }
    assert after == before


def test_analysis_metrics_include_required_counts_and_mean_fold_ap() -> None:
    analysis = _load_module(
        "v3_manual_analysis_test", PACKAGE_ROOT / "analyze_completed_v1.py"
    )
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "family_id": ["one"] * 4 + ["two"] * 4,
            "probability_yes": [0.9, 0.8, 0.7, 0.1, 0.6, 0.4, 0.3, 0.2],
        }
    )
    tables = analysis._analysis_tables({"synthetic": frame}, {1: 0.5, 2: 0.5})
    pooled = tables["pooled_metrics"].loc[
        tables["pooled_metrics"]["operating_point"] == "threshold_0.5"
    ].iloc[0]
    assert (pooled["tp"], pooled["fp"], pooled["tn"], pooled["fn"]) == (3, 1, 3, 1)
    assert pooled["precision"] == pytest.approx(0.75)
    assert pooled["recall"] == pytest.approx(0.75)
    assert pooled["f1"] == pytest.approx(0.75)
    assert pooled["f2"] == pytest.approx(0.75)
    assert np.isfinite(pooled["average_precision"])
    assert np.isfinite(pooled["roc_auc"])
    mean_fold = tables["mean_fold_metrics"]
    assert "mean_fold_average_precision" in mean_fold.columns
    assert set(tables) >= {
        "pooled_metrics",
        "fold_metrics",
        "mean_fold_metrics",
        "top_k_retrieval",
        "family_metrics",
    }
