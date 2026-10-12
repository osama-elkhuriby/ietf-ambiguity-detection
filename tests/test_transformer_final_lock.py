from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from models.classical.feature_pipeline import sha256_file
from models.transformer.final_evaluate import acquire_final_lock, load_sealed_test, run


def test_final_lock_refuses_second_claim(tmp_path: Path) -> None:
    lock = tmp_path / "final.lock.json"
    acquire_final_lock(lock, {"checkpoint_sha256": "first"})
    first = lock.read_bytes()
    with pytest.raises(RuntimeError, match="second run"):
        acquire_final_lock(lock, {"checkpoint_sha256": "second"})
    assert lock.read_bytes() == first


def test_final_evaluator_requires_explicit_final_flag_before_access() -> None:
    with pytest.raises(PermissionError, match="--final"):
        run(Namespace(final=False))


def test_sealed_loader_verifies_hash_and_role(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    frame = pd.DataFrame(
        {"record_id": ["dns-1"], "family_id": ["dns"], "split": ["test"]}
    )
    path = sealed / "test.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    manifest = {
        "artifact_role": "sealed_final_test_only",
        "sealed_file": {"path": "test.csv", "rows": 1, "sha256": sha256_file(path)},
    }
    (sealed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _ = load_sealed_test(sealed)
    assert loaded["record_id"].tolist() == ["dns-1"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("corrupt")
    with pytest.raises(RuntimeError, match="hash"):
        load_sealed_test(sealed)


def test_development_modules_do_not_reference_sealed_test_path() -> None:
    root = Path(__file__).resolve().parents[1]
    development_modules = [
        root / "models/transformer/train.py",
        root / "models/transformer/experiment.py",
        root / "models/transformer/tokenization.py",
        root / "models/transformer/modeling.py",
        root / "models/transformer/lr_screen_analysis.py",
        root / "models/transformer/protocol_audit.py",
        root / "models/transformer/protocol_comparison.py",
        root / "models/transformer/feasibility_audit.py",
        root / "models/transformer/ceiling_audit.py",
        root / "models/transformer/weighting_analysis.py",
        root / "models/classical/tfidf_comparator.py",
    ]
    calibration_module = root / "models/transformer/calibration.py"
    if calibration_module.exists():
        development_modules.append(calibration_module)
    forbidden = "transformer_partitioned/v1/sealed_test"
    for path in development_modules:
        source = path.read_text(encoding="utf-8").replace("\\", "/")
        assert forbidden not in source, path
        assert "load_sealed_test" not in source, path
