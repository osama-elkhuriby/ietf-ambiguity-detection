from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from experiments.transformer_final.method_specific_calibration_v1_1 import study
from models.transformer_final.common import read_json, sha256_file


def test_first_v1_artifact_passes_semantic_compatibility_validation() -> None:
    spec = read_json(study.V1_FIRST_SPEC_PATH)
    result = study.semantic_validate(spec)
    assert result["legacy_fold_file_sha256"] == study.LEGACY_FOLD_HASH
    assert result["custom_inner_split_sha256"] == study.CUSTOM_SPLIT_HASH
    assert result["fitting_rows"] == 910
    assert result["scoring_rows"] == result["prediction_rows"] == 439


def test_wrong_custom_hash_is_rejected_before_reuse() -> None:
    spec = copy.deepcopy(read_json(study.V1_FIRST_SPEC_PATH))
    spec["fold_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Custom inner-split"):
        study.semantic_validate(spec)


def test_wrong_legacy_hash_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = read_json(study.V1_FIRST_SPEC_PATH)
    original = study.read_json

    def altered(path: Path):
        value = original(path)
        if Path(path).resolve() == (study.V1_FIRST_RUN / "run_manifest.json").resolve():
            value = copy.deepcopy(value)
            value["fold_manifest_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(study, "read_json", altered)
    with pytest.raises(RuntimeError, match="Legacy fold-file hash"):
        study.semantic_validate(spec)


def test_altered_fold_membership_is_rejected() -> None:
    spec = copy.deepcopy(read_json(study.V1_FIRST_SPEC_PATH))
    moved = spec["scoring_record_ids"].pop()
    spec["fitting_record_ids"].append(moved)
    with pytest.raises(RuntimeError, match="Fitting/scoring identities"):
        study.semantic_validate(spec)


@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_duplicate_or_missing_prediction_ids_are_rejected(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    spec = read_json(study.V1_FIRST_SPEC_PATH)
    original = pd.read_csv

    def altered(*args, **kwargs):
        frame = original(*args, **kwargs)
        if Path(args[0]).resolve() == (
            study.V1_FIRST_RUN / "predictions" / "held_out_predictions.csv"
        ).resolve():
            if mode == "duplicate":
                frame.iloc[-1] = frame.iloc[0]
            else:
                frame = frame.iloc[:-1].copy()
        return frame

    monkeypatch.setattr(study.pd, "read_csv", altered)
    with pytest.raises(RuntimeError, match="Prediction identities"):
        study.semantic_validate(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fitting_rows", 909, "row counts"),
        ("loss", "weighted_ce", "Model configuration"),
    ],
)
def test_incorrect_counts_and_model_configuration_are_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    spec = read_json(study.V1_FIRST_SPEC_PATH)
    original = study.read_json

    def altered(path: Path):
        result = original(path)
        if Path(path).resolve() == (study.V1_FIRST_RUN / "run_manifest.json").resolve():
            result = copy.deepcopy(result)
            result[field] = value
        return result

    monkeypatch.setattr(study, "read_json", altered)
    with pytest.raises(RuntimeError, match=message):
        study.semantic_validate(spec)


def test_all_frozen_inner_splits_are_family_isolated_and_complete() -> None:
    frame, _, _, _ = study._dataset()
    audit = study._verify_reconstructed_splits(frame)
    assert len(audit) == 15
    indexed = frame.set_index("record_id")
    for split in study._inner_splits():
        fitting = set(
            indexed.loc[split["fitting_record_ids"], "family_id"].astype(str)
        )
        scoring = set(
            indexed.loc[split["scoring_record_ids"], "family_id"].astype(str)
        )
        assert not fitting & scoring
        assert not set(split["fitting_record_ids"]) & set(split["scoring_record_ids"])


def test_family_leakage_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _, _, _ = study._dataset()
    split = study._frozen_split(1, 1)
    altered = frame.copy()
    fitting_id = split["fitting_record_ids"][0]
    scoring_id = split["scoring_record_ids"][0]
    scoring_family = altered.loc[
        altered["record_id"] == scoring_id, "family_id"
    ].iloc[0]
    altered.loc[altered["record_id"] == fitting_id, "family_id"] = scoring_family
    monkeypatch.setattr(study, "_FROZEN_SPLIT_AUDIT", None)
    with pytest.raises(RuntimeError, match="family|Family"):
        study._verify_reconstructed_splits(altered)


def test_historical_immutability_rejects_incorrect_hash() -> None:
    path = study.V1_FAILURE_PATH.resolve()
    with pytest.raises(RuntimeError, match="Preserved V1 artifact changed"):
        study._verify_v1_preserved({str(path): "0" * 64})
    study._verify_v1_preserved({str(path): sha256_file(path)})


def test_v1_failure_and_training_completion_remain_distinct() -> None:
    failure = read_json(study.V1_FAILURE_PATH)
    manifest = read_json(study.V1_FIRST_RUN / "run_manifest.json")
    assert failure["status"] == "FAILED_VALIDATION"
    assert manifest["epochs_completed"] == 5
    assert manifest["runtime_seconds"] > 0


def test_reuse_spec_cannot_be_launched() -> None:
    if not study.STATUS_PATH.is_file():
        pytest.skip("V1.1 package has not been prepared yet")
    status = read_json(study.STATUS_PATH)
    record = next(value for value in status["cells"].values() if value["reuse_by_reference"])
    with pytest.raises(RuntimeError, match="must never be launched"):
        study.run_one(Path(record["spec_path"]))


def test_attempt_identities_and_new_output_paths_are_unique() -> None:
    if not study.STATUS_PATH.is_file():
        pytest.skip("V1.1 package has not been prepared yet")
    status = read_json(study.STATUS_PATH)
    specs = [
        read_json(Path(value["spec_path"])) for value in status["cells"].values()
    ]
    attempts = [spec["attempt_id"] for spec in specs]
    new_paths = [spec["run_path"] for spec in specs if not spec["reuse_by_reference"]]
    assert len(attempts) == len(set(attempts)) == 30
    assert len(new_paths) == len(set(new_paths)) == 29


def test_run_all_stops_immediately_on_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_status = {
        "cells": {
            "one": {"sequence": 1, "status": "PLANNED", "attempt_id": "one", "spec_path": "one.json"},
            "two": {"sequence": 2, "status": "PLANNED", "attempt_id": "two", "spec_path": "two.json"},
        }
    }
    calls: list[str] = []
    monkeypatch.setattr(study, "validate_pretraining", lambda: {"status": "PASS"})
    monkeypatch.setattr(study, "read_json", lambda path: fake_status)

    def fail(path: Path):
        calls.append(str(path))
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(study, "run_one", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        study.run_all()
    assert calls == ["one.json"]
