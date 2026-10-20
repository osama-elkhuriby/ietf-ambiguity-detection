"""Analysis-only repair layered over an immutable frozen parent experiment.

The parent analysis algorithm is executed directly.  During that call only its
metadata merge, output writers, and final output validator are rebound.  The
bindings are restored even when analysis fails.  No training module, model, or
checkpoint is loaded here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import (
    analysis as parent_analysis,
)
from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import (
    core as parent_core,
)

REPAIR_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPAIR_ROOT / "outputs"
REPAIR_MANIFEST_PATH = REPAIR_ROOT / "ANALYSIS_REPAIR_MANIFEST.json"
PARENT_ROOT = parent_core.EXPERIMENT_ROOT
PARENT_IDENTIFIER = "final_annotation_human_reviewed_100_v1"
REPAIR_IDENTIFIER = "final_annotation_human_reviewed_100_v1_analysis_fix_v1"
IDENTITY_COLUMNS = ["item_id", "record_id", "family_id", "document_id", "revision"]

# These are aliases, not modified implementations.  Regression tests pin their
# behavior so the repair cannot drift ECDF or threshold-selection mathematics.
empirical_cdf = parent_analysis.empirical_cdf
select_precision_threshold = parent_analysis.select_precision_threshold


def merge_metadata(prediction: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative metadata while proving shared identities agree."""
    required = {*IDENTITY_COLUMNS, "label", "fold"}
    missing_dataset = sorted(required - set(dataset.columns))
    missing_prediction = sorted(required - set(prediction.columns))
    if missing_dataset or missing_prediction:
        raise RuntimeError(
            "Metadata merge schema mismatch: "
            f"dataset_missing={missing_dataset}, prediction_missing={missing_prediction}"
        )
    if (
        dataset.duplicated(IDENTITY_COLUMNS).any()
        or dataset["record_id"].duplicated().any()
    ):
        raise RuntimeError(
            "Authoritative dataset metadata identity keys are not unique"
        )
    if (
        prediction.duplicated(IDENTITY_COLUMNS).any()
        or prediction["record_id"].duplicated().any()
    ):
        raise RuntimeError("Prediction metadata identity keys are not unique")

    prediction = prediction.copy()
    metadata = dataset.drop(columns=["label", "fold"]).copy()
    merged = prediction.merge(
        metadata,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] != "both", "record_id"].astype(str).tolist()
    if unmatched:
        raise RuntimeError(
            "Prediction metadata disagree with the authoritative frozen dataset: "
            f"{unmatched[:10]}"
        )
    merged.drop(columns="_merge", inplace=True)

    authoritative = dataset.set_index("record_id")
    expected_labels = merged["record_id"].map(authoritative["label"]).astype(int)
    if not expected_labels.eq(merged["label"].astype(int)).all():
        raise RuntimeError(
            "Prediction labels disagree with authoritative development labels"
        )
    expected_folds = merged["record_id"].map(authoritative["fold"]).astype(int)
    if not expected_folds.eq(merged["fold"].astype(int)).all():
        raise RuntimeError(
            "Prediction folds disagree with authoritative frozen fold assignments"
        )
    merged["outer_fold"] = expected_folds
    return merged


def _write_output_csv(name: str, frame: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        OUTPUT_ROOT / name,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def _write_output_text(
    path: str | Path, value: str, *, exclusive: bool = False
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    parent_core._write_text(OUTPUT_ROOT / Path(path).name, value, exclusive=exclusive)


def _relative_parent(path: Path) -> str:
    return path.relative_to(parent_core.ROOT).as_posix()


def _verify_completion_hashes(spec: dict[str, Any]) -> dict[str, str]:
    completion_path = Path(spec["completion_path"])
    completion = parent_core._read_json(completion_path)
    paths = {
        "run_manifest_sha256": Path(spec["run_path"]) / "run_manifest.json",
        "predictions_sha256": parent_core.prediction_path(spec),
        "audit_sha256": Path(spec["audit_path"]),
    }
    if completion.get("protocol_sha256") != spec["protocol_sha256"]:
        raise RuntimeError(f"Completion protocol mismatch: {spec['cell_id']}")
    for field, path in paths.items():
        if not path.is_file() or parent_core.sha256_file(path) != completion.get(field):
            raise RuntimeError(
                f"Completion hash mismatch for {spec['cell_id']}: {field}"
            )
    return {
        "cell_id": spec["cell_id"],
        "completion_manifest": _relative_parent(completion_path),
        "prediction_file": _relative_parent(paths["predictions_sha256"]),
        "prediction_sha256": completion["predictions_sha256"],
    }


def _verify_repair_manifest(reused: list[dict[str, str]]) -> dict[str, Any]:
    manifest = parent_core._read_json(REPAIR_MANIFEST_PATH)
    if manifest.get("repair_version") != REPAIR_IDENTIFIER:
        raise RuntimeError("Wrong analysis-repair manifest version")
    if manifest.get("parent_experiment_identifier") != _relative_parent(PARENT_ROOT):
        raise RuntimeError("Analysis repair references the wrong parent experiment")
    if manifest.get("analysis_output_directory") != _relative_parent(OUTPUT_ROOT):
        raise RuntimeError("Analysis repair output directory differs from its manifest")
    parent_protocol = parent_core.sha256_file(parent_core.PROTOCOL_PATH)
    if manifest.get("parent_protocol_sha256") != parent_protocol:
        raise RuntimeError("Analysis repair references the wrong parent protocol")
    frozen_record = parent_core._read_json(PARENT_ROOT / "FROZEN.json")
    if manifest.get("parent_dataset_sha256") != frozen_record.get("dataset_sha256"):
        raise RuntimeError("Analysis repair references the wrong authoritative dataset")
    if manifest.get("parent_effective_cohort_sha256") != frozen_record.get(
        "effective_dataset_sha256"
    ):
        raise RuntimeError("Analysis repair references the wrong effective cohort")
    original_analysis = PARENT_ROOT / "analysis.py"
    if manifest.get(
        "parent_original_analysis_source_sha256"
    ) != parent_core.sha256_file(original_analysis):
        raise RuntimeError(
            "Parent analysis source differs from the approved repair basis"
        )
    if manifest.get("repaired_analysis_source_sha256") != parent_core.sha256_file(
        Path(__file__)
    ):
        raise RuntimeError("Repaired analysis source fingerprint mismatch")
    listed = sorted(manifest.get("reused_prediction_manifests", []))
    actual = sorted(row["completion_manifest"] for row in reused)
    if listed != actual:
        raise RuntimeError(
            "Repair manifest does not list the exact reused completion manifests"
        )
    return manifest


@lru_cache(maxsize=1)
def verify_parent_artifacts() -> dict[str, Any]:
    """Validate the immutable parent and all reused prediction evidence."""
    frozen = parent_core.verify_frozen()
    valid: dict[str, list[dict[str, Any]]] = {"outer": [], "inner": []}
    reused: list[dict[str, str]] = []
    for _, spec in parent_core.iter_specs():
        valid[spec["kind"]].append(parent_core.validate_cell(spec))
        reused.append(_verify_completion_hashes(spec))
    if len(valid["outer"]) != 10 or len(valid["inner"]) != 30:
        raise RuntimeError("Expected exactly 10 valid outer and 30 valid inner cells")

    dataset = parent_analysis._load_dataset()
    if len(dataset) != 2143 or dataset["record_id"].duplicated().any():
        raise RuntimeError(
            "Frozen effective cohort is not the expected 2,143 unique records"
        )
    if dataset["family_id"].nunique() != 9:
        raise RuntimeError(
            "Frozen effective cohort does not contain exactly nine families"
        )

    coverage: dict[str, set[str]] = {}
    for method in parent_core.METHODS:
        raw = parent_analysis._load_predictions("outer", method)
        pooled_parts: list[pd.DataFrame] = []
        for fold in range(1, 6):
            frame = merge_metadata(raw[fold], dataset)
            fold_manifest = parent_core._read_json(
                parent_core.DATASET_DIR / "folds" / f"fold_{fold}.json"
            )
            if set(frame["record_id"].astype(str)) != set(
                fold_manifest["scoring_record_ids"]
            ):
                raise RuntimeError(
                    f"Outer fold record mismatch for {method}, fold {fold}"
                )
            if set(frame["family_id"].astype(str)) != set(
                fold_manifest["held_out_families"]
            ):
                raise RuntimeError(
                    f"Outer fold family mismatch for {method}, fold {fold}"
                )
            if not frame["outer_fold"].eq(fold).all():
                raise RuntimeError(
                    f"Outer fold assignment mismatch for {method}, fold {fold}"
                )
            pooled_parts.append(frame)
        pooled = pd.concat(pooled_parts, ignore_index=True)
        if len(pooled) != 2143 or pooled["record_id"].duplicated().any():
            raise RuntimeError(
                f"Outer OOF coverage is not unique and complete for {method}"
            )
        coverage[method] = set(pooled["record_id"].astype(str))
    if coverage["focal"] != coverage["partial4"] or coverage["focal"] != set(
        dataset["record_id"].astype(str)
    ):
        raise RuntimeError("Individual-model outer OOF record coverage differs")

    manifest = _verify_repair_manifest(reused)
    return {
        "status": "PASS",
        "parent_protocol": frozen,
        "outer_complete_valid": len(valid["outer"]),
        "inner_complete_valid": len(valid["inner"]),
        "reused_prediction_manifests": len(reused),
        "pooled_records_per_individual_model": 2143,
        "development_families": 9,
        "repair_version": manifest["repair_version"],
        "analysis_output_directory": str(OUTPUT_ROOT),
    }


def validate_analysis() -> dict[str, Any]:
    """Validate repaired outputs without reading or writing parent result files."""
    verify_parent_artifacts()
    dataset = parent_analysis._load_dataset().set_index("record_id")
    required_tables = [
        "OUTER_FOLD_METRICS.csv",
        "INNER_FOLD_CALIBRATION.csv",
        "SELECTED_THRESHOLDS.csv",
        "FINAL_NESTED_METRICS.csv",
        "ENSEMBLE_VALIDATION.csv",
        "TOP_K_COMPARISON.csv",
        "FAMILY_ROBUSTNESS.csv",
        "UNCERTAINTY_INTERVALS.csv",
        "GPU_TRAINING_TIME.csv",
    ]
    for name in required_tables:
        if not (OUTPUT_ROOT / name).is_file():
            raise FileNotFoundError(f"Missing repaired analysis artifact: {name}")

    family = pd.read_csv(OUTPUT_ROOT / "FAMILY_ROBUSTNESS.csv", keep_default_na=False)
    if len(family) != 27 or set(family["rfc_family"]) != set(dataset["family_id"]):
        raise RuntimeError("Repaired family-robustness coverage is incomplete")

    checked_errors = 0
    checked_disagreements = 0
    for method in (*parent_core.METHODS, "ensemble"):
        prefix = parent_analysis.DISPLAY[method]
        for error_name, error_type in (
            ("FALSE_POSITIVES", "false_positive"),
            ("FALSE_NEGATIVES", "false_negative"),
        ):
            path = OUTPUT_ROOT / f"{prefix}_TOP20_{error_name}.csv"
            frame = pd.read_csv(
                path, dtype={"item_id": str, "record_id": str}, keep_default_na=False
            )
            if frame.columns.tolist() != parent_analysis.ERROR_COLUMNS:
                raise RuntimeError(f"Error-report schema mismatch: {path.name}")
            if frame["record_id"].duplicated().any() or len(frame) > 20:
                raise RuntimeError(f"Duplicate/oversized error list: {path.name}")
            parent_analysis._assert_sorted(frame, error_type)
            for row in frame.itertuples(index=False):
                source = dataset.loc[row.record_id]
                if (
                    row.target_clause != source.target_clause
                    or row.previous_clause != source.previous_clause
                    or row.next_clause != source.next_clause
                ):
                    raise RuntimeError(
                        f"Text truncation or drift in {path.name}: {row.record_id}"
                    )
                if row.rfc_family != source.family_id:
                    raise RuntimeError(f"Family drift in {path.name}")
                if error_type == "false_positive" and not (
                    row.actual_label == "No" and row.predicted_label == "Yes"
                ):
                    raise RuntimeError(f"Invalid FP semantics in {path.name}")
                if error_type == "false_negative" and not (
                    row.actual_label == "Yes" and row.predicted_label == "No"
                ):
                    raise RuntimeError(f"Invalid FN semantics in {path.name}")
            checked_errors += 1
        for suffix, label in (
            ("NO_HIGH_YES_SCORE", "No"),
            ("YES_LOW_YES_SCORE", "Yes"),
        ):
            path = OUTPUT_ROOT / f"{prefix}_TOP20_{suffix}.csv"
            frame = pd.read_csv(
                path, dtype={"item_id": str, "record_id": str}, keep_default_na=False
            )
            if (
                frame["record_id"].duplicated().any()
                or len(frame) > 20
                or set(frame["actual_label"]) - {label}
            ):
                raise RuntimeError(f"Invalid disagreement list: {path.name}")
            checked_disagreements += 1

    report = {
        "status": "PASS",
        "repair_version": REPAIR_IDENTIFIER,
        "validated_threshold_dependent_error_files": checked_errors,
        "validated_threshold_independent_disagreement_files": checked_disagreements,
        "authoritative_text_preservation": "PASS",
        "family_metadata_preservation": "PASS",
        "outer_oof_only": "PASS",
        "reserved_tcp_dns_absent": "PASS",
        "parent_protocol_integrity": "PASS",
        "outputs_isolated_from_parent": "PASS",
    }
    _write_output_text(
        OUTPUT_ROOT / "VALIDATION_REPORT.md",
        "# Validation report\n\n"
        + "\n".join(f"- {key}: {value}." for key, value in report.items())
        + "\n",
    )
    return report


def analyze(*, bootstrap_replicates: int | None = None) -> dict[str, Any]:
    """Run the unchanged parent analysis with the approved isolated repair."""
    verification = verify_parent_artifacts()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    originals = {
        "_merge_metadata": parent_analysis._merge_metadata,
        "_write_csv": parent_analysis._write_csv,
        "_write_text": parent_analysis._write_text,
        "validate_analysis": parent_analysis.validate_analysis,
    }
    try:
        parent_analysis._merge_metadata = merge_metadata
        parent_analysis._write_csv = _write_output_csv
        parent_analysis._write_text = _write_output_text
        parent_analysis.validate_analysis = validate_analysis
        result = parent_analysis.analyze(bootstrap_replicates=bootstrap_replicates)
    finally:
        for name, value in originals.items():
            setattr(parent_analysis, name, value)
    return {**result, "analysis_repair": verification}
