"""Build immutable zero/few-shot inputs from final_annotation_ready.csv."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT, normalize_for_audit

DEFAULT_SOURCE = ROOT / "results/annotation/final_annotation_ready.csv"
DEFAULT_OUTPUT = ROOT / "corpus/processed/llm_final_annotation_ready/v1"
EXPECTED_SOURCE_SHA256 = (
    "c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712"
)
EXPECTED_UNRESOLVED_IDS = ("2161",)

TRAIN_FAMILIES = (
    "draft-ietf-6man-rfc2460bis",
    "draft-ietf-httpbis-cache-header",
    "draft-ietf-httpbis-http2bis",
    "draft-ietf-httpbis-semantics",
    "draft-ietf-quic-http",
    "draft-ietf-quic-recovery",
    "draft-ietf-quic-tls",
    "draft-ietf-quic-transport",
    "draft-ietf-tls-tls13",
)
TCP_FAMILY = "draft-ietf-tcpm-rfc793bis"
DNS_FAMILY = "draft-ietf-dnsop-svcb-https"

# Keep the reviewed family grouping used by the earlier experiments. Few-shot
# demonstrations are selected only from the fitting side of each grouped fold.
FAMILY_TO_FOLD = {
    "draft-ietf-6man-rfc2460bis": 4,
    "draft-ietf-httpbis-cache-header": 4,
    "draft-ietf-httpbis-http2bis": 4,
    "draft-ietf-httpbis-semantics": 2,
    "draft-ietf-quic-http": 3,
    "draft-ietf-quic-recovery": 5,
    "draft-ietf-quic-tls": 5,
    "draft-ietf-quic-transport": 1,
    "draft-ietf-tls-tls13": 4,
}
REQUIRED_COLUMNS = (
    "item_id",
    "document",
    "section_heading",
    "previous_clause",
    "target_clause",
    "next_clause",
    "final_label",
    "final_source",
    "final_status",
)
OUTPUT_COLUMNS = (
    "record_id",
    "source_item_id",
    "family_id",
    "document_id",
    "revision",
    "split",
    "section_heading",
    "previous_clause",
    "target_clause",
    "next_clause",
    "label",
    "label_tier",
    "sample_weight",
    "final_source",
    "final_status",
    "source_row_sha256",
)
DOCUMENT_PATTERN = re.compile(r"^(?P<family>.+)-(?P<revision>\d+)$")


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _portable_source_path(source: Path) -> str:
    try:
        value = source.relative_to(ROOT)
    except ValueError:
        value = source
    return str(value).replace("\\", "/")


def _row_sha256(row: pd.Series) -> str:
    payload = {column: str(row[column]) for column in REQUIRED_COLUMNS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_document(document: str) -> tuple[str, int]:
    match = DOCUMENT_PATTERN.fullmatch(document.strip())
    if match is None:
        raise ValueError(f"Document has no numeric revision suffix: {document!r}")
    return match.group("family"), int(match.group("revision"))


def _partition_for_family(family: str) -> str:
    if family in TRAIN_FAMILIES:
        return "train"
    if family == TCP_FAMILY:
        return "validation"
    if family == DNS_FAMILY:
        return "test"
    raise ValueError(f"Unexpected document family: {family}")


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "positive_labels": int(frame["label"].sum()),
        "negative_labels": int((frame["label"] == 0).sum()),
        "positive_prevalence": float(frame["label"].mean()),
        "families": sorted(frame["family_id"].astype(str).unique()),
    }


def _cross_partition_overlap(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    normalized = {
        name: set(frame["target_clause"].map(normalize_for_audit))
        for name, frame in frames.items()
    }
    pairs: dict[str, int] = {}
    names = tuple(frames)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairs[f"{left}_vs_{right}"] = len(normalized[left] & normalized[right])
    return {
        "normalized_target_overlap": pairs,
        "all_clear": all(value == 0 for value in pairs.values()),
    }


def _fold_payload(train: pd.DataFrame, train_sha256: str) -> dict[str, Any]:
    if set(train["family_id"]) != set(FAMILY_TO_FOLD):
        raise RuntimeError("Development families do not match the frozen fold map")
    summary: dict[str, Any] = {}
    for fold in range(1, 6):
        held_out = train["family_id"].map(FAMILY_TO_FOLD).eq(fold)
        subset = train.loc[held_out]
        positives = int(subset["label"].sum())
        if subset.empty or positives == 0 or positives == len(subset):
            raise RuntimeError(f"Fold {fold} does not contain both labels")
        summary[str(fold)] = {
            "rows": len(subset),
            "positive_labels": positives,
            "families": sorted(subset["family_id"].unique()),
        }
    return {
        "schema_version": 1,
        "name": "final-annotation-ready-grouped-five-fold-v1",
        "method": "frozen document-family assignment",
        "n_splits": 5,
        "train_split_sha256": train_sha256,
        "family_to_fold": FAMILY_TO_FOLD,
        "fold_summary_before_duplicate_exclusion": summary,
    }


def _write_partition(
    directory: Path,
    frame: pd.DataFrame,
    *,
    filename: str,
    artifact_role: str,
    source_sha256: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    path = directory / filename
    frame.to_csv(path, index=False, lineterminator="\n")
    split = str(frame["split"].iloc[0])
    manifest = {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "source_annotation_sha256": source_sha256,
        "training_blocked": True,
        "parameter_updates_permitted": False,
        "permitted_splits": [split],
        "split_files": {
            split: {
                "path": filename,
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
        },
        "label_counts": _counts(frame),
    }
    _json(directory / "manifest.json", manifest)
    return manifest


def build_final_annotation_dataset(
    *,
    source: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT,
    expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
    expected_unresolved_ids: Iterable[str] = EXPECTED_UNRESOLVED_IDS,
) -> dict[str, Any]:
    """Validate the source and atomically build isolated train/TCP/DNS artifacts."""

    source = source.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Dataset output is immutable: {output_root}")
    source_sha256 = sha256_file(source)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "final_annotation_ready.csv changed: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )

    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Source is missing columns: {sorted(missing)}")
    if raw.empty or raw["item_id"].duplicated().any():
        raise ValueError("Source must contain unique non-empty item IDs")
    if raw["item_id"].str.strip().eq("").any():
        raise ValueError("Source contains a blank item ID")
    if raw["target_clause"].str.strip().eq("").any():
        raise ValueError("Source contains a blank target clause")
    if set(raw["final_label"]) - {"Yes", "No"}:
        raise ValueError("Source labels must be exactly Yes or No")

    unresolved = raw.loc[
        (raw["final_status"] != "resolved") | (raw["final_source"] == "unresolved")
    ].copy()
    observed_unresolved = tuple(sorted(unresolved["item_id"].astype(str)))
    required_unresolved = tuple(sorted(str(value) for value in expected_unresolved_ids))
    if observed_unresolved != required_unresolved:
        raise RuntimeError(
            "Unresolved-row contract changed: "
            f"expected {required_unresolved}, got {observed_unresolved}"
        )
    resolved = raw.drop(index=unresolved.index).copy()
    if set(resolved["final_status"]) != {"resolved"}:
        raise RuntimeError("Non-resolved rows remain after exclusion")

    parsed = resolved["document"].map(_parse_document)
    resolved["family_id"] = parsed.map(lambda value: value[0])
    resolved["revision"] = parsed.map(lambda value: value[1])
    resolved["split"] = resolved["family_id"].map(_partition_for_family)
    resolved["record_id"] = resolved["item_id"].map(
        lambda value: f"final-annotation-{value}"
    )
    resolved["source_item_id"] = resolved["item_id"]
    resolved["document_id"] = resolved["document"]
    resolved["label"] = resolved["final_label"].map({"No": 0, "Yes": 1})
    resolved["label_tier"] = resolved["final_source"]
    resolved["sample_weight"] = 1.0
    resolved["source_row_sha256"] = resolved.apply(_row_sha256, axis=1)

    prepared = resolved.loc[:, OUTPUT_COLUMNS].copy()
    if prepared["record_id"].duplicated().any():
        raise RuntimeError("Prepared record IDs are not unique")
    partitions = {
        split: prepared.loc[prepared["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    if any(frame.empty for frame in partitions.values()):
        raise RuntimeError("Every isolated partition must be non-empty")
    overlap = _cross_partition_overlap(partitions)
    if not overlap["all_clear"]:
        raise RuntimeError("Exact normalized target text crosses a reserved boundary")

    temporary = output_root.with_name(
        f".{output_root.name}.building-{uuid.uuid4().hex}"
    )
    try:
        temporary.mkdir(parents=True)
        development = _write_partition(
            temporary / "development",
            partitions["train"],
            filename="train.csv",
            artifact_role="development_zero_few_shot_evaluation_only",
            source_sha256=source_sha256,
        )
        tcp = _write_partition(
            temporary / "reserved_tcp",
            partitions["validation"],
            filename="validation.csv",
            artifact_role="reserved_tcp_validation_not_for_selection",
            source_sha256=source_sha256,
        )
        dns = _write_partition(
            temporary / "sealed_dns",
            partitions["test"],
            filename="test.csv",
            artifact_role="sealed_dns_final_test_only",
            source_sha256=source_sha256,
        )
        exclusions_path = temporary / "excluded_unresolved.csv"
        unresolved.loc[:, REQUIRED_COLUMNS].to_csv(
            exclusions_path, index=False, lineterminator="\n"
        )

        train_path = temporary / "development/train.csv"
        train_sha256 = sha256_file(train_path)
        folds_path = temporary / "development/folds_v1.json"
        _json(folds_path, _fold_payload(partitions["train"], train_sha256))
        manifest = {
            "schema_version": 1,
            "dataset_name": "final-annotation-ready-llm-in-context-v1",
            "source_path": _portable_source_path(source),
            "source_sha256": source_sha256,
            "source_rows": len(raw),
            "resolved_rows": len(resolved),
            "excluded_unresolved_item_ids": list(observed_unresolved),
            "excluded_unresolved_sha256": sha256_file(exclusions_path),
            "partition_counts": {
                "development_train": _counts(partitions["train"]),
                "reserved_tcp_validation": _counts(partitions["validation"]),
                "sealed_dns_test": _counts(partitions["test"]),
            },
            "cross_partition_audit": overlap,
            "artifacts": {
                "development_manifest_sha256": sha256_file(
                    temporary / "development/manifest.json"
                ),
                "development_train_sha256": development["split_files"]["train"][
                    "sha256"
                ],
                "fold_assignment_sha256": sha256_file(folds_path),
                "reserved_tcp_manifest_sha256": sha256_file(
                    temporary / "reserved_tcp/manifest.json"
                ),
                "reserved_tcp_sha256": tcp["split_files"]["validation"]["sha256"],
                "sealed_dns_manifest_sha256": sha256_file(
                    temporary / "sealed_dns/manifest.json"
                ),
                "sealed_dns_sha256": dns["split_files"]["test"]["sha256"],
            },
            "label_contract": {
                "column": "final_label",
                "mapping": {"No": 0, "Yes": 1},
                "provenance": "human annotation supported by two AI models",
                "human_annotation_involved": True,
                "independent_human_verification": False,
                "operational_reference_not_external_ground_truth": True,
            },
            "in_context_policy": {
                "parameter_updates": False,
                "input_fields": [
                    "previous_clause",
                    "target_clause",
                    "next_clause",
                ],
                "prohibited_input_fields": [
                    "label",
                    "label_tier",
                    "final_source",
                    "final_status",
                    "source_item_id",
                    "source_row_sha256",
                    "family_id",
                    "document_id",
                    "split",
                ],
                "tcp_opened_by_evaluation": False,
                "dns_opened_by_evaluation": False,
            },
        }
        _json(temporary / "manifest.json", manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def verify_final_annotation_dataset(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Verify every generated file against the immutable root manifest."""

    output_root = output_root.resolve()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source_path"])
    if not source.is_absolute():
        source = ROOT / source
    if sha256_file(source) != manifest["source_sha256"]:
        raise RuntimeError("Source annotation hash no longer matches the dataset")
    artifacts = manifest["artifacts"]
    checks = {
        "development_manifest_sha256": output_root / "development/manifest.json",
        "development_train_sha256": output_root / "development/train.csv",
        "fold_assignment_sha256": output_root / "development/folds_v1.json",
        "reserved_tcp_manifest_sha256": output_root / "reserved_tcp/manifest.json",
        "reserved_tcp_sha256": output_root / "reserved_tcp/validation.csv",
        "sealed_dns_manifest_sha256": output_root / "sealed_dns/manifest.json",
        "sealed_dns_sha256": output_root / "sealed_dns/test.csv",
    }
    for key, path in checks.items():
        if sha256_file(path) != artifacts[key]:
            raise RuntimeError(f"Dataset artifact hash mismatch: {path}")
    if sha256_file(output_root / "excluded_unresolved.csv") != manifest[
        "excluded_unresolved_sha256"
    ]:
        raise RuntimeError("Unresolved-row exclusion artifact hash mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_final_annotation_dataset(args.output)
        print(json.dumps(manifest["partition_counts"], indent=2, sort_keys=True))
        return 0
    manifest = build_final_annotation_dataset(source=args.source, output_root=args.output)
    print(f"Prepared dataset: {args.output.resolve()}")
    print(json.dumps(manifest["partition_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
