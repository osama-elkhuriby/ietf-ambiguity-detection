"""Build and verify immutable Dataset-A manifests for Experiment 1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    git_commit,
    immutable_directory,
    read_json,
    repo_path,
    resolve_path,
    sha256_file,
    utc_now,
    write_json,
)
from .config import (
    DEFAULT_CONFIG,
    DNS_FAMILY,
    LABEL_MAP,
    TCP_FAMILY,
    TRAIN_FAMILIES,
    load_config,
)
from .leakage import audit_leakage, build_fold_manifests

DATASET_COLUMNS = (
    "item_id",
    "record_id",
    "family_id",
    "document_id",
    "revision",
    "clause_ids",
    "section_ids",
    "lineage_ids",
    "target_clause",
    "section_heading",
    "previous_clause",
    "next_clause",
    "context_source_ids",
    "context_resolution_status",
    "alignment_source_path",
    "source_hashes",
    "label",
)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    except ValueError as error:
        raise RuntimeError(
            f"Required columns are missing from {path}: {error}"
        ) from error


def _unique(frame: pd.DataFrame, column: str, source: str) -> None:
    if (frame[column] == "").any():
        raise RuntimeError(f"{source} contains blank {column}")
    duplicated = (
        frame.loc[frame[column].duplicated(keep=False), column].unique().tolist()
    )
    if duplicated:
        raise RuntimeError(
            f"{source} contains duplicate {column} values: {duplicated[:5]}"
        )


def assemble_dataset(
    source: pd.DataFrame,
    key: pd.DataFrame,
    metadata: pd.DataFrame,
    context: pd.DataFrame,
    *,
    expected_unresolved_item_ids: set[str],
    family_to_fold: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strictly join approved metadata and expose labels only for training families."""
    source = source.copy()
    key = key.copy()
    metadata = metadata.copy()
    context = context.copy()
    for frame in (source, key, metadata, context):
        frame.fillna("", inplace=True)
    source["item_id"] = source["item_id"].astype(str)
    key["item_id"] = key["item_id"].astype(str)
    _unique(source, "item_id", "Dataset A")
    _unique(key, "item_id", "private key")
    _unique(key, "record_id", "private key")
    _unique(metadata, "record_id", "metadata")
    _unique(context, "record_id", "canonical context")

    unresolved_mask = (source["final_status"].str.casefold() != "resolved") | (
        source["final_source"].str.casefold() == "unresolved"
    )
    unresolved_ids = set(source.loc[unresolved_mask, "item_id"].astype(str))
    if unresolved_ids != expected_unresolved_item_ids:
        raise RuntimeError(
            "Unexpected unresolved provenance records: "
            f"expected {sorted(expected_unresolved_item_ids)}, got {sorted(unresolved_ids)}"
        )

    joined = source.drop(columns=["final_label"]).merge(
        key[["item_id", "record_id", "family_id"]],
        on="item_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if set(joined["_merge"]) != {"both"}:
        missing = joined.loc[joined["_merge"] != "both", "item_id"].tolist()
        raise RuntimeError(
            f"Dataset A IDs are missing from the private key: {missing[:5]}"
        )
    joined.drop(columns="_merge", inplace=True)
    _unique(joined, "record_id", "joined Dataset A")

    joined = joined.merge(
        metadata,
        on=["record_id", "family_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if set(joined["_merge"]) != {"both"}:
        missing = joined.loc[joined["_merge"] != "both", "record_id"].tolist()
        raise RuntimeError(f"Approved metadata is missing records: {missing[:5]}")
    joined.drop(columns="_merge", inplace=True)
    if not (joined["document"] == joined["document_id"]).all():
        raise RuntimeError(
            "Document identities disagree between Dataset A and approved metadata"
        )
    if not (joined["target_clause"] == joined["clause_text"]).all():
        raise RuntimeError(
            "Target text disagrees between Dataset A and approved metadata"
        )

    context_columns = [
        "record_id",
        "family_id",
        "target_clause",
        "section_heading",
        "previous_clause",
        "next_clause",
        "context_source_ids",
        "context_resolution_status",
        "alignment_source_path",
        "source_hashes",
    ]
    renamed_context = context[context_columns].rename(
        columns={
            "target_clause": "canonical_target_clause",
            "section_heading": "canonical_section_heading",
            "previous_clause": "canonical_previous_clause",
            "next_clause": "canonical_next_clause",
        }
    )
    joined = joined.merge(
        renamed_context,
        on=["record_id", "family_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if set(joined["_merge"]) != {"both"}:
        missing = joined.loc[joined["_merge"] != "both", "record_id"].tolist()
        raise RuntimeError(f"Canonical context is missing records: {missing[:5]}")
    joined.drop(columns="_merge", inplace=True)
    if not (joined["target_clause"] == joined["canonical_target_clause"]).all():
        raise RuntimeError("Canonical context target text does not match Dataset A")

    known = set(TRAIN_FAMILIES) | {TCP_FAMILY, DNS_FAMILY}
    observed = set(joined["family_id"])
    if observed != known:
        raise RuntimeError(
            f"Unexpected family membership: missing={sorted(known - observed)}, extra={sorted(observed - known)}"
        )
    reserved_counts = {
        "tcp_rows": int((joined["family_id"] == TCP_FAMILY).sum()),
        "dns_rows": int((joined["family_id"] == DNS_FAMILY).sum()),
    }

    effective = joined.loc[joined["family_id"].isin(TRAIN_FAMILIES)].copy()
    unresolved_effective = effective.loc[
        effective["item_id"].isin(unresolved_ids), ["item_id", "record_id", "family_id"]
    ]
    effective = effective.loc[~effective["item_id"].isin(unresolved_ids)].copy()
    # Labels are joined only after TCP and DNS have been excluded.
    labels = source.loc[
        source["item_id"].isin(effective["item_id"]), ["item_id", "final_label"]
    ].copy()
    unexpected_labels = sorted(set(labels["final_label"]) - set(LABEL_MAP))
    if unexpected_labels:
        raise RuntimeError(
            f"Dataset A contains unexpected supervised labels: {unexpected_labels}"
        )
    labels["label"] = labels["final_label"].map(LABEL_MAP).astype(int)
    effective = effective.merge(
        labels[["item_id", "label"]], on="item_id", validate="one_to_one"
    )
    effective["section_heading"] = effective.pop("canonical_section_heading")
    effective["previous_clause"] = effective.pop("canonical_previous_clause")
    effective["next_clause"] = effective.pop("canonical_next_clause")
    effective.drop(
        columns=[
            "document",
            "clause_text",
            "canonical_target_clause",
            "final_status",
            "final_source",
        ],
        inplace=True,
    )
    effective["fold"] = effective["family_id"].map(family_to_fold)
    if effective["fold"].isna().any():
        raise RuntimeError("At least one training family has no fold assignment")
    effective["fold"] = effective["fold"].astype(int)
    effective.sort_values(
        "item_id", key=lambda values: values.astype(int), inplace=True
    )
    effective.reset_index(drop=True, inplace=True)
    _unique(effective, "record_id", "effective dataset")
    report = {
        "source_rows": len(source),
        "joined_rows": len(joined),
        "key_rows": len(key),
        "annotated_cohort_omissions": len(key) - len(source),
        "reserved_partition_counts": reserved_counts,
        "unresolved_exclusions": unresolved_effective.to_dict(orient="records"),
        "effective_rows": len(effective),
        "effective_labels": {
            str(key): int(value)
            for key, value in sorted(Counter(effective["label"]).items())
        },
    }
    return effective, report


def _load_sources(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    data = config["data"]
    source_path = resolve_path(data["source_csv"])
    if sha256_file(source_path) != data["source_sha256"]:
        raise RuntimeError("Dataset A hash mismatch; refusing to continue")
    source = _read_csv(
        source_path,
        [
            "item_id",
            "document",
            "target_clause",
            "final_label",
            "final_source",
            "final_status",
        ],
    )
    if len(source) != int(data["expected_source_rows"]):
        raise RuntimeError(
            f"Dataset A row count changed: expected {data['expected_source_rows']}, got {len(source)}"
        )
    key = _read_csv(
        resolve_path(data["private_key_csv"]), ["item_id", "record_id", "family_id"]
    )
    if len(key) - len(source) != int(data["expected_annotated_cohort_omissions"]):
        raise RuntimeError(
            "Annotated-cohort omission count changed; omitted candidates will not be restored"
        )
    metadata = _read_csv(
        resolve_path(data["metadata_csv"]),
        [
            "record_id",
            "family_id",
            "document_id",
            "revision",
            "clause_ids",
            "clause_text",
            "section_ids",
            "lineage_ids",
        ],
    )
    context = _read_csv(
        resolve_path(data["canonical_context_csv"]),
        [
            "record_id",
            "family_id",
            "target_clause",
            "section_heading",
            "previous_clause",
            "next_clause",
            "context_source_ids",
            "context_resolution_status",
            "alignment_source_path",
            "source_hashes",
        ],
    )
    fold_path = resolve_path(config["folds"]["fold_source"])
    if sha256_file(fold_path) != config["folds"]["fold_source_sha256"]:
        raise RuntimeError("Historical fold-source hash mismatch")
    payload = read_json(fold_path)
    mapping = {
        str(family): int(fold)
        for family, fold in payload["family_to_fold"].items()
        if family in TRAIN_FAMILIES
    }
    if set(mapping) != set(TRAIN_FAMILIES) or set(mapping.values()) != set(range(1, 6)):
        raise RuntimeError("Five-fold family assignment is incomplete")
    return source, key, metadata, context, mapping


def prepare_dataset(
    config_path: Path | str = DEFAULT_CONFIG, output: Path | None = None
) -> Path:
    config = load_config(config_path)
    source, key, metadata, context, family_to_fold = _load_sources(config)
    frame, validation = assemble_dataset(
        source,
        key,
        metadata,
        context,
        expected_unresolved_item_ids=set(
            config["data"]["expected_unresolved_item_ids"]
        ),
        family_to_fold=family_to_fold,
    )
    if (
        validation["annotated_cohort_omissions"]
        != config["data"]["expected_annotated_cohort_omissions"]
    ):
        raise RuntimeError("Candidate omission contract changed")
    audit = audit_leakage(
        frame.reset_index(drop=True),
        near_threshold=float(config["folds"]["near_duplicate_threshold"]),
    )
    if audit["training_blocked"]:
        raise RuntimeError("Leakage audit found an unhandled family/document overlap")
    folds = build_fold_manifests(frame, audit)
    destination = resolve_path(output or config["outputs"]["dataset_dir"])
    with immutable_directory(destination) as building:
        dataset_path = building / "dataset.csv"
        frame.loc[:, [*DATASET_COLUMNS, "fold"]].to_csv(
            dataset_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        write_json(building / "validation_report.json", validation)
        write_json(building / "leakage_audit.json", audit)
        fold_summaries: list[dict[str, Any]] = []
        for fold in folds:
            fold_number = int(fold["fold"])
            fold_path = building / "folds" / f"fold_{fold['fold']}.json"
            write_json(fold_path, fold)
            assignments = frame[["item_id", "record_id", "family_id", "fold"]].copy()
            exclusions = {
                item["record_id"]: "|".join(item["reasons"])
                for item in fold["fitting_exclusions"]
            }
            assignments["role"] = assignments["fold"].map(
                lambda value, current_fold=fold_number: (
                    "evaluation" if int(value) == current_fold else "fitting"
                )
            )
            assignments["included"] = ~assignments["record_id"].isin(exclusions)
            assignments.loc[assignments["role"] == "evaluation", "included"] = True
            assignments["exclusion_reason"] = (
                assignments["record_id"].map(exclusions).fillna("")
            )
            assignments.to_csv(
                building / "folds" / f"fold_{fold['fold']}_assignments.csv",
                index=False,
                lineterminator="\n",
            )
            fold_summaries.append(
                {
                    "fold": fold["fold"],
                    **fold["counts"],
                    "held_out_families": fold["held_out_families"],
                }
            )
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "git_commit": git_commit(),
            "experiment": config["experiment"],
            "config_path": repo_path(Path(config["_config_path"])),
            "config_sha256": config["_config_sha256"],
            "source_path": repo_path(resolve_path(config["data"]["source_csv"])),
            "original_dataset_sha256": config["data"]["source_sha256"],
            "approved_metadata_hashes": config["data"]["approved_source_hashes"],
            "fold_source_path": config["folds"]["fold_source"],
            "fold_source_sha256": config["folds"]["fold_source_sha256"],
            "dataset_file": "dataset.csv",
            "effective_dataset_sha256": sha256_file(dataset_path),
            "dataset_columns": [*DATASET_COLUMNS, "fold"],
            "validation_report": "validation_report.json",
            "validation_report_sha256": sha256_file(
                building / "validation_report.json"
            ),
            "leakage_audit": "leakage_audit.json",
            "leakage_audit_sha256": sha256_file(building / "leakage_audit.json"),
            "unresolved_exclusions": validation["unresolved_exclusions"],
            "label_mapping": LABEL_MAP,
            "reserved_partitions": {
                "tcp": {
                    "family": TCP_FAMILY,
                    "use": "reserved_no_training_tuning_or_calibration",
                },
                "dns": {
                    "family": DNS_FAMILY,
                    "use": "disabled_incompatible_sealed_evaluation",
                },
            },
            "effective_summary": {
                "rows": len(frame),
                "labels": validation["effective_labels"],
                "families": sorted(frame["family_id"].unique()),
                "folds": fold_summaries,
            },
            "private_key_reporting": "Only hashes, join counts, and required stable identifiers are persisted; votes and private annotations are omitted.",
        }
        write_json(building / "manifest.json", manifest)
    return destination


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    manifest = read_json(dataset_dir / "manifest.json")
    dataset_path = dataset_dir / manifest["dataset_file"]
    if sha256_file(dataset_path) != manifest["effective_dataset_sha256"]:
        raise RuntimeError("Effective dataset hash mismatch")
    if (
        sha256_file(dataset_dir / manifest["validation_report"])
        != manifest["validation_report_sha256"]
    ):
        raise RuntimeError("Validation-report hash mismatch")
    if (
        sha256_file(dataset_dir / manifest["leakage_audit"])
        != manifest["leakage_audit_sha256"]
    ):
        raise RuntimeError("Leakage-audit hash mismatch")
    frame = pd.read_csv(
        dataset_path, dtype={"item_id": str, "record_id": str}, keep_default_na=False
    )
    if len(frame) != int(manifest["effective_summary"]["rows"]):
        raise RuntimeError("Effective dataset row count mismatch")
    if frame["record_id"].duplicated().any() or frame["item_id"].duplicated().any():
        raise RuntimeError("Effective dataset contains duplicated scoring IDs")
    if set(frame["family_id"]) != set(TRAIN_FAMILIES):
        raise RuntimeError("Reserved or unknown family found in effective dataset")
    if set(frame["label"].astype(int)) - {0, 1}:
        raise RuntimeError("Effective dataset labels are not binary")
    scoring_by_fold: list[str] = []
    for fold in range(1, 6):
        fold_manifest = read_json(dataset_dir / "folds" / f"fold_{fold}.json")
        scoring_by_fold.extend(fold_manifest["scoring_record_ids"])
        if set(fold_manifest["held_out_families"]) & set(
            fold_manifest["fitting_families"]
        ):
            raise RuntimeError(f"Fold {fold} mixes fitting and held-out families")
        if (
            fold_manifest["selection_policy"]["checkpoint_selection"]
            != "none_final_epoch_only"
        ):
            raise RuntimeError(f"Fold {fold} enables checkpoint selection")
    if Counter(scoring_by_fold) != Counter(frame["record_id"].astype(str)):
        raise RuntimeError(
            "Fold manifests do not score every effective record exactly once"
        )
    return manifest


def load_effective_dataset(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    dataset_dir = resolve_path(config["outputs"]["dataset_dir"])
    manifest = verify_dataset(dataset_dir)
    frame = pd.read_csv(
        dataset_dir / manifest["dataset_file"],
        dtype={"item_id": str, "record_id": str},
        keep_default_na=False,
    )
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame, manifest, dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    destination = resolve_path(args.output or config["outputs"]["dataset_dir"])
    if args.verify_existing:
        manifest = verify_dataset(destination)
        print(json.dumps(manifest["effective_summary"], indent=2, sort_keys=True))
    else:
        created = prepare_dataset(args.config, args.output)
        print(f"Immutable Dataset-A artifacts written to {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
