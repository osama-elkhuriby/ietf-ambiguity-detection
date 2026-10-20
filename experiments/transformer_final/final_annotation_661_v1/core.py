"""Preparation, governance, freezing, and cell validation for this experiment.

This module never imports Transformers and never starts training.  The only code
path allowed to train is runner.py, which requires an explicit train-* command.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.transformer_final.common import hash_directory, sha256_file, stable_json
from models.transformer_final.leakage import audit_leakage, build_fold_manifests

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
PROTOCOL_PATH = EXPERIMENT_ROOT / "EXPERIMENT_PROTOCOL.json"
SOURCE_PROVENANCE_PATH = EXPERIMENT_ROOT / "SOURCE_PROVENANCE.json"
DATASET_DIR = EXPERIMENT_ROOT / "manifests" / "dataset_v1"
INNER_DIR = EXPERIMENT_ROOT / "manifests" / "inner"
CELL_SPECS_DIR = EXPERIMENT_ROOT / "cell_specs"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
AUDITS_DIR = EXPERIMENT_ROOT / "audits"
LOGS_DIR = EXPERIMENT_ROOT / "logs"
FAILURES_DIR = EXPERIMENT_ROOT / "failure_manifests"
COMPLETIONS_DIR = EXPERIMENT_ROOT / "completion_manifests"

REQUIRED_SOURCE_COLUMNS = [
    "item_id",
    "document",
    "section_heading",
    "previous_clause",
    "target_clause",
    "next_clause",
    "final_label",
    "final_source",
    "final_status",
]
DATASET_COLUMNS = [
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
    "final_source",
    "final_status",
    "fold",
]
METHODS = ("focal", "partial4")
METHOD_SLUG = {"focal": "focal", "partial4": "partial4"}
EXPECTED_DATASET_SHA256 = (
    "8e395e83087c28cdc8e5b514f6fb4eca96e54a31fb525be770810db6f9e22e1d"
)
EXPECTED_MODEL_SHA256 = (
    "206ee7bb2c7805bcc8fd3bbbc18a09e185bafc19d4e112f31d4a088136717dd7"
)


def as_path(value: str | os.PathLike[str] | Path) -> Path:
    """Normalize path-like input (regression guard for the historical str bug)."""
    return Path(value).expanduser().resolve()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Any:
    with as_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_text(path: str | Path, value: str, *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def load_local_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    path = as_path(path)
    config = _read_json(path)
    if config.get("experiment") != "final_annotation_661_v1":
        raise RuntimeError("Wrong experiment configuration")
    if config.get("schema_version") != 1:
        raise RuntimeError("Unsupported configuration schema")
    data = config["data"]
    if data["source_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Configuration does not pin the authoritative dataset")
    training = config["training"]
    expected_training = {
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "scheduler": "linear",
        "warmup_ratio": 0.1,
        "epochs": 5,
        "early_stopping": False,
        "max_grad_norm": 1.0,
        "seeds": [2718],
        "precision": "fp32",
        "threshold": 0.5,
        "diagnostic_numerics": True,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError(f"Frozen training setting changed: {key}")
    if config["tokenization"] != {
        "input_mode": "target_only",
        "max_length": 256,
        "target_overflow": "deterministic_head_tail",
        "dynamic_padding": True,
        "add_vocabulary_tokens": False,
    }:
        raise RuntimeError("Target-only tokenization policy changed")
    if config["model"]["source_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Approved pretrained model fingerprint changed in config")
    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    return config


def _resolve_repo_path(value: str | Path) -> Path:
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _verify_static_sources(config: dict[str, Any], *, hash_model: bool) -> dict[str, Any]:
    source = _resolve_repo_path(config["data"]["source_csv"])
    if not source.is_file():
        raise FileNotFoundError(f"Authoritative dataset is missing: {source}")
    actual = sha256_file(source)
    if actual != config["data"]["source_sha256"]:
        raise RuntimeError(f"Authoritative dataset SHA-256 mismatch: {actual}")
    verified: dict[str, Any] = {"authoritative_dataset": actual}
    for key, expected in config["data"]["approved_source_hashes"].items():
        path = _resolve_repo_path(config["data"][key])
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Approved {key} SHA-256 mismatch: {actual}")
        verified[key] = actual
    fold_path = _resolve_repo_path(config["folds"]["fold_source"])
    fold_hash = sha256_file(fold_path)
    if fold_hash != config["folds"]["fold_source_sha256"]:
        raise RuntimeError("Frozen family-fold mapping changed")
    verified["fold_source"] = fold_hash
    model_path = _resolve_repo_path(config["model"]["default_local_path"])
    required = ("config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Approved offline model is incomplete: {missing}")
    verified["model_path"] = str(model_path)
    if hash_model:
        model_hash, model_files = hash_directory(model_path)
        if model_hash != config["model"]["source_sha256"]:
            raise RuntimeError(f"Approved model SHA-256 mismatch: {model_hash}")
        verified["model_source_sha256"] = model_hash
        verified["model_file_hashes"] = model_files
    return verified


def _read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            usecols=list(columns) if columns is not None else None,
            dtype=str,
            keep_default_na=False,
        )
    except ValueError as exc:
        raise RuntimeError(f"CSV schema mismatch for {path}: {exc}") from exc


def _unique_nonblank(frame: pd.DataFrame, column: str, source: str) -> None:
    if frame[column].astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{source} contains blank {column}")
    if frame[column].duplicated().any():
        values = frame.loc[frame[column].duplicated(keep=False), column].head().tolist()
        raise RuntimeError(f"{source} contains duplicate {column}: {values}")


def build_effective_dataset(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join governance metadata while exposing labels only after reserved exclusion."""
    data = config["data"]
    source_path = _resolve_repo_path(data["source_csv"])
    source = _read_csv(source_path)
    if source.columns.tolist() != REQUIRED_SOURCE_COLUMNS:
        raise RuntimeError(
            f"Authoritative schema mismatch: expected {REQUIRED_SOURCE_COLUMNS}, "
            f"got {source.columns.tolist()}"
        )
    source.fillna("", inplace=True)
    source["item_id"] = source["item_id"].astype(str)
    _unique_nonblank(source, "item_id", "authoritative dataset")
    if len(source) != int(data["expected_source_rows"]):
        raise RuntimeError(f"Expected 2406 source rows, found {len(source)}")
    labels = Counter(source["final_label"])
    if labels != Counter({"No": 2247, "Yes": 159}):
        raise RuntimeError(f"Unexpected authoritative labels: {dict(labels)}")
    unresolved = source.loc[
        (source["final_status"].str.casefold() != "resolved")
        | (source["final_source"].str.casefold() == "unresolved"),
        "item_id",
    ].tolist()
    if sorted(unresolved) != sorted(data["expected_unresolved_item_ids"]):
        raise RuntimeError(f"Unexpected unresolved records: {unresolved}")
    if source["target_clause"].str.strip().eq("").any():
        raise RuntimeError("Authoritative dataset has an empty target clause")

    key = _read_csv(
        _resolve_repo_path(data["private_key_csv"]),
        ["item_id", "record_id", "family_id"],
    )
    key.fillna("", inplace=True)
    key["item_id"] = key["item_id"].astype(str)
    _unique_nonblank(key, "item_id", "private key")
    _unique_nonblank(key, "record_id", "private key")
    # Deliberately omit final_label here.  Family governance is applied first.
    joined = source.drop(columns=["final_label"]).merge(
        key, on="item_id", how="left", validate="one_to_one", indicator=True
    )
    if set(joined["_merge"]) != {"both"}:
        missing = joined.loc[joined["_merge"] != "both", "item_id"].tolist()
        raise RuntimeError(f"Source IDs missing from governance key: {missing[:10]}")
    joined.drop(columns="_merge", inplace=True)

    known = set(config["partitions"]["train_families"]) | {
        config["partitions"]["tcp_family"],
        config["partitions"]["dns_family"],
    }
    if not set(joined["family_id"]).issubset(known):
        raise RuntimeError("Unknown RFC family entered authoritative source")
    family_counts_all = joined["family_id"].value_counts().to_dict()
    # TCP and DNS labels are never selected, grouped, or counted below.
    effective = joined.loc[
        joined["family_id"].isin(config["partitions"]["train_families"])
        & ~joined["item_id"].isin(unresolved)
    ].copy()
    supervised = source.loc[
        source["item_id"].isin(effective["item_id"]),
        ["item_id", "final_label"],
    ].copy()
    if set(supervised["final_label"]) != {"No", "Yes"}:
        raise RuntimeError("Development labels are not exactly Yes/No")
    supervised["label"] = supervised["final_label"].map({"No": 0, "Yes": 1})
    effective = effective.merge(
        supervised[["item_id", "label"]], on="item_id", validate="one_to_one"
    )

    metadata = _read_csv(
        _resolve_repo_path(data["metadata_csv"]),
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
    effective = effective.merge(
        metadata,
        on=["record_id", "family_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if set(effective["_merge"]) != {"both"}:
        raise RuntimeError("Approved lineage metadata is incomplete")
    effective.drop(columns="_merge", inplace=True)
    if not (effective["document"] == effective["document_id"]).all():
        raise RuntimeError("Authoritative and governed document identifiers disagree")
    text_drift = effective.loc[
        effective["target_clause"] != effective["clause_text"],
        ["item_id", "record_id", "family_id", "target_clause", "clause_text"],
    ].copy()

    context = _read_csv(
        _resolve_repo_path(data["canonical_context_csv"]),
        [
            "record_id",
            "family_id",
            "context_source_ids",
            "context_resolution_status",
            "alignment_source_path",
            "source_hashes",
        ],
    )
    effective = effective.merge(
        context,
        on=["record_id", "family_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if set(effective["_merge"]) != {"both"}:
        raise RuntimeError("Approved context provenance is incomplete")
    effective.drop(columns=["_merge", "document", "clause_text"], inplace=True)

    fold_source = _read_json(_resolve_repo_path(config["folds"]["fold_source"]))
    family_to_fold = {
        str(family): int(fold)
        for family, fold in fold_source["family_to_fold"].items()
        if family in config["partitions"]["train_families"]
    }
    if set(family_to_fold) != set(config["partitions"]["train_families"]):
        raise RuntimeError("Frozen family-to-fold map is incomplete")
    effective["fold"] = effective["family_id"].map(family_to_fold).astype(int)
    effective["label"] = effective["label"].astype(int)
    effective.sort_values("item_id", key=lambda x: x.astype(int), inplace=True)
    effective.reset_index(drop=True, inplace=True)
    effective = effective.loc[:, DATASET_COLUMNS]
    _unique_nonblank(effective, "record_id", "effective dataset")

    missing_from_prior = sorted(
        set(_read_csv(ROOT / "results/annotation/final_annotation_ready.csv", ["item_id"])["item_id"])
        - set(source["item_id"]),
        key=int,
    )
    audit = {
        "schema_version": 1,
        "authoritative_path": str(source_path),
        "authoritative_sha256": sha256_file(source_path),
        "source_rows": len(source),
        "source_yes": int(labels["Yes"]),
        "source_no": int(labels["No"]),
        "source_prevalence": labels["Yes"] / len(source),
        "unknown_or_missing_labels": int(
            (~source["final_label"].isin(["Yes", "No"])).sum()
        ),
        "unresolved_item_ids": unresolved,
        "duplicate_item_ids": int(source["item_id"].duplicated().sum()),
        "duplicate_clause_text_rows": int(
            source["target_clause"].duplicated(keep=False).sum()
        ),
        "empty_target_clauses": int(source["target_clause"].str.strip().eq("").sum()),
        "document_identifiers": int(source["document"].nunique()),
        "governance_key_rows": len(key),
        "source_omissions_from_governance_universe": len(key) - len(source),
        "rows_removed_since_prior_annotation_source": len(missing_from_prior),
        "removed_item_ids": missing_from_prior,
        "family_counts_without_label_access": family_counts_all,
        "reserved_counts_without_label_access": {
            "tcp_rows": int(family_counts_all.get(config["partitions"]["tcp_family"], 0)),
            "dns_rows": int(family_counts_all.get(config["partitions"]["dns_family"], 0)),
        },
        "development_rows": len(effective),
        "development_yes": int(effective["label"].sum()),
        "development_no": int((effective["label"] == 0).sum()),
        "development_prevalence": float(effective["label"].mean()),
        "development_families": sorted(effective["family_id"].unique()),
        "positive_counts_by_development_family": {
            str(k): int(v)
            for k, v in effective.groupby("family_id")["label"].sum().items()
        },
        "authoritative_text_drift_from_legacy_metadata": [
            {
                "item_id": row.item_id,
                "record_id": row.record_id,
                "family_id": row.family_id,
                "legacy_text_sha256": _sha256_text(row.clause_text),
                "authoritative_text_sha256": _sha256_text(row.target_clause),
            }
            for row in text_drift.itertuples(index=False)
        ],
        "text_authority_policy": "All human-readable text is taken from final_annotation.csv; stable record/family/lineage identifiers come from frozen governance metadata.",
        "reserved_label_access_for_feasibility": False,
    }
    return effective, audit


def _partition_counts(
    frame: pd.DataFrame, manifest: dict[str, Any], *, scope: str, outer_fold: int
) -> dict[str, Any]:
    fit = frame.loc[frame["record_id"].isin(manifest["fitting_record_ids"])]
    score = frame.loc[frame["record_id"].isin(manifest["scoring_record_ids"])]
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "inner_fold": manifest.get("inner_fold", ""),
        "held_out_families": "|".join(manifest["held_out_families"]),
        "training_families": "|".join(manifest["fitting_families"]),
        "training_rows": len(fit),
        "training_yes": int(fit["label"].sum()),
        "training_no": int((fit["label"] == 0).sum()),
        "scoring_rows": len(score),
        "scoring_yes": int(score["label"].sum()),
        "scoring_no": int((score["label"] == 0).sum()),
        "scoring_prevalence": float(score["label"].mean()),
        "leakage_exclusions": len(manifest["fitting_exclusions"]),
        "context_sanitizations": len(manifest["context_sanitizations"]),
    }


def _make_inner_manifests(
    frame: pd.DataFrame, outer_fold: int, near_threshold: float
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    outer_fit = frame.loc[frame["fold"] != outer_fold].copy().reset_index(drop=True)
    families = outer_fit["family_id"].astype(str).to_numpy()
    if len(set(families)) < 3:
        raise RuntimeError(f"Outer {outer_fold} has fewer than three fitting families")
    splitter = GroupKFold(n_splits=3)
    mapping: dict[str, int] = {}
    for inner_fold, (_, scoring_indices) in enumerate(
        splitter.split(np.zeros(len(outer_fit)), groups=families), start=1
    ):
        for family in outer_fit.iloc[scoring_indices]["family_id"].unique():
            if family in mapping:
                raise RuntimeError("An RFC family entered multiple inner folds")
            mapping[str(family)] = inner_fold
    outer_fit["fold"] = outer_fit["family_id"].map(mapping).astype(int)
    leakage = audit_leakage(outer_fit.reset_index(drop=True), near_threshold=near_threshold)
    if leakage["training_blocked"]:
        raise RuntimeError(f"Inner leakage audit blocked outer fold {outer_fold}")
    manifests = build_fold_manifests(outer_fit, leakage)
    scoring_ids: list[str] = []
    for manifest in manifests:
        manifest["outer_fold"] = outer_fold
        manifest["inner_fold"] = int(manifest["fold"])
        scoring_ids.extend(manifest["scoring_record_ids"])
        row = _partition_counts(
            outer_fit, manifest, scope="inner", outer_fold=outer_fold
        )
        if min(row["training_yes"], row["training_no"], row["scoring_yes"], row["scoring_no"]) <= 0:
            raise RuntimeError(
                f"Scientifically invalid inner partition outer={outer_fold}, "
                f"inner={manifest['inner_fold']}: {row}"
            )
    if Counter(scoring_ids) != Counter(outer_fit["record_id"].astype(str)):
        raise RuntimeError(f"Inner OOF coverage failed for outer fold {outer_fold}")
    return manifests, leakage, outer_fit


def _source_snapshot() -> dict[str, str]:
    paths = sorted(
        path
        for path in EXPERIMENT_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    paths.append(CONFIG_PATH)
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
    }


def _historical_provenance() -> dict[str, Any]:
    paths = [
        ROOT / "experiments/transformer_final/recall_improvement_study_v1/cell_launcher.py",
        ROOT / "experiments/transformer_final/recall_improvement_study_v1/study_protocol.py",
        ROOT / "experiments/transformer_final/recall_improvement_study_v3_optimizer_fix.py",
        ROOT / "experiments/transformer_final/method_specific_calibration_v1/study.py",
        ROOT / "experiments/transformer_final/method_specific_calibration_v1_1/study.py",
        ROOT / "models/transformer_final/train.py",
        ROOT / "models/transformer_final/leakage.py",
        ROOT / "models/transformer_final/modeling.py",
    ]
    files = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "historical_references_read_only": files,
        "reuse_summary": {
            "family_governance": "frozen folds_v1 family boundaries",
            "leakage": "models.transformer_final.leakage label-blind audit/manifests",
            "training_loop": "guarded fixed-epoch FP32 core with experiment-local method wrapper",
            "focal": "gamma=2; fitting-side class weights; mean reduction",
            "partial4": "encoder layers 8-11 plus pooler/classifier; trainable-only AdamW",
            "calibration": "method-specific three-fold family OOF",
            "ensemble": "outer-specific fitting-side ECDF, equal weights",
        },
        "historical_artifacts_modified": False,
    }


def _manual_execution(protocol_hash: str) -> str:
    py = r"C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe"
    module = "experiments.transformer_final.final_annotation_661_v1.cli"
    return rf"""# Manual execution — final_annotation_661_v1

All transformer training is manual, sequential, CUDA-only, FP32, and fail-fast.
No preparation command starts training. The frozen protocol SHA-256 is
`{protocol_hash}`.

```powershell
Set-Location -LiteralPath 'D:\RFC\ietf-ambiguity-detection'
$python = '{py}'

# 1–3. Read-only preflight and fold-count verification
& $python -m {module} preflight
& $python -m {module} status
Import-Csv -LiteralPath 'experiments\transformer_final\final_annotation_661_v1\FOLD_CLASS_COUNTS.csv' | Format-Table

# 4–6. Ten outer cells, sequential; no retries
& $python -m {module} train-outer --all
& $python -m {module} status
& $python -m {module} validate-outer --require-all

# 7–9. Thirty inner cells, sequential; no retries
& $python -m {module} train-inner --all
& $python -m {module} status
& $python -m {module} validate-inner --require-all

# 10–12. CPU-only analysis, Top-20 generation, integrity validation
& $python -m {module} analyze
& $python -m {module} validate-analysis

# 13. Inspect failures and attempt logs
Get-ChildItem -LiteralPath 'experiments\transformer_final\final_annotation_661_v1\failure_manifests' -File
Get-ChildItem -LiteralPath 'experiments\transformer_final\final_annotation_661_v1\logs' -File

# 14. After investigating an interruption, run only the still-pending cell
& $python -m {module} train-outer --method focal --fold 1
& $python -m {module} train-inner --method focal --outer-fold 1 --inner-fold 1

# 15. Re-verify frozen source/protocol and historical immutability
& $python -m {module} preflight
git status --short -- experiments/transformer_final/recall_improvement_study_v3_optimizer_fix experiments/transformer_final/method_specific_calibration_v1 experiments/transformer_final/method_specific_calibration_v1_1
```

If a cell fails, inspect its log and failure manifest. The runner does not retry,
does not overwrite completed artifacts, and stops the sequence immediately. A
completed cell is skipped only after its manifest and held-out predictions pass
validation. Do not delete or replace a completed run; create a new experiment
version if frozen code or protocol must change.
"""


def _dataset_audit_markdown(audit: dict[str, Any]) -> str:
    return f"""# Dataset audit

- Authority: `{audit['authoritative_path']}`
- SHA-256: `{audit['authoritative_sha256']}`
- Source: {audit['source_rows']} rows; {audit['source_yes']} Yes; {audit['source_no']} No; {audit['source_prevalence']:.6%} positive.
- Unknown/missing labels: {audit['unknown_or_missing_labels']}.
- Duplicate item IDs: {audit['duplicate_item_ids']}.
- Duplicate target-clause rows: {audit['duplicate_clause_text_rows']}.
- Empty target clauses: {audit['empty_target_clauses']}.
- Document identifiers: {audit['document_identifiers']}.
- Authorized development cohort: {audit['development_rows']} rows; {audit['development_yes']} Yes; {audit['development_no']} No; {audit['development_prevalence']:.6%} positive.
- Unresolved exclusion: item_id {', '.join(audit['unresolved_item_ids'])}.
- Reserved counts (labels not accessed for feasibility): TCP {audit['reserved_counts_without_label_access']['tcp_rows']}; DNS {audit['reserved_counts_without_label_access']['dns_rows']}.
- Rows removed relative to the preceding annotation source: {audit['rows_removed_since_prior_annotation_source']}.
- Authoritative target-text corrections relative to legacy metadata: {len(audit['authoritative_text_drift_from_legacy_metadata'])}.

The authoritative CSV supplies every human-readable text and final label. Frozen
governance sources supply stable record, family, document-lineage, and fold IDs.
TCP remains reserved and DNS remains sealed; neither partition contributes labels
to development feasibility, fitting, calibration, or analysis.
"""


def _preflight_markdown(
    audit: dict[str, Any], leakage: dict[str, Any], counts: pd.DataFrame
) -> str:
    outer = counts.loc[counts["scope"] == "outer"]
    inner = counts.loc[counts["scope"] == "inner"]
    feasible = bool(
        (outer[["training_yes", "training_no", "scoring_yes", "scoring_no"]].min().min() > 0)
        and (inner[["training_yes", "training_no", "scoring_yes", "scoring_no"]].min().min() > 0)
        and not leakage["training_blocked"]
    )
    return f"""# Preflight report

- Dataset identity: PASS (`{audit['authoritative_sha256']}`).
- Exact schema and Yes/No semantics: PASS.
- Authorized development cohort: {audit['development_rows']} rows ({audit['development_yes']} Yes, {audit['development_no']} No).
- Five frozen family-isolated outer folds: PASS.
- Fifteen three-way family-isolated inner designs: PASS.
- Both classes in every fitting and scoring partition: PASS.
- Leakage audit blocked: {str(leakage['training_blocked']).lower()}.
- Reserved TCP or sealed DNS in development: no.
- Nested design feasible: {str(feasible).lower()}.
- Prepared cells: 10 outer and 30 inner.
- Training executed by preparation: no.
"""


def _create_specs(protocol_hash: str) -> list[dict[str, Any]]:
    config = load_local_config()
    specs: list[dict[str, Any]] = []
    model_path = str(_resolve_repo_path(config["model"]["default_local_path"]))
    for kind in ("outer", "inner"):
        outer_values = range(1, 6)
        inner_values: Iterable[int | None] = (None,) if kind == "outer" else range(1, 4)
        for method in METHODS:
            for outer in outer_values:
                for inner in inner_values:
                    if kind == "outer":
                        split_path = DATASET_DIR / "folds" / f"fold_{outer}.json"
                        train_fold = outer
                        cell_id = f"outer-{method}-fold{outer}-seed2718"
                        run_path = RUNS_DIR / "outer" / method / f"fold_{outer}" / f"attempt_{cell_id}"
                    else:
                        assert inner is not None
                        split_path = INNER_DIR / f"outer_{outer}" / "folds" / f"fold_{inner}.json"
                        train_fold = inner
                        cell_id = f"inner-{method}-outer{outer}-inner{inner}-seed2718"
                        run_path = RUNS_DIR / "inner" / method / f"outer_{outer}" / f"inner_{inner}" / f"attempt_{cell_id}"
                    spec = {
                        "schema_version": 1,
                        "cell_id": cell_id,
                        "kind": kind,
                        "method": method,
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "train_fold": train_fold,
                        "seed": 2718,
                        "input_mode": "target_only",
                        "precision": "fp32",
                        "config_path": str(CONFIG_PATH),
                        "protocol_path": str(PROTOCOL_PATH),
                        "protocol_sha256": protocol_hash,
                        "dataset_path": str(DATASET_DIR / "dataset.csv"),
                        "dataset_manifest_path": str(DATASET_DIR / "manifest.json"),
                        "split_manifest_path": str(split_path),
                        "split_manifest_sha256": sha256_file(split_path),
                        "model_path": model_path,
                        "model_source_sha256": EXPECTED_MODEL_SHA256,
                        "run_path": str(run_path),
                        "audit_path": str(AUDITS_DIR / f"{cell_id}.json"),
                        "completion_path": str(COMPLETIONS_DIR / f"{cell_id}.json"),
                    }
                    specs.append(spec)
    return specs


def prepare() -> dict[str, Any]:
    """Create and freeze all preparation artifacts. Never trains or loads a model."""
    if PROTOCOL_PATH.exists():
        raise FileExistsError(
            f"This experiment is already frozen: {PROTOCOL_PATH}. Validate it instead."
        )
    config = load_local_config()
    static = _verify_static_sources(config, hash_model=True)
    frame, dataset_audit = build_effective_dataset(config)
    near = float(config["folds"]["near_duplicate_threshold"])
    leakage = audit_leakage(frame.reset_index(drop=True), near_threshold=near)
    if leakage["training_blocked"]:
        raise RuntimeError("Outer leakage audit blocks training")
    outer_manifests = build_fold_manifests(frame, leakage)
    counts: list[dict[str, Any]] = []
    for manifest in outer_manifests:
        row = _partition_counts(
            frame, manifest, scope="outer", outer_fold=int(manifest["fold"])
        )
        if min(row["training_yes"], row["training_no"], row["scoring_yes"], row["scoring_no"]) <= 0:
            raise RuntimeError(f"Scientifically invalid outer partition: {row}")
        counts.append(row)

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=False)
    dataset_path = DATASET_DIR / "dataset.csv"
    frame.to_csv(dataset_path, index=False, encoding="utf-8", lineterminator="\n")
    _write_json(DATASET_DIR / "leakage_audit.json", leakage, exclusive=True)
    for manifest in outer_manifests:
        fold = int(manifest["fold"])
        _write_json(DATASET_DIR / "folds" / f"fold_{fold}.json", manifest, exclusive=True)
        assignments = frame[["item_id", "record_id", "family_id", "fold"]].copy()
        excluded = {
            value["record_id"]: "|".join(value["reasons"])
            for value in manifest["fitting_exclusions"]
        }
        assignments["role"] = np.where(assignments["fold"] == fold, "evaluation", "fitting")
        assignments["included"] = ~assignments["record_id"].isin(excluded)
        assignments.loc[assignments["role"] == "evaluation", "included"] = True
        assignments["exclusion_reason"] = assignments["record_id"].map(excluded).fillna("")
        path = DATASET_DIR / "folds" / f"fold_{fold}_assignments.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    inner_hashes: dict[str, dict[str, str]] = {}
    inner_audits: dict[str, Any] = {}
    for outer in range(1, 6):
        manifests, inner_leakage, inner_frame = _make_inner_manifests(frame, outer, near)
        inner_audits[str(outer)] = inner_leakage
        inner_hashes[str(outer)] = {}
        outer_dir = INNER_DIR / f"outer_{outer}"
        outer_dir.mkdir(parents=True, exist_ok=False)
        inner_frame.to_csv(
            outer_dir / "dataset.csv", index=False, encoding="utf-8", lineterminator="\n"
        )
        _write_json(outer_dir / "leakage_audit.json", inner_leakage, exclusive=True)
        for manifest in manifests:
            inner = int(manifest["inner_fold"])
            path = outer_dir / "folds" / f"fold_{inner}.json"
            _write_json(path, manifest, exclusive=True)
            inner_hashes[str(outer)][str(inner)] = sha256_file(path)
            counts.append(
                _partition_counts(inner_frame, manifest, scope="inner", outer_fold=outer)
            )

    counts_frame = pd.DataFrame(counts)
    counts_frame.to_csv(
        EXPERIMENT_ROOT / "FOLD_CLASS_COUNTS.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    label_rows = [
        {"scope": "authoritative_source", "label": "Yes", "count": dataset_audit["source_yes"]},
        {"scope": "authoritative_source", "label": "No", "count": dataset_audit["source_no"]},
        {"scope": "authorized_development", "label": "Yes", "count": dataset_audit["development_yes"]},
        {"scope": "authorized_development", "label": "No", "count": dataset_audit["development_no"]},
    ]
    pd.DataFrame(label_rows).to_csv(
        EXPERIMENT_ROOT / "LABEL_DISTRIBUTION.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    dataset_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "authoritative_source_path": str(_resolve_repo_path(config["data"]["source_csv"])),
        "original_dataset_sha256": dataset_audit["authoritative_sha256"],
        "effective_dataset_sha256": sha256_file(dataset_path),
        "dataset_file": "dataset.csv",
        "rows": len(frame),
        "labels": {"0": int((frame["label"] == 0).sum()), "1": int(frame["label"].sum())},
        "families": sorted(frame["family_id"].unique()),
        "reserved_partitions": {
            "tcp": "reserved_no_training_tuning_calibration_or_analysis",
            "dns": "sealed_no_label_consumption",
        },
        "outer_fold_manifest_sha256": {
            str(fold): sha256_file(DATASET_DIR / "folds" / f"fold_{fold}.json")
            for fold in range(1, 6)
        },
        "inner_fold_manifest_sha256": inner_hashes,
        "leakage_audit_sha256": sha256_file(DATASET_DIR / "leakage_audit.json"),
        "columns": DATASET_COLUMNS,
    }
    _write_json(DATASET_DIR / "manifest.json", dataset_manifest, exclusive=True)
    _write_json(EXPERIMENT_ROOT / "DATASET_MANIFEST.json", dataset_manifest, exclusive=True)
    _write_json(EXPERIMENT_ROOT / "LEAKAGE_AUDIT.json", {
        "outer": leakage,
        "inner_by_outer_fold": inner_audits,
        "training_blocked": False,
        "reserved_or_sealed_in_development": False,
    }, exclusive=True)
    _write_text(EXPERIMENT_ROOT / "DATASET_AUDIT.md", _dataset_audit_markdown(dataset_audit), exclusive=True)

    provenance = _historical_provenance()
    provenance["authoritative_dataset"] = {
        "path": dataset_audit["authoritative_path"],
        "sha256": dataset_audit["authoritative_sha256"],
    }
    provenance["approved_static_sources"] = static
    _write_json(SOURCE_PROVENANCE_PATH, provenance, exclusive=True)

    source_snapshot = _source_snapshot()
    protocol = {
        "schema_version": 1,
        "experiment_id": "final_annotation_661_v1",
        "created_at": utc_now(),
        "dataset": dataset_manifest,
        "dataset_manifest_sha256": sha256_file(DATASET_DIR / "manifest.json"),
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": _sha256_text(stable_json(source_snapshot)),
        "source_provenance_sha256": sha256_file(SOURCE_PROVENANCE_PATH),
        "training": config["training"],
        "model": config["model"],
        "tokenization": config["tokenization"],
        "methods": config["methods"],
        "outer_cv": {
            "folds": 5,
            "groups": "frozen RFC family mapping",
            "row_level_random_split": False,
        },
        "inner_cv": {
            "folds_per_outer": 3,
            "algorithm": "GroupKFold(shuffle=False)",
            "groups": "RFC family",
            "assignment_uses_labels": False,
        },
        "calibration": config["calibration"],
        "ensemble": config["ensemble"],
        "evaluation": config["evaluation"],
        "leakage_policy": config["folds"]["policy"],
        "error_analysis": {
            "primary_policy": "outer-specific method threshold selected from inner OOF at precision target 0.30",
            "false_positive": "actual No, predicted Yes; descending exact decision score; stable record_id tie-break",
            "false_negative": "actual Yes, predicted No; ascending exact decision score; stable record_id tie-break",
            "threshold_independent": "No/high-Yes and Yes/low-Yes score lists",
            "text": "untruncated authoritative source text",
        },
        "execution": {
            "manual_only": True,
            "sequential": True,
            "automatic_retries": False,
            "cuda_required": True,
            "cpu_fallback": False,
        },
    }
    _write_json(PROTOCOL_PATH, protocol, exclusive=True)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    _write_text(EXPERIMENT_ROOT / "PROTOCOL_SHA256.txt", protocol_hash + "\n", exclusive=True)

    specs = _create_specs(protocol_hash)
    CELL_SPECS_DIR.mkdir(parents=True, exist_ok=False)
    for index, spec in enumerate(specs, start=1):
        _write_json(CELL_SPECS_DIR / f"{index:02d}_{spec['cell_id']}.json", spec, exclusive=True)
    status = {
        "schema_version": 1,
        "experiment_id": "final_annotation_661_v1",
        "protocol_sha256": protocol_hash,
        "prepared_at": utc_now(),
        "training_executed_during_preparation": False,
        "outer_cells": 10,
        "inner_cells": 30,
        "cells": [{"cell_id": s["cell_id"], "kind": s["kind"], "status": "pending"} for s in specs],
    }
    _write_json(EXPERIMENT_ROOT / "TRAINING_STATUS.json", status, exclusive=True)
    _write_text(EXPERIMENT_ROOT / "MANUAL_EXECUTION.md", _manual_execution(protocol_hash), exclusive=True)
    _write_text(EXPERIMENT_ROOT / "PREFLIGHT_REPORT.md", _preflight_markdown(dataset_audit, leakage, counts_frame), exclusive=True)
    _write_json(EXPERIMENT_ROOT / "PREPARATION_AUDIT.json", dataset_audit, exclusive=True)
    _write_json(EXPERIMENT_ROOT / "FROZEN.json", {
        "frozen_at": utc_now(),
        "protocol_sha256": protocol_hash,
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "dataset_sha256": dataset_audit["authoritative_sha256"],
        "effective_dataset_sha256": dataset_manifest["effective_dataset_sha256"],
        "outer_cells": 10,
        "inner_cells": 30,
        "training_executed": False,
    }, exclusive=True)
    return {
        "protocol_sha256": protocol_hash,
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "development_rows": len(frame),
        "development_yes": int(frame["label"].sum()),
        "development_no": int((frame["label"] == 0).sum()),
        "outer_cells": 10,
        "inner_cells": 30,
        "training_executed": False,
    }


def verify_frozen() -> dict[str, Any]:
    config = load_local_config()
    static = _verify_static_sources(config, hash_model=True)
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("Experiment is not prepared/frozen")
    protocol = _read_json(PROTOCOL_PATH)
    expected_protocol = (EXPERIMENT_ROOT / "PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip()
    actual_protocol = sha256_file(PROTOCOL_PATH)
    if actual_protocol != expected_protocol:
        raise RuntimeError("Frozen protocol fingerprint mismatch")
    current_sources = _source_snapshot()
    if current_sources != protocol["source_snapshot"]:
        changed = sorted(set(current_sources) | set(protocol["source_snapshot"]))
        changed = [key for key in changed if current_sources.get(key) != protocol["source_snapshot"].get(key)]
        raise RuntimeError(f"Frozen experiment source changed: {changed}")
    manifest = _read_json(DATASET_DIR / "manifest.json")
    if sha256_file(DATASET_DIR / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest fingerprint mismatch")
    if sha256_file(DATASET_DIR / "dataset.csv") != manifest["effective_dataset_sha256"]:
        raise RuntimeError("Effective dataset fingerprint mismatch")
    if sha256_file(_resolve_repo_path(config["data"]["source_csv"])) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Authoritative dataset changed after freezing")
    for fold, expected in manifest["outer_fold_manifest_sha256"].items():
        if sha256_file(DATASET_DIR / "folds" / f"fold_{fold}.json") != expected:
            raise RuntimeError(f"Outer fold {fold} changed")
    for outer, folds in manifest["inner_fold_manifest_sha256"].items():
        for inner, expected in folds.items():
            path = INNER_DIR / f"outer_{outer}" / "folds" / f"fold_{inner}.json"
            if sha256_file(path) != expected:
                raise RuntimeError(f"Inner split outer={outer}, inner={inner} changed")
    return {
        "status": "PASS",
        "protocol_sha256": actual_protocol,
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "effective_dataset_sha256": manifest["effective_dataset_sha256"],
        "development_rows": manifest["rows"],
        "development_yes": manifest["labels"]["1"],
        "development_no": manifest["labels"]["0"],
        "model_sha256": static["model_source_sha256"],
        "training_started": any(RUNS_DIR.rglob("run_manifest.json")) if RUNS_DIR.exists() else False,
    }


def preflight() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        result = verify_frozen()
        counts = pd.read_csv(EXPERIMENT_ROOT / "FOLD_CLASS_COUNTS.csv")
        columns = ["training_yes", "training_no", "scoring_yes", "scoring_no"]
        if counts[columns].min().min() <= 0:
            raise RuntimeError("At least one frozen partition is single-class")
        result["outer_partitions"] = int((counts["scope"] == "outer").sum())
        result["inner_partitions"] = int((counts["scope"] == "inner").sum())
        result["nested_feasible"] = True
        return result
    config = load_local_config()
    static = _verify_static_sources(config, hash_model=True)
    frame, audit = build_effective_dataset(config)
    leakage = audit_leakage(
        frame.reset_index(drop=True),
        near_threshold=float(config["folds"]["near_duplicate_threshold"]),
    )
    return {
        "status": "PASS_NOT_YET_FROZEN",
        "dataset_sha256": audit["authoritative_sha256"],
        "development_rows": len(frame),
        "development_yes": int(frame["label"].sum()),
        "development_no": int((frame["label"] == 0).sum()),
        "leakage_blocked": leakage["training_blocked"],
        "model_sha256": static["model_source_sha256"],
        "training_started": False,
    }


def iter_specs(kind: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted(CELL_SPECS_DIR.glob("*.json")):
        spec = _read_json(path)
        if kind is None or spec["kind"] == kind:
            result.append((path, spec))
    return result


def prediction_path(spec: dict[str, Any]) -> Path:
    return Path(spec["run_path"]) / "predictions" / "held_out_predictions.csv"


def validate_cell(spec: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(Path(spec["protocol_path"])) != spec["protocol_sha256"]:
        raise RuntimeError(f"Protocol changed for {spec['cell_id']}")
    if sha256_file(Path(spec["split_manifest_path"])) != spec["split_manifest_sha256"]:
        raise RuntimeError(f"Split changed for {spec['cell_id']}")
    run = Path(spec["run_path"])
    manifest_path = run / "run_manifest.json"
    predictions_path = prediction_path(spec)
    audit_path = Path(spec["audit_path"])
    completion_path = Path(spec["completion_path"])
    for path in (manifest_path, predictions_path, audit_path, completion_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete cell {spec['cell_id']}: {path}")
    split = _read_json(spec["split_manifest_path"])
    predictions = pd.read_csv(predictions_path, dtype={"item_id": str, "record_id": str})
    if predictions["record_id"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction IDs in {spec['cell_id']}")
    if set(predictions["record_id"].astype(str)) != set(split["scoring_record_ids"]):
        raise RuntimeError(f"Prediction alignment failed for {spec['cell_id']}")
    if not np.isfinite(predictions["probability_yes"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite predictions in {spec['cell_id']}")
    if not predictions["probability_yes"].between(0, 1).all():
        raise RuntimeError(f"Out-of-range predictions in {spec['cell_id']}")
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("cell_id") != spec["cell_id"]:
        raise RuntimeError(f"Run manifest cell mismatch for {spec['cell_id']}")
    if run_manifest.get("protocol_sha256") != spec["protocol_sha256"]:
        raise RuntimeError(f"Run protocol mismatch for {spec['cell_id']}")
    if run_manifest.get("fresh_base_initialization") is not True:
        raise RuntimeError(f"Fresh initialization not proven for {spec['cell_id']}")
    if run_manifest.get("device", "").split(":")[0] != "cuda":
        raise RuntimeError(f"Cell did not use CUDA: {spec['cell_id']}")
    return {
        "cell_id": spec["cell_id"],
        "rows": len(predictions),
        "runtime_seconds": run_manifest.get("runtime_seconds"),
        "peak_gpu_memory_allocated_mb": run_manifest.get("peak_gpu_memory_allocated_mb"),
        "status": "complete_valid",
    }


def validate_cells(kind: str, *, require_all: bool) -> dict[str, Any]:
    verify_frozen()
    complete: list[dict[str, Any]] = []
    pending: list[str] = []
    invalid: list[dict[str, str]] = []
    for _, spec in iter_specs(kind):
        if not Path(spec["run_path"]).exists():
            pending.append(spec["cell_id"])
            continue
        try:
            complete.append(validate_cell(spec))
        except Exception as exc:  # noqa: BLE001 - aggregate validation must report every artifact failure
            invalid.append({"cell_id": spec["cell_id"], "error": str(exc)})
    if invalid or (require_all and pending):
        raise RuntimeError(
            json.dumps({"invalid": invalid, "pending": pending}, indent=2)
        )
    return {
        "kind": kind,
        "expected": 10 if kind == "outer" else 30,
        "complete_valid": len(complete),
        "pending": pending,
        "invalid": invalid,
    }


def status() -> dict[str, Any]:
    result: dict[str, Any] = {"protocol": verify_frozen(), "outer": {}, "inner": {}}
    for kind in ("outer", "inner"):
        counts = Counter()
        details = []
        for _, spec in iter_specs(kind):
            if Path(spec["completion_path"]).is_file():
                state = "complete"
            elif Path(spec["run_path"]).exists():
                state = "incomplete_or_invalid"
            elif (FAILURES_DIR / f"{spec['cell_id']}.json").is_file():
                state = "failed"
            else:
                state = "pending"
            counts[state] += 1
            details.append({"cell_id": spec["cell_id"], "status": state})
        result[kind] = {"counts": dict(counts), "cells": details}
    return result

