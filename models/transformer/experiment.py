"""Leakage-safe data loading, weighting, and grouped development folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import normalize_for_audit, require_training_ready

DevelopmentSplit = Literal["train", "validation"]
WeightStrategy = Literal["unweighted", "class", "confidence", "combined"]


@dataclass(frozen=True)
class DevelopmentFold:
    """One family-grouped fold and its leakage-safe scoring subset."""

    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    scoring_indices: np.ndarray
    held_out_families: tuple[str, ...]
    excluded_exact_duplicate_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class NestedDevelopmentFold:
    """Outer evaluation fold with an independent inner-family selector."""

    outer: DevelopmentFold
    fitting_indices: np.ndarray
    selection_indices: np.ndarray
    selection_scoring_indices: np.ndarray
    inner_selection_family: str
    excluded_inner_duplicate_record_ids: tuple[str, ...]


def _load_manifest(dataset_dir: Path) -> tuple[dict[str, object], Path]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_training_ready(manifest)
    if "split_files" not in manifest:
        raise RuntimeError("Dataset manifest has no isolated split files")
    return manifest, manifest_path


def load_development_split(dataset_dir: Path, split: DevelopmentSplit) -> pd.DataFrame:
    """Load and verify only a development split; the test split is not accepted."""

    if split not in {"train", "validation"}:
        raise ValueError("Development code may load only train or validation")
    manifest, _ = _load_manifest(dataset_dir)
    split_info = manifest["split_files"][split]  # type: ignore[index]
    split_path = dataset_dir / split_info["path"]
    if sha256_file(split_path) != split_info["sha256"]:
        raise RuntimeError(f"{split} split hash does not match its manifest")
    frame = pd.read_csv(split_path).fillna("")
    if len(frame) != int(split_info["rows"]):
        raise RuntimeError(f"{split} split row count does not match its manifest")
    if set(frame["split"]) != {split}:
        raise RuntimeError(f"Isolated {split} file contains rows from another split")
    if frame["record_id"].duplicated().any():
        raise RuntimeError(f"Isolated {split} file contains duplicate record IDs")
    return frame


def compute_training_weights(
    frame: pd.DataFrame, strategy: WeightStrategy
) -> np.ndarray:
    """Calculate fold-local weights and normalize their mean to exactly one."""

    if strategy not in {"unweighted", "class", "confidence", "combined"}:
        raise ValueError(f"Unknown weight strategy: {strategy}")
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Training labels must be binary")
    weights = np.ones(len(frame), dtype=np.float64)
    if strategy in {"class", "combined"}:
        positives = int(labels.sum())
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("Class weighting requires both classes")
        weights[labels == 1] *= negatives / positives
    if strategy in {"confidence", "combined"}:
        confidence = frame["sample_weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(confidence).all() or (confidence <= 0).any():
            raise ValueError("Sample weights must be positive and finite")
        weights *= confidence
    mean = float(weights.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Training weights cannot be normalized")
    return weights / mean


def make_grouped_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, seed: int = 1729
) -> list[DevelopmentFold]:
    """Create family-held-out folds and exclude exact train/score text overlap."""

    if set(frame["split"]) != {"train"}:
        raise ValueError("Grouped folds can be created only from the training split")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    groups = frame["family_id"].astype(str).to_numpy()
    normalized = frame["target_clause"].map(normalize_for_audit).to_numpy()
    record_ids = frame["record_id"].astype(str).to_numpy()
    folds: list[DevelopmentFold] = []
    for number, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(frame)), labels, groups),
        start=1,
    ):
        train_text = set(normalized[train_indices])
        duplicate_mask = np.array(
            [normalized[index] in train_text for index in validation_indices],
            dtype=bool,
        )
        scoring_indices = validation_indices[~duplicate_mask]
        if not len(scoring_indices):
            raise RuntimeError(f"Fold {number} has no leakage-safe scoring rows")
        folds.append(
            DevelopmentFold(
                number=number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                scoring_indices=scoring_indices,
                held_out_families=tuple(sorted(set(groups[validation_indices]))),
                excluded_exact_duplicate_record_ids=tuple(
                    sorted(record_ids[validation_indices[duplicate_mask]])
                ),
            )
        )
    return folds


def load_frozen_folds(
    frame: pd.DataFrame,
    assignment_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_train_sha256: str | None = None,
) -> list[DevelopmentFold]:
    """Load a reviewed family-to-fold map and reconstruct leakage-safe indices."""

    if expected_sha256 and sha256_file(assignment_path) != expected_sha256:
        raise RuntimeError("Frozen fold assignment hash does not match the config")
    payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    if payload.get("n_splits") != 5:
        raise ValueError("Frozen fold assignment must define exactly five folds")
    if expected_train_sha256 and payload.get("train_split_sha256") != expected_train_sha256:
        raise RuntimeError("Frozen folds target a different training split")
    mapping = payload.get("family_to_fold")
    if not isinstance(mapping, dict):
        raise TypeError("Frozen fold assignment requires family_to_fold")
    observed_families = set(frame["family_id"].astype(str))
    if set(mapping) != observed_families:
        raise ValueError("Frozen fold families do not match the training data")
    fold_numbers = {int(value) for value in mapping.values()}
    if fold_numbers != set(range(1, 6)):
        raise ValueError("Frozen fold assignment must use fold numbers 1 through 5")

    normalized = frame["target_clause"].map(normalize_for_audit).to_numpy()
    record_ids = frame["record_id"].astype(str).to_numpy()
    groups = frame["family_id"].astype(str).to_numpy()
    row_folds = np.array([int(mapping[family]) for family in groups], dtype=np.int64)
    folds: list[DevelopmentFold] = []
    for number in range(1, 6):
        validation_indices = np.flatnonzero(row_folds == number)
        train_indices = np.flatnonzero(row_folds != number)
        train_text = set(normalized[train_indices])
        duplicate_mask = np.array(
            [normalized[index] in train_text for index in validation_indices], dtype=bool
        )
        scoring_indices = validation_indices[~duplicate_mask]
        if not len(scoring_indices):
            raise RuntimeError(f"Fold {number} has no leakage-safe scoring rows")
        folds.append(
            DevelopmentFold(
                number=number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                scoring_indices=scoring_indices,
                held_out_families=tuple(sorted(set(groups[validation_indices]))),
                excluded_exact_duplicate_record_ids=tuple(
                    sorted(record_ids[validation_indices[duplicate_mask]])
                ),
            )
        )
    return folds


def load_nested_folds(
    frame: pd.DataFrame,
    outer_folds: list[DevelopmentFold],
    assignment_path: Path,
    *,
    expected_sha256: str,
    expected_train_sha256: str,
) -> list[NestedDevelopmentFold]:
    """Build hash-pinned inner-family selection splits inside each outer fold."""

    if sha256_file(assignment_path) != expected_sha256:
        raise RuntimeError("Inner selection assignment hash does not match the config")
    payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    if payload.get("train_split_sha256") != expected_train_sha256:
        raise RuntimeError("Inner selection assignment targets a different train split")
    mapping = payload.get("outer_fold_to_inner_family")
    if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(1, 6)}:
        raise ValueError("Inner assignment must map every outer fold exactly once")
    eligibility = payload.get("eligibility")
    if not isinstance(eligibility, dict):
        raise TypeError("Inner assignment requires eligibility constraints")
    minimum_before = int(eligibility["minimum_positive_labels_before_duplicate_exclusion"])
    minimum_after = int(eligibility["minimum_positive_labels_after_duplicate_exclusion"])
    maximum_rows = int(eligibility["maximum_family_rows"])
    families = frame["family_id"].astype(str).to_numpy()
    normalized = frame["target_clause"].map(normalize_for_audit).to_numpy()
    record_ids = frame["record_id"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int64)
    nested: list[NestedDevelopmentFold] = []
    for outer in outer_folds:
        inner_family = str(mapping[str(outer.number)])
        if inner_family in outer.held_out_families:
            raise ValueError(f"Outer fold {outer.number} uses an outer family for selection")
        outer_training = outer.train_indices
        selection_mask = families[outer_training] == inner_family
        selection_indices = outer_training[selection_mask]
        fitting_indices = outer_training[~selection_mask]
        if not len(selection_indices) or not len(fitting_indices):
            raise ValueError(f"Outer fold {outer.number} has an invalid inner split")
        if len(selection_indices) > maximum_rows:
            raise ValueError(f"Outer fold {outer.number} inner family exceeds row cap")
        if int(labels[selection_indices].sum()) < minimum_before:
            raise ValueError(f"Outer fold {outer.number} inner family has too few positives")
        fitting_text = set(normalized[fitting_indices])
        duplicate_mask = np.array(
            [normalized[index] in fitting_text for index in selection_indices], dtype=bool
        )
        scoring_indices = selection_indices[~duplicate_mask]
        if int(labels[scoring_indices].sum()) < minimum_after:
            raise ValueError(
                f"Outer fold {outer.number} inner scoring set has too few positives"
            )
        if set(fitting_indices) & set(selection_indices):
            raise RuntimeError("Fitting and inner-selection rows overlap")
        if set(outer.validation_indices) & (
            set(fitting_indices) | set(selection_indices)
        ):
            raise RuntimeError("Outer evaluation rows entered inner model selection")
        nested.append(
            NestedDevelopmentFold(
                outer=outer,
                fitting_indices=fitting_indices,
                selection_indices=selection_indices,
                selection_scoring_indices=scoring_indices,
                inner_selection_family=inner_family,
                excluded_inner_duplicate_record_ids=tuple(
                    sorted(record_ids[selection_indices[duplicate_mask]])
                ),
            )
        )
    return nested
