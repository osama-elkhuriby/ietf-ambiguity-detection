from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from models.transformer.experiment import (
    compute_training_weights,
    load_development_split,
    load_frozen_folds,
    load_nested_folds,
    make_grouped_folds,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "splits").mkdir(parents=True)
    columns = ["record_id", "family_id", "split", "target_clause", "label"]
    train = pd.DataFrame(
        [
            ["a0", "a", "train", "same clause", 0],
            ["a1", "a", "train", "alpha", 1],
            ["b0", "b", "train", "same clause", 0],
            ["b1", "b", "train", "beta", 1],
            ["c0", "c", "train", "gamma", 0],
            ["c1", "c", "train", "delta", 1],
        ],
        columns=columns,
    )
    validation = pd.DataFrame(
        [["v0", "v", "validation", "validation clause", 0]], columns=columns
    )
    test = pd.DataFrame(
        [["t0", "t", "test", "sealed test clause", 1]], columns=columns
    )
    split_files = {}
    for split, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        path = dataset_dir / "splits" / f"{split}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        split_files[split] = {
            "path": f"splits/{split}.csv",
            "rows": len(frame),
            "sha256": _hash(path),
        }
    manifest = {"training_blocked": False, "split_files": split_files}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_dir


def test_development_loader_refuses_test_and_verifies_hash(tmp_path: Path) -> None:
    dataset_dir = _isolated_dataset(tmp_path)
    assert len(load_development_split(dataset_dir, "train")) == 6
    with pytest.raises(ValueError, match="train or validation"):
        load_development_split(dataset_dir, "test")  # type: ignore[arg-type]
    with (dataset_dir / "splits/train.csv").open("a", encoding="utf-8") as stream:
        stream.write("corruption")
    with pytest.raises(RuntimeError, match="hash"):
        load_development_split(dataset_dir, "train")


def test_training_weights_are_fold_local_and_mean_one() -> None:
    frame = pd.DataFrame(
        {"label": [0, 0, 0, 1], "sample_weight": [1.0, 0.5, 0.5, 1.0]}
    )
    for strategy in ("unweighted", "class", "confidence", "combined"):
        weights = compute_training_weights(frame, strategy)
        assert weights.mean() == pytest.approx(1.0)
        assert np.isfinite(weights).all()
    class_weights = compute_training_weights(frame, "class")
    assert class_weights[-1] == pytest.approx(3 * class_weights[0])


def test_grouped_folds_hold_out_families_and_exclude_exact_overlap(
    tmp_path: Path,
) -> None:
    frame = load_development_split(_isolated_dataset(tmp_path), "train")
    folds = make_grouped_folds(frame, n_splits=3, seed=7)
    assert len(folds) == 3
    excluded = set()
    for fold in folds:
        train_families = set(frame.iloc[fold.train_indices]["family_id"])
        score_families = set(frame.iloc[fold.validation_indices]["family_id"])
        assert train_families.isdisjoint(score_families)
        excluded.update(fold.excluded_exact_duplicate_record_ids)
    assert excluded == {"a0", "b0"}


def test_frozen_fold_map_is_hashed_stable_and_excludes_overlap(tmp_path: Path) -> None:
    frame = load_development_split(_isolated_dataset(tmp_path), "train")
    assignment = tmp_path / "folds.json"
    assignment.write_text(
        json.dumps(
            {
                "n_splits": 5,
                "train_split_sha256": "train-hash",
                "family_to_fold": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            }
        ),
        encoding="utf-8",
    )
    extra = pd.DataFrame(
        [
            ["d0", "d", "train", "epsilon", 0],
            ["d1", "d", "train", "zeta", 1],
            ["e0", "e", "train", "eta", 0],
            ["e1", "e", "train", "theta", 1],
        ],
        columns=["record_id", "family_id", "split", "target_clause", "label"],
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    digest = _hash(assignment)
    folds = load_frozen_folds(
        frame,
        assignment,
        expected_sha256=digest,
        expected_train_sha256="train-hash",
    )
    assert [fold.held_out_families for fold in folds] == [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("e",),
    ]
    assert {record for fold in folds for record in fold.excluded_exact_duplicate_record_ids} == {
        "a0",
        "b0",
    }
    with pytest.raises(RuntimeError, match="hash"):
        load_frozen_folds(frame, assignment, expected_sha256="wrong")


def test_committed_fold_assignment_is_pinned_and_excludes_tcp() -> None:
    root = Path(__file__).resolve().parents[1]
    assignment_path = root / "experiments/transformer/configs/folds_v1.json"
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    split_policy = json.loads(
        (root / "experiments/transformer/configs/split_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(assignment["family_to_fold"]) == set(
        split_policy["assignments"]["train"]
    )
    assert set(assignment["family_to_fold"]).isdisjoint(
        split_policy["assignments"]["validation"]
    )
    assert set(assignment["family_to_fold"].values()) == {1, 2, 3, 4, 5}
    digest = _hash(assignment_path)
    for name in ("deberta_clause_only.yaml", "deberta_local_context.yaml"):
        config = yaml.safe_load(
            (root / "experiments/transformer/configs" / name).read_text(
                encoding="utf-8"
            )
        )
        assert config["selection"]["fold_assignment_sha256"] == digest
        assert config["selection"]["fold_assignment"].endswith("folds_v1.json")


def test_nested_inner_selection_is_disjoint_from_outer_evaluation(
    tmp_path: Path,
) -> None:
    frame = load_development_split(_isolated_dataset(tmp_path), "train")
    extra = pd.DataFrame(
        [
            ["d0", "d", "train", "epsilon", 0],
            ["d1", "d", "train", "zeta", 1],
            ["e0", "e", "train", "eta", 0],
            ["e1", "e", "train", "theta", 1],
        ],
        columns=["record_id", "family_id", "split", "target_clause", "label"],
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    outer_path = tmp_path / "outer.json"
    outer_path.write_text(
        json.dumps(
            {
                "n_splits": 5,
                "train_split_sha256": "train-hash",
                "family_to_fold": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            }
        ),
        encoding="utf-8",
    )
    outer = load_frozen_folds(
        frame,
        outer_path,
        expected_sha256=_hash(outer_path),
        expected_train_sha256="train-hash",
    )
    inner_path = tmp_path / "inner.json"
    inner_path.write_text(
        json.dumps(
            {
                "train_split_sha256": "train-hash",
                "eligibility": {
                    "minimum_positive_labels_before_duplicate_exclusion": 1,
                    "minimum_positive_labels_after_duplicate_exclusion": 1,
                    "maximum_family_rows": 2,
                },
                "outer_fold_to_inner_family": {
                    "1": "b",
                    "2": "c",
                    "3": "d",
                    "4": "e",
                    "5": "a",
                },
            }
        ),
        encoding="utf-8",
    )
    nested = load_nested_folds(
        frame,
        outer,
        inner_path,
        expected_sha256=_hash(inner_path),
        expected_train_sha256="train-hash",
    )
    for item in nested:
        assert set(item.fitting_indices).isdisjoint(item.selection_indices)
        assert set(item.outer.validation_indices).isdisjoint(item.fitting_indices)
        assert set(item.outer.validation_indices).isdisjoint(item.selection_indices)
        assert item.inner_selection_family not in item.outer.held_out_families


def test_committed_inner_assignment_is_hash_pinned_and_has_enough_positives() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "corpus/processed/transformer_partitioned/v1/development"
    frame = load_development_split(dataset_dir, "train")
    outer_path = root / "experiments/transformer/configs/folds_v1.json"
    outer = load_frozen_folds(
        frame,
        outer_path,
        expected_sha256=_hash(outer_path),
        expected_train_sha256=_hash(dataset_dir / "train.csv"),
    )
    inner_path = root / "experiments/transformer/configs/inner_selection_v1.json"
    nested = load_nested_folds(
        frame,
        outer,
        inner_path,
        expected_sha256=_hash(inner_path),
        expected_train_sha256=_hash(dataset_dir / "train.csv"),
    )
    assert [
        int(frame.iloc[item.selection_scoring_indices]["label"].sum())
        for item in nested
    ] == [47, 34, 80, 34, 34]
    for name in ("deberta_clause_only.yaml", "deberta_local_context.yaml"):
        config = yaml.safe_load(
            (root / "experiments/transformer/configs" / name).read_text(
                encoding="utf-8"
            )
        )
        assert config["selection"]["inner_selection_assignment_sha256"] == _hash(
            inner_path
        )


def test_training_materializes_outer_loader_only_after_checkpoint_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "models/transformer/train.py").read_text(encoding="utf-8")
    epoch_loop = source.index("for epoch in range")
    best_checkpoint_load = source.index("best_model =")
    outer_loader = source.index("outer_validation_loader =")
    assert epoch_loop < best_checkpoint_load < outer_loader
    assert "outer_validation_loader" not in source[epoch_loop:best_checkpoint_load]
    assert 'score = float(selection_metrics[training["checkpoint_metric"]])' in source
