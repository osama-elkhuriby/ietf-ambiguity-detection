"""Run target-only TF-IDF on the transformer's exact nested fitting IDs."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.classical.feature_pipeline import sha256_file
from models.classical.tfidf_comparator import _vectorizer, binary_metrics
from models.transformer.dataset import ROOT
from models.transformer.experiment import (
    compute_training_weights,
    load_development_split,
    load_frozen_folds,
    load_nested_folds,
)

DEFAULT_CONFIG = ROOT / "experiments/classical/configs/tfidf_logreg_v1.json"
DEFAULT_TRANSFORMER_CONFIG = (
    ROOT / "experiments/transformer/configs/deberta_clause_only.yaml"
)
DEFAULT_OUTPUT = ROOT / "experiments/classical/results/tfidf-matched-nested-v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, transformer_config_path: Path, output: Path) -> Path:
    """Fit five immutable class-weighted nested-data comparator folds."""

    import yaml

    config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer_config = yaml.safe_load(
        transformer_config_path.read_text(encoding="utf-8")
    )
    dataset_dir = (ROOT / config["development_dataset_dir"]).resolve()
    train = load_development_split(dataset_dir, "train")
    train_path = dataset_dir / "train.csv"
    fold_path = (ROOT / transformer_config["selection"]["fold_assignment"]).resolve()
    inner_path = (
        ROOT / transformer_config["selection"]["inner_selection_assignment"]
    ).resolve()
    folds = load_frozen_folds(
        train,
        fold_path,
        expected_sha256=transformer_config["selection"]["fold_assignment_sha256"],
        expected_train_sha256=sha256_file(train_path),
    )
    nested_folds = load_nested_folds(
        train,
        folds,
        inner_path,
        expected_sha256=transformer_config["selection"][
            "inner_selection_assignment_sha256"
        ],
        expected_train_sha256=sha256_file(train_path),
    )
    if output.exists():
        raise FileExistsError(f"Matched TF-IDF output is immutable: {output}")
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        summary: list[dict[str, Any]] = []
        for nested in nested_folds:
            fold = nested.outer
            fitting = train.iloc[nested.fitting_indices].reset_index(drop=True)
            validation = train.iloc[fold.validation_indices].reset_index(drop=True)
            scoring_ids = set(train.iloc[fold.scoring_indices]["record_id"].astype(str))
            vectorizer = _vectorizer(config["tfidf"])
            fitting_matrix = vectorizer.fit_transform(fitting["target_clause"].astype(str))
            validation_matrix = vectorizer.transform(
                validation["target_clause"].astype(str)
            )
            weights = compute_training_weights(fitting, "class")
            model = LogisticRegression(
                C=float(config["logistic_regression"]["c"]),
                solver=str(config["logistic_regression"]["solver"]),
                max_iter=int(config["logistic_regression"]["max_iter"]),
                random_state=int(config["seed"]),
            )
            model.fit(
                fitting_matrix,
                fitting["label"].to_numpy(dtype=np.int64),
                sample_weight=weights,
            )
            probabilities = model.predict_proba(validation_matrix)[:, 1]
            predictions = validation[
                ["record_id", "family_id", "label", "label_tier"]
            ].copy()
            predictions["raw_logit"] = model.decision_function(validation_matrix)
            predictions["ambiguity_probability"] = probabilities
            predictions["predicted_label_0_5"] = (probabilities >= 0.5).astype(int)
            predictions["included_in_scoring"] = predictions["record_id"].astype(
                str
            ).isin(scoring_ids)
            scored = predictions[predictions["included_in_scoring"]]
            metrics = binary_metrics(
                scored["label"].to_numpy(dtype=np.int64),
                scored["ambiguity_probability"].to_numpy(dtype=np.float64),
            )
            fold_dir = temporary / f"fold-{fold.number}"
            fold_dir.mkdir()
            predictions.to_csv(
                fold_dir / "validation_predictions.csv", index=False, lineterminator="\n"
            )
            _write_json(fold_dir / "metrics.json", metrics)
            _write_json(
                fold_dir / "run_config.json",
                {
                    "variant": "target_clause_only",
                    "weight_strategy": "class",
                    "training_ids": "exact_transformer_nested_fitting_ids",
                    "fold": fold.number,
                    "seed": config["seed"],
                    "tfidf": config["tfidf"],
                    "logistic_regression": config["logistic_regression"],
                },
            )
            _write_json(
                fold_dir / "run_manifest.json",
                {
                    "schema_version": 1,
                    "fold": fold.number,
                    "fitting_rows": len(fitting),
                    "fitting_positives": int(fitting["label"].sum()),
                    "inner_family_excluded": nested.inner_selection_family,
                    "outer_rows": len(validation),
                    "outer_scored_rows": len(scored),
                    "train_split_sha256": sha256_file(train_path),
                    "fold_assignment_sha256": sha256_file(fold_path),
                    "inner_selection_assignment_sha256": sha256_file(inner_path),
                },
            )
            joblib.dump(vectorizer, fold_dir / "tfidf_vectorizer.joblib")
            joblib.dump(model, fold_dir / "logistic_regression.joblib")
            summary.append({"fold": fold.number, **metrics})
        pd.DataFrame(summary).to_csv(
            temporary / "fold_metrics.csv", index=False, lineterminator="\n"
        )
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "scope": "development_train_only",
                "comparison": "matched_nested_fitting_ids",
                "config_sha256": sha256_file(config_path),
                "transformer_config_sha256": sha256_file(transformer_config_path),
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--transformer-config", type=Path, default=DEFAULT_TRANSFORMER_CONFIG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run(
        args.config.resolve(), args.transformer_config.resolve(), args.output.resolve()
    )
    print(f"Matched TF-IDF written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
