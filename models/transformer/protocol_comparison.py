"""Compare saved learning-rate screens without loading training or test data."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split
from models.transformer.lr_screen_analysis import (
    CLASSICAL_ROOT,
    DEVELOPMENT_DIR,
    LEARNING_RATES,
    PRIMARY_SEED,
    SECOND_SEED,
    _paired_pr_auc_bootstrap,
    _read_transformer_run,
    lr_slug,
)

V1_ROOT = ROOT / "results/transformer/studies/lr-screen-v1"
V2_ROOT = ROOT / "results/transformer/studies/lr-screen-v2"


def _metadata() -> pd.DataFrame:
    development = load_development_split(DEVELOPMENT_DIR, "train")
    metadata = development[["record_id", "document_id", "section_ids"]].copy()
    metadata["section_cluster"] = (
        metadata["document_id"].astype(str)
        + "::"
        + metadata["section_ids"].astype(str)
    )
    return metadata


def _load_oof(root: Path, rate: float, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        predictions, _ = _read_transformer_run(root, rate, fold, seed)
        parts.append(
            predictions[["record_id", "label", "ambiguity_probability"]].assign(
                fold=fold
            )
        )
    return pd.concat(parts, ignore_index=True)


def paired_screen_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    first_name: str,
    second_name: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Compare two OOF prediction sets on identical scored records."""

    left = first.rename(columns={"ambiguity_probability": "probability_first"})
    right = second[["record_id", "ambiguity_probability"]].rename(
        columns={"ambiguity_probability": "probability_second"}
    )
    paired = left.merge(right, on="record_id", how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(second):
        raise RuntimeError("Compared screens do not contain identical scored records")
    paired = paired.merge(metadata, on="record_id", how="left", validate="one_to_one")
    if paired["section_cluster"].isna().any():
        raise RuntimeError("Missing section cluster for a scored record")
    paired = paired.reset_index(drop=True)

    per_fold: list[dict[str, float | int]] = []
    for fold in range(1, 6):
        subset = paired[paired["fold"] == fold]
        labels = subset["label"].to_numpy(dtype=np.int64)
        first_score = float(
            average_precision_score(labels, subset["probability_first"])
        )
        second_score = float(
            average_precision_score(labels, subset["probability_second"])
        )
        per_fold.append(
            {
                "fold": fold,
                "first_pr_auc": first_score,
                "second_pr_auc": second_score,
                "difference": first_score - second_score,
            }
        )
    differences = np.asarray([row["difference"] for row in per_fold])
    first_pooled = float(
        average_precision_score(paired["label"], paired["probability_first"])
    )
    second_pooled = float(
        average_precision_score(paired["label"], paired["probability_second"])
    )
    return {
        "first": first_name,
        "second": second_name,
        "rows": len(paired),
        "per_fold": per_fold,
        "first_mean_fold_pr_auc": float(
            np.mean([row["first_pr_auc"] for row in per_fold])
        ),
        "second_mean_fold_pr_auc": float(
            np.mean([row["second_pr_auc"] for row in per_fold])
        ),
        "mean_fold_difference": float(np.mean(differences)),
        "standard_error_fold_differences": float(
            np.std(differences, ddof=1) / math.sqrt(len(differences))
        ),
        "first_pooled_pr_auc": first_pooled,
        "second_pooled_pr_auc": second_pooled,
        "pooled_difference": first_pooled - second_pooled,
        "paired_row_bootstrap": _paired_pr_auc_bootstrap(
            paired,
            "probability_first",
            "probability_second",
            repetitions=repetitions,
            seed=seed,
        ),
        "section_cluster_bootstrap": _paired_pr_auc_bootstrap(
            paired,
            "probability_first",
            "probability_second",
            repetitions=repetitions,
            seed=seed + 1,
            cluster_column="section_cluster",
        ),
    }


def _tfidf_oof() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        path = (
            CLASSICAL_ROOT
            / "unweighted"
            / f"fold-{fold}"
            / "validation_predictions.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["included_in_scoring"]].copy()
        parts.append(
            frame[["record_id", "label", "ambiguity_probability"]].assign(fold=fold)
        )
    return pd.concat(parts, ignore_index=True)


def run(output_dir: Path, *, repetitions: int = 2000) -> Path:
    """Write immutable v1/v2 and second-seed protocol comparisons."""

    if output_dir.exists():
        raise FileExistsError(f"Comparison output is immutable: {output_dir}")
    metadata = _metadata()
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        v1_v2: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES):
            v1_v2[lr_slug(rate)] = paired_screen_comparison(
                _load_oof(V2_ROOT, rate, PRIMARY_SEED),
                _load_oof(V1_ROOT, rate, PRIMARY_SEED),
                metadata,
                first_name="v2_nested_selection_seed_1729",
                second_name="v1_outer_fold_selection_seed_1729",
                repetitions=repetitions,
                seed=4000 + rate_index * 10,
            )

        seed_stability: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES[:2]):
            seed_stability[lr_slug(rate)] = paired_screen_comparison(
                _load_oof(V2_ROOT, rate, SECOND_SEED),
                _load_oof(V2_ROOT, rate, PRIMARY_SEED),
                metadata,
                first_name="v2_seed_2718",
                second_name="v2_seed_1729",
                repetitions=repetitions,
                seed=5000 + rate_index * 10,
            )

        tfidf = _tfidf_oof()
        seed_1729_vs_tfidf: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES):
            seed_1729_vs_tfidf[lr_slug(rate)] = paired_screen_comparison(
                _load_oof(V2_ROOT, rate, PRIMARY_SEED),
                tfidf,
                metadata,
                first_name="v2_seed_1729",
                second_name="tfidf_unweighted",
                repetitions=repetitions,
                seed=5500 + rate_index * 10,
            )

        seed_2718_vs_tfidf: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES[:2]):
            seed_2718_vs_tfidf[lr_slug(rate)] = paired_screen_comparison(
                _load_oof(V2_ROOT, rate, SECOND_SEED),
                tfidf,
                metadata,
                first_name="v2_seed_2718",
                second_name="tfidf_unweighted",
                repetitions=repetitions,
                seed=6000 + rate_index * 10,
            )

        top_two_seed_2718 = {
            rate: seed_2718_vs_tfidf[rate]["first_mean_fold_pr_auc"]
            for rate in ("1e-5", "2e-5")
        }
        seed2_differences = [
            seed_stability["1e-5"]["per_fold"][index]["first_pr_auc"]
            - seed_stability["2e-5"]["per_fold"][index]["first_pr_auc"]
            for index in range(5)
        ]
        summary = {
            "schema_version": 1,
            "bootstrap_repetitions": repetitions,
            "cluster_definition": "document_id + section_ids, resampled within fold",
            "v1_vs_v2": v1_v2,
            "seed_2718_vs_seed_1729": seed_stability,
            "seed_1729_vs_tfidf": seed_1729_vs_tfidf,
            "seed_2718_vs_tfidf": seed_2718_vs_tfidf,
            "second_seed_top_two": {
                "mean_fold_pr_auc": top_two_seed_2718,
                "one_e_minus_five_minus_two_e_minus_five_per_fold": seed2_differences,
                "mean_difference": float(np.mean(seed2_differences)),
                "standard_error": float(
                    np.std(seed2_differences, ddof=1) / math.sqrt(5)
                ),
                "higher_mean_rate": max(top_two_seed_2718, key=top_two_seed_2718.get),
                "frozen_selected_rate": "1e-5",
                "selection_changed_by_second_seed": False,
            },
        }
        summary_path = temporary / "comparison.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "artifacts": {summary_path.name: sha256_file(summary_path)},
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=V2_ROOT / "protocol-comparison-final",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(run(args.output.resolve(), repetitions=args.bootstrap_repetitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
