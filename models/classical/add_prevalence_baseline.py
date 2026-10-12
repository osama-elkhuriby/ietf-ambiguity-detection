"""Create a non-mutating prevalence-baseline annotation for an existing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.classical.tfidf_comparator import binary_metrics, bootstrap_intervals


def build_annotation(
    predictions_path: Path,
    metrics_path: Path,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Recompute metrics and add prevalence without altering the source run."""

    predictions = pd.read_csv(predictions_path)
    required = {"label", "ambiguity_probability", "included_in_scoring"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions are missing columns: {sorted(required - set(predictions))}")
    include = predictions["included_in_scoring"]
    if include.dtype != bool:
        include = include.astype(str).str.lower().eq("true")
    scored = predictions[include]
    labels = scored["label"].to_numpy(dtype=np.int64)
    probabilities = scored["ambiguity_probability"].to_numpy(dtype=np.float64)
    metrics = binary_metrics(labels, probabilities)
    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name in ("pr_auc", "precision", "recall", "f1", "f2", "mcc"):
        if not np.isclose(metrics[name], source_metrics[name], rtol=0, atol=1e-7):
            raise RuntimeError(f"Recomputed {name} does not match source metrics")
    metrics["bootstrap_95_percentile"] = bootstrap_intervals(
        labels,
        probabilities,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    return {
        "schema_version": 1,
        "annotation_type": "non_mutating_prevalence_baseline",
        "reason": "The completed immutable run predated the no-skill baseline rule.",
        "source_hashes": {
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
        "metrics_with_no_skill_baseline": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=314160)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Annotation already exists: {args.output}")
    annotation = build_annotation(
        args.predictions,
        args.metrics,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
