"""Binary metrics with explicit undefined-value handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or len(labels) == 0:
        raise ValueError("Labels and probabilities must be aligned, non-empty vectors")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Labels must be binary")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    prevalence = positives / len(labels)
    warnings: list[str] = []
    if positives == 0:
        ap = None
        warnings.append("average_precision_undefined_no_positives")
    else:
        ap = float(average_precision_score(labels, probabilities))
    if positives == 0 or negatives == 0:
        roc_auc = None
        warnings.append("roc_auc_undefined_single_class")
    else:
        roc_auc = float(roc_auc_score(labels, probabilities))
    mcc = (
        float(matthews_corrcoef(labels, predictions))
        if len(np.unique(labels)) > 1 and len(np.unique(predictions)) > 1
        else 0.0
    )
    return {
        "rows": len(labels),
        "positive_labels": positives,
        "negative_labels": negatives,
        "positive_prevalence": prevalence,
        "average_precision": ap,
        "pr_auc": ap,
        "roc_auc": roc_auc,
        "ap_prevalence_lift": None
        if ap is None or prevalence == 0
        else ap / prevalence,
        "threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "mcc": mcc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "undefined_metric_warnings": warnings,
    }


def metrics_for_predictions(
    frame: pd.DataFrame, *, threshold: float = 0.5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall = binary_metrics(
        frame["label"].to_numpy(),
        frame["probability_yes"].to_numpy(),
        threshold=threshold,
    )
    families = []
    for family, subset in frame.groupby("family_id", sort=True):
        families.append(
            {
                "family_id": str(family),
                **binary_metrics(
                    subset["label"].to_numpy(),
                    subset["probability_yes"].to_numpy(),
                    threshold=threshold,
                ),
            }
        )
    return overall, families
