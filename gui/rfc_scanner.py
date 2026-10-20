"""RFC parsing and Logistic Regression ranking for the local review GUI."""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from alignment.clause_segmentation import segment_document
from alignment.segmentation import parse_document
from models.classical.annotation_logreg_experiment2 import (
    TRAINING_FEATURE_COLUMNS,
    _transform_components,
    extract_v2_features,
)
from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT

MODEL_DIRECTORY = ROOT / "results/ML/v2_engineered_tfidf"
MODEL_PATH = MODEL_DIRECTORY / "final_model.joblib"
MAX_RFC_BYTES = 5_000_000
MAX_CANDIDATES = 5_000
MAX_RESPONSE_ROWS = MAX_CANDIDATES


@dataclass(frozen=True)
class LoadedClassifier:
    components: dict[str, Any]
    metadata: dict[str, Any]


def load_classifier(model_path: Path = MODEL_PATH) -> LoadedClassifier:
    """Load and validate the repository's locally trained classifier artifact."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved Logistic Regression model not found: {model_path}. "
            "Restore the local results/ML/v2_engineered_tfidf artifact first."
        )
    manifest_path = model_path.parent / "manifest.json"
    metrics_path = model_path.parent / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Model provenance files are missing beside {model_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if manifest.get("feature_set") != "engineered_tfidf":
        raise ValueError("Saved model is not the engineered + TF-IDF classifier")
    if manifest.get("feature_columns") != list(TRAINING_FEATURE_COLUMNS):
        raise ValueError("Model feature schema does not match the current pipeline")
    if manifest.get("validation_groups") != 11 or metrics.get("rows") != 2406:
        raise ValueError("Model cohort metadata does not match the documented study")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", InconsistentVersionWarning)
        components = joblib.load(model_path)
    compatibility_warnings = [
        str(item.message)
        for item in captured
        if issubclass(item.category, InconsistentVersionWarning)
    ]
    artifact_versions = sorted(
        {
            match.group(1)
            for warning in compatibility_warnings
            if (match := re.search(r"from version ([\d.]+)", warning))
        }
    )
    required = {"vectorizer", "numerical_scaler", "classifier", "feature_mode"}
    if not isinstance(components, dict) or not required.issubset(components):
        raise ValueError("Saved model artifact has an unexpected structure")
    classifier = components["classifier"]
    if components["feature_mode"] != "engineered_tfidf":
        raise ValueError("Saved classifier feature mode does not match its manifest")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError("Saved classifier does not expose binary classes [0, 1]")
    if classifier.n_features_in_ != (
        len(components["vectorizer"].vocabulary_) + len(TRAINING_FEATURE_COLUMNS)
    ):
        raise ValueError("Saved classifier input dimensions do not match its features")

    validation = metrics["validation"]
    metadata = {
        "name": "Balanced Logistic Regression · engineered features + TF–IDF",
        "artifact": model_path.name,
        "artifact_sha256": sha256_file(model_path),
        "average_precision": float(validation["pooled_oof_average_precision"]),
        "selection_rationale": (
            "Active single-RFC scorer: the saved Logistic Regression artifact. "
            "The newer transformer ensemble is a research candidate, not a "
            "live inference artifact; its 0.674 AP used a different cohort "
            "and is not directly comparable."
        ),
        "validation_rows": int(metrics["rows"]),
        "validation_positives": int(metrics["positive_rows"]),
        "validation_groups": int(metrics["validation_group_count"]),
        "validation_protocol": validation["strategy"],
        "prediction_threshold": float(validation["prediction_threshold"]),
        "artifact_sklearn_versions": artifact_versions,
        "runtime_sklearn_version": sklearn.__version__,
        "sklearn_compatibility_warnings": compatibility_warnings,
    }
    return LoadedClassifier(components=components, metadata=metadata)


def _prepare_candidate_frame(clauses: list[Any]) -> pd.DataFrame:
    positions_by_section: dict[str, list[int]] = {}
    for position, clause in enumerate(clauses):
        positions_by_section.setdefault(clause.section_id, []).append(position)

    contexts: dict[int, tuple[str, str]] = {}
    for positions in positions_by_section.values():
        for offset, position in enumerate(positions):
            previous = clauses[positions[offset - 1]].raw_text if offset > 0 else ""
            following = (
                clauses[positions[offset + 1]].raw_text
                if offset + 1 < len(positions)
                else ""
            )
            contexts[position] = (previous, following)

    rows: list[dict[str, str]] = []
    for position, clause in enumerate(clauses):
        explicit_normative = clause.classification == "NORMATIVE"
        if not explicit_normative:
            continue
        previous, following = contexts[position]
        rows.append(
            {
                "clause_id": clause.clause_id,
                "candidate_type": "EXPLICIT_NORMATIVE",
                "candidate_reason": clause.classification_reason,
                "section_id": clause.section_id,
                "section_heading": clause.section_heading,
                "section_path": " / ".join(clause.section_path),
                "target_clause": clause.raw_text,
                "previous_clause": previous,
                "next_clause": following,
                "line_start": str(clause.source_location.line_start or ""),
                "line_end": str(clause.source_location.line_end or ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "clause_id",
            "candidate_type",
            "candidate_reason",
            "section_id",
            "section_heading",
            "section_path",
            "target_clause",
            "previous_clause",
            "next_clause",
            "line_start",
            "line_end",
        ],
    )


def _score_candidate_rows(
    candidates: list[dict[str, Any]],
    loaded_model: LoadedClassifier,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    frame = pd.DataFrame(candidates)
    features = extract_v2_features(frame)
    matrix = _transform_components(
        "engineered_tfidf",
        loaded_model.components,
        frame["target_clause"],
        features,
        embeddings=None,
    )
    scores = np.asarray(
        loaded_model.components["classifier"].predict_proba(matrix)[:, 1],
        dtype=np.float64,
    )
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise RuntimeError("Classifier returned invalid RFC clause scores")
    threshold = float(loaded_model.metadata["prediction_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Model prediction threshold must be between 0 and 1")

    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        is_ambiguous = bool(score >= threshold)
        ranked.append(
            {
                **candidate,
                "relative_score": float(score),
                "prediction_threshold": threshold,
                "decision": "YES" if is_ambiguous else "NO",
                "prediction": "AMBIGUOUS" if is_ambiguous else "NOT_AMBIGUOUS",
                "qwen_justification": candidate.get("qwen_justification", ""),
            }
        )
    ranked.sort(key=lambda row: (-row["relative_score"], row["clause_id"]))
    return ranked


def scan_rfc_text(
    *,
    filename: str,
    content: str,
    loaded_model: LoadedClassifier,
) -> dict[str, Any]:
    """Score ambiguity only for clauses classified as explicitly normative."""
    if Path(filename).suffix.casefold() != ".txt":
        raise ValueError("Upload a rendered RFC plaintext file with a .txt extension")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The uploaded RFC is empty")
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_RFC_BYTES:
        raise ValueError(f"RFC plaintext must not exceed {MAX_RFC_BYTES} bytes")

    with tempfile.TemporaryDirectory(prefix="ietf-rfc-review-") as temporary:
        source = Path(temporary) / "uploaded-rfc.txt"
        normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
        source.write_text(normalized_content, encoding="utf-8")
        document = parse_document(source, document_type="rfc")
    clause_document = segment_document(document)
    candidates = _prepare_candidate_frame(clause_document.clauses)
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"RFC produced {len(candidates)} candidate clauses; "
            f"the limit is {MAX_CANDIDATES}"
        )

    scored_candidates = candidates
    threshold = float(loaded_model.metadata["prediction_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Model prediction threshold must be between 0 and 1")

    ranked = _score_candidate_rows(
        scored_candidates.to_dict(orient="records"),
        loaded_model,
    )
    if len(ranked) > MAX_RESPONSE_ROWS:
        raise ValueError(
            f"RFC produced {len(ranked)} candidate rows, above the response limit "
            f"of {MAX_RESPONSE_ROWS}"
        )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    clause_summary = clause_document.summary()
    ambiguous_count = sum(row["decision"] == "YES" for row in ranked)
    ambiguous_rows = [row for row in ranked if row["decision"] == "YES"]
    return {
        "document_id": document.document_id,
        "title": document.title or document.document_id,
        "candidate_count": len(ranked),
        "scored_count": len(scored_candidates),
        "ambiguous_count": ambiguous_count,
        "not_ambiguous_count": len(scored_candidates) - ambiguous_count,
        "prediction_threshold": threshold,
        "normative_count": clause_summary["normative"],
        "contextual_count": clause_summary["contextual"],
        "excluded_count": clause_summary["excluded"],
        "warnings": list(document.warnings),
        "rows": ambiguous_rows,
    }
