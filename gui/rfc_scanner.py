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
MAX_RESPONSE_ROWS = 2_000


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
        "validation_rows": int(metrics["rows"]),
        "validation_positives": int(metrics["positive_rows"]),
        "validation_groups": int(metrics["validation_group_count"]),
        "validation_protocol": validation["strategy"],
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
        if clause.classification != "NORMATIVE":
            continue
        previous, following = contexts[position]
        rows.append(
            {
                "clause_id": clause.clause_id,
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
    return pd.DataFrame(rows)


def scan_rfc_text(
    *,
    filename: str,
    content: str,
    loaded_model: LoadedClassifier,
) -> dict[str, Any]:
    """Parse an RFC plaintext upload and rank its explicit normative clauses."""
    if Path(filename).suffix.casefold() != ".txt":
        raise ValueError("Upload a rendered RFC plaintext file with a .txt extension")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The uploaded RFC is empty")
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_RFC_BYTES:
        raise ValueError(f"RFC plaintext must not exceed {MAX_RFC_BYTES} bytes")

    with tempfile.TemporaryDirectory(prefix="ietf-rfc-review-") as temporary:
        source = Path(temporary) / "uploaded-rfc.txt"
        source.write_text(content, encoding="utf-8")
        document = parse_document(source, document_type="rfc")
    clause_document = segment_document(document)
    candidates = _prepare_candidate_frame(clause_document.clauses)
    if candidates.empty:
        raise ValueError(
            "No explicitly normative clauses were found. "
            "The parser may need another RFC plaintext rendering."
        )
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"RFC produced {len(candidates)} normative clauses; "
            f"the limit is {MAX_CANDIDATES}"
        )

    features = extract_v2_features(candidates)
    matrix = _transform_components(
        "engineered_tfidf",
        loaded_model.components,
        candidates["target_clause"],
        features,
        embeddings=None,
    )
    scores = np.asarray(
        loaded_model.components["classifier"].predict_proba(matrix)[:, 1],
        dtype=np.float64,
    )
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise RuntimeError("Classifier returned invalid RFC clause scores")

    ranked: list[dict[str, Any]] = []
    for candidate, score in zip(
        candidates.to_dict(orient="records"), scores, strict=True
    ):
        ranked.append(
            {
                **candidate,
                "relative_score": float(score),
                "qwen_justification": "",
            }
        )
    ranked.sort(key=lambda row: (-row["relative_score"], row["clause_id"]))
    if len(ranked) > MAX_RESPONSE_ROWS:
        raise ValueError(
            f"RFC produced {len(ranked)} candidate rows, above the response limit "
            f"of {MAX_RESPONSE_ROWS}"
        )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    clause_summary = clause_document.summary()
    return {
        "document_id": document.document_id,
        "title": document.title or document.document_id,
        "candidate_count": len(ranked),
        "normative_count": clause_summary["normative"],
        "manual_review_count": clause_summary["manual_review"],
        "contextual_count": clause_summary["contextual"],
        "excluded_count": clause_summary["excluded"],
        "warnings": list(document.warnings),
        "rows": ranked,
    }
