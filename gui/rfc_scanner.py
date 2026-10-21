"""RFC parsing and transformer ensemble classification for the local review GUI."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alignment.clause_segmentation import segment_document
from alignment.segmentation import parse_document
from experiments.transformer_final.final_annotation_human_reviewed_100_v1.analysis import (
    empirical_cdf,
    select_precision_threshold,
)
from models.transformer_final.tokenization import encode_example
from models.transformer.dataset import ROOT

EXPERIMENT_DIRECTORY = (
    ROOT
    / "experiments/transformer_final/final_annotation_human_reviewed_100_v1"
)
MAX_RFC_BYTES = 5_000_000
MAX_CANDIDATES = 5_000
MAX_RESPONSE_ROWS = MAX_CANDIDATES
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 3
MINIMUM_PRECISION_TARGET = 0.30
VOTE_THRESHOLD = 3
INFERENCE_BATCH_SIZE = 16


@dataclass(frozen=True)
class FoldClassifier:
    fold: int
    focal_model: Any
    partial4_model: Any
    focal_reference: np.ndarray
    partial4_reference: np.ndarray
    threshold: float


@dataclass(frozen=True)
class LoadedClassifier:
    components: dict[str, Any]
    metadata: dict[str, Any]


def _read_inner_predictions(
    experiment_directory: Path, method: str, outer_fold: int
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for inner_fold in range(1, INNER_FOLD_COUNT + 1):
        parent = (
            experiment_directory
            / "runs"
            / "inner"
            / method
            / f"outer_{outer_fold}"
            / f"inner_{inner_fold}"
        )
        paths = list(parent.glob("*/predictions/held_out_predictions.csv"))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected one {method} inner-fold prediction file for "
                f"outer fold {outer_fold}, inner fold {inner_fold}; found {len(paths)}"
            )
        frame = pd.read_csv(paths[0], dtype={"record_id": str})
        required = {"record_id", "label", "probability_yes"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Unexpected prediction schema in {paths[0]}")
        if frame["record_id"].duplicated().any():
            raise ValueError(f"Duplicate inner predictions in {paths[0]}")
        predictions.append(frame[["record_id", "label", "probability_yes"]])

    combined = pd.concat(predictions, ignore_index=True)
    if combined["record_id"].duplicated().any():
        raise ValueError(
            f"Inner OOF predictions overlap for {method}, outer fold {outer_fold}"
        )
    probabilities = combined["probability_yes"].to_numpy(dtype=np.float64)
    labels = combined["label"].to_numpy(dtype=np.int64)
    if (
        not np.isfinite(probabilities).all()
        or not np.isin(labels, [0, 1]).all()
        or not combined["probability_yes"].between(0, 1).all()
    ):
        raise ValueError(
            f"Invalid inner OOF predictions for {method}, outer fold {outer_fold}"
        )
    return combined


def _build_fold_calibration(
    focal_inner: pd.DataFrame, partial4_inner: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float]:
    focal = focal_inner.rename(columns={"probability_yes": "focal_score"})
    partial4 = partial4_inner.rename(columns={"probability_yes": "partial4_score"})
    paired = focal.merge(
        partial4,
        on=["record_id", "label"],
        validate="one_to_one",
    )
    if len(paired) != len(focal_inner) or len(paired) != len(partial4_inner):
        raise ValueError("Focal and Partial-4 inner OOF cohorts do not align")
    focal_reference = paired["focal_score"].to_numpy(dtype=np.float64)
    partial4_reference = paired["partial4_score"].to_numpy(dtype=np.float64)
    ensemble_scores = (
        0.5 * empirical_cdf(focal_reference, focal_reference)
        + 0.5 * empirical_cdf(partial4_reference, partial4_reference)
    )
    selected = select_precision_threshold(
        paired["label"].to_numpy(dtype=np.int64),
        ensemble_scores,
        MINIMUM_PRECISION_TARGET,
    )
    if not selected["feasible"] or not np.isfinite(selected["threshold"]):
        raise RuntimeError("The inner OOF ensemble has no feasible 0.30 threshold")
    return focal_reference, partial4_reference, float(selected["threshold"])


def _verified_checkpoint_path(
    experiment_directory: Path, method: str, outer_fold: int
) -> Path:
    run_parent = (
        experiment_directory
        / "runs"
        / "outer"
        / method
        / f"fold_{outer_fold}"
    )
    checkpoint_paths = list(run_parent.glob("*/checkpoints/final_epoch"))
    if len(checkpoint_paths) != 1:
        raise FileNotFoundError(
            f"Expected one {method} checkpoint for outer fold {outer_fold}; "
            f"found {len(checkpoint_paths)} under {run_parent}"
        )
    checkpoint = checkpoint_paths[0]
    metadata_path = checkpoint.parent / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {metadata_path}")
    run_manifest_path = checkpoint.parent.parent / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint run manifest is missing: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_metadata_hash = (
        run_manifest.get("artifact_hashes", {})
        .get("checkpoint_metadata", {})
        .get("sha256")
    )
    if (
        not expected_metadata_hash
        or _sha256_file(metadata_path) != expected_metadata_hash
    ):
        raise ValueError(f"Checkpoint metadata hash mismatch: {metadata_path}")
    checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for filename, expected_hash in checkpoint_metadata.get("files", {}).items():
        artifact_path = checkpoint / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Checkpoint file is missing: {artifact_path}")
        if _sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"Checkpoint file hash mismatch: {artifact_path}")
    if not checkpoint_metadata.get("files"):
        raise ValueError(f"Checkpoint file hashes are missing: {metadata_path}")
    return checkpoint


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as artifact:
        while True:
            chunk = artifact.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_classifier() -> LoadedClassifier:
    """Load the validated five-fold focal + Partial-4 transformer classifier."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import (
        core as experiment_core,
    )

    # Validate the frozen protocol and every OOF prediction artifact before inference.
    experiment_core.verify_frozen()
    for kind, expected_count in (("outer", 10), ("inner", 30)):
        specs = experiment_core.iter_specs(kind)
        if len(specs) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {kind} cells, found {len(specs)}"
            )
        for _, spec in specs:
            experiment_core.validate_cell(spec)
    config = experiment_core.load_local_config(EXPERIMENT_DIRECTORY / "config.json")
    model_path = Path(config["model"]["default_local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The pinned DeBERTa tokenizer has no padding token")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds: list[FoldClassifier] = []
    for outer_fold in range(1, OUTER_FOLD_COUNT + 1):
        focal_inner = _read_inner_predictions(
            EXPERIMENT_DIRECTORY, "focal", outer_fold
        )
        partial4_inner = _read_inner_predictions(
            EXPERIMENT_DIRECTORY, "partial4", outer_fold
        )
        focal_reference, partial4_reference, threshold = _build_fold_calibration(
            focal_inner, partial4_inner
        )
        models: dict[str, Any] = {}
        for method in ("focal", "partial4"):
            checkpoint = _verified_checkpoint_path(
                EXPERIMENT_DIRECTORY, method, outer_fold
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            label2id = {
                str(label): int(index)
                for label, index in model.config.label2id.items()
            }
            id2label = {
                int(index): str(label)
                for index, label in model.config.id2label.items()
            }
            if (
                model.config.num_labels != 2
                or label2id != {"No": 0, "Yes": 1}
                or id2label != {0: "No", 1: "Yes"}
            ):
                raise ValueError(f"Unexpected binary label mapping in {checkpoint}")
            model.to(device)
            model.eval()
            models[method] = model
        folds.append(
            FoldClassifier(
                fold=outer_fold,
                focal_model=models["focal"],
                partial4_model=models["partial4"],
                focal_reference=focal_reference,
                partial4_reference=partial4_reference,
                threshold=threshold,
            )
        )

    metadata = {
        "name": "DeBERTa-v3 Focal + Partial-4 five-fold ensemble",
        "average_precision": 0.6735728,
        "roc_auc": 0.9118418,
        "validation_rows": 2143,
        "validation_positives": 138,
        "validation_groups": 9,
        "validation_protocol": "Nested grouped family OOF; current human-reviewed source",
        "minimum_precision_target": MINIMUM_PRECISION_TARGET,
        "vote_threshold": VOTE_THRESHOLD,
        "vote_count": OUTER_FOLD_COUNT,
        "prediction_threshold": VOTE_THRESHOLD / OUTER_FOLD_COUNT,
        "score_semantics": "mean ECDF ranking score; not a calibrated probability",
        "decision_policy": (
            "YES when at least 3 of 5 outer-fold models exceed their own "
            "inner-OOF threshold selected for a 0.30 precision target"
        ),
        "inference_device": str(device),
        "selection_rationale": (
            "Uses the current-source human-reviewed transformer study. Each "
            "fold's Focal + Partial-4 ECDF threshold is fitted only on its "
            "inner OOF predictions. Reported OOF metrics evaluate one "
            "held-out outer-fold score per clause, not the runtime mean score "
            "or 3-of-5 vote; neither deployment aggregate has been measured "
            "or guarantees production precision."
        ),
    }
    return LoadedClassifier(
        components={"folds": folds, "tokenizer": tokenizer, "device": device},
        metadata=metadata,
    )


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


def _predict_model_probabilities(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
) -> np.ndarray:
    import torch

    from models.transformer_final.modeling import (
        collate_clauses,
        supported_model_fields,
    )

    if tokenizer.pad_token_id is None:
        raise ValueError("The pinned DeBERTa tokenizer has no padding token")
    model_fields = supported_model_fields(model)
    if "input_ids" not in model_fields:
        raise RuntimeError("Transformer model does not accept input_ids")

    probabilities: list[np.ndarray] = []
    for batch_start in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[batch_start : batch_start + INFERENCE_BATCH_SIZE]
        examples = []
        for offset, text in enumerate(batch_texts, start=batch_start):
            encoded = encode_example(
                tokenizer,
                target=text,
                section="",
                previous="",
                following="",
                input_mode="target_only",
                max_length=256,
            )
            example: dict[str, Any] = {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "labels": 0,
                "row_position": offset,
                "target_truncated": encoded.target_truncated,
                "context_truncated": encoded.context_truncated,
            }
            if encoded.token_type_ids is not None:
                example["token_type_ids"] = encoded.token_type_ids
            examples.append(example)
        padded = collate_clauses(examples, pad_token_id=tokenizer.pad_token_id)
        model_inputs = {
            name: padded[name].to(device)
            for name in model_fields
            if name in padded
        }
        with torch.inference_mode():
            logits = model(**model_inputs).logits
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError("Transformer returned invalid binary logits")
            batch_probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
        probabilities.append(batch_probabilities.cpu().numpy().astype(np.float64))

    scores = np.concatenate(probabilities) if probabilities else np.empty(0)
    if scores.shape != (len(texts),) or not np.isfinite(scores).all():
        raise RuntimeError("Transformer returned invalid RFC clause scores")
    if np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("Transformer probabilities are outside [0, 1]")
    return scores


def _fold_ensemble_scores(
    fold: FoldClassifier,
    focal_probabilities: np.ndarray,
    partial4_probabilities: np.ndarray,
) -> np.ndarray:
    if focal_probabilities.shape != partial4_probabilities.shape:
        raise ValueError("Focal and Partial-4 predictions do not align")
    if not 0.0 <= fold.threshold <= 1.0:
        raise ValueError(f"Invalid threshold for outer fold {fold.fold}")
    return (
        0.5 * empirical_cdf(fold.focal_reference, focal_probabilities)
        + 0.5 * empirical_cdf(fold.partial4_reference, partial4_probabilities)
    )


def _score_candidate_rows(
    candidates: list[dict[str, Any]],
    loaded_model: LoadedClassifier,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    folds = loaded_model.components.get("folds")
    if not isinstance(folds, list) or len(folds) != OUTER_FOLD_COUNT:
        raise ValueError("Loaded transformer model must contain five outer folds")
    tokenizer = loaded_model.components.get("tokenizer")
    device = loaded_model.components.get("device")
    if tokenizer is None or device is None:
        raise ValueError("Loaded transformer tokenizer or inference device is missing")
    vote_threshold = int(loaded_model.metadata["vote_threshold"])
    if not 1 <= vote_threshold <= len(folds):
        raise ValueError("Model vote threshold is outside the fold count")

    texts = [str(candidate["target_clause"]) for candidate in candidates]
    rank_scores = np.zeros(len(candidates), dtype=np.float64)
    fold_votes = np.zeros(len(candidates), dtype=np.int64)
    fold_thresholds: list[float] = []
    for fold in folds:
        focal_probabilities = _predict_model_probabilities(
            fold.focal_model, tokenizer, texts, device
        )
        partial4_probabilities = _predict_model_probabilities(
            fold.partial4_model, tokenizer, texts, device
        )
        fold_scores = _fold_ensemble_scores(
            fold, focal_probabilities, partial4_probabilities
        )
        if fold_scores.shape != (len(candidates),) or not np.isfinite(fold_scores).all():
            raise RuntimeError(f"Invalid ensemble scores from outer fold {fold.fold}")
        rank_scores += fold_scores
        fold_votes += fold_scores >= fold.threshold
        fold_thresholds.append(float(fold.threshold))
    rank_scores /= len(folds)
    vote_fraction = fold_votes / len(folds)
    threshold = vote_threshold / len(folds)

    ranked = []
    for candidate, score, votes, fraction in zip(
        candidates, rank_scores, fold_votes, vote_fraction, strict=True
    ):
        is_ambiguous = bool(votes >= vote_threshold)
        ranked.append(
            {
                **candidate,
                "relative_score": float(score),
                "fold_vote_fraction": float(fraction),
                "fold_votes": int(votes),
                "fold_vote_count": len(folds),
                "fold_thresholds": fold_thresholds,
                "vote_threshold": vote_threshold,
                "vote_cutoff_fraction": threshold,
                "prediction_threshold": threshold,
                "decision": "YES" if is_ambiguous else "NO",
                "prediction": "AMBIGUOUS" if is_ambiguous else "NOT_AMBIGUOUS",
                "qwen_justification": candidate.get("qwen_justification", ""),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["relative_score"],
            -row["fold_votes"],
            row["clause_id"],
        )
    )
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
        "vote_cutoff_fraction": threshold,
        "prediction_threshold": threshold,
        "vote_threshold": int(loaded_model.metadata["vote_threshold"]),
        "vote_count": int(loaded_model.metadata["vote_count"]),
        "normative_count": clause_summary["normative"],
        "contextual_count": clause_summary["contextual"],
        "excluded_count": clause_summary["excluded"],
        "warnings": list(document.warnings),
        "rows": ambiguous_rows,
    }
