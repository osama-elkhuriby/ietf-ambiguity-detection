"""LLM refinement for rule-based MANUAL_REVIEW clauses."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = Any  # type: ignore[misc, assignment]


# ============================================================
# Configuration
# ============================================================

LLM_BATCH_SIZE = 5

# Be stricter when accepting NORMATIVE.
NORMATIVE_THRESHOLD = 0.80
CONTEXTUAL_THRESHOLD = 0.70


# ============================================================
# Environment
# ============================================================

def _load_env_file() -> None:
    """
    Load project-level .env variables without requiring python-dotenv.

    Expected project structure:

        project/
            .env
            models/
                llm/
                    manual_review_refiner.py
    """

    env_path = Path(__file__).resolve().parents[2] / ".env"

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = [
            part.strip()
            for part in line.split("=", 1)
        ]

        if (
            key
            and value
            and key not in os.environ
        ):
            os.environ[key] = value.strip("\"'")


_load_env_file()


def _resolve_llm_config() -> tuple[
    str | None,
    str | None,
    str | None,
]:
    """
    Resolve API key, base URL, and model.

    Preferred variables:

        LLM_API_KEY
        LLM_BASE_URL
        LLM_MODEL

    Other provider variables are accepted as fallbacks.
    """

    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("API_KEY")
    )

    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("MODEL")
        or "openai/gpt-oss-120b"
    )

    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GROQ_BASE_URL")
        or "https://api.groq.com/openai/v1"
    )

    if api_key:
        os.environ.setdefault(
            "LLM_API_KEY",
            api_key,
        )

    if model:
        os.environ.setdefault(
            "LLM_MODEL",
            model,
        )

    if base_url:
        os.environ.setdefault(
            "LLM_BASE_URL",
            base_url,
        )

    return (
        api_key,
        base_url,
        model,
    )


LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = (
    _resolve_llm_config()
)


# ============================================================
# Result model
# ============================================================

@dataclass(slots=True)
class LLMClassification:
    clause_id: str

    # Final classification after our confidence gate.
    classification: str

    # Classification originally returned by the LLM.
    raw_classification: str

    # Self-reported model confidence.
    confidence: float

    reason: str


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You classify TARGET clauses from IETF Internet-Drafts and RFCs.

The rule-based classifier has already processed these clauses.

You receive only clauses for which the rule-based classifier
could not make a sufficiently confident decision.

Your task is NOT to decide whether text is technically important,
part of a standard, or associated with a normative protocol component.

Your task is ONLY to determine whether the TARGET natural-language
clause itself imposes normative behavior.


CLASSIFICATIONS
===============

NORMATIVE

The TARGET clause imposes a requirement, permission, prohibition,
recommendation, conformance condition, or explicit behavioral
constraint on a protocol participant or implementation.

Examples of protocol actors include:

- client
- server
- sender
- receiver
- endpoint
- peer
- implementation
- protocol participant


CONTEXTUAL

The TARGET clause explains, describes, defines, summarizes, motivates,
justifies, reports a fact, describes a property, or provides
background information WITHOUT imposing a behavioral requirement.


UNCERTAIN

There is insufficient evidence to reliably determine whether the
TARGET clause imposes normative behavior.

Use UNCERTAIN instead of guessing.


CRITICAL DISTINCTION
====================

"NORMATIVE" does NOT mean:

- the text occurs in an RFC or Internet-Draft;
- the text occurs in a normative section;
- the text defines a protocol component;
- the text is technically important;
- implementations need to understand the concept;
- the statement affects interoperability.

"NORMATIVE" means that the TARGET natural-language clause itself
clearly imposes or defines participant or implementation behavior.


STRICT DECISION RULES
=====================

1. Before selecting NORMATIVE, ask:

   "Does this TARGET clause clearly constrain what a protocol
   participant or implementation is required, permitted,
   recommended, or prohibited to do?"

   If the answer is not clearly yes, do NOT select NORMATIVE.


2. Do NOT classify a clause as NORMATIVE merely because it appears
   in a technical, security, processing, message-format,
   interoperability, or protocol section.


3. Section headings are context only.

   Never use a section heading alone as evidence of normativity.


4. A statement describing how a protocol, function, algorithm,
   structure, state, field, or message works is NOT automatically
   normative.


5. A statement describing:

   - interoperability
   - compatibility
   - sufficiency
   - capability
   - availability
   - expected behavior
   - deployment
   - implementation status
   - design rationale
   - protocol architecture

   is normally CONTEXTUAL unless it directly imposes behavior.


6. Words such as:

   sufficient
   necessary
   useful
   compatible
   supported
   available
   expected
   common
   intended
   appropriate
   possible
   capable

   do NOT themselves make a statement normative.


7. Example:

   "The set of curves specified here is sufficient for
   interoperability with currently deployed TLS implementations."

   Classification:
   CONTEXTUAL

   Reason:
   It describes an interoperability property without imposing
   participant behavior.


8. Example:

   "The deprotection functions reverse the process."

   Classification:
   CONTEXTUAL

   Reason:
   It describes function behavior without imposing a requirement.


9. A factual description of current implementations is normally
   CONTEXTUAL.

   Example:

   "Most existing implementations support this extension."

   Classification:
   CONTEXTUAL


10. Technical importance does NOT imply normativity.


11. BCP 14 keywords are strong evidence of NORMATIVE usage when
    they actually impose behavior:

    MUST
    MUST NOT
    REQUIRED
    SHALL
    SHALL NOT
    SHOULD
    SHOULD NOT
    RECOMMENDED
    NOT RECOMMENDED
    MAY
    OPTIONAL


12. Merely mentioning a BCP 14 keyword does NOT automatically
    make a clause normative.


13. Absence of BCP 14 keywords does NOT automatically imply
    CONTEXTUAL.


14. A clause may be implicitly NORMATIVE without a BCP 14 keyword,
    but ONLY when a behavioral or conformance constraint is clear.


15. Example:

    "The client MUST abort the connection if validation fails."

    Classification:
    NORMATIVE

    Reason:
    It imposes mandatory client behavior.


16. Example:

    "A compliant implementation rejects certificates with an
    invalid signature."

    Classification:
    NORMATIVE

    Reason:
    It defines required conformance behavior.


17. Be particularly careful with declarative statements.

    Example:

    "The server sends X after receiving Y."

    This could be descriptive or prescriptive.

    Do not automatically classify it as NORMATIVE.

    Use surrounding context only if necessary to determine whether
    the TARGET clearly specifies required behavior.


18. Previous and next clauses may help interpretation, but you
    classify ONLY the TARGET clause.


19. Do NOT transfer normativity from neighboring clauses.

    A nearby MUST statement does not make surrounding descriptive
    sentences normative.


20. Do NOT invent implicit requirements that are not stated or
    clearly implied by the TARGET.


21. If both NORMATIVE and CONTEXTUAL interpretations remain
    plausible, return UNCERTAIN.


22. Prefer UNCERTAIN over unsupported NORMATIVE classification.


23. A clearly descriptive sentence should be CONTEXTUAL, not
    UNCERTAIN.


FORMAL / NON-PROSE CONTENT
=========================

The intended task is classification of natural-language clauses.

If a TARGET consists primarily of:

- protocol grammar
- ABNF
- structure declarations
- field declarations
- vector declarations
- schema notation
- source code
- pseudocode
- formal syntax

do NOT infer a behavioral requirement merely because it defines
a protocol component.

For example:

PskIdentity identities<7..2^16-1>;
PskBinderEntry binders<33..2^16-1>;

This is formal protocol syntax, not a natural-language normative
requirement.

If such text unexpectedly reaches this classifier, return:

classification = UNCERTAIN

with low confidence.

The parser/segmenter should normally exclude such content before
this stage.


CONFIDENCE
==========

Return confidence between 0.0 and 1.0.

Use approximately:

0.90 - 1.00
Very clear evidence.

0.80 - 0.89
Strong evidence.

0.70 - 0.79
Reasonably strong evidence.

0.60 - 0.69
Moderate evidence.

Below 0.60
Insufficient evidence for a reliable definitive decision.

If confidence is below 0.60, classification should normally
be UNCERTAIN.

Do NOT inflate confidence merely because the sentence sounds
technical.


REASON
======

The reason must:

- refer only to the TARGET clause;
- explain the selected classification;
- be one short sentence;
- contain no more than 20 words;
- avoid extended reasoning.


OUTPUT
======

Return JSON only.

Do not return Markdown.

Do not include explanations outside the JSON.

Expected format:

{
  "results": [
    {
      "clause_id": "...",
      "classification": "NORMATIVE",
      "confidence": 0.92,
      "reason": "Imposes mandatory behavior on the protocol participant."
    }
  ]
}

classification must be exactly one of:

NORMATIVE
CONTEXTUAL
UNCERTAIN
""".strip()


# ============================================================
# Client
# ============================================================

def make_client() -> OpenAI:
    """Create an OpenAI-compatible API client."""

    try:
        from openai import OpenAI as OpenAIClient

    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional dependency 'openai' is not installed. "
            "Install it with: pip install openai"
        ) from exc

    api_key, base_url, _ = (
        _resolve_llm_config()
    )

    # Ollama does not require authentication,
    # but OpenAI's Python client expects a key.
    if not api_key:

        if (
            base_url
            and (
                "127.0.0.1" in base_url
                or "localhost" in base_url
            )
        ):
            api_key = "ollama"

        else:
            raise RuntimeError(
                "No API key configured. "
                "Set LLM_API_KEY in the project .env file."
            )

    kwargs: dict[str, Any] = {
        "api_key": api_key,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIClient(
        **kwargs
    )


# ============================================================
# LLM input
# ============================================================

def prepare_clause_for_llm(
    clause: dict[str, Any],
) -> dict[str, Any]:
    """
    Send only information useful for semantic classification.
    """

    item: dict[str, Any] = {
        "clause_id": clause["clause_id"],
        "text": clause["raw_text"],
        "section_id": clause.get(
            "section_id"
        ),
        "section_heading": clause.get(
            "section_heading"
        ),
    }

    if clause.get(
        "reference_targets"
    ):
        item["references"] = (
            clause[
                "reference_targets"
            ]
        )

    return item


def prepare_batch(
    all_clauses: list[dict[str, Any]],
    target_clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Add the previous and next clauses from the same section
    as limited local context.
    """

    ordered = sorted(
        all_clauses,
        key=lambda clause: clause.get(
            "index",
            0,
        ),
    )

    index_by_id = {
        clause["clause_id"]: index
        for index, clause
        in enumerate(ordered)
    }

    items: list[
        dict[str, Any]
    ] = []

    for target in target_clauses:

        clause_id = target[
            "clause_id"
        ]

        if clause_id not in index_by_id:
            continue

        position = index_by_id[
            clause_id
        ]

        previous_clause = None
        next_clause = None

        # Previous
        if position > 0:

            previous = ordered[
                position - 1
            ]

            if (
                previous.get("section_id")
                == target.get("section_id")
            ):
                previous_clause = (
                    previous.get(
                        "raw_text"
                    )
                )

        # Next
        if (
            position + 1
            < len(ordered)
        ):

            following = ordered[
                position + 1
            ]

            if (
                following.get("section_id")
                == target.get("section_id")
            ):
                next_clause = (
                    following.get(
                        "raw_text"
                    )
                )

        item = (
            prepare_clause_for_llm(
                target
            )
        )

        if previous_clause:
            item[
                "previous_clause"
            ] = previous_clause

        if next_clause:
            item[
                "next_clause"
            ] = next_clause

        items.append(
            item
        )

    return {
        "target_clauses": items,
    }


# ============================================================
# Confidence handling
# ============================================================

def _parse_confidence(
    value: Any,
) -> float:
    """
    Convert confidence to a float between 0 and 1.

    Also tolerates accidental percentage values such as 85.
    """

    try:
        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    # Handle 85 -> 0.85
    if (
        confidence > 1.0
        and confidence <= 100.0
    ):
        confidence /= 100.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    return confidence


def _apply_confidence_threshold(
    classification: str,
    confidence: float,
) -> str:
    """
    Convert low-confidence predictions to UNCERTAIN.
    """

    if classification == "NORMATIVE":

        if (
            confidence
            < NORMATIVE_THRESHOLD
        ):
            return "UNCERTAIN"

        return "NORMATIVE"

    if classification == "CONTEXTUAL":

        if (
            confidence
            < CONTEXTUAL_THRESHOLD
        ):
            return "UNCERTAIN"

        return "CONTEXTUAL"

    return "UNCERTAIN"


# ============================================================
# JSON handling
# ============================================================

def _clean_json_response(
    content: str,
) -> str:
    """
    Remove accidental Markdown JSON fences.
    """

    cleaned = content.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[
            len("```json"):
        ].strip()

    elif cleaned.startswith("```"):
        cleaned = cleaned[
            len("```"):
        ].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[
            :-3
        ].strip()

    return cleaned


# ============================================================
# LLM inference
# ============================================================

def classify_batch_with_llm(
    client: OpenAI,
    payload: dict[str, Any],
) -> list[LLMClassification]:
    """
    Send one batch to the LLM and parse its classifications.
    """

    _, _, model = (
        _resolve_llm_config()
    )

    if not model:
        raise RuntimeError(
            "LLM_MODEL is not configured."
        )

    response = (
        client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={
                "type": "json_object",
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    content = (
        _clean_json_response(
            content
        )
    )

    try:
        data = json.loads(
            content
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LLM returned invalid JSON. "
            f"Response begins with: "
            f"{content[:500]}"
        ) from exc

    raw_results = data.get(
        "results"
    )

    if not isinstance(
        raw_results,
        list,
    ):
        raise RuntimeError(
            "LLM response does not contain "
            "a valid 'results' list."
        )

    valid_labels = {
        "NORMATIVE",
        "CONTEXTUAL",
        "UNCERTAIN",
    }

    results: list[
        LLMClassification
    ] = []

    for item in raw_results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        clause_id = item.get(
            "clause_id"
        )

        if not clause_id:
            continue

        # -----------------------------------
        # Raw classification
        # -----------------------------------

        raw_classification = str(
            item.get(
                "classification",
                "UNCERTAIN",
            )
        ).strip().upper()

        if (
            raw_classification
            not in valid_labels
        ):
            raw_classification = (
                "UNCERTAIN"
            )

        # -----------------------------------
        # Confidence
        # -----------------------------------

        confidence = (
            _parse_confidence(
                item.get(
                    "confidence",
                    0.0,
                )
            )
        )

        # -----------------------------------
        # Confidence gate
        # -----------------------------------

        classification = (
            _apply_confidence_threshold(
                raw_classification,
                confidence,
            )
        )

        reason = str(
            item.get(
                "reason",
                "No reason supplied.",
            )
        ).strip()

        results.append(
            LLMClassification(
                clause_id=str(
                    clause_id
                ),
                classification=classification,
                raw_classification=raw_classification,
                confidence=confidence,
                reason=reason,
            )
        )

    return results


# ============================================================
# Clause refinement
# ============================================================

def refine_manual_review_clauses(
    clauses: list[dict[str, Any]],
    batch_size: int = LLM_BATCH_SIZE,
    reprocess: bool = False,
) -> list[dict[str, Any]]:
    """
    Refine rule-based MANUAL_REVIEW clauses.

    Normal mode:
        Only unresolved MANUAL_REVIEW clauses that have not already
        been sent to the LLM are processed.

    Reprocess mode:
        Re-run every clause whose ORIGINAL rule classification
        was MANUAL_REVIEW.
    """

    manual: list[
        dict[str, Any]
    ] = []

    for clause in clauses:

        current_classification = (
            clause.get(
                "classification"
            )
        )

        original_rule_classification = (
            clause.get(
                "rule_classification",
                current_classification,
            )
        )

        existing_refinement = (
            clause.get(
                "llm_refinement",
                {},
            )
        )

        already_processed = (
            isinstance(
                existing_refinement,
                dict,
            )
            and existing_refinement.get(
                "used"
            ) is True
        )

        # -----------------------------------
        # Reprocess mode
        # -----------------------------------

        if reprocess:

            if (
                original_rule_classification
                != "MANUAL_REVIEW"
            ):
                continue

            manual.append(
                clause
            )

            continue

        # -----------------------------------
        # Normal mode
        # -----------------------------------

        if (
            current_classification
            != "MANUAL_REVIEW"
        ):
            continue

        if already_processed:
            continue

        manual.append(
            clause
        )

    print(
        f"LLM refinement: "
        f"{len(manual)} clause(s) selected "
        f"(reprocess={reprocess})."
    )

    if not manual:
        return clauses

    client = make_client()

    results_by_id: dict[
        str,
        LLMClassification,
    ] = {}

    total_batches = (
        len(manual)
        + batch_size
        - 1
    ) // batch_size

    # ========================================================
    # Batch processing
    # ========================================================

    for start in range(
        0,
        len(manual),
        batch_size,
    ):

        targets = manual[
            start:start + batch_size
        ]

        payload = prepare_batch(
            all_clauses=clauses,
            target_clauses=targets,
        )

        batch_number = (
            start // batch_size
        ) + 1

        print(
            f"Processing LLM batch "
            f"{batch_number}/{total_batches} "
            f"({len(targets)} clauses)"
        )

        try:
            results = (
                classify_batch_with_llm(
                    client,
                    payload,
                )
            )

        except Exception as exc:
            print(
                f"WARNING: LLM batch "
                f"{batch_number} failed: "
                f"{exc}"
            )

            continue

        returned_ids = {
            result.clause_id
            for result in results
        }

        requested_ids = {
            target["clause_id"]
            for target in targets
        }

        missing_ids = (
            requested_ids
            - returned_ids
        )

        if missing_ids:
            print(
                f"WARNING: LLM batch "
                f"{batch_number} omitted "
                f"{len(missing_ids)} clause(s)."
            )

        for result in results:
            results_by_id[
                result.clause_id
            ] = result

    # ========================================================
    # Merge LLM results
    # ========================================================

    _, _, current_model = (
        _resolve_llm_config()
    )

    for clause in clauses:

        clause_id = clause.get(
            "clause_id"
        )

        result = (
            results_by_id.get(
                clause_id
            )
        )

        if result is None:
            continue

        # Preserve original rule output.
        clause.setdefault(
            "rule_classification",
            clause.get(
                "classification"
            ),
        )

        clause.setdefault(
            "rule_classification_reason",
            clause.get(
                "classification_reason"
            ),
        )

        # -----------------------------------
        # Final classification
        # -----------------------------------

        if (
            result.classification
            == "UNCERTAIN"
        ):

            clause[
                "classification"
            ] = "MANUAL_REVIEW"

            clause[
                "classification_reason"
            ] = "llm_remains_uncertain"

            clause[
                "manual_review"
            ] = True

        else:

            clause[
                "classification"
            ] = result.classification

            clause[
                "classification_reason"
            ] = (
                "llm_manual_review_resolution"
            )

            clause[
                "manual_review"
            ] = False

        # -----------------------------------
        # LLM provenance
        # -----------------------------------

        clause[
            "llm_refinement"
        ] = {
            "used": True,
            "model": current_model,
            "raw_classification": (
                result.raw_classification
            ),
            "classification": (
                result.classification
            ),
            "confidence": (
                result.confidence
            ),
            "threshold_applied": (
                result.raw_classification
                != result.classification
            ),
            "thresholds": {
                "normative": (
                    NORMATIVE_THRESHOLD
                ),
                "contextual": (
                    CONTEXTUAL_THRESHOLD
                ),
            },
            "reason": result.reason,
        }

    return clauses


# ============================================================
# Document refinement
# ============================================================

def refine_document_manual_review_clauses(
    document: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Refine MANUAL_REVIEW clauses inside one processed document.
    """

    clauses = document.get(
        "clauses"
    )

    if not isinstance(
        clauses,
        list,
    ):
        raise ValueError(
            "Document payload must contain "
            "a top-level 'clauses' list."
        )

    document["clauses"] = (
        refine_manual_review_clauses(
            clauses,
            **kwargs,
        )
    )

    return document


# ============================================================
# File operations
# ============================================================

def save_document_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    """Save one processed document."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def refine_document_manual_review_clauses_in_place(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Load, refine, and overwrite one processed JSON document.
    """

    source = Path(
        path
    )

    payload = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    refined = (
        refine_document_manual_review_clauses(
            payload,
            **kwargs,
        )
    )

    save_document_json(
        source,
        refined,
    )

    return refined


# ============================================================
# Multi-file helpers
# ============================================================

def _is_clause_document(
    path: Path,
) -> bool:
    """
    Check that a JSON file contains a top-level clauses list.
    """

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ):
        return False

    return (
        isinstance(
            payload,
            dict,
        )
        and isinstance(
            payload.get(
                "clauses"
            ),
            list,
        )
    )


def _discover_json_files(
    input_dir: Path,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """
    Discover JSON files.

    Examples:

        *.json
        draft-ietf-tls-*.json
    """

    if recursive:
        candidates = (
            input_dir.rglob(
                pattern
            )
        )

    else:
        candidates = (
            input_dir.glob(
                pattern
            )
        )

    files = [
        path
        for path in candidates
        if path.is_file()
    ]

    return sorted(
        files
    )


def _process_single_file(
    input_path: Path,
    output_path: Path | None,
    batch_size: int,
    reprocess: bool,
    skip_existing: bool,
) -> str:
    """
    Process one clause JSON file.

    Returns:

        processed
        skipped
        invalid
        failed
    """

    # -----------------------------------
    # Validate
    # -----------------------------------

    if not _is_clause_document(
        input_path
    ):
        print(
            f"SKIP: Not a clause document: "
            f"{input_path}"
        )

        return "invalid"

    # -----------------------------------
    # Skip existing
    # -----------------------------------

    if (
        output_path is not None
        and skip_existing
        and output_path.exists()
    ):
        print(
            f"SKIP: Output already exists: "
            f"{output_path}"
        )

        return "skipped"

    print()
    print(
        "=" * 72
    )

    print(
        f"Processing: {input_path}"
    )

    if output_path is None:
        print(
            "Output: IN PLACE"
        )
    else:
        print(
            f"Output: {output_path}"
        )

    print(
        "=" * 72
    )

    try:
        payload = json.loads(
            input_path.read_text(
                encoding="utf-8"
            )
        )

        refined = (
            refine_document_manual_review_clauses(
                payload,
                batch_size=batch_size,
                reprocess=reprocess,
            )
        )

        destination = (
            input_path
            if output_path is None
            else output_path
        )

        save_document_json(
            destination,
            refined,
        )

        print(
            f"SUCCESS: {destination}"
        )

        return "processed"

    except Exception as exc:
        print(
            f"ERROR processing "
            f"{input_path}: {exc}"
        )

        return "failed"


def _process_directory(
    input_dir: Path,
    output_dir: Path | None,
    pattern: str,
    recursive: bool,
    batch_size: int,
    reprocess: bool,
    skip_existing: bool,
) -> dict[str, int]:
    """
    Process all matching clause JSON files.

    Relative directory structure is preserved.

    Example:

        clauses/TLS13/file.json

    becomes:

        clauses_llm/TLS13/file.json
    """

    files = _discover_json_files(
        input_dir=input_dir,
        pattern=pattern,
        recursive=recursive,
    )

    print()
    print(
        f"Discovered "
        f"{len(files)} candidate file(s)."
    )

    print(
        f"Input directory: "
        f"{input_dir}"
    )

    if output_dir is None:
        print(
            "Output mode: IN PLACE"
        )

    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Output directory: "
            f"{output_dir}"
        )

    print(
        f"Pattern: {pattern}"
    )

    print(
        f"Recursive: {recursive}"
    )

    print(
        f"Reprocess: {reprocess}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    stats = {
        "processed": 0,
        "skipped": 0,
        "invalid": 0,
        "failed": 0,
    }

    total = len(
        files
    )

    for file_number, input_path in enumerate(
        files,
        start=1,
    ):

        # -----------------------------------
        # Avoid recursively processing output
        # directory if it sits under input.
        # -----------------------------------

        if output_dir is not None:

            try:
                input_path.relative_to(
                    output_dir
                )

                continue

            except ValueError:
                pass

        print()
        print(
            f"[{file_number}/{total}] "
            f"{input_path.name}"
        )

        # -----------------------------------
        # Output path
        # -----------------------------------

        if output_dir is None:

            output_path = None

        else:

            # IMPORTANT:
            # preserve TLS13/, QUIC/, etc.
            relative_path = (
                input_path.relative_to(
                    input_dir
                )
            )

            output_path = (
                output_dir
                / relative_path
            )

        status = (
            _process_single_file(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                reprocess=reprocess,
                skip_existing=skip_existing,
            )
        )

        stats[
            status
        ] += 1

    return stats


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    """
    Command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Refine rule-based MANUAL_REVIEW clauses "
            "using an LLM. Input may be either one "
            "processed JSON file or a directory."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Processed clause JSON file "
            "or directory."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "For file input: output JSON file. "
            "For directory input: output directory. "
            "If omitted, input files are modified "
            "in place."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLM_BATCH_SIZE,
        help=(
            "Number of clauses sent to the LLM "
            "in each request."
        ),
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
        help=(
            "Re-run every clause whose original "
            "rule classification was MANUAL_REVIEW."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Search subdirectories recursively "
            "when input is a directory."
        ),
    )

    parser.add_argument(
        "--pattern",
        default="*.json",
        help=(
            "Glob pattern for directory mode. "
            "Default: *.json"
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a file when its destination "
            "already exists."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def _main() -> int:
    args = _parse_args()

    # -----------------------------------
    # Basic validation
    # -----------------------------------

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be "
            "greater than zero."
        )

    input_path = (
        args.input.resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: "
            f"{input_path}"
        )

    # ========================================================
    # Single-file mode
    # ========================================================

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            != ".json"
        ):
            raise ValueError(
                "Input file must be JSON."
            )

        output_path = (
            args.output.resolve()
            if args.output
            else None
        )

        # If -o is an existing directory,
        # put the input filename inside it.
        if (
            output_path is not None
            and output_path.exists()
            and output_path.is_dir()
        ):
            output_path = (
                output_path
                / input_path.name
            )

        status = (
            _process_single_file(
                input_path=input_path,
                output_path=output_path,
                batch_size=args.batch_size,
                reprocess=args.reprocess,
                skip_existing=args.skip_existing,
            )
        )

        if status in {
            "processed",
            "skipped",
        }:
            return 0

        return 1

    # ========================================================
    # Directory mode
    # ========================================================

    if input_path.is_dir():

        output_dir = (
            args.output.resolve()
            if args.output
            else None
        )

        if (
            output_dir is not None
            and output_dir.exists()
            and not output_dir.is_dir()
        ):
            raise ValueError(
                "When input is a directory, "
                "--output must be a directory."
            )

        stats = (
            _process_directory(
                input_dir=input_path,
                output_dir=output_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                batch_size=args.batch_size,
                reprocess=args.reprocess,
                skip_existing=args.skip_existing,
            )
        )

        print()
        print(
            "=" * 72
        )
        print(
            "MULTI-FILE REFINEMENT SUMMARY"
        )
        print(
            "=" * 72
        )

        print(
            f"Processed: "
            f"{stats['processed']}"
        )

        print(
            f"Skipped existing: "
            f"{stats['skipped']}"
        )

        print(
            f"Invalid/non-clause JSON: "
            f"{stats['invalid']}"
        )

        print(
            f"Failed: "
            f"{stats['failed']}"
        )

        print(
            "=" * 72
        )

        if (
            stats["failed"]
            > 0
        ):
            return 1

        return 0

    raise ValueError(
        f"Unsupported input path: "
        f"{input_path}"
    )


if __name__ == "__main__":
    raise SystemExit(
        _main()
    )