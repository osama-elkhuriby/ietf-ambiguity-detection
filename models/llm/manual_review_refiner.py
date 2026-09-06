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
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = Any  # type: ignore[misc,assignment]


# ============================================================
# Configuration
# ============================================================

LLM_BATCH_SIZE = 20

# Conservative thresholds.
# NORMATIVE is intentionally stricter because false normative
# classifications are particularly undesirable in this stage.
NORMATIVE_THRESHOLD = 0.80
CONTEXTUAL_THRESHOLD = 0.70


# ============================================================
# Environment loading
# ============================================================

def _load_env_file() -> None:
    """Load project-level .env variables if present without requiring dotenv."""

    env_path = Path(__file__).resolve().parents[2] / ".env"

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():

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

        if key and value and key not in os.environ:
            os.environ[key] = value.strip('"\'')
            

_load_env_file()


def _resolve_llm_config() -> tuple[str | None, str | None, str | None]:
    """
    Resolve LLM configuration.

    Supported environment variables:

    LLM_API_KEY
    LLM_MODEL
    LLM_BASE_URL

    Compatibility fallbacks are also supported.
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

    return api_key, base_url, model


LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = _resolve_llm_config()


# ============================================================
# Result representation
# ============================================================

@dataclass(slots=True)
class LLMClassification:
    clause_id: str

    # Final classification after confidence filtering
    classification: str

    # Original classification returned by the model
    raw_classification: str

    # Self-reported confidence from the model
    confidence: float

    reason: str


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You classify TARGET clauses from IETF Internet-Drafts and RFCs.

The rule-based classifier has already processed these clauses.

You are receiving only clauses for which the rule-based classifier
could not make a sufficiently confident decision.

Your task is NOT to identify whether a sentence is technically important.

Your task is ONLY to determine whether the TARGET clause itself imposes
normative behavior.

Classify every TARGET clause into exactly one of:

NORMATIVE

The clause imposes a requirement, permission, prohibition,
recommendation, conformance condition, or explicit behavioral
constraint on a protocol participant or implementation.

Typical actors include:

- client
- server
- sender
- receiver
- endpoint
- peer
- implementation
- protocol participant

A NORMATIVE clause constrains what one of these actors is required,
permitted, recommended, expected, or prohibited to do.


CONTEXTUAL

The clause explains, describes, defines, summarizes, motivates,
justifies, reports a fact, describes a property, or provides
background information WITHOUT imposing a behavioral requirement.

Describing protocol behavior, design rationale, interoperability,
capability, data structures, algorithms, or implementation facts
does NOT automatically make a clause normative.


UNCERTAIN

There is insufficient evidence to determine reliably whether the
TARGET clause imposes a normative behavioral constraint.

Use UNCERTAIN instead of guessing.


============================================================
STRICT DECISION RULES
============================================================

1. NORMATIVE requires evidence of an obligation, permission,
   prohibition, recommendation, conformance condition, or behavioral
   constraint.

2. Ask this question before selecting NORMATIVE:

   "Does the TARGET clause constrain what an implementation or
   protocol participant is required, permitted, recommended,
   expected, or prohibited to do?"

   If the answer is not clearly yes, do NOT classify it as NORMATIVE.

3. Do NOT classify a clause as NORMATIVE merely because it occurs
   in a technical, protocol-processing, security, message-format,
   or interoperability section.

4. Section headings are CONTEXT ONLY.

   A section heading must never be sufficient evidence for a
   NORMATIVE classification.

5. A statement describing how something works is NOT automatically
   normative.

   For example:

   "The deprotection functions reverse the process."

   Classification:
   CONTEXTUAL

   Reason:
   It describes function behavior without imposing a requirement.

6. A statement describing a protocol property is NOT automatically
   normative.

   Statements about:

   - interoperability
   - compatibility
   - sufficiency
   - capability
   - availability
   - performance
   - deployment
   - expected operation
   - existing implementations
   - design intention
   - protocol architecture

   are normally CONTEXTUAL unless they explicitly constrain
   participant or implementation behavior.

7. The words:

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

   do NOT by themselves imply a normative requirement.

8. Example:

   "The set of curves specified here is sufficient for
   interoperability with currently deployed TLS implementations."

   Classification:
   CONTEXTUAL

   Reason:
   It states an interoperability property without imposing
   implementation behavior.

9. A factual statement about implementations or deployment is
   normally CONTEXTUAL.

   Example:

   "Most existing implementations support this extension."

   Classification:
   CONTEXTUAL

10. A definition is normally CONTEXTUAL unless the definition itself
    imposes implementation or protocol behavior.

11. A sentence describing a field, structure, message, function,
    algorithm, or state is normally CONTEXTUAL when it merely explains
    what that object represents or does.

12. Do NOT infer a hidden requirement merely because following the
    statement may be important for interoperability.

13. Technical importance does NOT imply normativity.

14. BCP 14 keywords are strong evidence when they impose requirements:

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

15. However, merely mentioning a BCP 14 keyword does NOT automatically
    make a clause normative.

    For example, a sentence explaining what "MUST" means is
    CONTEXTUAL.

16. Absence of BCP 14 keywords does NOT automatically imply
    CONTEXTUAL.

17. A clause may be implicitly NORMATIVE without a BCP 14 keyword,
    but ONLY when the behavioral or conformance constraint is clear.

18. Example of explicit normative language:

    "The client MUST abort the connection if validation fails."

    Classification:
    NORMATIVE

    Reason:
    It imposes mandatory behavior on the client.

19. Example of implicit normative language:

    "A compliant implementation rejects certificates with an
    invalid signature."

    Classification:
    NORMATIVE

    Reason:
    It defines required behavior for compliant implementations.

20. Be especially careful with declarative statements.

    A grammatical statement of fact such as:

    "The server sends X after Y."

    may either describe protocol behavior or impose a rule.

    Do NOT automatically classify such statements as NORMATIVE.

    Use surrounding context only when needed to determine whether
    the statement is clearly prescribing behavior.

21. Previous and next clauses may help resolve meaning, but the
    classification must apply to the TARGET clause.

22. Do NOT transfer normativity from a nearby clause to the TARGET
    clause.

    A nearby MUST statement does not make every sentence around it
    normative.

23. Do NOT infer requirements that are not stated or clearly implied
    by the TARGET clause.

24. When both NORMATIVE and CONTEXTUAL interpretations remain
    plausible, return UNCERTAIN.

25. Prefer UNCERTAIN over an unsupported NORMATIVE classification.

26. Be conservative when selecting NORMATIVE.

27. A clear descriptive explanation should be CONTEXTUAL, not
    UNCERTAIN.

28. Use UNCERTAIN only when genuine semantic ambiguity remains about
    whether the TARGET clause imposes behavior.


============================================================
CONFIDENCE
============================================================

Return a confidence value between 0.0 and 1.0.

The confidence represents how strongly the available evidence supports
a definitive NORMATIVE or CONTEXTUAL decision.

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
The evidence is insufficient for a reliable definitive decision.
Classification should normally be UNCERTAIN.

Do NOT use high confidence simply because the clause sounds technical.

Do NOT inflate confidence.

If significant interpretation or assumptions are required,
lower the confidence.


============================================================
REASON
============================================================

The reason must:

- refer only to the TARGET clause;
- explain why it is normative, contextual, or uncertain;
- be one short sentence;
- contain no more than 20 words;
- avoid unnecessary chain-of-thought or extended reasoning.


============================================================
OUTPUT
============================================================

Return JSON only.

Do not include Markdown.

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

The classification value must be exactly one of:

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

    api_key, base_url, _ = _resolve_llm_config()

    # Ollama itself does not need authentication, but the OpenAI
    # Python client expects a non-empty API key.
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
                "Put LLM_API_KEY in the project .env file."
            )

    kwargs: dict[str, Any] = {
        "api_key": api_key,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIClient(**kwargs)


# ============================================================
# LLM input preparation
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
        "section_id": clause.get("section_id"),
        "section_heading": clause.get("section_heading"),
    }

    if clause.get("reference_targets"):
        item["references"] = clause["reference_targets"]

    return item


def prepare_batch(
    all_clauses: list[dict[str, Any]],
    target_clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Construct LLM batch with neighboring context.

    Only previous/next clauses from the same section are included.
    """

    ordered = sorted(
        all_clauses,
        key=lambda x: x.get("index", 0),
    )

    index_by_id = {
        clause["clause_id"]: i
        for i, clause in enumerate(ordered)
    }

    items: list[dict[str, Any]] = []

    for target in target_clauses:

        clause_id = target["clause_id"]

        if clause_id not in index_by_id:
            continue

        position = index_by_id[clause_id]

        previous_clause = None
        next_clause = None

        # ----------------------------------------------------
        # Previous clause
        # ----------------------------------------------------

        if position > 0:

            previous = ordered[position - 1]

            if (
                previous.get("section_id")
                == target.get("section_id")
            ):
                previous_clause = previous.get("raw_text")

        # ----------------------------------------------------
        # Next clause
        # ----------------------------------------------------

        if position + 1 < len(ordered):

            following = ordered[position + 1]

            if (
                following.get("section_id")
                == target.get("section_id")
            ):
                next_clause = following.get("raw_text")

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        item = prepare_clause_for_llm(target)

        if previous_clause:
            item["previous_clause"] = previous_clause

        if next_clause:
            item["next_clause"] = next_clause

        items.append(item)

    return {
        "target_clauses": items,
    }


# ============================================================
# Confidence parsing
# ============================================================

def _parse_confidence(value: Any) -> float:
    """
    Convert model confidence into a number between 0 and 1.

    Handles both:

        0.85

    and accidental percentage-style values such as:

        85
    """

    try:
        confidence = float(value)

    except (TypeError, ValueError):
        return 0.0

    # Gracefully interpret percentage-style output.
    if 1.0 < confidence <= 100.0:
        confidence = confidence / 100.0

    # Clamp to valid probability-like range.
    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return confidence


# ============================================================
# Threshold policy
# ============================================================

def _apply_confidence_threshold(
    classification: str,
    confidence: float,
) -> str:
    """
    Apply conservative decision thresholds.

    NORMATIVE requires stronger confidence because an incorrect
    normative assignment is particularly undesirable.
    """

    if classification == "NORMATIVE":

        if confidence < NORMATIVE_THRESHOLD:
            return "UNCERTAIN"

        return "NORMATIVE"

    if classification == "CONTEXTUAL":

        if confidence < CONTEXTUAL_THRESHOLD:
            return "UNCERTAIN"

        return "CONTEXTUAL"

    return "UNCERTAIN"


# ============================================================
# LLM inference
# ============================================================

def classify_batch_with_llm(
    client: OpenAI,
    payload: dict[str, Any],
) -> list[LLMClassification]:
    """
    Classify one batch of MANUAL_REVIEW clauses.
    """

    _, _, model = _resolve_llm_config()

    if not model:
        raise RuntimeError(
            "LLM_MODEL environment variable is not configured."
        )

    response = client.chat.completions.create(
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

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    try:
        data = json.loads(content)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"LLM returned invalid JSON: {content[:500]}"
        ) from exc

    results: list[LLMClassification] = []

    valid_labels = {
        "NORMATIVE",
        "CONTEXTUAL",
        "UNCERTAIN",
    }

    raw_results = data.get("results", [])

    if not isinstance(raw_results, list):
        raise RuntimeError(
            "LLM JSON response does not contain a valid 'results' list."
        )

    for item in raw_results:

        if not isinstance(item, dict):
            continue

        clause_id = item.get("clause_id")

        if not clause_id:
            continue

        # ----------------------------------------------------
        # Raw model classification
        # ----------------------------------------------------

        raw_classification = str(
            item.get(
                "classification",
                "UNCERTAIN",
            )
        ).strip().upper()

        if raw_classification not in valid_labels:
            raw_classification = "UNCERTAIN"

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = _parse_confidence(
            item.get(
                "confidence",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Apply conservative threshold
        # ----------------------------------------------------

        classification = _apply_confidence_threshold(
            raw_classification,
            confidence,
        )

        # ----------------------------------------------------
        # Reason
        # ----------------------------------------------------

        reason = str(
            item.get(
                "reason",
                "No reason supplied.",
            )
        ).strip()

        results.append(
            LLMClassification(
                clause_id=str(clause_id),
                classification=classification,
                raw_classification=raw_classification,
                confidence=confidence,
                reason=reason,
            )
        )

    return results


# ============================================================
# Refinement
# ============================================================

def refine_manual_review_clauses(
    clauses: list[dict[str, Any]],
    batch_size: int = LLM_BATCH_SIZE,
    reprocess: bool = False,
) -> list[dict[str, Any]]:
    """
    Refine MANUAL_REVIEW clauses using the LLM.

    By default, clauses previously processed by the LLM are skipped.

    Set reprocess=True to run them again.
    """

    manual: list[dict[str, Any]] = []

    for clause in clauses:

        if clause.get("classification") != "MANUAL_REVIEW":
            continue

        existing_refinement = clause.get(
            "llm_refinement",
            {},
        )

        already_processed = (
            isinstance(existing_refinement, dict)
            and existing_refinement.get("used") is True
        )

        if already_processed and not reprocess:
            continue

        manual.append(clause)

    print(
        f"LLM refinement: "
        f"{len(manual)} MANUAL_REVIEW clauses."
    )

    if not manual:
        return clauses

    client = make_client()

    results_by_id: dict[
        str,
        LLMClassification,
    ] = {}

    # --------------------------------------------------------
    # Batch processing
    # --------------------------------------------------------

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
            f"{batch_number} "
            f"({len(targets)} clauses)"
        )

        try:

            results = classify_batch_with_llm(
                client,
                payload,
            )

        except Exception as exc:

            print(
                f"WARNING: LLM batch "
                f"{batch_number} failed: {exc}"
            )

            continue

        for result in results:
            results_by_id[
                result.clause_id
            ] = result

    # --------------------------------------------------------
    # Merge results
    # --------------------------------------------------------

    _, _, current_model = _resolve_llm_config()

    for clause in clauses:

        clause_id = clause.get("clause_id")

        result = results_by_id.get(
            clause_id
        )

        if result is None:
            continue

        # Preserve rule-based classification only if it has not
        # already been stored.
        clause.setdefault(
            "rule_classification",
            clause.get("classification"),
        )

        clause.setdefault(
            "rule_classification_reason",
            clause.get("classification_reason"),
        )

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        if result.classification == "UNCERTAIN":

            clause["classification"] = "MANUAL_REVIEW"

            clause[
                "classification_reason"
            ] = "llm_remains_uncertain"

            clause["manual_review"] = True

        else:

            clause[
                "classification"
            ] = result.classification

            clause[
                "classification_reason"
            ] = "llm_manual_review_resolution"

            clause["manual_review"] = False

        # ----------------------------------------------------
        # Detailed provenance
        # ----------------------------------------------------

        clause["llm_refinement"] = {
            "used": True,
            "model": current_model,
            "raw_classification": result.raw_classification,
            "classification": result.classification,
            "confidence": result.confidence,
            "threshold_applied": (
                result.raw_classification
                != result.classification
            ),
            "thresholds": {
                "normative": NORMATIVE_THRESHOLD,
                "contextual": CONTEXTUAL_THRESHOLD,
            },
            "reason": result.reason,
        }

    return clauses


# ============================================================
# Document-level operations
# ============================================================

def refine_document_manual_review_clauses(
    document: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Refine clauses inside a complete processed document.
    """

    clauses = document.get("clauses")

    if not isinstance(clauses, list):

        raise ValueError(
            "Document payload must contain a 'clauses' list."
        )

    document["clauses"] = refine_manual_review_clauses(
        clauses,
        **kwargs,
    )

    return document


def save_document_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    """Save JSON document."""

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
    Load, refine, and overwrite a processed clause JSON file.
    """

    source = Path(path)

    payload = json.loads(
        source.read_text(
            encoding="utf-8",
        )
    )

    refined = refine_document_manual_review_clauses(
        payload,
        **kwargs,
    )

    save_document_json(
        source,
        refined,
    )

    return refined


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Refine MANUAL_REVIEW clauses "
            "with an LLM."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a processed "
            "clause JSON file."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Optional output copy. "
            "By default, the input file "
            "is updated in place."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLM_BATCH_SIZE,
        help=(
            "Number of clauses per LLM batch."
        ),
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
        help=(
            "Re-run the LLM on MANUAL_REVIEW "
            "clauses that already contain "
            "llm_refinement results."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def _main() -> int:

    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.input}"
        )

    # --------------------------------------------------------
    # In-place mode
    # --------------------------------------------------------

    if args.output is None:

        refine_document_manual_review_clauses_in_place(
            args.input,
            batch_size=args.batch_size,
            reprocess=args.reprocess,
        )

        print(
            f"Updated original file in place: "
            f"{args.input}"
        )

        return 0

    # --------------------------------------------------------
    # Separate output mode
    # --------------------------------------------------------

    payload = json.loads(
        args.input.read_text(
            encoding="utf-8",
        )
    )

    refined = refine_document_manual_review_clauses(
        payload,
        batch_size=args.batch_size,
        reprocess=args.reprocess,
    )

    save_document_json(
        args.output,
        refined,
    )

    print(
        f"Wrote refined output to "
        f"{args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())