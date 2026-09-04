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


def _load_env_file() -> None:
    """Load project-level .env variables if present without requiring dotenv."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key and value and key not in os.environ:
            os.environ[key] = value.strip('"\'')


_load_env_file()


def _resolve_llm_config() -> tuple[str | None, str | None, str | None]:
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
        os.environ.setdefault("LLM_API_KEY", api_key)
    if model:
        os.environ.setdefault("LLM_MODEL", model)
    if base_url:
        os.environ.setdefault("LLM_BASE_URL", base_url)
    return api_key, base_url, model


LLM_BATCH_SIZE = 20
LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = _resolve_llm_config()


@dataclass(slots=True)
class LLMClassification:
    clause_id: str
    classification: str
    reason: str


SYSTEM_PROMPT = """
You classify clauses from IETF Internet-Drafts and RFCs.

The rule-based classifier has already processed these clauses.
You are only receiving clauses for which the rule-based classifier
could not make a confident decision.

Classify each TARGET clause into exactly one of:

NORMATIVE
The clause imposes, constrains, permits, recommends, or prohibits
behavior of a protocol participant or implementation.

CONTEXTUAL
The clause explains, describes, defines, summarizes, motivates,
or provides factual/background information without imposing a
requirement.

UNCERTAIN
There is insufficient evidence to reliably determine whether the
clause is normative or contextual.

Important rules:

1. Do NOT classify a clause as NORMATIVE merely because it occurs
   in a protocol-related section.

2. Absence of MUST, SHOULD, MAY, SHALL, etc. does NOT automatically
   make a clause contextual.

3. A description of how a protocol or data structure works is not
   automatically a normative requirement.

4. A statement that constrains implementation or participant behavior
   can be normative even without an explicit BCP 14 keyword.

5. Use the section heading and nearby clauses as context.

6. If the evidence is genuinely insufficient, return UNCERTAIN
   instead of guessing.

Return JSON only.

Expected format:

{
  "results": [
    {
      "clause_id": "...",
      "classification": "NORMATIVE | CONTEXTUAL | UNCERTAIN",
      "reason": "short explanation"
    }
  ]
}
""".strip()


def make_client() -> OpenAI:
    try:
        from openai import OpenAI as OpenAIClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional dependency 'openai' is not installed. "
            "Install it with: pip install openai"
        ) from exc

    api_key, base_url, _ = _resolve_llm_config()
    if not api_key:
        raise RuntimeError(
            "No API key configured. Put API_KEY in the project .env file or set LLM_API_KEY."
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIClient(**kwargs)


def prepare_clause_for_llm(clause: dict[str, Any]) -> dict[str, Any]:
    """Send only the information useful for semantic classification."""
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
    ordered = sorted(all_clauses, key=lambda x: x.get("index", 0))
    index_by_id = {clause["clause_id"]: i for i, clause in enumerate(ordered)}

    items = []
    for target in target_clauses:
        clause_id = target["clause_id"]
        position = index_by_id[clause_id]

        previous_clause = None
        next_clause = None

        if position > 0:
            previous = ordered[position - 1]
            if previous.get("section_id") == target.get("section_id"):
                previous_clause = previous.get("raw_text")

        if position + 1 < len(ordered):
            following = ordered[position + 1]
            if following.get("section_id") == target.get("section_id"):
                next_clause = following.get("raw_text")

        item = prepare_clause_for_llm(target)
        if previous_clause:
            item["previous_clause"] = previous_clause
        if next_clause:
            item["next_clause"] = next_clause
        items.append(item)

    return {"target_clauses": items}


def classify_batch_with_llm(
    client: OpenAI,
    payload: dict[str, Any],
) -> list[LLMClassification]:
    _, _, model = _resolve_llm_config()
    if not model:
        raise RuntimeError("LLM_MODEL environment variable is not configured.")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty response.")

    data = json.loads(content)
    results: list[LLMClassification] = []
    valid_labels = {"NORMATIVE", "CONTEXTUAL", "UNCERTAIN"}

    for item in data.get("results", []):
        classification = item["classification"].upper()
        if classification not in valid_labels:
            classification = "UNCERTAIN"
        results.append(
            LLMClassification(
                clause_id=item["clause_id"],
                classification=classification,
                reason=item.get("reason", "No reason supplied."),
            )
        )

    return results


def refine_manual_review_clauses(
    clauses: list[dict[str, Any]],
    batch_size: int = LLM_BATCH_SIZE,
) -> list[dict[str, Any]]:
    manual = [clause for clause in clauses if clause.get("classification") == "MANUAL_REVIEW"]
    print(f"LLM refinement: {len(manual)} MANUAL_REVIEW clauses.")

    if not manual:
        return clauses

    client = make_client()
    results_by_id: dict[str, LLMClassification] = {}

    for start in range(0, len(manual), batch_size):
        targets = manual[start : start + batch_size]
        payload = prepare_batch(all_clauses=clauses, target_clauses=targets)
        batch_number = start // batch_size + 1
        print(f"Processing LLM batch {batch_number} ({len(targets)} clauses)")

        try:
            results = classify_batch_with_llm(client, payload)
        except Exception as exc:
            print(f"WARNING: LLM batch {batch_number} failed: {exc}")
            continue

        for result in results:
            results_by_id[result.clause_id] = result

    for clause in clauses:
        result = results_by_id.get(clause.get("clause_id"))
        if result is None:
            continue

        clause["rule_classification"] = clause["classification"]
        clause["rule_classification_reason"] = clause.get("classification_reason")

        if result.classification == "UNCERTAIN":
            clause["classification"] = "MANUAL_REVIEW"
            clause["classification_reason"] = "llm_remains_uncertain"
            clause["manual_review"] = True
        else:
            clause["classification"] = result.classification
            clause["classification_reason"] = "llm_manual_review_resolution"
            clause["manual_review"] = False

        clause["llm_refinement"] = {
            "used": True,
            "model": LLM_MODEL,
            "classification": result.classification,
            "reason": result.reason,
        }

    return clauses


def refine_document_manual_review_clauses(document: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    clauses = document.get("clauses")
    if not isinstance(clauses, list):
        raise ValueError("Document payload must contain a 'clauses' list.")
    document["clauses"] = refine_manual_review_clauses(clauses, **kwargs)
    return document


def save_document_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refine_document_manual_review_clauses_in_place(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    refined = refine_document_manual_review_clauses(payload, **kwargs)
    save_document_json(source, refined)
    return refined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine MANUAL_REVIEW clauses with an LLM.")
    parser.add_argument("input", type=Path, help="Path to a processed clause JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output copy. By default, the input file is updated in place.",
    )
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE, help="Number of clauses per LLM batch.")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()

    if args.output is None:
        refined = refine_document_manual_review_clauses_in_place(args.input, batch_size=args.batch_size)
        print(f"Updated original file in place: {args.input}")
        return 0

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    refined = refine_document_manual_review_clauses(payload, batch_size=args.batch_size)
    save_document_json(args.output, refined)
    print(f"Wrote refined output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
