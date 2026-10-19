"""Versioned prompts for zero/few-shot ambiguity classification."""

from __future__ import annotations

from typing import Any

import pandas as pd

ORDINARY_ENGINEER_V1 = (
    "Classify whether the TARGET clause is ambiguous for a competent ordinary "
    "engineer who must implement the instruction and is not assumed to be a senior "
    "domain specialist. Read PREVIOUS CONTEXT and FOLLOWING CONTEXT to resolve the "
    "meaning, but judge only the TARGET clause. Answer yes if at least one criterion "
    "holds: A1, two competent engineers could reasonably implement the instruction "
    "in materially different ways because of its wording; A2, the clause is very "
    "vague or difficult to understand on a first careful reading and could be written "
    "more clearly without changing its intended meaning; A3, the clause depends on a "
    "subjective qualifier or value without a strong operational or measurable "
    "criterion. Answer no when apparent vagueness is resolved by an explicit citation "
    "or section reference, or by an identifiable local pointer such as mentioned "
    "above, specified below, or listed as follows. A procedure defined in another "
    "cited section is not ambiguous merely because that definition is not repeated. "
    "Explicitly permitted implementation discretion, including MAY, is not ambiguity "
    "by itself. Technical complexity, long wording, or the need for domain knowledge "
    "alone is not ambiguity. Apply the criteria consistently. Return exactly one "
    "lowercase word, yes or no, with no punctuation or explanation."
)

ORDINARY_ENGINEER_STRICT_V2 = (
    "Classify whether the TARGET clause contains an unresolved ambiguity that can "
    "materially change an ordinary competent engineer's implementation. Read "
    "PREVIOUS CONTEXT and FOLLOWING CONTEXT to resolve the meaning, but judge only "
    "the TARGET clause. Ambiguity is uncommon, so default to no unless the wording "
    "itself supports a specific unresolved conflict. Answer yes only when at least "
    "one of these is true: A1, the same mandatory or recommended instruction permits "
    "two materially different implementations because a condition, actor, action, "
    "scope, or outcome is unclear; A2, a required behavioral boundary depends on an "
    "undefined subjective qualifier and different reasonable values can affect "
    "interoperability or conformance; A3, even after using the supplied context and "
    "explicit references, the instruction cannot be translated into a determinate "
    "implementation rule. Answer no for intentionally permitted choices or policy "
    "freedom, including MAY, implementation-defined strategies, configurable values, "
    "experimentation, or alternatives whose selection is explicitly left to the "
    "implementer. Answer no when a citation, section reference, mentioned-above, "
    "specified-below, or listed-as-follows pointer resolves the apparent vagueness. "
    "Answer no for technical complexity, long wording, unfamiliar terminology, or "
    "the need for domain knowledge alone. SHOULD or SHOULD NOT is not ambiguous merely "
    "because exceptions are possible. Before answering yes, identify internally the "
    "two conflicting implementation outcomes and verify that the specification did "
    "not intentionally allow that choice. Return exactly one lowercase word, yes or "
    "no, with no punctuation or explanation."
)

PROMPT_CONTRACTS = {
    "ordinary_engineer_v1": ORDINARY_ENGINEER_V1,
    "ordinary_engineer_strict_v2": ORDINARY_ENGINEER_STRICT_V2,
}


def system_prompt_for_contract(contract: str) -> str:
    try:
        return PROMPT_CONTRACTS[contract]
    except KeyError as error:
        raise ValueError(f"Unsupported prompt contract: {contract}") from error


def user_message(*, target: str, previous: str = "", following: str = "") -> str:
    """Render only the three clause fields allowed to enter the model prompt."""

    return (
        f"PREVIOUS CONTEXT:\n{previous.strip() or '[none]'}\n\n"
        f"TARGET:\n{target.strip()}\n\n"
        f"FOLLOWING CONTEXT:\n{following.strip() or '[none]'}"
    )


def render_messages(
    row: Any,
    demonstrations: pd.DataFrame,
    *,
    prompt_contract: str,
) -> list[dict[str, str]]:
    """Render fold-safe labeled demonstrations followed by one unlabeled query."""

    messages = [
        {"role": "system", "content": system_prompt_for_contract(prompt_contract)}
    ]
    for demo in demonstrations.itertuples(index=False):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": "LABELED EXAMPLE:\n"
                    + user_message(
                        target=str(demo.target_clause),
                        previous=str(demo.previous_clause),
                        following=str(demo.next_clause),
                    ),
                },
                {
                    "role": "assistant",
                    "content": "yes" if int(demo.label) else "no",
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "CLAUSE TO CLASSIFY:\n"
            + user_message(
                target=str(row.target_clause),
                previous=str(row.previous_clause),
                following=str(row.next_clause),
            ),
        }
    )
    return messages
