# LLM-only baseline

Per proposal SS6.1: zero/few-shot structured prompting producing an explicit evidence
span, ambiguity subtype, confidence, and rationale. Abstract behind a provider-independent
interface (SS6.4) so at least one open-parameter model and one hosted model can be
compared — don't hardcode a single provider's SDK into the detection logic itself.

Track real cost-per-clause from the very first pilot run (see `experiments/pilot/docs/pilot_memo_template.md`)
— this number is what H2 and RQ4 (hybrid routing reducing LLM calls) get judged against.
