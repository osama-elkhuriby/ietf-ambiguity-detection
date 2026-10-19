# Annotation v2 implementation and bounded-run report

Status: implementation, dry run, hardware preflight, review packet, and bounded
pilot orchestration complete. No full-corpus annotation or classifier training ran.
No TCP, sealed DNS, or non-allowlisted lineage data was opened by the v2 pipeline.

## Outcome

The complete requested panel could not be run safely on the observed machine state.
All 72 pilot cells were therefore recorded as `MODEL_BLOCKED_PREFLIGHT` with
`semantic_label=null`. This is a completed blocked pilot, not a completed annotation
pilot. There are no pilot labels from which to calculate positive rates, context
transitions, or pairwise agreement.

The only live inference comprised two synthetic preflight requests. No development
clause was sent to any v2 model.

## Frozen experiment

- Experiment ID: `ietf-ambiguity-annotation-v2-requested-panel`
- Requested panel: `qwen3.5:27b`, `gemma4:31b`, `gpt-oss:120b`
- Sampling seed: `20261014`
- Development cohort: 2,312 candidates = 2,253 definitive v1 majority records +
  59 v1 consensus-uncertain records
- Configuration SHA-256:
  `c40da4868872b8f9690e0bcfd482efa744030d70f04c0bb102ca05f4e4bf9925`
- Rubric SHA-256:
  `b8dd1a3aa771058a5c33ca9349dfb702246cbd61b23e5072117572c163743266`
- Schema SHA-256:
  `454d1da6517780c1f4cc07737b6677672d8945cb730a42d386f6bbd4094d97db`
- Ordered review sample SHA-256:
  `cf9b711f17ed3d51cfcfa3a54e9245cce9613ecd84c9a7353242d1962aca562c`
- Ordered pilot sample SHA-256:
  `2955788ef80924967e1593918cf804120eb76d8ef4d4e6ef41d3201535334883`

The 40 random lineages were selected before the 60 targeted lineages. All requested
target quotas were achieved exactly after excluding earlier selections: 8 each for
unanimous positive, unanimous negative, Gemma-only positive, GPT-OSS-only positive,
two-positive disagreement, and v1 uncertain/abstention; 6 each for explicit-reference
and segmentation/structural cases. No top-up was needed. The 12-lineage pilot was
frozen as four random plus eight targeted records before v2 inference.

The exact allowlist, exclusions, versions, limits, settings, source hashes, sample
hashes, and count reconciliation are in `configs/requested_panel_v2.json`. V1 is
unchanged. The v2 code reads historical responses only by an already-allowed exact
record ID; it does not scan the global consensus file.

## Rubric and context

The exact system prompt and synthetic examples are frozen in `RUBRIC_V2.md`; the JSON
schema is in `schemas/annotation_response_v2.json`. V2 is not equivalent to v1. Every
judgment is conditional on the supplied same-version bundle. A failed lookup or absent
excerpt is not proof of whole-specification ambiguity. Source-supported U-2 ambiguity
requires affirmative evidence in the reviewed relevant scope; otherwise potentially
decisive unavailable information produces `INSUFFICIENT_CONTEXT`.

The two variants share identical task instructions:

- `local`: heading plus same-section previous/target/next clauses.
- `resolved`: local content plus same-block structure, explicit internal-section
  bodies, and deterministic same-revision definition candidates. External citations
  are marked unavailable unless an exact permitted version can be established; no
  later draft or final RFC is substituted.

Every context item records document/revision, section, clause/block IDs, source span,
retrieval reason, source hash, and content hash. Retrieval failures and budget
omissions are explicit. The target is never dropped. The pilot dry run materialized
24 context bundles, with zero budget omissions, and 72 exact model-specific rendered
requests. It made zero network/model requests.

## Schema, execution, and aggregation

The strict output fields are `label`, `ambiguity_subtype`, `evidence_span`,
`interpretation_a`, `interpretation_b`, `missing_information`, `justification`, and
`context_issue`; every free-text field has a finite maximum. `num_predict` is 768,
`num_ctx` is 16,384, temperature is 0, and one retry is allowed per cell.

Retries are fresh requests containing the original bounded input and only a concise
validation error; malformed text is never appended. Each attempt is a separate
append-only artifact. A valid result names its accepted attempt. API reasoning is
stored separately from final structured content and never parsed as the answer.

Operational failures and blocked/not-run states have null semantic labels. Under the
conservative policy, any valid `INSUFFICIENT_CONTEXT` causes `REVIEW_REQUIRED`, and
fewer than three valid judgments causes `REVIEW_REQUIRED_INCOMPLETE_PANEL`. Two models
cannot be called a unanimous three-model panel. A v1-compatible view exists only as a
separately named comparison policy. GPT-OSS-20B is configured only as a separately
named possible fallback experiment and cannot fill the requested 120B panel.

## Hardware and model feasibility

Observed software/hardware: Ollama 0.40.0; Windows 11; 31.71 GiB physical RAM with
about 21.42 GiB available initially; NVIDIA RTX 5000 Ada Generation with 32,760 MiB
VRAM and 31,408 MiB initially free. The frozen reserve was 8 GiB RAM, the heavy-paging
delta ceiling 1 GiB, and the projected 24-cell per-model pilot ceiling 6 hours.

| Requested model | Exact digest | Quantization | Stored size | Synthetic probe | Result |
| --- | --- | --- | ---: | ---: | --- |
| qwen3.5:27b | `64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59` | Q4_K_M | 16.28 GiB | 226.609 s | blocked: available RAM fell to 3.64 GiB, below reserve |
| gemma4:31b | `17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97` | Q4_K_M | 18.99 GiB | 299.235 s | blocked: RAM fell to 1.77 GiB and paging grew by 3.41 GiB |
| gpt-oss:120b | `714d86958b6e870a9320e78126ac1ce3477727a1cd9bf7194b8ace0d84c62fd8` | MXFP4 | 60.88 GiB | not loaded | blocked before load: conservative peak 87.10 GiB exceeds combined headroom |

Qwen returned structurally valid synthetic `CLEAR`; Gemma returned structurally valid
synthetic `INSUFFICIENT_CONTEXT`. These are schema/adherence probes, not corpus labels
or evidence of semantic accuracy. GPU memory rose by only about 1 MiB for Qwen and
25 MiB for Gemma between the captured before/after snapshots, while system RAM was
nearly exhausted. This indicates no material GPU residency during those measurements;
an exact layer/offload percentage was not captured and remains unknown.

The initial preflight decision omitted applying the already-frozen post-load RAM
reserve to Qwen. This was found before pilot inference. The original report is
preserved, and `preflight_report_final.json` applies the unchanged threshold with zero
additional requests. No ceiling was raised.

## Bounded pilot accounting

| Item | Count |
| --- | ---: |
| Requested cells (12 × 2 × 3) | 72 |
| Materialized cell records | 72 |
| Valid semantic judgments | 0 |
| Malformed/refused/timed-out/transport-failed cells | 0 |
| Blocked by preflight | 72 |
| Not run because request budget was exhausted | 0 |
| Pilot inference requests | 0 |
| Preflight inference requests | 2 |
| Total live inference requests | 2 |

Pilot AMBIGUOUS/CLEAR/INSUFFICIENT_CONTEXT counts and positive rates are unavailable
(`n=0` valid). All 24 record/context aggregates are
`REVIEW_REQUIRED_INCOMPLETE_PANEL`. Local-to-resolved semantic transitions and all
pairwise contingency tables have denominator zero. A v1-to-v2 label comparison is
also unavailable. Even with completed labels, it would be a compatibility comparison,
not an accuracy comparison, because both models and rubric changed.

## Review packet

`review/blinded_review_packet_v1.json` contains one resolved-context bundle for each
of the 100 items. It excludes model identities, votes, rationales, classifier outputs,
confidence tiers, and sampling strata. `private/blinded_review_mapping_v1.json` holds
the identity/random-targeted mapping separately. A single reviewer's decisions would
be a limited expert reference, not reproducible gold truth; targeted-case rates must
not be presented as population accuracy.

## Verification

- Focused v2 tests: 10 passed.
- Full repository suite: 251 passed in 127.63 seconds.
- Ruff scope: `models/annotation_v2` and `tests/test_annotation_v2.py`; clean.
- Dry run: 24/24 context cells and 6/6 synthetic schema cases; 72 rendered requests;
  zero live requests.
- Pytest emitted one non-test warning because the pre-existing `.pytest_cache`
  directory was not writable; a workspace basetemp was used successfully.

Tests cover label-dependent schema rules, bounded fields, malformed/refused/exhausted
outputs, semantic abstention versus failure, fresh retries, append-only resume,
cache invalidation, exact join rejection, conservative aggregation, same-version
provenance, blocked family access, and prompt leakage. They establish software
behavior, not model understanding.

## Commands

Freeze (already completed; intentionally refuses replacement):

```powershell
python -m models.annotation_v2.cli freeze
```

Dry run, review packet, and preflight:

```powershell
python -m models.annotation_v2.cli dry-run
python -m models.annotation_v2.cli review-packet
python -m models.annotation_v2.cli preflight
python -m models.annotation_v2.cli audit-preflight
```

Resume the bounded pilot after a future preflight in a new output namespace passes:

```powershell
python -m models.annotation_v2.cli --output-root corpus/processed/llm_annotations/v2/requested-panel-safe-host preflight
python -m models.annotation_v2.cli --output-root corpus/processed/llm_annotations/v2/requested-panel-safe-host audit-preflight
python -m models.annotation_v2.cli --output-root corpus/processed/llm_annotations/v2/requested-panel-safe-host annotate --scope pilot
```

The implemented resumable full-development command is below, but it must not be run
until a bounded pilot passes and governance approval is given:

```powershell
python -m models.annotation_v2.cli --output-root corpus/processed/llm_annotations/v2/requested-panel-safe-host annotate --scope full --confirm-full
```

## Outstanding blockers and methodological limits

1. The requested models did not satisfy the frozen memory/paging gates on this host.
   More usable system RAM and verified GPU offload are required before retrying.
2. Exact Ollama offload percentages were not captured; the GPU/RAM deltas strongly
   suggest CPU residency but do not identify the configuration cause.
3. No v2 development labels exist, so semantic adherence, context effects, agreement,
   and v1 compatibility cannot yet be estimated.
4. Resolved retrieval is intentionally conservative and cannot establish exhaustive
   specification coverage. External citations remain unavailable unless the exact
   same-version permitted source can be proven.
5. Higher agreement, larger models, or fewer abstentions would not demonstrate label
   validity. Blinded expert evidence is still needed.

No v1 artifact was overwritten, no fallback model was substituted, no full corpus was
annotated, and no classifier was retrained, recalibrated, or re-thresholded.
