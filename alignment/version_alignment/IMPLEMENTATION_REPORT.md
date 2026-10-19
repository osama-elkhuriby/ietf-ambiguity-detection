# TLS-only alignment MVP completion report

Completed and independently validated on 2026-09-09: TLS drafts 00 through 28,
followed by RFC8446. These are **provisional alignments, not measured accuracy**.
The 150-record sample has not been manually labeled. No ambiguity labels, revision
reasons, final benchmark, or ambiguity-detection models were produced.

## Run and validated artifacts

New immutable directory, relative to the repository root:

`corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5`

- [Manifest](../../corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5/run_manifest.json): `COMPLETE`, with 185 artifact checksums.
- [Machine-readable report](../../corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5/report.json).
- [Independent validation](../../corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5-validation.json): `PASS`.
- [Unreviewed sample](../../corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5/review_sample.jsonl): 150 unique records.

| Validated item | Result |
|---|---:|
| Documents / consecutive pairs | 30 / 29 |
| Clause-version occurrences / lineage nodes | 52,137 |
| Candidate hypotheses | 877,103 |
| Alignment relationships | 52,953 |
| Draft feature files / rows | 29 / 50,138 |
| Token windows independently checked | 52,361 |
| Protected original files unchanged | 394 |
| Missing or multiply assigned endpoint occurrences | 0 |
| Feature guard failures / unresolved diagnostics | 0 |

Exactly 28 existing TLS `clauses_llm` snapshots were preferred. Only TLS-22 and
RFC8446 were prepared using the existing deterministic parser and segmentation
pipeline, inside this run's `inputs/` directory. No LLM refinement ran. Original
text, identifiers, source spans, classifications and extension fields are preserved
in snapshots with an alignment envelope and normalized text.

Final inventory and SHA-256 checks found exactly the original 394 files under
`corpus/raw`, `corpus/metadata`, `corpus/processed/clauses`, and
`corpus/processed/clauses_llm`: no additions, removals, or content changes.

Generated artifacts include `run_manifest.json`, `model.json`, input snapshots,
`candidates.jsonl`, `alignments.jsonl`, `lineage.jsonl`, `review_queue.jsonl`,
`review_sample.jsonl`, `features/*.jsonl`, `embedding_windows/*.jsonl`, per-pair
candidates/alignments/reports, empty `diagnostics.jsonl`, and `report.json`.
The run occupies approximately 1.68 GB, including consolidated and per-pair copies.
These are ignored local generated data. This report and audit sidecars are outside
the completed immutable run.

## Files added and modified

New package files below are relative to `alignment/version_alignment/`; other
paths include their repository prefix. The CLI is opt-in and does not replace
existing corpus or refinement commands.

| Added files | Purpose and verification |
|---|---|
| `__init__.py`, `__main__.py` | Version identifiers and entry point; integration tests and pilot. |
| `models.py`, `schema.json` | Typed document, clause, alignment and evidence contracts; malformed-input, cardinality and schema-sync tests. |
| `config.py`, `defaults.yaml` | Validated retrieval, scoring, solver, review and embedding configuration; config tests. |
| `inputs.py` | TLS discovery, revision/date validation and exclusive preparation; synthetic and actual TLS-22/RFC8446 preparation tests. |
| `normalize.py` | Conservative normalization with original text retained; normalization tests. |
| `scoring.py`, `candidates.py` | Component scores, local/global bidirectional retrieval and bounded groups; scoring, candidate, generic-group and hash-seed regressions. |
| `embeddings.py` | Pinned offline MiniLM, direct token-ID windows and deterministic hashed caches; embedding/window tests and real TLS-08 regression. |
| `matching.py` | Bipartite assignment and MILP for competing groups; matching tests and all 29 solver reports. |
| `review.py` | Relations, alternatives, confidence, mandatory review and stratified sampling; relation/confidence tests and output audit. |
| `lineage.py` | Persistent forward-only identity and split/merge ancestry; prefix-invariance tests and full occurrence audit. |
| `temporal.py`, `features.py` | Evidence guards, reference resolution and prefix-only history; date, reference and same-day chronology tests. |
| `io.py`, `cli.py` | Exclusive JSON/JSONL writes, orchestration, failure records, reports and manifest; integration/failure tests and full pilot. |
| `validate_run.py` | Read-only hash, coverage, lineage, temporal and window audit; serialized-tokenizer regression and full validation. |
| `README.md`, `SCHEMA.md`, `LEAKAGE.md`, `IMPLEMENTATION_REPORT.md` | Usage, configuration, contracts, leakage policy, results and limitations. |
| `eval/alignment_metrics.py` | Workload reports and optional reviewed-scope candidate recall/link metrics; synthetic evaluation tests. |
| Former pilot alignment annotation guide | Retired on 2026-10-13 after the unused pilot experiment was removed; current review limitations are documented in the alignment README. |
| `tests/test_alignment_contracts.py` | Schema, configuration and invalid-relation tests. |
| `tests/test_alignment_embeddings.py` | Windows, real model, caches, TLS-08 and padding-audit regressions. |
| `tests/test_alignment_inputs.py` | Timeline/discovery and missing-input preparation tests. |
| `tests/test_alignment_integration.py` | Consecutive revisions, immutable runs, output consistency, lineage prefixes and failures. |
| `tests/test_alignment_matching.py` | Scoring, retrieval, assignment, all relation types, confidence and three-process hash-seed stability. |
| `tests/test_alignment_temporal.py` | Future/undated/non-feature rejection, reference resolution and actual same-day TLS-00/01. |
| `tests/fixtures/alignment/tls08_overlength.json` | Original 935-token failure clause, token IDs, hashes and decode-expansion evidence. |

| Modified existing file | Change and behavior impact |
|---|---|
| `alignment/segmentation/models.py` | Add the already-emitted `formal_syntax` block-type declaration. No segmentation algorithm change; existing segmentation tests pass. |
| `alignment/README.md` | Link the module and replace future publication proximity with observable age/revision count in feature guidance. |
| `eval/README.md` | Explain current TLS evaluation and distinguish future generalization protocols. |
| `DECISIONS.md` | Record approved scope, direct-ID windows, leakage controls and deferred work. |

The tracked diff contains those four files: 34 insertions and 3 deletions. New
implementation files remain uncommitted. Unrelated `.vscode/extensions.json` and
`Proposal/Literature_Review updated.pdf` were left untouched. No commit, push, PR,
remote write, other-family processing or dependency installation occurred during
recovery/completion.

## Algorithm and long-clause correction

Distinctive exact/structural anchors precede semantic and lexical retrieval.
Candidates retain individual scores and do not themselves select links.
One-to-one components use bipartite assignment with unmatched options. Adjacent
split/merge groups are bounded to three members with coverage and contribution
checks; competing groups use non-overlapping MILP selection. Moves use structural
correspondence and anchor-order evidence. Close alternatives remain uncertain.

Pinned model:
`sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
`model.json` records all 11 snapshot artifact hashes; weights SHA-256:
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
The run uses deterministic CPU execution, batch size 32, sentence-transformers
6.0.1, transformers 5.16.1, torch 2.12.0 and NumPy 2.4.6. The manifest also records
SciPy 1.17.1, Pydantic 2.13.5, RapidFuzz 3.14.6 and NetworkX 3.6.1.

The original TLS-08 error arose because decoding a 254-token window and
retokenizing it yielded 256 content tokens before special tokens. Inference now
slices original token IDs directly and forwards prepared features through the
encoder module stack. It never relies on decode/retokenize equivalence or silent
truncation. The current tokenizer API's BERT special-token template is checked.

| Window statistic / setting | Value |
|---|---:|
| Effective limit including special tokens | 256 |
| Content limit / special tokens | 254 / 2 |
| Overlap / deterministic stride | 32 / 222 |
| Original source tokens, before overlap and specials | 1,501,382 |
| Long clause-version instances | 171 |
| Maximum clause token count | 1,004 |
| Windows / maximum encoded window length | 52,361 / 256 |
| Cached documents / newly encoded documents | 17 / 13 |
| Long instances cached / newly encoded | 72 / 99 |

Each window preserves clause/document ID, source token bounds, encoded length and
token hash. All boundaries, hashes and complete coverage passed independent audit.
The failing TLS-08 clause now has windows `[0,254)`, `[222,476)`, `[444,698)`,
`[666,920)`, `[888,935)` and encoded lengths 256, 256, 256, 256, 49.

Empty/whitespace clauses fail explicitly; empty batches are supported. Tests cover
exact limits, one token over, very long text, special tokens, protocol punctuation,
invalid overlap, strict progress and full coverage. Normalized window vectors are
summed and normalized again; pooling remains an approximation.

## Relationships, confidence and manual work

| Relationship | Count |
|---|---:|
| UNCHANGED | 36,592 |
| MODIFIED | 4,452 |
| INSERTED | 2,356 |
| DELETED | 2,025 |
| SPLIT | 261 |
| MERGED | 260 |
| MOVED | 1,808 |
| UNCERTAIN | 5,199 |
| Total | 52,953 |

| Confidence | Count | Share |
|---|---:|---:|
| HIGH | 41,377 | 78.14% |
| MEDIUM | 1,996 | 3.77% |
| LOW | 9,580 | 18.09% |

Confidence is rule-based and uncalibrated; probability is null.
**26,884 relationships require review, or 50.77% of all relationship records.**
This includes complex, moved, uncertain, class-transition, protected-token-change
and insertion/deletion cases plus deterministic high-confidence audit sampling.
High confidence does not remove mandatory review for these categories.

Every applicable endpoint occurrence has exactly one disposition. Accepted
correspondence relationships are 81.91% of records; this is operational coverage,
not accuracy. The 5,199 uncertain records remain unresolved manual work.

The 150-record sample contains 12 deletions, 12 insertions, 24 merges, 24
modifications, 23 moves, 22 splits, 11 uncertain and 22 unchanged cases. Bands:
HIGH 58, MEDIUM 57, LOW 35. Reviewer and reviewed-relation fields remain empty.
This sample overrepresents difficult cases; raw sample accuracy would not be a
corpus-weighted estimate.

Candidate recall and link precision/recall/F1 are **null**: no complete reviewed
gold scope was supplied. The current evaluator accepts complete reviewed
document-pair scopes. Isolated sampled relationships cannot establish full recall.

## Runtime and diagnostics

Reported pipeline runtime: **693.598690 seconds (11m 33.60s)**. Sum of pair times:
**655.628153 seconds**. Other time includes preparation, initialization and
consolidated/feature output. The report timer ends before final manifest hashing;
creation time to manifest filesystem timestamp is approximately 694.91 seconds.
The latter is a filesystem observation, not a separate instrumented benchmark.

Seventeen document caches were reused, so this is a mixed-cache run, not a
cold-start benchmark. Pair times include target embedding, retrieval, selection
and relationship construction, ending before pair JSONL serialization. Unrounded
values remain in `report.json`.

| Pair | Seconds |
|---|---:|
| 00 -> 01 | 3.615 |
| 01 -> 02 | 5.362 |
| 02 -> 03 | 3.717 |
| 03 -> 04 | 4.100 |
| 04 -> 05 | 3.627 |
| 05 -> 06 | 4.289 |
| 06 -> 07 | 5.173 |
| 07 -> 08 | 4.844 |
| 08 -> 09 | 4.527 |
| 09 -> 10 | 4.191 |
| 10 -> 11 | 6.225 |
| 11 -> 12 | 4.851 |
| 12 -> 13 | 5.718 |
| 13 -> 14 | 6.362 |
| 14 -> 15 | 4.597 |
| 15 -> 16 | 5.330 |
| 16 -> 17 | 36.777 |
| 17 -> 18 | 35.550 |
| 18 -> 19 | 39.688 |
| 19 -> 20 | 42.291 |
| 20 -> 21 | 46.119 |
| 21 -> 22 | 46.916 |
| 22 -> 23 | 48.254 |
| 23 -> 24 | 45.433 |
| 24 -> 25 | 46.233 |
| 25 -> 26 | 45.879 |
| 26 -> 27 | 49.549 |
| 27 -> 28 | 48.340 |
| 28 -> RFC8446 | 48.072 |

All components reported optimal solutions: 10,576 bipartite components, 619
complex components and 30,345 locked anchors. No solver failure, timeout, fallback,
skipped pair or skipped clause occurred. Semantic mode did not fall back to lexical
mode. Input warning lists and `diagnostics.jsonl` are empty.

## Exact tests and failure history

Commands used the existing Conda environment at
`C:/Users/mosta/.conda/envs/ietf-ambiguity-detection`, through `conda run
--no-capture-output -p ...` on Windows for native-library paths.

| Command / check | Result |
|---|---|
| `python -m pytest tests/test_alignment_embeddings.py -q` | **29 passed, 2 warnings in 6.11s**, no skips. |
| `python -m pytest tests/ -q` | **121 passed, 2 warnings in 11.76s**, no skips. |
| `python -m ruff check alignment/version_alignment eval/alignment_metrics.py tests/test_alignment*.py` | **All checks passed!** |
| Full semantic command below | **Exit 0**, 29 pairs and complete manifest. |
| Independent `validate_run(...)` with protected baseline | **PASS** for hashes, endpoint coverage, lineage, temporal guards and complete token coverage. |
| `git diff --check` | **Exit 0**, no whitespace errors; only Git LF/CRLF conversion notices. |

The warnings are third-party `SwigPyPacked` and `SwigPyObject` deprecations about
missing `__module__` attributes. The real pinned-model test ran without network
access. Before the audit fix, the suite also passed 120 tests in 209.34s; 121 is
the final count. Earlier lexical/semantic 00 -> 01 and 27 -> 28 smoke runs remain
preserved in `smoke-early-*` and `smoke-late-*`; the full run includes both pairs.

Preserved records under `corpus/processed/alignments/draft-ietf-tls-tls13/`:

| Failure record | Cause and correction |
|---|---|
| `pilot-semantic-v1/failure.json` | Original TLS-08 overflow after seven pairs; direct-ID windows and actual-clause regression replace decode/retokenize. |
| `pilot-semantic-v3/failure.json` | Stopped after 15 pairs because TLS-00 and TLS-01 share 2014-04-17; added validated numerical-predecessor admission and preflight tests. |
| `pilot-semantic-v4/failure.json` | Stopped after 16 pairs because overlap ties depended on Python hash-set iteration; added positional tie-breaking and a three-hash-seed regression. |
| `pilot-semantic-v5-validation-failure.json` | Standalone audit tokenizer retained 128-token padding. The first clause actually has 18 source tokens and its saved hash matched. Disable serialized padding and truncation in audit; add regression. |

The v5 run itself has no `failure.json`. Its initial audit issue was a validator
error, not embedding truncation. Only the validator and its test changed after
the run; the timeline was not regenerated. Its manifest and all 185 artifact
hashes remain intact. The successful external audit records corrected validator
SHA-256 `c924c07ccd7c8dc4f2336e0efc5a3c41546121fcdbde88778086d86e18446e90`.
The run manifest retains the validator hash captured at run start.

## Temporal controls and remaining limitations

Features select only the target draft and earlier numerical revisions and enforce
availability dates. The same-day TLS-00/01 exception applies only to validated
earlier revisions in the same family. Same-day external evidence, future revisions,
RFC text, later errata/discussions, undated evidence, label support and annotation
metadata are rejected by the feature guard. RFC8446 supports offline alignment
only and is excluded from all draft feature prefixes. All 29 prefixes and their
serialized dependencies passed checks.

References resolve against the local catalog to an admissible historical version;
missing references remain unresolved. Observable age and revisions seen replace
future publication proximity. Actual days to RFC, future revision count/stability,
revision reasons and publication outcomes are not prediction features.

Remaining work and limits:

- Review the 150 records in a separate annotation copy, correcting endpoints as
  well as relation types. Independently review difficult cases. Do not edit the
  automatic run or mark isolated samples as complete gold scopes. Complete reviewed
  pairs are needed for the current recall evaluator.
- Confidence thresholds are development defaults. Many-to-many changes, movement
  heuristics, surface actor/condition extraction and group/window pooling need
  empirical error analysis and calibration.
- TLS-22 and RFC8446 use deterministic classifications; preferred neighboring
  snapshots have LLM refinement. Review must account for that difference.
- Automatic correction application, full symbolic-reference expansion, discussion
  retrieval and comprehensive gold evaluation remain future work.
- Modern MiniLM pretraining may include historical RFC material. Explicit evidence
  guards do not establish historical training-data purity or that the encoder
  existed at draft publication time. Lexical mode is the reproducible model-free
  ablation; no historical deployment claim is made.
- One TLS family cannot establish family or protocol generalization. Family splits,
  cross-family temporal evaluation and protocol holdouts remain future work. No
  other family, ambiguity ground truth, final benchmark or ambiguity-model training
  was included.

Reproduce into a **new** output directory; existing paths are rejected:

```powershell
python -m alignment.version_alignment --backend semantic --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --prepare-missing --output corpus/processed/alignments/draft-ietf-tls-tls13/NEW_RUN_NAME
python -m alignment.version_alignment.validate_run corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5 --protected corpus/processed/alignments/recovery-protected-inputs-v3.json
```
