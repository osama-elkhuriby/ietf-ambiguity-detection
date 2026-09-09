# TLS 1.3 clause alignment MVP

Aligns only `draft-ietf-tls-tls13` revisions and, where valid, the final draft to
RFC8446. Relationships are correspondence predictions, never ambiguity labels or
revision reasons. Other families, benchmark splits, training and cross-protocol
evaluation are outside this milestone.

See [the completion report](IMPLEMENTATION_REPORT.md) for the validated 30-document,
29-pair pilot, exact tests, runtimes and remaining manual review.

## Commands

Activate the repository Conda environment. On Windows, direct execution of its
Python binary can load incorrect native DLLs; use activation or `conda run`.
The semantic backend initializes PyArrow before torch on Windows to avoid an
observed native import crash. No installed package is patched.

```powershell
conda activate ietf-ambiguity-detection
python -m alignment.version_alignment --backend lexical --pair draft-ietf-tls-tls13-00 draft-ietf-tls-tls13-01 --output corpus/processed/alignments/draft-ietf-tls-tls13/my-early-smoke
python -m alignment.version_alignment --backend semantic --allow-download --pair draft-ietf-tls-tls13-27 draft-ietf-tls-tls13-28 --output corpus/processed/alignments/draft-ietf-tls-tls13/my-late-smoke
python -m alignment.version_alignment --backend semantic --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --prepare-missing --output corpus/processed/alignments/draft-ietf-tls-tls13/my-full-pilot
```

The pinned MiniLM commit above is the public snapshot selected for this MVP.
`--allow-download` resolves and records an immutable revision. Subsequent local-only
runs use `--model-revision`; `model.json` records all snapshot artifact hashes.
No hosted LLM call is made. `--prepare-only --prepare-missing` validates and prepares
the timeline without alignment or embeddings. `--config PATH` accepts YAML overrides.

Output directories must be new. Failed runs preserve `failure.json`; retry with a
different directory. Only `run_manifest.json` with `status=COMPLETE` establishes a
finished alignment run. Run artifacts are written exclusively, not overwritten.

## Inputs

- Discover TLS outputs recursively in `corpus/processed/clauses_llm`.
- Never silently substitute older `clauses` classifications.
- Validate revision sequence from 00, chronology, duplicate identities, positions,
  raw-source hashes, classifications, and block-relative spans.
- Preserve existing clause IDs and all rule/LLM extras in immutable input snapshots.
- Prepare missing TLS-22/RFC8446 using existing deterministic parsing/segmentation,
  only inside the run directory, only with `--prepare-missing`. No LLM refinement.
- Do not rewrite corpus metadata, links, sources, or existing clause outputs.
- An absent RFC entry permits draft-only operation; a conflicting association fails.

Normalization preserves NFC text, uppercase modals, negation, numbers, constants,
references, conditions, identifiers and formal multiline layout. Only explicit
bullet glyphs are removed; numeric/alphabetic list markers are retained. Original
text and provenance spans remain unchanged.

## Algorithm

Bidirectional retrieval first searches title-corresponding and nearby sections,
then a bounded global fallback. Unique distinctive exact matches supported by the
same section title are locked. Repeated wording is not locked on text alone.
Candidates retain semantic, lexical, structural, positional, block and protected-token
scores. Number/modal changes penalize correspondence modestly and trigger review;
they are not automatic rejection or ambiguity signals.

Adjacent split/merge groups require bidirectional lexical coverage, per-member
coverage and unique contribution by every member. Group semantic similarity uses
normalized sums of member embeddings; lexical scoring uses full concatenated text.
One-to-one components use bipartite assignment with unmatched choices. Components
containing competing groups use SciPy MILP with non-overlap constraints. A solver
failure aborts explicitly. Order is a weak preference, not a hard constraint.

Moves use title correspondence elsewhere or reliable anchor inversions, not section
renumbering alone. Close alternatives remain `UNCERTAIN`. Lineage IDs preserve
earlier prefixes, with explicit descendant branches and merges.

Long text uses original token-ID slices, passed directly through the encoder module
stack without decoding or retokenizing. Effective limits include required special
tokens and the minimum declared tokenizer/model capacity. Every source token remains
covered. Token bounds and hashes are saved both in the cache and run-local
`embedding_windows/`. Empty clauses fail explicitly; empty batches are supported.
Normalized window vectors are summed and normalized. CPU deterministic settings,
versioned cache keys and artifact checksums support repeatability. Older v1/v2
embedding caches and failed runs are retained and are not reused by this v3 path.

## Configuration

`defaults.yaml` is an exported reference; `Config` is the validated default source.
Unknown keys, nonfinite values and inconsistent threshold ordering are errors.

| Settings | Defaults | Pilot rationale |
|---|---|---|
| Local/global top-k; nearby sections | 10 / 5; 1 | Bounded search with global movement recovery |
| Candidate/match/high score | .35 / .62 / .88 | Broad retrieval, conservative assignment, high band |
| Uncertain/high margin | .025 / .12 | Preserve competing alternatives |
| Semantic/lexical weights | .38 / .38 | Balance meaning and protocol text evidence |
| Structure/position weights | .12 / .04 | Structural support without fixed ordering |
| Block/protected weights | .04 / .04 | Compatibility and bounded change penalties |
| Group size/coverage/member coverage | 3 / .75 / .55 | Limit unrelated grouping |
| Group penalty/minimum member tokens | .12 / 4 | Prefer simple, distinctive correspondence |
| Solver limit | 30 seconds/component | Fail clearly if optimum not obtained |
| Batch/window overlap | 32 / 32 tokens | Deterministic CPU execution |
| High-confidence audit | 5% | Seeded control sample |
| Review sample/seed | 150 / 17 | Relation/confidence-stratified pilot review |

These are uncalibrated development defaults, not measured accuracy guarantees.
Weights renormalize over available components in lexical mode. No corpus-wide
TF-IDF fitting or test-fitted preprocessing is used.

## Outputs and evaluation

The run manifest records input/model/package/code hashes, effective configuration,
and artifact checksums. `pairs/` contains candidate graphs, relationships and reports.
Root files consolidate alignments, lineage, review queues, a 150-record review sample,
basic guarded features, diagnostics and a report. `review_sample.jsonl` is unreviewed,
not gold. It includes full endpoint clause records and competing links.

All split/merge/move, uncertain, class-transition and protected-token-change cases
are reviewed; insertion/deletion residuals are also conservatively reviewed. High
confidence receives deterministic audit sampling. Confidence is rule-based with
`calibrated_probability=null`.

Reports include relationship counts, confidence bands, review rate, operational
coverage, runtime and solver diagnostics. Accuracy and candidate recall are null
without reviewed gold. `--gold FILE.jsonl` requires alignment-shaped records with
`reviewer` and `scope_complete=true` for **complete reviewed document-pair scopes**.
A partial 150-record sample cannot establish full recall and must not set that flag.
`eval.alignment_metrics.evaluate` is also available as a Python API.

Synthetic metrics verify code, not empirical TLS accuracy. No document-family or
protocol-family generalization, ambiguity evaluation or final benchmark is claimed.

Known limitations: surface actor/condition signatures are not semantic parsing;
move detection and group/window pooling are approximate; many-to-many rewrites
remain uncertain; cross-document coverage is incomplete; modern encoder pretraining
may include historical RFC material; classification variability creates review work.
Calibration training and correction application are outside this MVP.

See `SCHEMA.md`, `LEAKAGE.md`, and the pilot alignment annotation guide.

```powershell
python -m pytest tests/ -q
python -m ruff check alignment/version_alignment eval/alignment_metrics.py tests/test_alignment*.py
```

Tests use deterministic synthetic fixtures and fake embeddings. A preparation
regression uses collected TLS sources offline. Tests make no external API calls.

The real TLS-08 overlength fixture is `tests/fixtures/alignment/tls08_overlength.json`.
A local-model test reproduces the original WordPiece expansion and verifies direct-ID
inference; it skips only if the pinned model is absent, without downloading.
After a full run, validate all artifact/model hashes, endpoint coverage, lineage,
review sampling, temporal dependencies and every token-window boundary with:

```powershell
python -m alignment.version_alignment.validate_run corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5 --protected corpus/processed/alignments/recovery-protected-inputs-v3.json
```
