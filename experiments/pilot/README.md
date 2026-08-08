# Pilot experiment (Phase 1)

Small end-to-end validation on one or two **DNS (`dnsop`)** Internet-Drafts before Phase 2
corpus build. Protocol-family choice and reuse strategy: `../../DECISIONS.md`.

## Goals

Answer, with evidence, whether these assumptions hold:

1. Datatracker revision history is complete enough for modeling (`ietfdata` + plaintext `.txt`).
2. Clauses can be aligned across revisions with acceptable manual effort.
3. The annotation scheme (class + tier + revision-reason + subtype) is usable and stable.
4. Heuristic and LLM baselines produce signals worth routing on (not random noise).
5. Positive labels at **draft stage** can be grounded (errata on published RFC, revision diffs, RFCScope categories).

## Suggested drafts

Pick drafts that **published as RFCs** so errata and RFCScope labels can be mapped back through
alignment (see `docs/annotation_guide_v0.md`, §3):

- `draft-ietf-dnsop-svcb-https` (SVCB/HTTPS RRs; multiple revisions; rich dnsop discussion)
- One additional dnsop draft the team knows has verified errata on the final RFC

Confirm exact names on [Datatracker](https://datatracker.ietf.org/).

## Workflow

| Step | Action | Artifact |
|------|--------|----------|
| 1 | Fetch revisions | `corpus/raw/<draft-name>/` via `corpus/fetch_datatracker.py` |
| 2 | Inspect text | `versions.json`, sample `-NN.txt` files (section numbering, boilerplate) |
| 3 | Hand-align | 15–30 clause pairs across 2+ revisions → `data/processed/` (local; gitignored) |
| 4 | Label | `labels/pilot_labels.csv` from `labels/pilot_labels_template.csv` |
| 5 | Baselines | Heuristic + one LLM prompt on the same clauses |
| 6 | Memo | `docs/pilot_memo_template.md` → filled memo for advisors |

### Fetch example

From repo root, with network access:

```bash
conda activate ietf-ambiguity-detection
python corpus/fetch_datatracker.py draft-ietf-dnsop-svcb-https
```

Optional output directory (e.g. keep pilot copies separate):

```bash
python corpus/fetch_datatracker.py --output-dir experiments/pilot/data/raw draft-ietf-dnsop-svcb-https
```

Commit lightweight `versions.json` manifests if you need a record of what was pulled; do not commit full `.txt` corpora.

### Alignment (pilot)

Before custom code, diff two revisions with [iddiff](https://author-tools.ietf.org/iddiff) and
note whether its hunks match your intended **clause** boundaries. Record the working clause
definition in the annotation guide (§1).

### Labeling

- Guide: `docs/annotation_guide_v0.md`
- Template: `labels/pilot_labels_template.csv`
- Target: **15–20 clauses minimum** for the first memo; expand if inter-annotator agreement is still unclear.

### Baselines

- Heuristic: `models/heuristic/` (implement minimal pilot script when alignment exists)
- LLM: one structured prompt with evidence span; log **cost per clause** in the memo

Do **not** start transformer fine-tuning until the label schema stabilizes (`models/transformer/README.md`).

## Directory layout

```text
experiments/pilot/
  docs/           Annotation guide, memo template
  labels/         CSV annotations (commit templates + small pilot sets only)
  data/
    raw/          Optional mirror of fetched drafts (gitignored)
    processed/    Aligned clause tables (gitignored)
  notebooks/      Exploratory only — not source of truth for results
  results/        Pilot metrics, prompts, configs (commit summaries, not secrets)
```

Raw text for all stages defaults under `corpus/raw/` (see root `README.md`).
