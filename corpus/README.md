# Stage 1 — Corpus

Run scripts from the repository root (paths are resolved automatically):

```bash
python corpus/build_corpus.py draft-ietf-dnsop-svcb-https
```

Or run steps individually:

| Script | Role |
|--------|------|
| `fetch_datatracker.py` | Internet-Draft revision history → `corpus/raw/drafts/`, `corpus/metadata/drafts.json` |
| `link_draft_rfc.py` | Draft↔RFC join with cross-check → `corpus/metadata/draft_rfc_links.json` |
| `fetch_rfc.py` | RFC index metadata + plaintext → `corpus/raw/rfcs/`, `corpus/metadata/rfcs.json` |
| `fetch_errata.py` | RFC Editor bulk errata (cached) → `corpus/metadata/errata.json` |
| `build_corpus.py` | Runs the four steps above for one or more draft names |
| `build_from_manifest.py` | Batch-build every draft family in `metadata/final_rfc_manifest.csv` |
| `sync_metadata.py` | Backfill RFC numbers, rebuild links, refresh manifest, write audit |
| `validate_corpus.py` | Audit storage layout, download completeness, and protocol coverage |

All network fetchers need normal internet access (Datatracker, rfc-editor.org).

## Structured storage layout

```text
corpus/
  metadata/                 JSON manifests (committed)
    drafts.json
    rfcs.json
    draft_rfc_links.json
    errata.json
    verified_technical_errata.json
    final_rfc_manifest.csv    curated Phase-2 target families
  raw/                        plaintext + caches (gitignored)
    drafts/<draft-name>/      one file per submitted revision (.txt/.xml/.html)
    rfcs/rfc<N>.txt
    rfcs/.rfcindex-cache/     RFC index XML cache (ietfdata)
    errata/errata_full.json   bulk RFC Editor errata cache
```

`local_txt_path` fields in metadata are stored repo-relative (for example
`corpus/raw/rfcs/RFC9460.txt`) so manifests remain valid across machines.

## Validate completeness and coverage

```bash
python corpus/validate_corpus.py
python corpus/validate_corpus.py --repair-paths --relocate-cache
python corpus/validate_corpus.py --json corpus/metadata/corpus_audit.json
```

The validator compares `final_rfc_manifest.csv` against on-disk plaintext and reports
gaps grouped by protocol area: transport, routing, dns, security, application.

After any fetch or rebuild, synchronize metadata:

```bash
python corpus/sync_metadata.py --refresh-manifest --write-audit
```

## Batch catch-up from manifest

```bash
python corpus/build_from_manifest.py --dry-run
python corpus/build_from_manifest.py
python corpus/build_from_manifest.py --area dns transport
```

To create the proposal's high-confidence errata label source without overwriting the
full errata metadata, filter the RFC Editor export to records whose status is
`Verified` and type is `Technical`:

```bash
python corpus/fetch_errata.py --verified-technical
```

The default targets come from `metadata/final_rfc_manifest.csv`, and the result is
written to `metadata/verified_technical_errata.json`. The full export is cached under
`corpus/raw/errata/` for 24 hours. Use `--refresh` to force a new download, or pass RFC
numbers explicitly to create a smaller result:

```bash
python corpus/fetch_errata.py --verified-technical --refresh
python corpus/fetch_errata.py --verified-technical 9000 9460
```

**Pilot mirror:** pass `--out-dir experiments/pilot/data/raw` (alias: `--output-dir`) to
`fetch_datatracker.py` to keep a copy under the pilot tree; metadata still updates
`corpus/metadata/drafts.json`.

Manifest schemas: `metadata/README.md`. Raw plaintext under `corpus/raw/` is **committed**
so collaborators can work offline; only re-fetchable caches (RFC index XML, bulk errata
JSON, Datatracker SQLite) are gitignored. See `corpus/raw/README.md`.

## TLS 1.3 ground-truth preparation

Build the TLS-only silver review set from verified RFC 8446 technical errata and
the frozen `draft-ietf-tls-tls13` alignment run:

```bash
python -m corpus.build_tls_silver_labels
```

Artifacts are separated by lifecycle under
`processed/ground_truth/draft-ietf-tls-tls13/`:

- `silver/`: immutable automatic candidates, errata mapping, and audit report
- `review/`: editable human-review template with pending adjudication fields
- `gold/`: reserved for completed, adjudicated ground truth

The automatic records are not gold: they retain `ambiguity_label=UNCERTAIN`,
`ground_truth=false`, and `requires_human_review=true`. Make a working copy of the
review template before annotation because rebuilding replaces the template.

## Additional family review preparation

For a completed family alignment, build normative review signals and the independent
review package with:

```bash
python -m corpus.build_family_silver_labels --family draft-ietf-httpbis-semantics --rfc RFC9110 --alignment-run corpus/processed/alignments/draft-ietf-httpbis-semantics/lexical-first-classifier-v1
python -m corpus.create_tls_review_batch --source corpus/processed/ground_truth/draft-ietf-httpbis-semantics/review/ground_truth_review_template.csv --classification-input-dir corpus/processed/alignments/draft-ietf-httpbis-semantics/lexical-first-classifier-v1/inputs --classification NORMATIVE --all-lineages --output-dir corpus/processed/ground_truth/draft-ietf-httpbis-semantics/review/all_candidates_v1
```

Normativity always comes from the frozen first-stage classification stored in the
alignment inputs. Verified technical errata can support a silver-positive candidate
only when their conservatively traced draft clauses are normative. Because errata are
sparse, the generic builder also emits `SILVER_CHANGE_CANDIDATE` records for normative
clauses with confident substantive revision links. A change is only a review signal,
not an automatic ambiguity label. Stable normative lineages are sampled as controls;
all records remain pending human review and are deduplicated by lineage.
