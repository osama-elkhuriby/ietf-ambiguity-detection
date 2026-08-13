# Corpus metadata (Stage 1)

JSON manifests produced by the fetch scripts. Paths in `local_txt_path` fields are
**repo-relative** (for example `corpus/raw/rfcs/RFC9460.txt`).

| File | Producer | Contents |
|------|----------|----------|
| `drafts.json` | `fetch_datatracker.py`, `build_corpus.py` | Per-draft title, WG, RFC link, revision list with dates/word counts and `local_txt_path` |
| `rfcs.json` | `fetch_rfc.py` | Per-RFC index fields including `draft` (final I-D name) and `local_txt_path` |
| `draft_rfc_links.json` | `link_draft_rfc.py`, `sync_metadata.py` | Join rows with `confidence`: `high`, `single-source`, or `conflict` |
| `errata.json` | `fetch_errata.py` | RFC Editor errata grouped by `rfc<N>`, filtered to RFCs in `rfcs.json` |
| `verified_technical_errata.json` | `fetch_errata.py --verified-technical` | Verified Technical errata across the curated manifest |
| `final_rfc_manifest.csv` | Curated + refreshed by `sync_metadata.py` | Phase-2 target families, protocol coverage, on-disk provenance |
| `corpus_audit.json` | `validate_corpus.py`, `sync_metadata.py --write-audit` | Machine-readable completeness and coverage report |

## Draft collection modes

| Mode | Meaning | Examples |
|------|---------|----------|
| `full-history` | All Datatracker submissions fetched under `corpus/raw/drafts/` | QUIC, TLS, HTTP, DNS, TCP, IPv6 |
| `rfc-only` | No Datatracker submission archive; RFC plaintext + errata only | OSPF v2 (RFC2328), BGP-4 (RFC4271) |

## Post-fetch workflow

```bash
python corpus/sync_metadata.py --refresh-manifest --write-audit
python corpus/validate_corpus.py
```

`sync_metadata.py` backfills missing `rfc_number` values from `rfcs.json`, rebuilds
high-confidence draft↔RFC links, updates manifest provenance columns (file sizes,
SHA-256, `match_verified`), and optionally writes `corpus_audit.json`.

## Errata labels

`verified_technical_errata.json` records the RFC Editor source URL, exact filters,
target-manifest provenance, target and record counts, and a `by_rfc` mapping. These
records are high-confidence evidence about published RFCs — not automatic draft-stage
ambiguity labels. Alignment back to the relevant draft revision and annotation are still
required.

Plaintext bodies live under `corpus/raw/` (gitignored). Rebuild with:

```bash
python corpus/build_from_manifest.py
```

See `../README.md` and `../../DECISIONS.md` for methodology and design rationale.
