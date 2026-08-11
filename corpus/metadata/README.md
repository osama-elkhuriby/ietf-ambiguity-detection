# Corpus metadata (Stage 1)

JSON manifests produced by the fetch scripts. Paths are relative to the repository root.

| File | Producer | Contents |
|------|----------|----------|
| `drafts.json` | `fetch_datatracker.py`, `build_corpus.py` | Per-draft title, WG, RFC link, revision list with dates/word counts and `local_txt_path` |
| `rfcs.json` | `fetch_rfc.py` | Per-RFC index fields including `draft` (final I-D name) and `local_txt_path` |
| `draft_rfc_links.json` | `link_draft_rfc.py` | Join rows with `confidence`: `high`, `single-source`, or `conflict` |
| `errata.json` | `fetch_errata.py` | RFC Editor errata grouped by `rfc<N>`, filtered to RFCs in `rfcs.json` |
| `verified_technical_errata.json` | `fetch_errata.py --verified-technical` | Only RFC Editor records with status `Verified` and type `Technical`, grouped across the curated final RFC manifest |

`verified_technical_errata.json` records the RFC Editor source URL, exact filters,
target-manifest provenance, target and record counts, and a `by_rfc` mapping. Each
erratum retains its ID, RFC, section, original and corrected text, notes, submission
and verification metadata, and direct RFC Editor URL. RFCs with no matching errata
remain present with an empty list.

These records are high-confidence evidence about published RFCs. They are not
automatically draft-stage ambiguity labels; alignment back to the relevant draft
revision and annotation/adjudication are still required.

Plaintext bodies live under `corpus/raw/` (gitignored). Rebuild with:

```bash
python corpus/build_corpus.py draft-ietf-dnsop-svcb-https
```
