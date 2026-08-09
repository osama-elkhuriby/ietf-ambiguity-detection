# Corpus metadata (Stage 1)

JSON manifests produced by the fetch scripts. Paths are relative to the repository root.

| File | Producer | Contents |
|------|----------|----------|
| `drafts.json` | `fetch_datatracker.py`, `build_corpus.py` | Per-draft title, WG, RFC link, revision list with dates/word counts and `local_txt_path` |
| `rfcs.json` | `fetch_rfc.py` | Per-RFC index fields including `draft` (final I-D name) and `local_txt_path` |
| `draft_rfc_links.json` | `link_draft_rfc.py` | Join rows with `confidence`: `high`, `single-source`, or `conflict` |
| `errata.json` | `fetch_errata.py` | RFC Editor errata grouped by `rfc<N>`, filtered to RFCs in `rfcs.json` |

Plaintext bodies live under `corpus/raw/` (gitignored). Rebuild with:

```bash
python corpus/build_corpus.py draft-ietf-dnsop-svcb-https
```
