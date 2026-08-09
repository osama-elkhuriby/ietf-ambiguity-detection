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

All network fetchers need normal internet access (Datatracker, rfc-editor.org).

**Pilot mirror:** pass `--out-dir experiments/pilot/data/raw` (alias: `--output-dir`) to
`fetch_datatracker.py` to keep a copy under the pilot tree; metadata still updates
`corpus/metadata/drafts.json`.

Manifest schemas: `metadata/README.md`. Large plaintext under `corpus/raw/` is gitignored;
RFCScope errata reuse for modeling is documented in `../DECISIONS.md`.
