# Stage 1 — Corpus

`fetch_datatracker.py` — pulls Internet-Draft version history via `ietfdata`. Verified
working against the real library API (see `../DECISIONS.md`, 2026-08-08). Needs normal
internet access to run (not usable from a locked-down sandbox).

Still to add here:
- Errata integration. Check what RFCScope's public dataset (github.com/HIPREL-Group/RFCScope)
  already covers before writing an independent RFC Editor errata scraper — see `../DECISIONS.md`.
- Final-RFC linking for drafts that were published (`dt.document_from_rfc()` /
  `dt.documents_from_bcp()` in `ietfdata.datatracker` look like the relevant entry points —
  confirm against the installed package before relying on them).
- `corpus/raw/` is where `fetch_datatracker.py` writes downloaded text — gitignored on purpose,
  since it's large and re-fetchable. Commit `versions.json` manifests if you want a lightweight
  record of exactly what was pulled and when.
