# Corpus raw storage

This directory holds Stage 1 plaintext fetched from the IETF Datatracker and RFC Editor.

**Tracked in git** (so collaborators can clone and work offline):

- `drafts/<draft-name>/` — one file per submitted Internet-Draft revision
- `rfcs/RFC<N>.txt` — published RFC plaintext

**Not tracked** (re-created automatically by fetch scripts):

- `rfcs/.rfcindex-cache/` — cached `rfc-index.xml`
- `errata/errata_full.json` — bulk RFC Editor errata export cache

Metadata manifests live in `../metadata/`. After cloning, verify completeness:

```bash
python corpus/validate_corpus.py
```

To rebuild from scratch (needs network):

```bash
python corpus/build_from_manifest.py
```
