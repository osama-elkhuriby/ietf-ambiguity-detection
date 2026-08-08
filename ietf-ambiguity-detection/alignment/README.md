# Stage 2 — Alignment

Clause segmentation, version-to-version clause matching, and revision-aware feature
computation (proposal SS6.2).

Before writing custom diff logic here, check `iddiff` (one of IETF's own author tools,
at author-tools.ietf.org) — it's built specifically for diffing two Internet-Draft
revisions and may cover a meaningful chunk of the "textual and structural changes"
computation this stage needs.

Planned contents:
- clause segmentation (spaCy-based)
- version matching (dynamic-programming or embedding-based, per SS6.4)
- revision feature extraction: edit frequency, semantic distance between versions,
  normative-strength changes, clause stability, proximity to publication (SS6.2)
