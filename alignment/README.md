# Stage 2 — Alignment

Clause segmentation, version-to-version clause matching, and revision-aware feature
computation (proposal SS6.2).

Before writing custom diff logic here, check `iddiff` (one of IETF's own author tools,
at author-tools.ietf.org) — it's built specifically for diffing two Internet-Draft
revisions and may cover a meaningful chunk of the "textual and structural changes"
computation this stage needs.

Planned contents:
- clause segmentation (spaCy-based, with RFC plaintext pre-processing: strip boilerplate, keep section paths)
- version matching: combine **iddiff** hunks with embedding/DP alignment for moved or split paragraphs (SS6.4)
- revision feature extraction: edit frequency, semantic distance between versions,
  normative-strength changes (BCP 14 keyword shifts), clause stability, proximity to publication (SS6.2)

**RFC / IETF specifics:**
- Prefer Datatracker **plaintext** (`.txt`) for the pilot; note that XML (`*.xml`) preserves structure
  but adds parsing cost — defer unless plaintext alignment fails.
- Track **section number drift** (e.g. 3.2 → 4.1) separately from text similarity; iddiff highlights
  moves that naive diff misses.
- Cross-RFC references: unresolved or wrong RFC numbers are **U-4**; flag reference targets for LLM context
  the way RFCScope builds cross-document context.
- Distinguish **editorial** vs **normative** edits when computing “ambiguity clarification” features —
  a `MUST` → `SHOULD` change is not merely editorial.
