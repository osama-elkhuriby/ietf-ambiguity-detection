# Stage 2 — Alignment

Document segmentation, version-to-version clause matching, and revision-aware feature
computation (proposal SS6.2).

Before writing custom diff logic here, check `iddiff` (one of IETF's own author tools,
at author-tools.ietf.org) — it's built specifically for diffing two Internet-Draft
revisions and may cover a meaningful chunk of the "textual and structural changes"
computation this stage needs.

Implemented foundation:

- deterministic Internet-Draft TXT/XML and RFC TXT parsing in `segmentation/`
- a shared document → section → block tree with stable block IDs and source locations
- profile-aware handling of legacy indented RFC headings and unnumbered RFC back matter
- RFC number, title, publication date, and category extraction
- XML2RFC `<xref>`/`<relref>`/`<eref>` and rendered-text reference extraction
- preservation of paragraphs, list items, definitions, artwork, source code, tables, and reference definitions

Planned next contents:

- clause refinement and normative/contextual classification
- RFCScope-style hierarchical, token-aware section partitioning
- version matching: combine **iddiff** hunks with embedding/DP alignment for moved or split paragraphs (SS6.4)
- revision feature extraction: edit frequency, semantic distance between versions,
  normative-strength changes (BCP 14 keyword shifts), clause stability, proximity to publication (SS6.2)

## Parse one document

Run from the repository root:

```bash
python -m alignment.segmentation \
  corpus/raw/drafts/draft-ietf-dnsop-svcb-https/draft-ietf-dnsop-svcb-https-12.xml \
  --output corpus/processed/sections/draft-ietf-dnsop-svcb-https-12.json
```

The input format is selected from `.txt` or `.xml`. If `--output` is omitted,
the canonical JSON document is written to stdout. A concise count of parsed
sections, content blocks, references, and warnings is written to stderr.

The parser removes only plaintext rendering noise such as repeated page headers
and footers. References, appendices, Security Considerations, and IANA
Considerations are retained and marked by `analysis_scope`; later stages decide
which scopes to analyze.

RFC plaintext is detected from an `RFC<number>.txt` filename or its title-page
header. A profile can also be selected explicitly:

```bash
python -m alignment.segmentation corpus/raw/rfcs/RFC9110.txt \
  --document-type rfc \
  --output corpus/processed/sections/RFC9110.json
```

The parser keeps legacy indented subsections and both numbered and unnumbered
reference groups. XML input is currently the XML2RFC source profile for
Internet-Drafts; RFC parsing uses the collected RFC Editor plaintext.

**RFC / IETF specifics:**
- Prefer Datatracker **plaintext** (`.txt`) for the pilot; note that XML (`*.xml`) preserves structure
  but adds parsing cost — defer unless plaintext alignment fails.
- Track **section number drift** (e.g. 3.2 → 4.1) separately from text similarity; iddiff highlights
  moves that naive diff misses.
- Cross-RFC references: unresolved or wrong RFC numbers are **U-4**; flag reference targets for LLM context
  the way RFCScope builds cross-document context.
- Distinguish **editorial** vs **normative** edits when computing “ambiguity clarification” features —
  a `MUST` → `SHOULD` change is not merely editorial.
