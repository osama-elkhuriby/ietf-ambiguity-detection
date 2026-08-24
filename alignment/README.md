# Stage 2 — Alignment

Document segmentation, version-to-version clause matching, and revision-aware feature
computation (proposal SS6.2).

Before writing custom diff logic here, check `iddiff` (one of IETF's own author tools,
at author-tools.ietf.org) — it's built specifically for diffing two Internet-Draft
revisions and may cover a meaningful chunk of the "textual and structural changes"
computation this stage needs.

Implemented foundation:

- deterministic Internet-Draft TXT/XML and RFC TXT parsing in `segmentation/`
- a shared document → section → block tree with deterministic per-source block IDs and locations
- profile-aware handling of legacy indented RFC headings and unnumbered RFC back matter
- RFC number, title, publication date, and category extraction
- XML2RFC `<xref>`/`<relref>`/`<eref>`, internal-section, and rendered-text reference extraction
- XML2RFC v2 external bibliography-entity stubs when network access is disabled
- preservation of paragraphs, nested list items, definitions, figure order, artwork,
  source code, table cells, and reference definitions
- deterministic block → sentence/clause refinement in `clause_segmentation/`
- stable clause IDs, source provenance, revision IDs, and block-relative character spans
- evidence-first `NORMATIVE`, `CONTEXTUAL`, `EXCLUDED`, and `MANUAL_REVIEW` classification
- uppercase RFC 2119/RFC 8174 detection, quoted/reference safeguards, section priors, and review reasons

Planned next contents:

- RFCScope-style hierarchical, token-aware section partitioning
- version matching: combine **iddiff** hunks with embedding/DP alignment for moved or split paragraphs (SS6.4)
- revision feature extraction: edit frequency, semantic distance between versions,
  normative-strength changes (BCP 14 keyword shifts), clause stability, proximity to publication (SS6.2)
- independent human annotation and calibration of the clause-classification heuristics

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
Considerations are retained and marked by `analysis_scope` (`front`, `body`,
`references`, `appendix`, or `back`); later stages decide
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

## Segment and classify clauses

Run the clause layer directly on any supported source:

```bash
python -m alignment.clause_segmentation corpus/raw/rfcs/RFC9110.txt \
  --output corpus/processed/clauses/RFC9110.json
```

This command first runs the structural parser, then refines prose paragraphs into
conservative sentence-level clauses. List items remain independently addressable
units, and conditions remain attached to the sentence containing their requirement.
Common RFC punctuation such as version numbers, section numbers, abbreviations, and
URLs is protected from accidental sentence splits.

Every output clause records its parent block/section, stable clause ID, source
location, block-relative character span, revision ID, references, detected BCP 14
terms, weak section prior, classification reason, and evidence. Formal blocks and
reference material are retained but labeled `EXCLUDED`; ambiguous lowercase modal
language and implicit requirements are routed to `MANUAL_REVIEW` rather than forced
into a high-confidence label.

The classifier implements the approved deterministic design and has automated
regression/corpus coverage. Agreement and accuracy against independently annotated
clauses remain an evaluation activity; the heuristic labels must not be treated as
human-validated ground truth until that calibration is complete.

**RFC / IETF specifics:**
- Prefer Datatracker **plaintext** (`.txt`) for the pilot; note that XML (`*.xml`) preserves structure
  but adds parsing cost — defer unless plaintext alignment fails.
- Track **section number drift** (e.g. 3.2 → 4.1) separately from text similarity; iddiff highlights
  moves that naive diff misses.
- Cross-RFC references: unresolved or wrong RFC numbers are **U-4**; flag reference targets for LLM context
  the way RFCScope builds cross-document context.
- Distinguish **editorial** vs **normative** edits when computing “ambiguity clarification” features —
  a `MUST` → `SHOULD` change is not merely editorial.
