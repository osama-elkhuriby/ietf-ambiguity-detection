# Project presentation

This folder contains an editable PowerPoint deck and its timed speaker guide.

## Files

- `ietf_ambiguity_detection_presentation.pptx` — 18-slide, 16:9 academic talk,
  designed for 25 minutes before Q&A.
- `gui_rfc9000_overview.png` and `gui_rfc9000_filtered_results.png` — captured
  local-GUI screenshots embedded on slide 15.
- `SPEAKER_NOTES.md` — slide-by-slide talk track and timing.
- `generate_presentation.py` — generator for the PowerPoint deck. Text, cards,
  diagrams, and bar charts are native PowerPoint objects and can be edited in
  PowerPoint or compatible presentation software.

## Regenerate

From the repository root, with Python and `python-pptx` installed:

```powershell
python paper/presentation/generate_presentation.py
```

The deck's updated reviewed-label DeBERTa slide reports the three methods evaluated
on the same 2,143-row, nine-family cohort. Classical and Qwen/hybrid results are
shown as separate studies rather than as a shared leaderboard; the 100 reviewed
labels were selected from historical OOF errors, and the analysis-repair run reuses
parent predictions. The complete source evidence and limitations are documented in
[`../main.tex`](../main.tex), [`../../README.md`](../../README.md), and
[`../../models/hybrid/README.md`](../../models/hybrid/README.md).

Slide 15 screenshots show the local GUI scanning the repository's RFC 9000
plaintext: 112 potentially ambiguous suggestions among 516 explicitly normative
clauses, with one filtered example. The scikit-learn version warning present during
capture is called out in the slide and speaker notes. Regeneration expects both
PNG screenshot inputs to remain beside `generate_presentation.py`.
