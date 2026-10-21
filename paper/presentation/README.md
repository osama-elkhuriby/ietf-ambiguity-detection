# Project presentation

This folder contains an editable PowerPoint deck and its timed speaker guide.

## Files

- `ietf_ambiguity_detection_presentation.pptx` — original 18-slide deck; it
  remains unchanged because it was open in PowerPoint during regeneration.
- `ietf_ambiguity_detection_presentation_updated.pptx` — regenerated 20-slide
  deck with the added LLM and hybrid-routing results; the prior deck was open
  during generation and could not be overwritten.
- `gui_rfc9000_overview.png` and `gui_rfc9000_filtered_results.png` — captured
  local-GUI screenshots embedded on slide 17.
- `SPEAKER_NOTES.md` — slide-by-slide talk track and timing.
- `generate_presentation.py` — generator for the PowerPoint deck. Text, cards,
  diagrams, and bar charts are native PowerPoint objects and can be edited in
  PowerPoint or compatible presentation software.

## Regenerate

From the repository root, with Python and `python-pptx` installed:

```powershell
python paper/presentation/generate_presentation.py
```

The deck reflects the current report. It includes the latest 2,425-row classical
rerun (best AP 0.3584 with engineered features plus TF–IDF and Linear SVM), the
matched revision-feature ablation, the annotation-disagreement audit, the
updated 2,143-row reviewed-label DeBERTa study, and separate LLM and hybrid
routing results. The routing slide is an exploratory replay of stored OOF scores,
not observed inference-call savings. Classical, Qwen/hybrid, and
transformer results remain separate studies rather than a shared leaderboard. The
deck also records that TCP and DNS entered the latest classical rerun and are no
longer untouched project-level holdouts. The complete evidence and limitations are documented in
[`../main.tex`](../main.tex), [`../../README.md`](../../README.md), and
[`../../models/hybrid/README.md`](../../models/hybrid/README.md).

Slide 17 screenshots show the local GUI scanning the repository's RFC 9000
plaintext: 112 potentially ambiguous suggestions among 516 explicitly normative
clauses, with one filtered example. The scikit-learn version warning present during
capture is called out in the slide and speaker notes. Regeneration expects both
PNG screenshot inputs to remain beside `generate_presentation.py`.
