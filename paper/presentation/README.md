# Project presentation

This folder contains an editable PowerPoint deck and its timed speaker guide.

## Files

- `ietf_ambiguity_detection_presentation.pptx` — 18-slide, 16:9 academic talk,
  designed for 25 minutes before Q&A.
- `SPEAKER_NOTES.md` — slide-by-slide talk track and timing.
- `generate_presentation.py` — generator for the PowerPoint deck. Text, cards,
  diagrams, and bar charts are native PowerPoint objects and can be edited in
  PowerPoint or compatible presentation software.

## Regenerate

From the repository root, with Python and `python-pptx` installed:

```powershell
python paper/presentation/generate_presentation.py
```

The deck keeps the three reported model-study cohorts separate. In particular, the
classical, Qwen/hybrid, and latest DeBERTa results are not presented as a shared
leaderboard. The complete source evidence and limitations are documented in
[`../main.tex`](../main.tex), [`../../README.md`](../../README.md), and
[`../../models/hybrid/README.md`](../../models/hybrid/README.md).
