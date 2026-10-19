# IEEE paper source

`main.tex` is a full IEEE conference-format manuscript grounded in the repository's
current artifacts as reviewed on 2026-10-19. `references.bib` contains the bibliography,
and `main.pdf` is the successfully compiled seven-page preview.

## Build

Use a TeX distribution containing `IEEEtran`, then run:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or:

```text
latexmk -pdf main.tex
```

Run these commands from this directory so BibTeX finds `references.bib`.

## Author action required before submission

1. Replace the red `[corresponding email]` marker in `main.tex` with the corresponding
   author's institutional email.
2. Select the actual IEEE venue and apply its page limit, copyright notice, anonymity,
   and artifact-availability rules. Use the anonymous author block if double-blind review
   is required.
3. Confirm author order, spelling, affiliations, acknowledgments, and consent.
4. Do not rewrite the development results as final-test performance. TCP validation and
   the DNS sealed test were unopened at the time of this manuscript.
5. If those evaluations are later authorized and completed, update the abstract, Results,
   Threats to Validity, Conclusion, artifact commit, and all associated tables together.
6. Obtain an independently adjudicated expert sample before making accuracy or deployment
   claims. The current labels are an operational human/AI-assisted reference.

The paper deliberately calls the repository's `average_precision_score` output
"average precision (AP)" rather than using the less precise historical label "PR-AUC."
