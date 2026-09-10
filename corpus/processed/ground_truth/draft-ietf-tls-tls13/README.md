# TLS 1.3 label lifecycle

This directory contains ground-truth preparation for only the
`draft-ietf-tls-tls13` family.

- `silver/` contains reproducible automatic evidence and candidates. Do not
  treat these records as ground truth.
- `review/` contains the pending human-annotation template. Copy it before
  editing because the builder overwrites the template.
- `gold/` is reserved for completed records that passed human review and
  adjudication.

Rebuild from the repository root with:

```bash
python -m corpus.build_tls_silver_labels
```

The promotion rule is `silver -> human review -> adjudication -> gold`. Never
promote records based only on an automatic errata or alignment match.
