# Dataset audit

- Authority: `D:\RFC\ietf-ambiguity-detection\results\annotation\final_annotation_ready.csv`
- SHA-256: `8ffa654c2c405dd3eef72901c600f1ef3823e6dbff663020bb824c57ff65bdbf`
- Source: 2425 rows; 160 Yes; 2265 No; 6.597938% positive.
- Unknown/missing labels: 0.
- Duplicate item IDs: 0.
- Duplicate target-clause rows: 0.
- Empty target clauses: 0.
- Document identifiers: 178.
- Authorized development cohort: 2143 rows; 138 Yes; 2005 No; 6.439571% positive.
- Unresolved exclusions: none.
- Reserved counts (labels not accessed for feasibility): TCP 131; DNS 151.
- Duplicate training-example rows (all text fields identical): 0.
- Authoritative target-text corrections relative to legacy metadata: 1.

The authoritative CSV supplies every human-readable text and final label. Frozen
governance sources supply stable record, family, document-lineage, and fold IDs.
TCP remains reserved and DNS remains sealed; neither partition contributes labels
to development feasibility, fitting, calibration, or analysis.
