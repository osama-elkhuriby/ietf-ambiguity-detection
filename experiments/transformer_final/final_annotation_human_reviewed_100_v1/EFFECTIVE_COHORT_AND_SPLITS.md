# Effective cohort and family splits

## Source and eligibility

- Authoritative source: `results/annotation/final_annotation_ready.csv`
- Source SHA-256: `8ffa654c2c405dd3eef72901c600f1ef3823e6dbff663020bb824c57ff65bdbf`
- Source rows: 2,425 (160 Yes, 2,265 No)
- Unique `item_id`: 2,425
- Invalid/missing labels: 0
- Duplicate IDs, target clauses, full training examples: 0 / 0 / 0
- Empty target clauses: 0
- Unresolved exclusions: none
- Reserved rows: TCP 131; DNS 151
- Effective development cohort: 2,143 (138 Yes, 2,005 No), nine families
- Effective dataset SHA-256: `4e20a2571bfe71f38d04e5677e067ce4d2eda2008f00c297349b929bb662b4af`

`item_id=2161` is currently `resolved`, has `final_source=manual_conflict_verdict`,
label No, and belongs to `draft-ietf-quic-transport` (outer fold 1). It is included;
no label was changed by preparation.

## Effective family counts

| Family | Rows | Yes | No |
|---|---:|---:|---:|
| draft-ietf-6man-rfc2460bis | 1 | 0 | 1 |
| draft-ietf-httpbis-cache-header | 5 | 2 | 3 |
| draft-ietf-httpbis-http2bis | 239 | 11 | 228 |
| draft-ietf-httpbis-semantics | 385 | 34 | 351 |
| draft-ietf-quic-http | 314 | 12 | 302 |
| draft-ietf-quic-recovery | 162 | 11 | 151 |
| draft-ietf-quic-tls | 189 | 8 | 181 |
| draft-ietf-quic-transport | 794 | 57 | 737 |
| draft-ietf-tls-tls13 | 54 | 3 | 51 |

## Outer splits

| Fold | Held-out families | Fit rows (Yes/No) | Score rows (Yes/No) | Fit exclusions |
|---:|---|---:|---:|---:|
| 1 | quic-transport | 1,348 (80/1,268) | 794 (57/737) | 1 |
| 2 | httpbis-semantics | 1,758 (104/1,654) | 385 (34/351) | 0 |
| 3 | quic-http | 1,826 (126/1,700) | 314 (12/302) | 3 |
| 4 | 6man-rfc2460bis, httpbis-cache-header, httpbis-http2bis, tls-tls13 | 1,841 (122/1,719) | 299 (16/283) | 3 |
| 5 | quic-recovery, quic-tls | 1,791 (118/1,673) | 351 (19/332) | 1 |

Every outer record is scored once. Every inner root is exactly its outer
manifest's `fitting_record_ids`: 1,348 / 1,758 / 1,826 / 1,841 / 1,791 rows.
The 15 detailed inner partitions, class counts, held-out families, exclusions,
and context sanitizations are in `FOLD_CLASS_COUNTS.csv`.

