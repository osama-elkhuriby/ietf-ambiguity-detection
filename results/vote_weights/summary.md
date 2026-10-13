# Vote-weight distribution audit

## Summary

- Full dataset: 2546 rows, 455 positives.
- Weighted positive fraction, computed as positive-row weight sum divided by total weight sum: 14.2557%.
- On the 9-family evaluated cohort (2,540 rows, 453 positives; the two <30-row families are excluded only from evaluation), weighted positive fraction: 14.2298%.
- Class(es) for which a majority of rows have weight 0.5: 0, 1.

## Distribution by label

| Label | Rows | Weight 0.5 | Weight 1.0 | Share at 0.5 | Mean weight |
|---:|---:|---:|---:|---:|---:|
| 0 | 2091 | 1325 | 766 | 63.4% | 0.683 |
| 1 | 455 | 435 | 20 | 95.6% | 0.522 |

## Distribution by family and label

The complete count table is in `weight_distribution.csv`.

| Family | Label | Rows | Weight 0.5 | Weight 1.0 | Share at 0.5 |
|---|---:|---:|---:|---:|---:|
| draft-ietf-6man-rfc2460bis | 0 | 1 | 1 | 0 | 100.0% |
| draft-ietf-dnsop-svcb-https | 0 | 120 | 84 | 36 | 70.0% |
| draft-ietf-dnsop-svcb-https | 1 | 34 | 34 | 0 | 100.0% |
| draft-ietf-httpbis-cache-header | 0 | 3 | 1 | 2 | 33.3% |
| draft-ietf-httpbis-cache-header | 1 | 2 | 2 | 0 | 100.0% |
| draft-ietf-httpbis-http2bis | 0 | 227 | 129 | 98 | 56.8% |
| draft-ietf-httpbis-http2bis | 1 | 35 | 32 | 3 | 91.4% |
| draft-ietf-httpbis-semantics | 0 | 327 | 190 | 137 | 58.1% |
| draft-ietf-httpbis-semantics | 1 | 80 | 76 | 4 | 95.0% |
| draft-ietf-quic-http | 0 | 290 | 178 | 112 | 61.4% |
| draft-ietf-quic-http | 1 | 49 | 49 | 0 | 100.0% |
| draft-ietf-quic-recovery | 0 | 130 | 82 | 48 | 63.1% |
| draft-ietf-quic-recovery | 1 | 34 | 33 | 1 | 97.1% |
| draft-ietf-quic-tls | 0 | 179 | 118 | 61 | 65.9% |
| draft-ietf-quic-tls | 1 | 19 | 18 | 1 | 94.7% |
| draft-ietf-quic-transport | 0 | 653 | 428 | 225 | 65.5% |
| draft-ietf-quic-transport | 1 | 170 | 162 | 8 | 95.3% |
| draft-ietf-tcpm-rfc793bis | 0 | 113 | 79 | 34 | 69.9% |
| draft-ietf-tcpm-rfc793bis | 1 | 26 | 23 | 3 | 88.5% |
| draft-ietf-tls-tls13 | 0 | 48 | 35 | 13 | 72.9% |
| draft-ietf-tls-tls13 | 1 | 6 | 6 | 0 | 100.0% |

## Decision gate

STOP: at least one class is mostly weight 0.5. No model was run, as required.
The label-1 class is 95.6% weight 0.5 (435/455 rows); the label-0 class is 63.4% weight 0.5 (1325/2091 rows).

Weights derive from the same LLM votes as the labels; gains reflect fit to consensus labels, not verified ambiguity detection. Labels come from a candidate cohort selected using future information.

No stored weight was used as a feature, matrix column, preprocessing input, metric weight, or selection criterion. No repository artifacts outside `results/vote_weights/` were changed by this audit.
