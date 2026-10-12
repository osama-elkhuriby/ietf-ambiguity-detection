# Revision-aware signal check

## Assumptions and protocol

- Only numbered draft inputs at or before each row revision are loaded; no RFC endpoint, errata, labels, votes, rationales, candidate-source values, or saved alignment/lineage outputs enter matching.
- The deterministic predecessor matcher uses same section ID when available, otherwise matching section headings; it ranks candidates by the mean of the repository-style lexical score and pinned MiniLM cosine similarity, with a 0.62 minimum score and stable position/ID tie-breaks.
- H0 is the pair `(t-1 -> t)`. H1 uses `(t-1 -> t)` only to locate the same clause at t-1, then measures the historical pair `(t-2 -> t-1)`; this identity lookup rule was explicitly approved.
- Candidate source is joined only after matching by `record_id`; it is diagnostic-only except in the isolated N1 diagnostic. It is never included in R1/R2.
- LOFO uses the fixed v3 family assignments. Families with fewer than 30 rows remain in every training fold but are never held out for scoring.
- Imputation, scaling, and one-hot encoders are fitted within each training fold. Logistic regression uses `class_weight='balanced'`, `max_iter=2000`, with threshold 0.5.
- Bootstrap intervals resample evaluated rows with replacement; one-class replicates are redrawn until 1,000 valid AP replicates are obtained.

## TLS matcher validation

Validation against existing row-level TLS alignment relationships could not be completed: the retained TLS run folders contain pair summary reports, but no `alignments.jsonl`, `lineage.jsonl`, or candidate link records. Aggregate relation counts do not identify per-clause predecessors. No alignment output was used by the matcher. The current matcher uses one-to-one predecessor links; split/merge recognition is outside this quick signal check.
TLS matcher runtime: 1.7 seconds; all-family matching runtime: 19.7 seconds (sum of per-family matcher wall-clock times).

## Coverage gate

Coverage summary CSV: `coverage_gate.csv` (76 family/label/source/tier rows).

Rows flagged for a >10 percentage-point difference in resolvable-predecessor rate between label 1 and label 0:

- H1, family `draft-ietf-quic-http`: -13.7 pp
- H1, family `draft-ietf-tcpm-rfc793bis`: -13.6 pp
- H0, family `draft-ietf-quic-http`: -13.5 pp
- H0, family `draft-ietf-tcpm-rfc793bis`: -15.4 pp
- No candidate-source-level differences exceed 10 pp.

## Quick LOFO signal check

The existing combined-matrix reference is pooled AP 0.293 for 11 families; the 9-held-out-family references are 0.292/0.291. The dataset prevalence reference is 0.179.

```csv
model,n_evaluated_rows,heldout_families,pooled_oof_ap,pooled_ap_ci95_low,pooled_ap_ci95_high,mean_family_ap,sd_family_ap,per_family_ap_json,recall_at_top20_reviewed,prevalence,threshold,note
N1_selection_artifact_diagnostic,2540,draft-ietf-dnsop-svcb-https|draft-ietf-httpbis-http2bis|draft-ietf-httpbis-semantics|draft-ietf-quic-http|draft-ietf-quic-recovery|draft-ietf-quic-tls|draft-ietf-quic-transport|draft-ietf-tcpm-rfc793bis|draft-ietf-tls-tls13,0.17913515380827608,0.15995764087457545,0.1987090979751796,0.18194868793618446,0.05341081564782159,"{""draft-ietf-dnsop-svcb-https"": 0.2744355317884729, ""draft-ietf-httpbis-http2bis"": 0.15929999480708312, ""draft-ietf-httpbis-semantics"": 0.2276509604095811, ""draft-ietf-quic-http"": 0.14162046999690833, ""draft-ietf-quic-recovery"": 0.21645459463273514, ""draft-ietf-quic-tls"": 0.11190855927698032, ""draft-ietf-quic-transport"": 0.2020314839826471, ""draft-ietf-tcpm-rfc793bis"": 0.1816876169394155, ""draft-ietf-tls-tls13"": 0.12244897959183673}",0.18543046357615894,0.17834645669291338,0.5,"N1 is a selection-artifact diagnostic, not a candidate model"
R1_H1_predecessor_only,2540,draft-ietf-dnsop-svcb-https|draft-ietf-httpbis-http2bis|draft-ietf-httpbis-semantics|draft-ietf-quic-http|draft-ietf-quic-recovery|draft-ietf-quic-tls|draft-ietf-quic-transport|draft-ietf-tcpm-rfc793bis|draft-ietf-tls-tls13,0.16815211577326153,0.15190857468999674,0.1897156673207369,0.1790183128521822,0.05220097812544942,"{""draft-ietf-dnsop-svcb-https"": 0.2121139401245015, ""draft-ietf-httpbis-http2bis"": 0.13566450187313583, ""draft-ietf-httpbis-semantics"": 0.1980646473374866, ""draft-ietf-quic-http"": 0.17328533358068798, ""draft-ietf-quic-recovery"": 0.22394818988459464, ""draft-ietf-quic-tls"": 0.10625252389285234, ""draft-ietf-quic-transport"": 0.19792800150498227, ""draft-ietf-tcpm-rfc793bis"": 0.25523640390650815, ""draft-ietf-tls-tls13"": 0.10867127356489059}",0.20309050772626933,0.17834645669291338,0.5,candidate_source excluded
R2_H0_predecessor_only,2540,draft-ietf-dnsop-svcb-https|draft-ietf-httpbis-http2bis|draft-ietf-httpbis-semantics|draft-ietf-quic-http|draft-ietf-quic-recovery|draft-ietf-quic-tls|draft-ietf-quic-transport|draft-ietf-tcpm-rfc793bis|draft-ietf-tls-tls13,0.16491735102618277,0.14857705218205058,0.1841493443982607,0.1828508830175518,0.0595806035370602,"{""draft-ietf-dnsop-svcb-https"": 0.2063342475632333, ""draft-ietf-httpbis-http2bis"": 0.13669472129937246, ""draft-ietf-httpbis-semantics"": 0.19966860968096634, ""draft-ietf-quic-http"": 0.18565986078234745, ""draft-ietf-quic-recovery"": 0.22537021801727683, ""draft-ietf-quic-tls"": 0.09594433826333373, ""draft-ietf-quic-transport"": 0.1987461538912144, ""draft-ietf-tcpm-rfc793bis"": 0.28612868654911067, ""draft-ietf-tls-tls13"": 0.1111111111111111}",0.18984547461368653,0.17834645669291338,0.5,candidate_source excluded
```

**N1_selection_artifact_diagnostic** is a diagnostic, not a candidate model. Candidate source is used only by this diagnostic and is not combined with any revision feature.

Rows are an enriched candidate cohort selected using future information; AP and prevalence do not transfer to the full clause population. Revision features measure change/churn signal and may not reflect ambiguity.

Discrepancy to resolve: this dataset contains only 5 errata rows, all label 0, all in TLS. The progress report stated 121 errata positives and 242 TLS records. These counts do not agree; the source/dataset definition needs reconciliation. As requested, the 5 errata rows form an unseen category in the fold where TLS is held out.
