# Contextual sentence-embedding experiment

## Protocol and assumptions

- Context text is resolved by the exact `ContextResolver` and `prompt_payload` implementation in `corpus/annotate_with_ollama.py`; the target text and same-revision previous/next clauses are passed exactly as in the annotation prompt. A fixed-seed random sample of 20 rows matched the generated prompt fields in all three text positions.
- Target text is loaded using only `record_id`, `clause_text`, and `section_ids` from reviewer CSVs. The model receives target text, context text, and the existing engineered feature values only; labels are used only as outcomes.
- No votes, rationales, errata, final RFC text, candidate-source values, sample weights, or revision/future information are used. No per-LLM vote columns or vote artifacts were found in the processed CSV/JSON outputs, so unanimous/2-1 subset metrics are unavailable.
- MiniLM and BGE-small are frozen local encoders, used offline and without fine-tuning. Each target/previous/next text is encoded separately. Empty neighbor strings map to all-zero vectors rather than an embedding of a special empty-text token.
- BGE-small is used without retrieval/query prefixes because these are clause representations for classification rather than query-document retrieval.
- Embeddings are computed once from text only and cached; all learned feature transforms (TF-IDF vocabulary/IDF, imputation, and standardization) are fit separately inside each outer LOFO training fold. Inner-C selection uses GroupKFold splits of outer-training families only.
- Families with fewer than 30 rows (6man: 1; cache-header: 5) remain in every training fold and are never held out. This yields 9 scored families, 2,540 scored rows, and 453 positives.
- Results use LogisticRegression(class_weight='balanced', max_iter=2000). Threshold metrics use the default 0.5 only; all other comparisons use threshold-free AP. Bootstrap intervals resample evaluated rows 1,000 times, and AP-difference intervals use paired row resamples against E0.
- The nested selector first chooses an embedding configuration using inner grouped OOF AP at C=1, then chooses its C from {0.01, 0.1, 1, 10} on those same inner grouped folds. Outer held-out families are not used in either selection step; the nested selector is the leakage-safe tuned result. The fixed-configuration E1-E5 comparisons use C=1.

## Prev/next context coverage

| split | family_id | n_rows | previous_n | previous_pct | next_n | next_pct |
| --- | --- | --- | --- | --- | --- | --- |
| test | draft-ietf-dnsop-svcb-https | 154.000 | 136.000 | 88.312 | 132.000 | 85.714 |
| train | draft-ietf-6man-rfc2460bis | 1.000 | 1.000 | 100.000 | 1.000 | 100.000 |
| train | draft-ietf-httpbis-cache-header | 5.000 | 5.000 | 100.000 | 5.000 | 100.000 |
| train | draft-ietf-httpbis-http2bis | 262.000 | 258.000 | 98.473 | 236.000 | 90.076 |
| train | draft-ietf-httpbis-semantics | 407.000 | 397.000 | 97.543 | 367.000 | 90.172 |
| train | draft-ietf-quic-http | 339.000 | 336.000 | 99.115 | 275.000 | 81.121 |
| train | draft-ietf-quic-recovery | 164.000 | 155.000 | 94.512 | 145.000 | 88.415 |
| train | draft-ietf-quic-tls | 198.000 | 185.000 | 93.434 | 160.000 | 80.808 |
| train | draft-ietf-quic-transport | 823.000 | 784.000 | 95.261 | 736.000 | 89.429 |
| train | draft-ietf-tls-tls13 | 54.000 | 53.000 | 98.148 | 52.000 | 96.296 |
| validation | draft-ietf-tcpm-rfc793bis | 139.000 | 134.000 | 96.403 | 120.000 | 86.331 |

## Prompt-text spot check

All 20 random sample rows matched the annotation payload’s target, previous-clause, and next-clause strings exactly. Three sample excerpts (clipped for readability):

1. `22a42ec080a706d658d9e82b7ffde0fdd8773df84fd9da3420ed400039f39c08`
   - Target: If a PING frame is received with a stream identifier field value other than 0x0, the recipient MUST respond with a connection error of type PROTOCOL_ERROR.
   - Previous: PING frames are not associated with any individual stream.
   - Next: Receipt of a PING frame with a length field value other than 8 MUST be treated as a connection error of type FRAME_SIZE_ERROR.
2. `9755e69a294308252631210f2df476a7eee1953752e0194fb1f8084d98b405f3`
   - Target: When this is true, a server MAY request that the client abort transmission of a request without error by sending a RST_STREAM with an error code of NO_ERROR after sending a comple…
   - Previous: A server can send a complete response prior to the client sending an entire request if the response does not depend on any portion of the request that has not been sent and receiv…
   - Next: Clients MUST NOT discard responses as a result of receiving such a RST_STREAM, though clients can always discard responses at their discretion for other reasons.
3. `0a43e8897cb8d331da617cd3b947c10872611f9079787bb9ae4934893665d31e`
   - Target: A client SHOULD NOT generate a payload in a DELETE request.
   - Previous: o a 200 (OK) status code if the action has been enacted and the response message includes a representation describing the status.
   - Next: A payload received in a DELETE request has no defined semantics, cannot alter the meaning or target of the request, and might lead some implementations to reject the request.

## Encoders

- `MiniLM`: `sentence-transformers/all-MiniLM-L6-v2` (cached revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions).
- `BGE-small`: `BAAI/bge-small-en-v1.5` (cached revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions).

## LOFO comparison

E0 is the existing combined text+engineered baseline reconstructed from fold-local target TF-IDF and fold-local transforms of the raw engineered features. The notebook reference is pooled AP 0.291 on 9 scored families; 0.293 is the 11-family result before applying the <30-row held-out rule.

| model | pooled_oof_ap | pooled_ap_ci95_low | pooled_ap_ci95_high | mean_family_ap | sd_family_ap | precision_at_0.5 | recall_at_0.5 | f1_at_0.5 | f2_at_0.5 | recall_at_top20_reviewed | confusion_tn_fp_fn_tp | ap_difference_vs_e0 | paired_ap_difference_ci95_low | paired_ap_difference_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_combined_reference | 0.291 | 0.260 | 0.335 | 0.294 | 0.093 | 0.299 | 0.468 | 0.365 | 0.420 | 0.351 | [1590, 497, 241, 212] | 0.000 | 0.000 | 0.000 |
| MiniLM_E1_target_only | 0.219 | 0.195 | 0.250 | 0.222 | 0.053 | 0.227 | 0.457 | 0.304 | 0.380 | 0.258 | [1384, 703, 246, 207] | -0.072 | -0.108 | -0.038 |
| MiniLM_E2_target_plus_engineered | 0.246 | 0.219 | 0.280 | 0.257 | 0.069 | 0.245 | 0.479 | 0.324 | 0.402 | 0.300 | [1417, 670, 236, 217] | -0.045 | -0.074 | -0.016 |
| MiniLM_E3_target_prev_next | 0.204 | 0.183 | 0.230 | 0.214 | 0.055 | 0.215 | 0.340 | 0.263 | 0.304 | 0.241 | [1524, 563, 299, 154] | -0.087 | -0.123 | -0.055 |
| MiniLM_E4_context_plus_engineered | 0.212 | 0.190 | 0.240 | 0.223 | 0.055 | 0.231 | 0.349 | 0.278 | 0.317 | 0.252 | [1562, 525, 295, 158] | -0.079 | -0.111 | -0.048 |
| MiniLM_E5_full_stack | 0.212 | 0.190 | 0.240 | 0.225 | 0.056 | 0.230 | 0.344 | 0.276 | 0.313 | 0.258 | [1565, 522, 297, 156] | -0.079 | -0.113 | -0.050 |
| BGE-small_E1_target_only | 0.224 | 0.199 | 0.256 | 0.237 | 0.044 | 0.217 | 0.422 | 0.287 | 0.355 | 0.269 | [1398, 689, 262, 191] | -0.067 | -0.103 | -0.033 |
| BGE-small_E2_target_plus_engineered | 0.245 | 0.218 | 0.281 | 0.258 | 0.067 | 0.243 | 0.470 | 0.321 | 0.396 | 0.296 | [1424, 663, 240, 213] | -0.046 | -0.082 | -0.015 |
| BGE-small_E3_target_prev_next | 0.204 | 0.183 | 0.229 | 0.211 | 0.045 | 0.215 | 0.311 | 0.255 | 0.286 | 0.267 | [1573, 514, 312, 141] | -0.087 | -0.123 | -0.056 |
| BGE-small_E4_context_plus_engineered | 0.210 | 0.186 | 0.238 | 0.218 | 0.062 | 0.221 | 0.307 | 0.257 | 0.285 | 0.258 | [1597, 490, 314, 139] | -0.081 | -0.115 | -0.049 |
| BGE-small_E5_full_stack | 0.211 | 0.189 | 0.239 | 0.218 | 0.061 | 0.220 | 0.302 | 0.255 | 0.281 | 0.258 | [1601, 486, 316, 137] | -0.080 | -0.114 | -0.051 |
| Nested_inner_CV_selected_embedding | 0.232 | 0.207 | 0.265 | 0.248 | 0.063 | 0.244 | 0.435 | 0.313 | 0.376 | 0.298 | [1477, 610, 256, 197] | -0.058 | -0.089 | -0.030 |

Per-family AP for each model is in `per_family_ap.csv`; the complete metrics table is in `results.csv`.

### Inner-C nested selection

| heldout_family | selected_configuration | inner_ap_at_c1 | selected_c |
| --- | --- | --- | --- |
| draft-ietf-dnsop-svcb-https | MiniLM_E2_target_plus_engineered | 0.261 | 0.010 |
| draft-ietf-httpbis-http2bis | MiniLM_E2_target_plus_engineered | 0.252 | 0.010 |
| draft-ietf-httpbis-semantics | BGE-small_E2_target_plus_engineered | 0.226 | 0.010 |
| draft-ietf-quic-http | BGE-small_E2_target_plus_engineered | 0.242 | 0.010 |
| draft-ietf-quic-recovery | BGE-small_E2_target_plus_engineered | 0.239 | 0.010 |
| draft-ietf-quic-tls | MiniLM_E2_target_plus_engineered | 0.266 | 0.010 |
| draft-ietf-quic-transport | BGE-small_E5_full_stack | 0.222 | 0.100 |
| draft-ietf-tcpm-rfc793bis | BGE-small_E2_target_plus_engineered | 0.230 | 0.010 |
| draft-ietf-tls-tls13 | MiniLM_E2_target_plus_engineered | 0.243 | 0.010 |

The selected configuration and C may vary by held-out family. This is a nested model-selection result, not a single globally fixed encoder/configuration.

Labels are LLM-consensus labels on a candidate cohort selected using future information; AP and prevalence do not transfer to the full clause population.

