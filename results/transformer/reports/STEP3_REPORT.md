# Step 3 weighting ablation report

Status: complete. The experiment used target-only DeBERTa-v3-small at LR 1e-5,
fp32, frozen folds and inner-family assignments, and seeds 1729, 2718, and 31415.
Ten existing unweighted cells were reused and exactly 50 missing cells were trained.
The sealed DNS artifact was not opened. Results measure agreement with LLM-consensus
silver labels, not human-verified ambiguity accuracy.

## Frozen selection result

Selection uses the seed-averaged PR-AUC within each fold and then averages the five
folds. Class weighting is the DeBERTa winner. Its advantage over the runner-up,
unweighted, is 0.0067; the paired-fold SE is 0.0055. Because the difference is not
less than one SE, the fallback to the simpler condition does not trigger.

| Weighting | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean-fold PR-AUC | Three-seed pooled ensemble PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.3361 | 0.2921 | 0.2124 | 0.3363 | 0.3008 | 0.2955 | 0.2663 |
| Class | 0.3462 | 0.3077 | 0.2048 | 0.3315 | 0.3208 | **0.3022** | 0.2874 |
| Confidence | 0.3200 | 0.3102 | 0.2058 | 0.3143 | 0.3034 | 0.2907 | 0.2708 |
| Combined | 0.3583 | 0.3053 | 0.1986 | 0.2954 | 0.3170 | 0.2949 | 0.3016 |

The selection statistic and pooled ensemble PR-AUC answer different questions. The
former is the pre-registered equal-fold selection statistic; the latter pools rows and
therefore gives larger families more influence. Combined has the largest pooled
ensemble PR-AUC but is not selected after observing that secondary aggregation.

## Required DeBERTa contrasts

Intervals below compare three-seed logit-average OOF scores with a within-fold,
section-cluster paired bootstrap (2,000 replicates). Mean-fold differences instead use
the pre-registered seed-averaged fold PR-AUC values.

| Contrast | Mean-fold difference | Paired-fold SE | Pooled difference | Section-cluster 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Unweighted - confidence | 0.0048 | 0.0071 | -0.0045 | [-0.0264, 0.0174] |
| Class - combined | 0.0073 | 0.0079 | -0.0143 | [-0.0342, 0.0045] |
| Class - unweighted | 0.0067 | 0.0055 | 0.0211 | [-0.0014, 0.0446] |

All three section-cluster intervals include zero. The frozen rule still selects class
weighting because it is defined on the seed-averaged five-fold statistic, not on the
secondary pooled bootstrap.

## TF-IDF comparison

For TF-IDF, confidence has the highest mean-fold PR-AUC (0.2879), followed by combined
(0.2878). Their difference is 0.0001 with paired-fold SE 0.0050, so the frozen rule
selects the simpler of those two: confidence. TF-IDF is deterministic and therefore
has one value per fold.

| Weighting | TF-IDF mean-fold PR-AUC | TF-IDF pooled PR-AUC | DeBERTa - TF-IDF mean-fold | DeBERTa - TF-IDF pooled | Section-cluster 95% interval on pooled difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.2823 | 0.2712 | 0.0322 | -0.0049 | [-0.0332, 0.0243] |
| Class | 0.2826 | 0.2785 | 0.0295 | 0.0089 | [-0.0296, 0.0451] |
| Confidence | 0.2879 | 0.2741 | 0.0136 | -0.0033 | [-0.0348, 0.0256] |
| Combined | 0.2878 | 0.2852 | 0.0336 | 0.0164 | [-0.0198, 0.0543] |

No architecture comparison excludes zero at the section-cluster level. H1 therefore
remains unsupported. This comparison also retains the documented asymmetry: TF-IDF
fits all outer-training families, whereas DeBERTa reserves one family for inner
checkpoint selection.

The three requested TF-IDF weighting contrasts are in `contrasts.json`. In brief,
their section-cluster pooled intervals are [-0.0102, 0.0025] for unweighted minus
confidence, [-0.0125, -0.0013] for class minus combined, and [0.0004, 0.0155] for class
minus unweighted. These pooled results do not override the frozen mean-fold selection.

## Routing and secondary probability metrics

At the minimum pooled single-score threshold reaching at least 90% ambiguous recall:

| Model / weighting | Referral rate | 95% row-bootstrap interval | Avoided LLM calls | Brier | ECE (10 bins) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeBERTa unweighted | 0.8429 | [0.8091, 0.8658] | 15.7% | 0.1555 | 0.1167 |
| DeBERTa class | 0.8582 | [0.8104, 0.8829] | 14.2% | 0.1698 | 0.1386 |
| DeBERTa confidence | 0.8402 | [0.8095, 0.8708] | 16.0% | 0.1644 | 0.1374 |
| DeBERTa combined | 0.8316 | [0.7681, 0.8838] | 16.8% | 0.1687 | 0.1353 |
| TF-IDF unweighted | 0.7546 | [0.7150, 0.7803] | 24.5% | 0.1396 | 0.0185 |
| TF-IDF class | 0.7510 | [0.6952, 0.7960] | 24.9% | 0.1920 | 0.2301 |
| TF-IDF confidence | 0.7335 | [0.6916, 0.7632] | 26.7% | 0.1412 | 0.0452 |
| TF-IDF combined | 0.7470 | [0.6880, 0.7942] | 25.3% | 0.1737 | 0.1884 |

The selected DeBERTa condition leaves only about 14% estimated call savings at 90%
recall, versus about 27% for selected TF-IDF. These are uncalibrated OOF development
estimates. Brier and ECE are secondary diagnostics and must not be interpreted as the
final calibration result.

## Runtime and artifacts

The 50 new transformer cells consumed 3,254.6 aggregate cell-seconds (54.2 minutes).
Including the ten reused cells, the 60-cell comparison represents 3,977.8 seconds.
Per-condition total cell runtime was 1,047.1 seconds unweighted (323.9 new), 892.0
class, 1,133.7 confidence, and 905.0 combined. Maximum allocated GPU memory was about
2,952 MB. Statistical analysis took 81.5 CPU seconds. Historical TF-IDF training time
was not recorded by Step 1, so no retrospective runtime is invented.

Every cell retains raw `logit_0` and `logit_1`. The analysis saves validated per-cell
metrics, raw-logit three-seed OOF ensembles, summaries, selection details, contrasts,
artifact hashes, and an explicit `dns_opened=false` manifest under
`results/weighting-ablation-v1/analysis-final/`.
