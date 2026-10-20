# final_annotation_human_reviewed_100_v1_analysis_fix_v1

Analysis-only erratum for the immutable parent experiment
`final_annotation_human_reviewed_100_v1`. It reuses the parent's 40 validated
prediction artifacts and changes only the metadata identity merge that formerly
created `family_id_x` / `family_id_y`.

The repaired merge requires exact agreement on `item_id`, `record_id`,
`family_id`, `document_id`, and `revision`, then independently verifies labels
and frozen fold assignments. The parent ECDF, ensemble, threshold, metric,
bootstrap, ranking, and error-analysis implementations execute unchanged.

All generated files are written beneath `outputs/` in this directory. The
parent experiment, checkpoints, predictions, manifests, and existing analysis
files are read-only inputs. This erratum is not retraining and does not create
independent evaluation evidence.

The CLI intentionally exposes only `preflight`, `analyze`, and
`validate-analysis`; it has no training command.

