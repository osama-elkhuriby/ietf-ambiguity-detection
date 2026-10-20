# V1.1 pretraining failure attempt 001

- Frozen protocol SHA-256: `7f7efb4d97552524ecb5da0626667f90ccda5da3547c13b901a578621a2b8c15`
- New GPU cells launched: 0
- Failure stage: post-preparation CPU validation
- Failure: `validate_completed` passed a stored path string directly to the
  project JSON reader, which requires a `pathlib.Path`.
- Disposition: all generated preparation artifacts and their frozen source files
  were archived here before the compatibility infrastructure was corrected.
- Scientific impact: none. No dataset, assignment, label, prediction, model,
  threshold, or training artifact was changed.

This attempt is historical evidence only and must not be executed.
