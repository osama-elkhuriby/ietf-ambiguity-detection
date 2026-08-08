# Hybrid ML/transformer -> LLM routing

Per proposal SS6.1: a calibrated high-recall selector (classical ML or transformer) routes
likely-or-uncertain clauses to the LLM verifier; clear low-risk clauses are resolved without
an LLM call. This is the component the proposal's H2 and RQ4 are about — it only makes sense
once `models/classical` or `models/transformer` produce calibrated scores to route on.
