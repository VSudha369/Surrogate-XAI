# Stage 3.5M validation checklist

- [x] Source compilation and AST parse
- [x] Frozen Stage 3M hash, teacher, seed, objective, and zero-counter gates
- [x] Teacher parameter/state immutability and no optimizer/backward path
- [x] No unknown training class
- [x] MSP and energy direction conventions
- [x] Prototype and Mahalanobis synthetic correctness
- [x] Covariance and diagonal-density regularization
- [x] Known-validation-only threshold fitting
- [x] ZD-STRICT and ZD-CALIBRATED separation
- [x] Strict partition rejected before Stage 08
- [x] Strict labels/metrics cannot influence fitting or thresholds
- [x] Evaluation lock required and fitting rejected afterward
- [x] All-predeclared scorer policy frozen before strict access
- [x] Deterministic scorer, threshold, OSCR, and bootstrap primitives
- [x] Resume invalidates on executable/configuration/input/output hash changes
- [x] Incomplete score stores are never reused
- [x] Score-store manifests bind executable/configuration/scorer semantics and exact global indices
- [x] P0-P3 pre-strict bundle recursively detects mutation of every stored component
- [x] Evaluation lock detects fit, threshold, policy, bundle, equivalence, counter-key, and permission-flag mutations
- [x] Strict stores bind the current lock and exact sealed index sequence
- [x] Stage 3M→Stage 3.5M P0-P3 inference-equivalence mutation matrix
- [x] Stages 09-11 require current known and strict score bundles
- [x] Strict shift is enforced as an exact unique subset of strict main before fresh scoring
- [x] Strict shift rows outside main, partial overlap, duplicates, and non-strict overlap abort
- [x] No concatenated/double-counted `strict_combined` population exists
- [x] Post-lock recovery preflight is read-only and requires the exact reviewed NOT_READY failure
- [x] Recovery rejects modified lock, Stage 03-07 evidence, fit, thresholds, known bundle, strict stores, or store-lock binding
- [x] Recovery rejects any strict-label file and contains no strict-signal/teacher/fitting path
- [x] Persisted overlapping predictions and score vectors pass exact/tolerance integrity checks
- [x] READY schema discloses post-lock recovery and all no-recomputation flags
- [x] Stage-11 transaction writes READY before its checkpoint and validates final hashes
- [x] Logging formatting regression
- [x] Frozen Stage 3M and Stage 2.6M validators remain passing
- [ ] Colab preflight against canonical Drive artifacts
- [ ] Human/GitHub leakage and locking review
- [ ] Full canonical Colab execution and independent Drive review

Local validation does not claim `MANYTX_STAGE3_5M_READY` or scientific completion.
