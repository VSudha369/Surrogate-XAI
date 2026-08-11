# Validation checklist

## Static and code-level

- [ ] The standalone script parses with Python 3.10+.
- [ ] `--synthetic-validation` produces finite gradients for A0–A3 without benchmark access.
- [ ] No `TODO`, pseudocode branch, omitted scientific function, or fabricated metric path is present.
- [ ] The Colab launcher uses `subprocess.check_call`, not `%run`.
- [ ] The notebook does not pass Jupyter kernel arguments to the standalone parser.

## Frozen inputs

- [ ] Branch root resolves to `MANYTX_ZERO_DAY_BRANCH_v1.0.3`.
- [ ] Benchmark SHA-256 equals the canonical hash.
- [ ] `STAGE2M_FINAL_STATUS.json` exactly verifies READY status, version, declared script/benchmark hashes, proceed decision, empty failed gates, final-test prohibitions, and nested strict guard zeros.
- [ ] Stage 2M `HASH_MANIFEST.json` independently matches the frozen SHA-256; no Stage 2M source script is required in the output tree.
- [ ] Stage 1B and Stage 2M are treated read-only.
- [ ] All six authorized frozen `.npy` partition arrays exist and their counts match the protocol; no full-HDF5 split reconstruction path exists.
- [ ] Train Known contains exactly 98 identities; Calibration Unknown contains 22 disjoint identities.

## v1.0.2 performance preflight

- [ ] Opaque Drive-to-`/content` copy is SHA-verified, fsynced, atomically finalized, and reported.
- [ ] Corrupt/partial local copies are rejected; insufficient disk never silently falls back.
- [ ] Authorized Train Known shard manifest and every shard SHA pass.
- [ ] Single-local versus sharded-local signals and metadata are bitwise/order equivalent, including duplicates and shard boundaries.
- [ ] Seed-42/epoch-1 exposure SHA, order, labels, equalization, and signals are identical across backends.
- [ ] Common initialized model inputs, logits, embeddings, and objective values are equivalent across backends.
- [ ] Vectorized circular shift exactly equals the reference loop for fixed shifts and remains gradient-compatible.
- [ ] Storage selection applies the 10% throughput and P95 latency gates and reports measured percentages.
- [ ] `SHARDED_FULL_RUN_CONTEXT_INITIALIZATION_PASS` proves Train Known shard initialization is non-recursive, occurs exactly once, registers its manifest, and is reused.
- [ ] Storage matrix cases A–D pass: automatic single local, automatic sharded Train Known, explicit single Drive, and forced sharded Train Known.
- [ ] P0/P1/P2/P3/Calibration Unknown resolve to one-file storage and do not inherit Train Known's 128-shard backend.
- [ ] DataLoader autotune uses independent RNG and tests safe worker/prefetch candidates.
- [ ] Evaluation batch autotune preserves top-1 predictions and strict logit tolerance with GPU-memory headroom.
- [ ] Performance preflight performs no optimizer update, checkpoint mutation, Calibration Unknown signal access, READY creation, or strict-row access.
- [ ] CUDA phase fields are sampled event durations; CPU enqueue observations and CPU/GPU utilization snapshots carry explicit sampling semantics.

## Strict zero-day

- [ ] Strict index arrays are never passed to `numpy.load` or HDF5 row readers.
- [ ] Frozen manifests provide strict file counts through the canonical split keys; byte-stream hashes are verified against path-bound records for the exact strict filenames.
- [ ] Signal, label, embedding, metric, and threshold strict violation counters are all zero and are not represented as complete HDF5 read instrumentation.
- [ ] No strict prediction, embedding, score, AUROC, or threshold artifact exists.
- [ ] Shard builder has no strict partition enum value or strict shard path and accepts only caller-supplied authorized frozen indices.

## Controlled experiment

- [ ] A0–A3 have identical parameter counts and architecture signatures.
- [ ] Same-seed initial model states are byte-identical.
- [ ] Sampler exposure SHA is identical across all arms per seed/epoch.
- [ ] Augmentation seed is identical across arms per seed/epoch.
- [ ] Group-level early stopping gives equal completed epochs across arms.
- [ ] Seed panel is exactly 42, 123, 2026 for every arm.
- [ ] Both equalization states appear in the training exposure.
- [ ] Calibration Unknown never contributes gradients, prototype updates, covariance fitting, or checkpoint selection.

## Metrics and diagnostics

- [ ] P0–P3 accuracy, top-5, cross-entropy, ECE, observed macro metrics, and fixed-98 metrics exist.
- [ ] P2/P3 missing identities are zero in the fixed frame.
- [ ] Full geometry and deterministic sampled cluster metrics exist with persisted indices.
- [ ] Train-to-P0/P1/P2/P3 centroid drift includes per-Tx and summary statistics.
- [ ] Day, receiver, combined, and equalization domain leakage results exist.
- [ ] All five required novelty scores fit geometry from Train Known only.
- [ ] Overall and five domain-matched Calibration Unknown analyses exist without thresholds.

## Statistics and selection

- [ ] All five declared arm comparisons are paired by seed.
- [ ] Paired bootstrap CI, Wilcoxon, paired Cohen dz, and BH-FDR columns exist.
- [ ] Raw classification, separation, domain, and Calibration Unknown components remain visible.
- [ ] Selection follows the declared hierarchy and practical/noninferiority rule.
- [ ] `NO_OBJECTIVE_CLEARLY_SUPERIOR` is retained when evidence is unresolved.

## Publication and freeze

- [ ] All required CSV tables and Markdown reports exist.
- [ ] Every required figure exists as PNG and PDF.
- [ ] Excel sheets, LaTeX tables, PDF report, and figure manifest exist.
- [ ] Three checkpoint types exist for all 12 arm/seed runs.
- [ ] File and SHA-256 manifests exist.
- [ ] `CANONICAL_STAGE3M_OBJECTIVE.json` contains the frozen policy and rationale.
- [ ] `STAGE2_6M_FINAL_STATUS.json` contains pipeline/script/configuration hashes, Stage 2M hashes, architecture signature, seed panel, and profile.
- [ ] READY is written only after all gates pass under the full profile.
- [ ] Final status separates canonical benchmark identity from runtime backend and includes preflight-manifest SHA.
