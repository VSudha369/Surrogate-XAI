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
- [ ] Stage 2M version, canonical script SHA, artifact hash-manifest SHA, READY marker, and proceed decision pass.
- [ ] Stage 1B and Stage 2M are treated read-only.
- [ ] All six authorized partition counts match the frozen protocol.
- [ ] Train Known contains exactly 98 identities; Calibration Unknown contains 22 disjoint identities.

## Strict zero-day

- [ ] Strict index arrays are never passed to `numpy.load` or HDF5 row readers.
- [ ] Frozen manifests provide strict file counts; byte-stream hashes are recorded.
- [ ] Signal, label, embedding, metric, and threshold strict counters are all zero.
- [ ] No strict prediction, embedding, score, AUROC, or threshold artifact exists.

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
- [ ] READY is written only after all gates pass under the full profile.
