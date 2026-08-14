# Stage 3M v1.0.0 Scientific Protocol

## Purpose

Stage 3M promotes exactly one frozen Stage 2.6M A3 teacher into an immutable canonical teacher. It answers which already-trained A3 seed should be used downstream. It does not repeat representation-objective selection, train a model, tune a threshold, evaluate a final test set, distill a surrogate, or perform XAI.

The canonical path is `PROMOTE_AND_VERIFY`, selected with `--teacher-source stage2_6m_promote`. Any provenance or compatibility mismatch causes a scientific abort; there is no automatic retraining or checkpoint repair.

## Frozen scientific contract

- Benchmark: `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`, SHA-256 `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`.
- Stage 2M: v1.0.5, executable SHA-256 `46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba`, artifact manifest SHA-256 `0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a`.
- Stage 2.6M artifact SHA-256: `83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660`.
- Frozen decision: `SELECT_CE_SUPCON_PROTOTYPE`; source arm A3 only; seeds 42, 123, and 2026.
- Loss: CE + 0.1 SupCon + 0.1 Prototype; temperature 0.07; prototype momentum 0.95.
- Input `[B,2,256]`; 98 known classes; normalized 128-dimensional embedding; 849,634 parameters.

Candidate checkpoint acceptance requires exact arm, seed, benchmark SHA, Stage 2M SHA, Stage 2.6M configuration SHA, architecture signature, loss coefficients, ordered state schema, tensor shapes, and strict state loading. Candidate checkpoint bytes are copied into Stage 3M without changing Stage 2.6M.

## Data policy

P0, P1, P2, and P3 are allowed for frozen forward-only known validation. Calibration Unknown is disabled by default and, if enabled, is descriptive only. Train Known is provenance-only in the canonical implementation. Strict Zero-Day Test and Strict Zero-Day Shift Test signals, labels, embeddings, metrics, and thresholds are structurally rejected and separately counted. Every counter must be zero at the READY gate.

## Google Drive predecessor audit

`--drive-audit` is a read-only predecessor command. It discovers the unique READY branch root, inventories Stage 1B/2M/2.6M, verifies hash manifests and report/table consistency, measures class coverage from the frozen per-class table, validates all three A3 checkpoints on CPU, and audits A3 embedding-store identity. It writes only evidence below `04_canonical_teacher/audit`, does not run Stages 04-10, and has no teacher-selection, export, training, strict-signal, surrogate, XAI, or READY path. `--preflight` requires this audit to pass before Stages 01-03.

The Stage 2.6M READY marker publishes canonical `strict_zero_day_*_violations` names, while the frozen structured final status stores the five facts as `strict_test_*_reads` inside `strict_zero_day_violation_counters`. Stage 3M maps and requires every real structured key explicitly; a missing counter is a provenance failure, never an implicit zero.

The audited frozen class coverage is P0=98, P1=98, P2=97 (missing 72), and P3=93 (missing 50, 52, 58, 71, 72). All observed classes have at least two samples. These are predecessor facts, not hard-coded replacements for Stage 3M evaluation.

## Deterministic selection

The selection policy is written and hashed before selection. Candidates are ordered by:

1. higher mean fixed-98 macro-F1 over P0–P3;
2. higher mean fixed-98 balanced accuracy over P0–P3;
3. lower mean absolute fixed-98 macro-F1 degradation from P0 to P1/P2/P3;
4. higher P0 representation Fisher ratio;
5. lower numeric seed.

Calibration Unknown and all strict/final data are excluded from selection.

## Representation frame and sampling provenance

Classification retains both observed-class and fixed-98 frames; absent identities receive zero only in the fixed-98 classification frame. Representation geometry is different: centroids and clustering statistics are defined only for identities with samples in a protocol. Silhouette, Davies–Bouldin, Calinski–Harabasz, intra/inter-class distance, Fisher ratio, separation, and compactness therefore use the explicitly labeled `OBSERVED_CLASSES` frame. P1/P2/P3 may legitimately omit identities. P0 must contain all 98 identities for every candidate because only P0 Fisher ratio participates in teacher selection.

Each representation sampling record binds the base seed, derived effective seed, candidate seed, protocol, observed-class coverage, per-class cap, sampled row count, sampled-position SHA-256, and sampled global-index SHA-256. Original transmitter labels need not be dense; the implementation uses an explicit label-to-centroid mapping.

## Stages and freeze criterion

The ten hash-bound resumable stages verify predecessors, audit candidates, prove architecture equivalence, evaluate known domains, calculate representation diagnostics, analyze degradation, optionally describe Calibration Unknown, select deterministically, export weights, and run the final READY audit. Stage manifests bind executable, configuration, benchmark, predecessor, input, and output hashes. Stage 10 writes and verifies its final hash manifest and READY marker before atomically writing `STAGE_10_CHECKPOINT.json`; an interrupted or corrupt final transaction is never considered reusable.

Stage 3M is implemented but not scientifically frozen until a canonical Drive run produces `MANYTX_STAGE3M_READY.txt` with complete P0–P3 artifacts, exact source/export weight equivalence, all required hashes, and five zero strict-access counters.
