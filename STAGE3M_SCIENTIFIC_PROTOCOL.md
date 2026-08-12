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

## Deterministic selection

The selection policy is written and hashed before selection. Candidates are ordered by:

1. higher mean fixed-98 macro-F1 over P0–P3;
2. higher mean fixed-98 balanced accuracy over P0–P3;
3. lower mean absolute fixed-98 macro-F1 degradation from P0 to P1/P2/P3;
4. higher P0 representation Fisher ratio;
5. lower numeric seed.

Calibration Unknown and all strict/final data are excluded from selection.

## Stages and freeze criterion

The ten hash-bound resumable stages verify predecessors, audit candidates, prove architecture equivalence, evaluate known domains, calculate representation diagnostics, analyze degradation, optionally describe Calibration Unknown, select deterministically, export weights, and run the final READY audit. Stage manifests bind executable, configuration, benchmark, predecessor, input, and output hashes.

Stage 3M is implemented but not scientifically frozen until a canonical Drive run produces `MANYTX_STAGE3M_READY.txt` with complete P0–P3 artifacts, exact source/export weight equivalence, all required hashes, and five zero strict-access counters.
