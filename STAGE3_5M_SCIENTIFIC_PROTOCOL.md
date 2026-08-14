# Stage 3.5M scientific protocol

Stage 3.5M compares deterministic post-hoc open-set scores under the frozen Stage 3M A3 seed-123 teacher. The teacher has 849,634 parameters, 98 classifier outputs, and a normalized 128-dimensional embedding. Its canonical SHA-256 is `ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4`. Teacher retraining, architecture search, an unknown training class, surrogate training, and XAI are forbidden.

## Scorers and direction

All scores increase with unknownness:

| ID | Score | Fit source |
|---|---|---|
| S0 | `1 - max softmax probability` | none |
| S1 | `-T logsumexp(logits/T)` energy | none |
| S2 | `1 - max cosine` to normalized class prototypes | Train Known |
| S3 | minimum regularized tied-covariance Mahalanobis distance | Train Known |
| S4 | minimum class-conditional diagonal-Gaussian NLL | Train Known |

S2-S4 are fit only from frozen-teacher Train Known embeddings. No gradient, optimizer, classifier update, or unknown-labelled training occurs.

## Protocol separation

`ZD-STRICT` is primary. P0-P3 Known Validation alone freezes one 95% known-acceptance threshold per scorer; 99% known-score quantiles are retained as declared sensitivity analyses. Strict emitters cannot fit scorers, thresholds, hyperparameters, or a winner. Because unknown-free diagnostics do not identify a defensible detector winner, the canonical scorer policy is `ALL_PREDECLARED`; every strict result is reported without post-hoc selection.

`ZD-CALIBRATED` is optional and separately labelled. Calibration Unknown may inform only its own table. The hash of the strict threshold manifest is checked before and after that analysis.

## Final strict transaction

Stages 01-07 cannot open strict index arrays or signals. Stage 04 first proves Stage 3M→Stage 3.5M known-inference equivalence for P0-P3 using exact global-index, label, and prediction identity when frozen primitive arrays are available, plus scalar accuracy and fixed-98 macro-F1 agreement. Stage 07 freezes a recursively verified `PRE_STRICT_KNOWN_SCORE_BUNDLE.json` covering every known score-store file, the threshold manifest, and scorer policy. Immediately before Stage 08, the pipeline writes and verifies `STRICT_ZERO_DAY_EVALUATION_LOCK.json`, binding those products together with the teacher, scorer state, executable, configuration, benchmark, predecessor hashes, exact zero-valued violation-counter keyset, and disabled post-lock fitting/calibration flags. The guard then permanently disables fitting and calibration. Stage 08 verifies sealed strict-index hashes, reads strict signals exactly for the final evaluation, and freezes a lock-bound `FINAL_STRICT_SCORE_BUNDLE.json`. Frozen partition membership supplies the binary unknown semantic; transmitter labels are not loaded.

Violation counters independently cover signal, label, embedding, metric, threshold, and fit access. All are violation counters and must remain zero. Stages 09-11 refuse to proceed unless both score bundles and the evaluation lock remain recursively hash-current. Stage 11 writes READY only after every stage and final hash manifest are current.

## Strict shifted-subset semantics and post-lock recovery

The frozen `strict_zero_day_shift_test` partition is a 3,000-row sensitivity subset of the 216,000-row `strict_zero_day_test`, not a disjoint peer. Fresh executions validate uniqueness, non-strict disjointness, and the exact subset relationship immediately after sealed indices are verified and before strict signal scoring. Metrics and confidence intervals are reported separately for the overall strict test and nested shift subset; no concatenated or double-counted `strict_combined` population exists.

The reviewed canonical run scored both strict stores once under its original lock before the earlier disjointness assertion aborted. `Stage3_5M_PostLock_Recovery_v1_0_0.py` therefore verifies and reuses those immutable stores. It cannot open strict signals, execute the teacher, fit scorers or thresholds, read strict labels, or replace the original lock. Recovery provenance and the original NOT_READY evidence are disclosed in the recovery manifest, final status, strict bundle, and READY marker.

Google Drive is the source of truth for frozen runtime artifacts; GitHub is the source of truth for executable code. Root resolution honors an explicit `--branch-root`, then `WISIG_BRANCH_ROOT`, then recursively discovers exactly one valid Stage-3M-READY root below `/content/drive/MyDrive`. Zero or multiple valid roots abort without selecting, copying, or renaming artifacts. The read-only Drive audit hashes predecessor evidence and the post-lock stores without inference or scientific writes.

Recovery statistics use an independent deterministic RNG stream derived from the fixed base seed, strict partition, and scorer. The effective seed for every stream is persisted. Stage 11 is a durable transaction: final status, final hash manifest, and exact READY bytes are verified; the Stage-11 checkpoint is then written and verified; only afterward is the historical NOT_READY marker removed. Interrupted states retain NOT_READY, while a verified Stage-11-plus-READY state with stale NOT_READY permits cleanup only. A second finalization of a complete transaction is a no-op.

## Metrics

Every scorer reports AUROC, AUPRC, unknown F1, known F1, macro-F1, FPR@95TPR, minimum detection error, OSCR, threshold, known acceptance, and unknown rejection. Deterministic stratified bootstrap confidence intervals are produced for AUROC, AUPRC, macro-F1, and OSCR. Closed-set known correctness remains separate and Stage 3M metrics are never overwritten.
