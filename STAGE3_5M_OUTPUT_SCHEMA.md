# Stage 3.5M output schema

All persistent outputs are confined to `<branch-root>/05_zero_day_open_set`. Stage 1B, Stage 2M, Stage 2.6M, and Stage 3M trees are read-only.

| Directory | Contents |
|---|---|
| `configs/` | frozen scientific configuration |
| `logs/` | execution log |
| `manifests/` | predecessor, exposure, fit, inference-equivalence, known/strict score-bundle, policy, lock, stage, final-status, and final-hash manifests |
| `scores/fitted/` | known-only scorer state |
| `scores/p0..p3/` | complete known-validation score stores |
| `scores/calibration_unknown/` | optional ZD-CALIBRATED store |
| `scores/strict_zero_day_*/` | stores created only after the Stage-08 lock |
| `thresholds/` | frozen ZD-STRICT known-only thresholds |
| `metrics/`, `tables/` | strict and known score results |
| `statistics/` | deterministic bootstrap intervals |
| `figures/`, `reports/`, `publication/` | measured-result publication bundle |
| `performance/` | verified local-cache report |

Every score store has an `INCOMPLETE` transaction marker during construction and a hash-bound `store_manifest.json` only when complete. Each manifest binds pipeline/executable/configuration identity, benchmark and teacher, fitted scorer, scorer order and definitions, energy temperature, partition, row count, strictness, exact global-index values, and—where strict—the current evaluation lock. `PRE_STRICT_KNOWN_SCORE_BUNDLE.json` recursively binds every P0-P3 store file before strict evaluation. `FINAL_STRICT_SCORE_BUNDLE.json` binds both strict stores to their exact sealed indices and the evaluation lock. Resume requires exact executable, configuration, predecessor, input, and output hashes; file existence is insufficient.

`STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json` records P0-P3 global-index, label, prediction, accuracy, and fixed-98 macro-F1 equivalence. If the frozen Stage 3M export lacks primitive arrays, this is explicitly marked and equivalence is limited to its frozen scalar evidence; no result is silently described as array-level equivalence.

Post-lock recovery additionally creates `STRICT_PARTITION_RELATIONSHIP_AUDIT.json`, `STRICT_OVERLAP_SCORE_CONSISTENCY.json`, and `POST_LOCK_RECOVERY_MANIFEST.json`. The relationship audit proves that the 3,000-row shifted set is an exact subset of the 216,000-row main set and both are disjoint from non-strict partitions. The overlap audit proves persisted predictions agree exactly and S0-S4 scores agree within the declared tolerance for all duplicated global indices. The recovery manifest binds the original failure marker, executable, configuration, lock, fit, thresholds, policy, known bundle, equivalence artifact, both audits, and both strict-store manifests while declaring every prohibited recomputation flag false.

`MANYTX_STAGE3_5M_READY.txt` is created only after the final status and self-consistent hash manifest. The hash manifest explicitly excludes itself, the Stage-11 checkpoint, and mutually exclusive READY/NOT_READY markers to avoid recursion. The Stage-11 checkpoint is written last and binds all final products.
