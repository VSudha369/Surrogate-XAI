# Stage 3.5M output schema

All persistent outputs are confined to `<branch-root>/05_zero_day_open_set`. Stage 1B, Stage 2M, Stage 2.6M, and Stage 3M trees are read-only.

| Directory | Contents |
|---|---|
| `configs/` | frozen scientific configuration |
| `logs/` | execution log |
| `manifests/` | predecessor, exposure, fit, policy, lock, stage, final-status, and final-hash manifests |
| `scores/fitted/` | known-only scorer state |
| `scores/p0..p3/` | complete known-validation score stores |
| `scores/calibration_unknown/` | optional ZD-CALIBRATED store |
| `scores/strict_zero_day_*/` | stores created only after the Stage-08 lock |
| `thresholds/` | frozen ZD-STRICT known-only thresholds |
| `metrics/`, `tables/` | strict and known score results |
| `statistics/` | deterministic bootstrap intervals |
| `figures/`, `reports/`, `publication/` | measured-result publication bundle |
| `performance/` | verified local-cache report |

Every score store has an `INCOMPLETE` transaction marker during construction and a hash-bound `store_manifest.json` only when complete. Resume requires exact executable, configuration, predecessor, input, and output hashes; file existence is insufficient.

`MANYTX_STAGE3_5M_READY.txt` is created only after the final status and self-consistent hash manifest. The hash manifest explicitly excludes itself, the Stage-11 checkpoint, and mutually exclusive READY/NOT_READY markers to avoid recursion. The Stage-11 checkpoint is written last and binds all final products.
