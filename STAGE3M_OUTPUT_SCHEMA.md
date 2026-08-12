# Stage 3M v1.0.0 Output Schema

All persistent runtime outputs are confined to `<branch-root>/04_canonical_teacher`. Stage 2.6M is read-only.

| Directory | Principal artifacts |
|---|---|
| `configs/` | Effective configuration and predeclared `CANONICAL_TEACHER_SELECTION_POLICY.json` |
| `checkpoints/candidates/seed_<seed>/` | Byte-identical promoted A3 best-selection checkpoints |
| `checkpoints/canonical/` | `canonical_teacher_v1_0.pt`, `canonical_teacher_state_dict.pt` |
| `manifests/` | Input freeze, candidate provenance, architecture equivalence, sampling, stage checkpoints, final status, hash manifests |
| `metrics/confusion_matrices/` | One 98×98 NumPy matrix per seed/protocol |
| `tables/` | Known/per-class/representation/degradation/selection CSV tables and optional Calibration Unknown diagnostics |
| `embeddings/seed_<seed>/<protocol>/` | Complete-manifest-bound float16 embedding/logit memmaps plus labels and global indices |
| `statistics/` | Across-seed known-domain descriptive summaries |
| `reports/` | Scientific report, model card, selection report, and robustness report |
| `figures/` | Measured PNG/PDF macro-F1, degradation, and representation figures |
| `publication/` | PDF report, XLSX tables, LaTeX tables, and figure manifest |
| `performance/` | Verified opaque local-cache report |
| `logs/` | Stage 3M execution log |

An embedding store is reusable only when `INCOMPLETE` is absent, all four arrays exist, `store_manifest.json` says `complete: true`, and its checkpoint SHA and row count match the current request.

`TEACHER_FREEZE.json` records selected seed, source arm/objective, frozen hashes, architecture, selection-policy hash, source/canonical checkpoint hashes, metrics, provenance chain, strict counters, and freeze time. `TEACHER_HASH_MANIFEST.json` hashes both canonical exports and the freeze/model-card artifacts.

`MANYTX_STAGE3M_READY.txt` is created last and is mutually exclusive with `MANYTX_STAGE3M_NOT_READY.txt`. It identifies the selected teacher and explicitly records final zero-day evaluation, surrogate training, and XAI as `NO`.
