# Stage 3M v1.0.0 Output Schema

All persistent runtime outputs are confined to `<branch-root>/04_canonical_teacher`. Stage 2.6M is read-only.

| Directory | Principal artifacts |
|---|---|
| `audit/` | Drive predecessor inventory/reconstruction, exact class coverage, CPU-only A3 checkpoint audit, report/table consistency, and Stage 2.6M embedding-store compatibility evidence |
| `configs/` | Effective configuration and predeclared `CANONICAL_TEACHER_SELECTION_POLICY.json` |
| `checkpoints/candidates/seed_<seed>/` | Byte-identical promoted A3 best-selection checkpoints |
| `checkpoints/canonical/` | `canonical_teacher_v1_0.pt`, `canonical_teacher_state_dict.pt` |
| `manifests/` | Input freeze, candidate provenance, architecture equivalence, sampling, stage checkpoints, final status, hash manifests |
| `metrics/confusion_matrices/` | One 98×98 NumPy matrix per seed/protocol |
| `tables/` | Known/per-class/representation/degradation/selection CSV tables and optional Calibration Unknown diagnostics; representation rows persist observed/missing class coverage and `OBSERVED_CLASSES` frame |
| `embeddings/seed_<seed>/<protocol>/` | Complete-manifest-bound float16 embedding/logit memmaps plus labels and global indices |
| `statistics/` | Across-seed known-domain descriptive summaries |
| `reports/` | Scientific report, model card, selection report, and robustness report |
| `figures/` | Measured PNG/PDF macro-F1, degradation, and representation figures |
| `publication/` | PDF report, XLSX tables, LaTeX tables, and figure manifest |
| `performance/` | Verified opaque local-cache report |
| `logs/` | Stage 3M execution log |

An embedding store is reusable only when `INCOMPLETE` is absent, all four arrays exist with exact expected shapes, `store_manifest.json` says `complete: true`, and its checkpoint SHA and row count match the current request. `REPRESENTATION_METRIC_SAMPLING.json` records exact effective RNG seeds and hashes both selected store positions and selected frozen global indices.

`--drive-audit` writes `STAGE3M_DRIVE_PREDECESSOR_AUDIT.json/.md`, `STAGE3M_DRIVE_CLASS_COVERAGE_AUDIT.json`, `STAGE3M_DRIVE_A3_CHECKPOINT_AUDIT.json`, and `STAGE3M_DRIVE_REPORT_CONSISTENCY.json` under `audit/`. It never writes a READY marker and never opens strict-zero-day signal or label arrays.

`TEACHER_FREEZE.json` records selected seed, source arm/objective, frozen hashes, architecture, selection-policy hash, source/canonical checkpoint hashes, metrics, provenance chain, strict counters, and freeze time. `TEACHER_HASH_MANIFEST.json` hashes both canonical exports and the freeze/model-card artifacts.

`STAGE3M_HASH_MANIFEST.json` declares and justifies four recursion/transaction exclusions: itself, the Stage-10 completion checkpoint, and mutually exclusive READY/NOT_READY markers. `MANYTX_STAGE3M_READY.txt` is created only after that manifest verifies. The Stage-10 checkpoint is written last and hash-binds READY, the final manifest, canonical exports, freeze metadata, status, and required publication outputs. READY identifies the selected teacher and explicitly records final zero-day evaluation, surrogate training, and XAI as `NO`.
