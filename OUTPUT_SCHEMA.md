# Output schema

All scientific runtime artifacts are written below the canonical branch's `03_representation_ablation/` directory. Nothing is written to `01_benchmark_engineering/` or `02_benchmark_diagnostics/`.

## Directory contract

```text
03_representation_ablation/
├── configs/
├── checkpoints/A0..A3/seed_<seed>/
├── logs/
├── manifests/
├── embeddings/A0..A3/seed_<seed>/<partition>/
├── metrics/
├── statistics/
├── tables/
├── figures/
├── reports/
├── publication/
└── cache/
```

The v1.0.2 runtime also creates `performance/` beneath the Stage 2.6M output and disposable cache/shard files only beneath `/content/wisig_stage2_6m_cache/`.

## Checkpoints

Each arm/seed directory contains `last.pt`, alternating `resume_slot_0.pt`/`resume_slot_1.pt`, `best_known_macro_f1.pt`, and `best_selection.pt`. Payloads include model, optimizer, scheduler, scaler, epoch, arm, seed, best metrics, frozen configuration, benchmark/Stage2M/script hashes, RNG states, EMA prototype state, sampler exposure state, architecture signature, loss coefficients, and synchronized group early-stopping state.

## Performance artifacts

`performance/` contains `local_cache_report.json`, storage A/B CSV/JSON, storage selection, DataLoader autotune, evaluation-batch autotune, batch/epoch timing, hardware utilization, performance summary JSON/Markdown, exposure/model-input equivalence evidence, authorized shard manifest copies, and `PERFORMANCE_PREFLIGHT_STATUS.json`. Missing historical baseline telemetry is labeled `BASELINE_NOT_MEASURED`; values are never fabricated.

`/content/wisig_stage2_6m_cache/LOCAL_CACHE_MANIFEST.json` and partition-local `LOCAL_SHARD_MANIFEST.json` files describe disposable runtime storage. These local files are not copied into scientific checkpoint identity and are never represented as a new benchmark.

## Embedding stores

Each authorized partition store contains:

- `embedding_normalized.npy`: `[N,128]`, float16, memory-mapped;
- `logits.npy`: `[N,98]`, float16, memory-mapped;
- `labels.npy`: known class indices or −1 for Calibration Unknown;
- `global_indices.npy`: frozen benchmark row indices;
- `receiver.npy`, `day.npy`, `equalized.npy`: domain metadata;
- `store_manifest.json`: checkpoint hash, shape, dtypes, and completion flag.

There is no strict-zero-day store.

## Required tables

- `arm_configuration.csv`: fixed arm/loss definition.
- `arm_parameter_equivalence.csv`: common parameter count/signature and unit checks.
- `training_history_summary.csv`: per arm/seed/epoch training and P0–P3 metrics.
- `seed_summary.csv`: one row per arm/seed for selection components.
- `known_protocol_results.csv`: arm/seed/protocol aggregate metrics.
- `known_protocol_per_class.csv`: fixed 98-class support, precision, recall, and F1.
- `fixed98_macro_results.csv`: observed versus fixed-frame results.
- `protocol_degradation.csv`: P0-relative P1/P2/P3 degradation.
- `embedding_separability.csv`: geometry and deterministic cluster metrics.
- `centroid_drift.csv`: per-Tx and summary drift rows.
- `domain_leakage_auc.csv`: Tx-balanced domain classifier results.
- `calibration_unknown_scores.csv`: overall threshold-free novelty diagnostics.
- `calibration_unknown_domain_matched.csv`: stratum and macro-domain-matched AUROC.
- `paired_statistical_tests.csv`: paired bootstrap, Wilcoxon, and BH-FDR.
- `effect_sizes.csv`: paired Cohen dz and confidence intervals.
- `objective_selection_matrix.csv`: raw selection hierarchy components and seed summaries.

## Required reports

The `reports/` directory contains input, sampler, model/loss, training, known evaluation, embedding, domain, Calibration Unknown, statistics, objective-selection, and reviewer-ready Markdown reports. The reviewer-ready report uses explicit measured-fact, statistical-inference, scientific-interpretation, and Stage-3M-recommendation sections.

## Figures

Every declared figure is saved as 300-DPI PNG and vector PDF: training loss, validation macro-F1, protocol results, fixed-98 comparison, degradation, silhouette, intra/inter distance, centroid margin, receiver leakage, equalization leakage, centroid drift, PCA by Tx/domain, Calibration Unknown score distributions, and the objective trade-off.

PCA is visualization only. It is not used in quantitative selection.

## Manifests

The manifest set includes input provenance, frozen config, strict guard, architecture, losses, sampler, seeds, checkpoints, statistics, per-stage manifests, final status, Stage 3M objective, file inventory, and SHA-256 inventory. `STRICT_TEST_GUARD.json` labels its numeric fields as violation counters and records the structural enforcement mechanisms.

`STAGE2_6M_FINAL_STATUS.json` includes pipeline/script/configuration provenance, both frozen Stage 2M hashes, architecture signature, seed panel, profile, canonical benchmark hash, runtime backend, local-cache/shard-equivalence status, DataLoader settings, evaluation batch size, vectorized-augmentation status, performance-preflight manifest SHA, decision, violation counters, and success gates. The artifact-manifest hash remains in the separate READY marker to avoid self-reference.

`CANONICAL_STAGE3M_OBJECTIVE.json` freezes the decision, selected objective/coefficients when resolved, architecture signature, embedding dimension, sampler, augmentation, optimizer, budget, seed evidence, and rationale.

## Publication artifacts

`publication/Stage2_6M_summary.xlsx` contains Configuration, Training, P0-P3, Fixed98, Separability, DomainRobustness, CalibrationUnknown, Statistics, and Selection sheets. The directory also contains `Stage2_6M_tables.tex`, `Stage2_6M_report.pdf`, and `figure_manifest.csv`.

## Terminal markers

- `MANYTX_STAGE2_6M_READY.txt` exists only for a full successful run.
- `MANYTX_STAGE2_6M_NOT_READY.txt` records a hard gate or execution failure.
- `STAGE2_6M_PERFORMANCE_PREFLIGHT_PASS` is printed only by a non-training performance preflight whose cache, equivalence, tuning, and strict guards pass.

READY includes the benchmark hash, strict-test violation counters, selected decision, and artifact-manifest hash.
