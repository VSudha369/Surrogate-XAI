# Stage 4M v1.0.0 scientific protocol

Stage 4M tests whether one deterministic half-width surrogate can preserve the frozen Stage 3M seed-123 A3 teacher sufficiently well for later surrogate-assisted XAI. The only scientific independent variable is the predeclared KD objective.

## Frozen inputs

- Benchmark: `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`, SHA-256 `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`.
- Teacher: A3 seed 123, 849,634 parameters, 98 classes, 128-D embedding, checkpoint SHA-256 `ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4`.
- Stage 3.5M is metadata-only. Its strict signals, labels, indices, scores, predictions, metrics, statistics, figures, and scientific reports are inaccessible.

## Student and objectives

The student preserves the teacher’s operation order, kernels, strides, GroupNorm, SiLU, pooling, and classifier interface at a fixed 0.50 width multiplier. Even channel rounding gives 16/32/64/128 channels and a native 64-D embedding. K2/K3 use a training-only 64→128 projection.

At temperature `T=4`, K0 is CE; K1 is `0.5 CE + 0.5 KD`; K2 is `0.4 CE + 0.4 KD + 0.2 representation`; K3 is `0.35 CE + 0.35 KD + 0.15 representation + 0.15 prototype`. No coefficient, temperature, architecture, optimizer, or seed search is permitted.

For every training batch, RF augmentation is applied exactly once. K1-K3 pass that exact augmented tensor to both the frozen teacher and student; the online teacher logits/embedding are detached sample-matched targets. K0 performs no teacher training forward. Clean Train Known teacher embeddings are used only to construct K3 class prototypes, while clean P0-P3 caches remain evaluation-only.

## Data and selection

All 12 canonical trainings use Train Known only for seeds 42, 123, and 2026. P0 alone controls early stopping, best epoch, arm selection, and seed selection. P1-P3 are evaluated only after freeze. Calibration Unknown is opened only after the immutable selection lock and is labelled `ZD_CALIBRATED_DIAGNOSTIC`; it cannot alter model weights or selection.

The canonical HDF5 benchmark remains read-only on Google Drive. Before Stage 04, only Train Known and P0-P3 are copied into deterministic, uncompressed 1,024-row HDF5 shards under `/content/wisig_stage4m_local_v1_0_0`; scientific membership, order, global indices, labels, receiver/day/equalization, transmitter metadata, sampler exposure, and augmentation are unchanged. This cache is a disposable runtime representation, not a scientific independent variable. Ordinary training signal reads from Drive after activation are forbidden and instrumented. Calibration Unknown has a separate post-lock cache; strict/final data cannot be staged.

Best epoch maximizes P0 teacher/student agreement with absolute tolerance 0.002, then minimizes P0 KL, maximizes fixed-98 macro-F1, and chooses the earlier epoch. Arm selection uses median agreement, median KL, agreement standard deviation, median F1, then K0<K1<K2<K3. The selected arm must meet the 40% parameter, 0.90 agreement, 0.8193773268 accuracy, and 0.8168258758 fixed-98 F1 gates.

## Optimization

The fixed schedule inherits Stage 2.6M’s AdamW (`lr=0.001`, `weight_decay=0.0001`), CosineAnnealingLR (`eta_min=1e-5`), batch 256, deterministic domain-balanced exposure, RF augmentation, CUDA AMP, and gradient clipping at norm 5. Stage 4M fixes maximum/minimum epochs to 40/5 and patience to 8.

## AMP numerical-safety policy

CUDA AMP remains frozen and enabled. Numerical overflow handling is runtime safety, not a scientific independent variable: after unscale and norm-5 clipping, a detected non-finite AMP gradient causes `GradScaler.step` to skip the underlying optimizer update, the scaler to back off, gradients to be cleared, and training to continue with the next batch. The run aborts only after more than 32 consecutive AMP overflows. With AMP disabled, any non-finite gradient remains an immediate scientific abort. The scheduler remains epoch-scoped, and a completed epoch must contain at least one successful optimizer update. Per-epoch and cumulative overflow accounting is persisted in checkpoints and history.

After every completed epoch, Drive receives an atomic exact-resume checkpoint, history, epoch status, and any improved best checkpoint. A tiny heartbeat is written every 25 batches without saving model state. Interruption during an epoch resumes from the preceding completed epoch and deterministically replays the incomplete epoch.

The pre-hotfix K0/seed-42 epoch-1 checkpoint from executable `a770fe52a83a408eedd7e8affaff120dd1d56eb5f243da52f05501252e4b4de3` is **not resume-compatible** because the new executable and regenerated preflight/predecessor provenance change the dependency graph. It is explicitly rejected, audit-recorded, removed, and K0/seed-42 restarts from epoch 1. Unknown stale checkpoints remain hard failures.

## Finality

Every reusable stage revalidates its full recorded input graph, provenance hashes, output hashes, and output sizes. Training checkpoints additionally bind `TRAINING_TARGET_POLICY.json`, preventing any pre-hotfix clean-target checkpoint from resuming.

Stage 12 writes final status, verifies the final hash manifest, writes READY, writes the durable Stage-12 checkpoint last, verifies it, and only then removes NOT_READY. A verified completed transaction is immutable: later failures cannot replace READY, and a normal second invocation returns `MANYTX_STAGE4M_ALREADY_READY` without scientific writes.
