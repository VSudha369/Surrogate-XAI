# Stage 4M predecessor Drive audit

Result: **STAGE4M_PREDECESSOR_DRIVE_AUDIT_PASS**

This was a read-only audit of the unique canonical root at `/content/drive/MyDrive/colab files /Surrogate-XAI/project_root/MANYTX_ZERO_DAY_BRANCH_v1.0.3`. Google Drive was not modified. No Stage 3.5M strict score, prediction, index, label, metric, bootstrap, figure, or scientific publication artifact was opened.

## Frozen continuity chain

- Stage 1B: `MANYTX_BENCHMARK_READY`; benchmark `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5`, SHA-256 `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`.
- Stage 2M: `MANYTX_STAGE2M_READY`; hash manifest SHA-256 `0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a`.
- Stage 2.6M: `MANYTX_STAGE2_6M_READY`; artifact SHA-256 `83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660`; decision `SELECT_CE_SUPCON_PROTOTYPE`.
- Stage 3M: `MANYTX_STAGE3M_READY`; hash manifest SHA-256 `5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0`; seed-123 A3 teacher selected.
- Stage 3.5M: `MANYTX_STAGE3_5M_READY`; post-lock recovery is provenance-valid and all six strict violation counters are zero.

## Actual benchmark contract

The frozen schema contains 1,020,643 rows. Signals are `signals/X`, `float32`, shape `[N, 2, 256]`; known labels are `metadata/known_class_index` with 98 classes. Authorized Stage 4M partitions are Train Known (388,139), P0 (68,495), P1 (153,529), P2 (27,088), P3 (8,992), and `calibration_unknown` (158,400). Their HDF5 split keys are recorded in the JSON audit. Strict split filenames were verified from metadata only; their contents and labels were not read.

## Actual teacher contract

The frozen teacher is `WiSigRepresentationNet`, input `[2, 256]`, 849,634 parameters, 98 logits, and a normalized 128-D embedding. The canonical checkpoint is 3,417,639 bytes with SHA-256 `ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4`; the state-dict is 3,414,971 bytes with SHA-256 `7d6c6ff609fb86618ae7b92bcd55b0c8a440ed2769561d4de9b4485802e639d7`. Both byte hashes were independently recomputed from the Drive objects and match `TEACHER_HASH_MANIFEST.json`.

## Actual optimization and runtime contract

Stage 2.6M used AdamW (`lr=0.001`, `weight_decay=0.0001`), CosineAnnealingLR (`T_max=40`, `eta_min=0.00001`), batch size 256, evaluation batch size 1024, CUDA AMP when available, and gradient clipping at norm 5.0. Its deterministic domain-balanced sampler uses four samples per transmitter, explicit DataLoader generators, and Python/NumPy worker seeds derived from `torch.initial_seed()`. The frozen Drive configuration selected two workers, prefetch factor two, pinned memory, persistent workers, a 128-shard local Train Known cache, and single-local evaluation partitions.

Stage 4M intentionally keeps the optimizer, scheduler, batching, augmentation, clipping, deterministic loader, cache, and exposure conventions where scientifically compatible. The Stage 4M protocol itself fixes minimum epochs to 5 and patience to 8, replaces Stage 2.6M objective selection with the predeclared K0-K3 KD panel, and uses only P0 for best-epoch/arm/seed selection.

## Resume and provenance contract

The predecessor saves optimizer, scheduler, AMP scaler, full RNG, sampler exposure, model, objective, configuration, architecture, and upstream hashes. Resume selects a hash-current synchronized epoch. Stage 4M will preserve exact-state resume and will add immutable teacher, predecessor-lock, student-architecture, objective-policy, and selection-policy hashes.

## Comparison outcome

No prompt-vs-Drive scientific mismatch and no GitHub-vs-Drive schema or provenance mismatch was found. The approved GitHub source commit is exactly `9fbe2ca80768a3407ea825495318141e9e668b30`. There are no unresolved ambiguities.
