# Scientific protocol

## Purpose

Stage 2.6M is a controlled objective ablation, not architecture search or final teacher training. It asks whether supervised contrastive and/or prototype compactness objectives improve known-transmitter discrimination, class geometry, and receiver/day/equalization robustness while creating a more useful Train-Known-fitted novelty geometry.

## Frozen inputs and exclusions

- Benchmark: `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`, SHA-256 `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`.
- Stage 2M executed version: `1.0.5`.
- Stage 2M canonical script SHA-256: `46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba`.
- Stage 2M artifact hash-manifest SHA-256: `0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a`.
- Stage 2M provenance is accepted only from the structured `manifests/STAGE2M_FINAL_STATUS.json` contract; the artifact `HASH_MANIFEST.json` is hashed independently. A Stage 2M source `.py` file is not expected inside the frozen output tree.
- Train Known is the sole gradient-bearing partition.
- P0–P3 are known-transmitter evaluation protocols.
- Calibration Unknown is a frozen-model, threshold-free secondary diagnostic.
- Strict zero-day signals, labels, embeddings, metrics, and thresholds are unavailable.
- All six authorized partitions must be loaded from frozen Stage 1B `.npy` index arrays. Missing arrays abort the run; splits are never reconstructed by scanning the full HDF5 metadata.

## Hypotheses

- H1: CE+SupCon improves inter-transmitter separation and known-Tx robustness relative to CE.
- H2: CE+Prototype improves known-manifold compactness and distance-based novelty structure relative to CE.
- H3: CE+SupCon+Prototype outperforms CE and may outperform each single auxiliary objective if they are complementary.
- H4: representation objectives reduce receiver/equalization nuisance information without sacrificing Tx identity.
- H5: better known-manifold geometry improves Calibration Unknown separability when all novelty geometry is fitted from Train Known only.

The pipeline reports non-confirmation without changing the hypotheses after seeing results.

## Causal controls

All arms share the same data, model architecture and initialization, 128-dimensional embedding, 98-class head, preprocessing, conservative augmentation, deterministic Tx/domain sampler, AdamW optimizer, cosine scheduler, AMP policy, epoch budget, synchronized early stopping, seed panel, validation protocols, checkpoint rules, and downstream diagnostics. Only the loss coefficients differ.

For each seed and epoch, the sampler materializes one ordered batch exposure. The same exposure SHA and augmentation seed are consumed by A0–A3. Early stopping is group-level: all arms stop together after the shared patience condition, avoiding unequal exposure.

## Architecture

Input `[B,2,256]` passes through learned I/Q mixing, a stride-2 temporal frontend, residual 64-channel block, 64→128 downsampling block, dilated 128-channel block, 128→256 downsampling block, global temporal mean and standard-deviation pooling, a 128-dimensional embedding projection, L2 normalization, and a 98-class classifier.

The model returns `logits`, `embedding_raw`, and `embedding_normalized`. SupCon and prototype losses operate in FP32 on normalized embeddings.

## Objectives

```text
A0: L = CE
A1: L = CE + 0.1 × SupCon(T=0.07)
A2: L = CE + 0.1 × Prototype
A3: L = CE + 0.1 × SupCon(T=0.07) + 0.1 × Prototype
```

Prototype targets use one normalized EMA prototype per known transmitter with momentum 0.95. An uninitialized class uses its detached batch centroid for that first loss evaluation and is initialized after the successful optimizer step. No prototype-separation or hidden auxiliary loss is present.

## Sampling and augmentation

Each batch selects transmitters as its primary units and draws multiple examples per Tx by cycling available `(receiver, day, equalization)` cells before reusing a cell. Both equalization states are retained and batch totals are recorded.

The single frozen augmentation policy applies small phase rotation, amplitude jitter, low-intensity AWGN relative to per-sample RMS, and bounded circular shift. The epoch augmentation random stream is identical across arms. Version 1.0.2 implements the identical per-sample circular shift with vectorized gather indexing; fixed-shift equivalence requires exact tensor equality and gradient compatibility.

## Execution-only performance layer

Performance settings are excluded from the scientific configuration SHA. Training batch size remains 256, validation remains every epoch, and no arm, seed, model, objective, optimizer, scheduler, augmentation magnitude, or protocol is changed.

The Drive benchmark is first verified against the canonical SHA. A local whole-file copy is permitted only as an opaque byte stream, is written through a temporary file, fsynced, SHA-verified, and atomically renamed. Optional shards are local disposable execution artifacts. Their builder has an explicit six-partition authorization enum and requires the frozen authorized index array; strict-zero-day partitions have no construction route. Under the conservative v1.0.2 runtime policy, only Train Known is eligible for sharded execution. P0–P3 and Calibration Unknown use a single-file backend and are never sharded merely because Train Known wins its storage benchmark. Shard construction receives the already-resolved benchmark and never re-enters context initialization.

Storage microbenchmarks and autotuners use independent RNG state that is captured and restored. They perform no optimizer update and do not alter scientific sampler exposure. Sharded storage is selected automatically only after bitwise signal/metadata equivalence, exposure equivalence, model-input equivalence, a throughput improvement of at least 10%, and an acceptable P95 latency result.

DataLoader wait and total epoch/checkpoint/validation durations are wall-clock measurements. CUDA forward, objective, backward, optimizer, and total phase durations use events sampled every 50 batches with one synchronization per sampled batch. CPU-side enqueue observations are labeled separately and are not used as true GPU-duration percentages. CPU utilization rows are interval samples since the preceding snapshot, and GPU utilization rows are explicitly labeled instantaneous `nvidia-smi` samples rather than epoch averages.

## Known evaluation

P0–P3 report accuracy, top-5 accuracy, cross-entropy, ECE, observed-class macro-F1/balanced accuracy, and fixed-98 macro-F1/balanced accuracy. The fixed frame assigns zero recall/F1 to missing identities, which is essential for P2 and P3.

Model checkpoints are ranked during training by macro-average fixed-98 macro-F1 across P0–P3, then worst-protocol fixed-98 macro-F1, then P0 fixed-98 macro-F1. Final objective selection uses downstream geometry and domain results as declared tie-breakers.

## Representation and domain analysis

Full embedding stores produce within/between variance, Fisher ratio, intra-class radius, inter-centroid distance, nearest-centroid margin, and centroid cosine similarity. Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics use deterministic class-stratified samples whose global indices and hashes are persisted.

Train-to-P0/P1/P2/P3 Tx-centroid drift is reported per class and with mean, median, IQR, and 95% CI. Transmitter-balanced SGD log-loss classifiers estimate P0-vs-P1 day, P0-vs-P2 receiver, P0-vs-P3 combined, and equalized-0-vs-1 leakage. AUROC closer to 0.5 is interpreted as beneficial only alongside retained/improved Tx performance.

## Calibration Unknown diagnostic

Train Known embeddings alone fit class centroids and a Ledoit–Wolf shrinkage covariance over class residuals. Required unknown-oriented scores are nearest-prototype Euclidean distance, cosine prototype distance, shrinkage Mahalanobis distance, energy, and one-minus maximum softmax probability.

P0 Known versus Calibration Unknown reports AUROC, AUPRC, Cohen's d, robust histogram overlap, and stratified-bootstrap AUROC CI. Domain-matched diagnostics cover day, receiver, equalization, day+receiver, and day+receiver+equalization common strata. No threshold is tuned.

## Statistical comparison and selection

Primary comparisons are A1-vs-A0, A2-vs-A0, A3-vs-A0, A3-vs-A1, and A3-vs-A2. Metrics are paired by seed. The pipeline reports paired bootstrap mean-difference CIs, two-sided Wilcoxon signed-rank results, paired Cohen dz, and Benjamini–Hochberg FDR.

Selection is lexicographic:

1. macro-average fixed-98 P0–P3 macro-F1;
2. P0 Fisher separation;
3. lower mean absolute domain-AUROC deviation from 0.5;
4. Calibration Unknown AUROC aggregate as a secondary criterion.

An auxiliary arm is selected over CE only if its primary mean gain is at least 0.002, its paired-bootstrap lower bound is no worse than −0.001, and P0 Fisher ratio does not regress. Otherwise the scientifically valid decision is `NO_OBJECTIVE_CLEARLY_SUPERIOR`.

## Reproducibility and numerical policy

Python, NumPy, and Torch RNGs are seeded; CUDA seeds and deterministic cuDNN settings are recorded. Deterministic algorithms use warning mode so any unavailable deterministic kernel is surfaced without silently changing the experiment.

AMP applies only on CUDA. CE, pairwise similarity, and prototype distances are computed in FP32. Gradient norms are checked and clipped. A non-finite forward loss uses the strict two-event abort gate. AMP gradient overflow is different: GradScaler canonically skips the optimizer step, reduces its scale, and may adapt for at most 32 consecutive overflows; a successful step resets that counter. An epoch with no successful optimizer step aborts. Prototype updates occur only after successful optimizer steps, no silent `nan_to_num` repair is used, and the epoch scheduler advances only after successful optimizer work.

## Claims boundary

The final report distinguishes measured fact, statistical inference, scientific interpretation, and Stage 3M recommendation. It does not claim final zero-day performance, threshold quality, XAI quality, surrogate fidelity, or deployment latency.

Strict-test counters are explicitly interpreted as violation counters produced by the structural guard. The access-control evidence is the authorized partition allowlist, strict-path prohibition, frozen-index-only resolver, strict-array non-loading rule, and output/artifact scan.
