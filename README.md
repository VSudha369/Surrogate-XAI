# Stage 2.6M — WiSig ManyTx controlled representation ablation

This repository contains the executable Stage 2.6M v1.0.1 experiment for the frozen `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3` benchmark. It compares exactly four representation objectives under one common RF temporal network:

| Arm | Objective |
|---|---|
| A0 | CE |
| A1 | CE + supervised contrastive loss |
| A2 | CE + EMA-prototype compactness loss |
| A3 | CE + supervised contrastive + EMA-prototype loss |

The supplied prior notebook was reviewed for project history and engineering conventions. Its embedded Stage 2 diagnostic bundle is RadioML-oriented, so it is not used as the active experiment. This implementation is purpose-built for WiSig ManyTx: 2×256 I/Q, 98 known transmitters, P0–P3 domain protocols, receiver/day/equalization metadata, and calibration-unknown diagnostics.

## Safety boundary

Training reads only `Train Known`. P0–P3 are frozen validation protocols. Calibration Unknown is embedded only after training and never contributes gradients, prototypes, covariance fitting, thresholds, or primary model selection.

`strict_zero_day_test_indices.npy` and `strict_zero_day_shift_test_indices.npy` are never loaded. The guard verifies only their existence, byte-stream hash, and count declared in frozen Stage 1B manifests. Its strict signal/label/embedding/metric/threshold fields are violation counters, while the primary evidence is structural: an authorized partition allowlist, strict-path prohibition, frozen-index-only resolution, output scanning, and a static forbidden-artifact guard.

## Canonical layout

```text
MANYTX_ZERO_DAY_BRANCH_v1.0.3/
├── 01_benchmark_engineering/
│   └── benchmark/
│       └── WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5
├── 02_benchmark_diagnostics/
└── 03_representation_ablation/          # created by this pipeline only
```

Stage 1B and Stage 2M are opened read-only. `manifests/STAGE2M_FINAL_STATUS.json` must satisfy the exact structured readiness, version, script/benchmark hash, proceed recommendation, failed-gate, final-test prohibition, and strict-guard contract. `manifests/HASH_MANIFEST.json` is hashed independently and must match the frozen Stage 2M artifact-manifest SHA-256.

## Colab execution

The easiest route is to copy the complete contents of `Stage2_6M_Colab_Launcher_v1_0_1.py` into one Colab cell. The launcher mounts Drive, installs only missing dependencies, verifies upstream inputs, reports the GPU, and invokes the standalone program with `subprocess.check_call`.

Environment overrides are explicit:

```python
import os
os.environ["WISIG_BRANCH_ROOT"] = "/content/drive/MyDrive/.../MANYTX_ZERO_DAY_BRANCH_v1.0.3"
os.environ["WISIG_STAGE2_6M_SCRIPT"] = "/content/drive/MyDrive/.../Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_1.py"
os.environ["WISIG_STAGE2_6M_PROFILE"] = "full"
```

Never invoke the standalone pipeline with `%run`; that can leak Jupyter's `-f kernel.json` into argument parsing.

## Direct execution

```bash
python Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_1.py \
  --branch-root "/path/to/MANYTX_ZERO_DAY_BRANCH_v1.0.3" \
  --profile full
```

Resume is enabled by default. A checkpoint is rejected unless benchmark, Stage 2M, script, configuration, arm, seed, architecture, and loss signatures all match. Use `--no-resume` to start a fresh synchronized run in the configured output directory.

For dependency and finite-gradient validation without benchmark access:

```bash
python Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_1.py \
  --synthetic-validation --device cpu
```

This command produces no scientific metrics and cannot create READY.

## Controlled training design

For each seed (42, 123, 2026), all arms start from the identical model state. Every epoch materializes one deterministic Tx-balanced list of training indices; every arm consumes that same ordered list and the same epoch augmentation seed. Group-level early stopping stops all arms together, preserving equal exposure and budget.

The network contains a learned I/Q mixing frontend, residual and dilated temporal blocks, global mean/standard-deviation pooling, a 128-dimensional projection, normalized embeddings, and a 98-class head. It stays within the requested compact parameter budget and has no architecture branches.

## Outputs

The ten internal stages create the required reports, tables, figures in PNG and PDF, manifests, three checkpoint types per arm/seed, memory-mapped embedding stores, Excel/LaTeX/PDF publication artifacts, and `CANONICAL_STAGE3M_OBJECTIVE.json`. See `OUTPUT_SCHEMA.md` for the complete contract.

`MANYTX_STAGE2_6M_READY.txt` is created only after all gates pass under the full profile. An unresolved scientific result is valid and is represented by `NO_OBJECTIVE_CLEARLY_SUPERIOR`; it is not replaced with a fabricated winner.

## Scientific interpretation

The final report separates measured facts, statistical inference, scientific interpretation, and the Stage 3M recommendation. Calibration Unknown is explicitly labeled a precursor diagnostic—not final zero-day evidence.
