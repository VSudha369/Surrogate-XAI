# Stage 2.6M — WiSig ManyTx controlled representation ablation

This repository contains the executable Stage 2.6M v1.0.2 performance-engineered experiment for the frozen `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3` benchmark. Versioned v1.0.1 sources remain as historical provenance. The scientific experiment still compares exactly four representation objectives under one common RF temporal network:

## Repository context

This work belongs to **Ultra-Low Latency Surrogate-Assisted Explainable AI for Zero-Day Threat Detection in Non-Stationary Physical Layer Communications**. The active path is the WiSig ManyTx transmitter-centric zero-day branch; earlier RadioML/AMC code is legacy provenance only. Stage 1B and Stage 2M are frozen, Stage 2.6M v1.0.2 is the current controlled-representation implementation candidate, and Stage 3M remains contingent on the Stage 2.6M selection result.

| Arm | Objective |
|---|---|
| A0 | CE |
| A1 | CE + supervised contrastive loss |
| A2 | CE + EMA-prototype compactness loss |
| A3 | CE + supervised contrastive + EMA-prototype loss |

The supplied prior notebook was reviewed for project history and engineering conventions. Its embedded Stage 2 diagnostic bundle is RadioML-oriented, so it is not used as the active experiment. This implementation is purpose-built for WiSig ManyTx: 2×256 I/Q, 98 known transmitters, P0–P3 domain protocols, receiver/day/equalization metadata, and calibration-unknown diagnostics.

## Safety boundary

Training reads only `Train Known`. P0–P3 are frozen validation protocols. Calibration Unknown is embedded only after training and never contributes gradients, prototypes, covariance fitting, thresholds, or primary model selection.

`strict_zero_day_test_indices.npy` and `strict_zero_day_shift_test_indices.npy` are never loaded. The guard verifies only their existence, byte-stream hash, and count declared in frozen Stage 1B manifests. Hashes are accepted only from records bound to the exact strict filename; counts may also use the frozen canonical split keys `strict_zero_day` and `strict_zero_day_shift`. This prevents unrelated manifest digests or split counts from being mistaken for strict-array declarations. Its strict signal/label/embedding/metric/threshold fields are violation counters, while the primary evidence is structural: an authorized partition allowlist, strict-path prohibition, frozen-index-only resolution, output scanning, and a static forbidden-artifact guard.

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

Mount Google Drive once in the parent Colab notebook and use `Stage2_6M_Colab_Launcher_v1_0_2.py`. The launcher detects an accessible existing mount, installs only missing dependencies, verifies upstream inputs, reports the GPU, and invokes the standalone program with `subprocess.check_call`.

```bash
git pull --ff-only origin codex/stage2-6m
git rev-parse HEAD
python -u Stage2_6M_Colab_Launcher_v1_0_2.py --performance-preflight
python -u Stage2_6M_Colab_Launcher_v1_0_2.py --reset-seed 42
python -u Stage2_6M_Colab_Launcher_v1_0_2.py
```

The preflight performs no model training and cannot create READY. It makes an opaque SHA-verified local copy, builds only authorized Train Known shards, checks bitwise backend/exposure/model-input equivalence, benchmarks storage, tunes execution-only loader/evaluation settings, and writes measured performance reports. The full v1.0.2 run requires a passing current preflight status.

Full-profile training resumes only from a complete synchronized four-arm epoch checkpoint. Two alternating resume slots retain the current and preceding epoch so an interruption during one arm rolls back to the latest common epoch. The v1.0.1 AMP recovery corrections explicitly accept checkpoints written by predecessor script SHA-256 values `421e3c64ce33b3b7929e10d5af84debe9e735c9b2a8709475080cfa0346fd6ac` and `7493e709bf0cd4a41b990b950f8603900ce4904299497c400ab5df7de346a141`; every other benchmark, Stage 2M, configuration, architecture, arm, and seed provenance gate remains mandatory, and all newly written checkpoints record the current script SHA-256. AMP gradient overflows use GradScaler's canonical skipped-step recovery and abort only after 32 consecutive overflows without a successful optimizer step; non-finite forward losses retain their stricter abort gate.

Environment overrides are explicit:

```python
import os
os.environ["WISIG_BRANCH_ROOT"] = "/content/drive/MyDrive/.../MANYTX_ZERO_DAY_BRANCH_v1.0.3"
os.environ["WISIG_STAGE2_6M_SCRIPT"] = "/content/Surrogate-XAI/Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"
os.environ["WISIG_STAGE2_6M_PROFILE"] = "full"
```

Never invoke the standalone pipeline with `%run`; that can leak Jupyter's `-f kernel.json` into argument parsing.

## Direct execution

```bash
python Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py \
  --branch-root "/path/to/MANYTX_ZERO_DAY_BRANCH_v1.0.3" \
  --profile full
```

Resume is enabled by default. A checkpoint is rejected unless benchmark, Stage 2M, script, configuration, arm, seed, architecture, and loss signatures all match. Use `--no-resume` to start a fresh synchronized run in the configured output directory.

For dependency and finite-gradient validation without benchmark access:

```bash
python Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py \
  --synthetic-validation --device cpu
```

This command produces no scientific metrics and cannot create READY.

## v1.0.2 storage and security design

The canonical HDF5 identity never changes. `/content/wisig_stage2_6m_cache` is disposable execution storage and is excluded from scientific configuration identity. The whole-HDF5 local operation is an opaque filesystem copy plus SHA-256 verification; it does not parse rows.

Shard generation accepts only an explicit authorized partition name and the already-verified frozen index array. Train Known is the only shard constructed during preflight. P0–P3 shards are delayed until known evaluation is required, and Calibration Unknown shards are structurally rejected until Stage 08. Strict-zero-day names are absent from the authorization enum, rejected by path guards, never supplied by split discovery, and never passed to an HDF5 signal reader. Local shards are never persisted to Drive.

`storage-mode=auto` selects sharded local storage only after equivalence passes, measured samples/s improves by at least 10%, and P95 latency remains within its gate. Otherwise the verified single local HDF5 is selected. `single_drive` is available only as an explicit controlled fallback; insufficient local disk never triggers a silent fallback.

## Controlled training design

For each seed (42, 123, 2026), all arms start from the identical model state. Every epoch materializes one deterministic Tx-balanced list of training indices; every arm consumes that same ordered list and the same epoch augmentation seed. Group-level early stopping stops all arms together, preserving equal exposure and budget.

The network contains a learned I/Q mixing frontend, residual and dilated temporal blocks, global mean/standard-deviation pooling, a 128-dimensional projection, normalized embeddings, and a 98-class head. It stays within the requested compact parameter budget and has no architecture branches.

## Outputs

The ten internal stages create the required reports, tables, figures in PNG and PDF, manifests, three checkpoint types per arm/seed, memory-mapped embedding stores, Excel/LaTeX/PDF publication artifacts, and `CANONICAL_STAGE3M_OBJECTIVE.json`. See `OUTPUT_SCHEMA.md` for the complete contract.

`MANYTX_STAGE2_6M_READY.txt` is created only after all gates pass under the full profile. An unresolved scientific result is valid and is represented by `NO_OBJECTIVE_CLEARLY_SUPERIOR`; it is not replaced with a fabricated winner.

## Scientific interpretation

The final report separates measured facts, statistical inference, scientific interpretation, and the Stage 3M recommendation. Calibration Unknown is explicitly labeled a precursor diagnostic—not final zero-day evidence.
