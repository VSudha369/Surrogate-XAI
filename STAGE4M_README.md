# Stage 4M — WiSig ManyTx surrogate KD

This package implements Stage 4M v1.0.0 on branch `codex/stage4m`. It is downstream of the frozen Stage 3M teacher and Stage 3.5M metadata lock. It does not perform strict zero-day evaluation or XAI.

## Colab preflight

Mount Drive in a notebook cell, clone the repository, check out `codex/stage4m`, and run:

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-u",
    "/content/Surrogate-XAI/Stage4M_Colab_Launcher_v1_0_0.py",
    "--repository-root", "/content/Surrogate-XAI",
    "--preflight",
])
```

Preflight verifies Drive/root uniqueness, predecessor identities, benchmark schema, teacher identity/forward contract, strict guards, GPU, and output containment. It performs no training, cache generation, architecture/objective freeze, selection, Calibration Unknown access, strict access, or READY creation.

Canonical K1-K3 training uses online frozen-teacher targets from the exact once-augmented tensor also given to the student. K0 does not execute a teacher training forward. Resume checkpoints bind this policy and the complete upstream dependency graph. Once Stage 12 is valid, later invocations preserve READY; a second normal invocation exits with `MANYTX_STAGE4M_ALREADY_READY` without writes.

Canonical training remains gated on review of this code and the Drive-backed preflight result.

## Local-SSD staging and resumable execution

Google Drive remains the canonical benchmark source and Stage 4M output destination. Before Stage 04, the launcher deterministically stages only `train_known`, P0, P1, P2, and P3 into 1,024-row uncompressed HDF5 shards under `/content/wisig_stage4m_local_v1_0_0`. The expected frozen layout is 633 shards. Every shard retains canonical global indices and domain metadata and is verified against a Drive-persisted manifest. Calibration Unknown is excluded and can be staged separately only after the canonical selection lock; strict/final partitions are rejected before source data are opened.

Run the explicit modes in the same Colab runtime:

```python
import subprocess, sys
launcher = "/content/Surrogate-XAI/Stage4M_Colab_Launcher_v1_0_0.py"
common = [sys.executable, "-u", launcher, "--repository-root", "/content/Surrogate-XAI"]
subprocess.check_call(common + ["--stage-local-data"])
subprocess.check_call(common + ["--verify-local-data"])
subprocess.check_call(common + ["--run", "--resume"])
```

Canonical results and recovery state remain under Drive `06_surrogate_kd`. Every completed epoch atomically updates `latest.pt`, `history.csv`, and `epoch_status.json`, with `best.pt` updated only on improvement. `STAGE4M_LIVE_PROGRESS.json` records lifecycle events and `STAGE4M_HEARTBEAT.json` is refreshed every 25 batches. A `KeyboardInterrupt` records `STAGE4M_INTERRUPTED.json` and resumes from the last completed epoch; it is not classified as a scientific failure.

## AMP-overflow recovery hotfix

The canonical Colab run exposed a CUDA-AMP overflow in K0/seed 42 after epoch 1. The hotfix preserves the frozen scientific design and AMP setting while treating isolated fp16 overflows as bounded numerical events: the underlying optimizer update is skipped, GradScaler backs off, counters are persisted, and the next batch proceeds. More than 32 consecutive overflows abort. The old epoch-1 checkpoint is intentionally rejected under the new executable/provenance and K0/seed 42 restarts cleanly.

After updating the branch, rerun Stage 4M preflight because preflight is executable-hash bound. Only after a new `STAGE4M_PREFLIGHT_PASS` should canonical training be restarted with normal resume enabled; Stages 01-03 are regenerated under the new executable, and the old K0 seed-42 checkpoint is audit-rejected automatically.
