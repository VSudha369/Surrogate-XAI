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

## AMP-overflow recovery hotfix

The canonical Colab run exposed a CUDA-AMP overflow in K0/seed 42 after epoch 1. The hotfix preserves the frozen scientific design and AMP setting while treating isolated fp16 overflows as bounded numerical events: the underlying optimizer update is skipped, GradScaler backs off, counters are persisted, and the next batch proceeds. More than 32 consecutive overflows abort. The old epoch-1 checkpoint is intentionally rejected under the new executable/provenance and K0/seed 42 restarts cleanly.

After updating the branch, rerun Stage 4M preflight because preflight is executable-hash bound. Only after a new `STAGE4M_PREFLIGHT_PASS` should canonical training be restarted with normal resume enabled; Stages 01-03 are regenerated under the new executable, and the old K0 seed-42 checkpoint is audit-rejected automatically.
