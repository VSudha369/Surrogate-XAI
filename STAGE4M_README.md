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

Preflight verifies Drive/root uniqueness, predecessor identities, benchmark schema, teacher identity/forward contract, strict guards, GPU, and output containment. It performs no training, cache generation, selection, Calibration Unknown access, strict access, or READY creation.

Canonical training remains gated on review of this code and the Drive-backed preflight result.
