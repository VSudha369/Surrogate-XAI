# Stage 3.5M — WiSig ManyTx zero-day/open-set detection

This package implements post-hoc open-set scoring under the scientifically frozen Stage 3M A3 seed-123 teacher. It does not retrain or modify the teacher and does not add an unknown training class.

Five deterministic scorers are compared: MSP, energy, normalized-prototype cosine distance, regularized Mahalanobis distance, and diagonal-Gaussian embedding NLL. Train Known fits embedding statistics; P0-P3 Known Validation freezes strict thresholds. Before strict access, the pipeline verifies Stage 3M→Stage 3.5M known-inference equivalence and freezes a recursively hash-bound P0-P3 score bundle. Calibration Unknown is optional and isolated as `ZD-CALIBRATED`. Strict data is inaccessible until the Stage-08 evaluation lock is written and verified; Stage 08 then creates a separate lock- and sealed-index-bound strict score bundle required by all later stages.

The strict shift partition is a nested 3,000-row sensitivity subset of the 216,000-row main strict test. It is never concatenated with main. The reviewed run already produced both immutable strict score stores under the original lock before aborting on the former disjointness assumption. Use only the dedicated recovery preflight for that run; ordinary `--resume` is intentionally invalid after the executable hotfix:

```bash
python -u Stage3_5M_PostLock_Recovery_v1_0_0.py --branch-root "$WISIG_BRANCH_ROOT" --repository-root /content/Surrogate-XAI --preflight
```

Do not invoke recovery finalization until the GitHub and recovery-preflight reviews approve it.

The launcher resolves the canonical Drive root in this order: explicit `--branch-root`, `WISIG_BRANCH_ROOT`, then exactly-one recursive discovery under `/content/drive/MyDrive`. It refuses zero or multiple valid roots. A read-only audit is available with `--drive-audit`; it performs no inference, fitting, deletion, or READY creation. For the reviewed consumed-strict state, the launcher prints `POST_LOCK_RECOVERY_REQUIRED` and will not dispatch ordinary `--resume`. It can dispatch only the read-only audit with `--postlock-recovery-preflight`; finalization is never automatic.

## Colab sequence

Run preflight first. It executes only Stages 01-02 and performs no signal inference:

```bash
python -u Stage3_5M_Colab_Launcher_v1_0_0.py --preflight
```

The ordinary resumable command below applies only to a fresh pre-strict execution. It is forbidden for the current reviewed post-lock recovery:

```bash
python -u Stage3_5M_Colab_Launcher_v1_0_0.py --resume
```

Do not claim Stage 3.5M complete until the full run produces `MANYTX_STAGE3_5M_READY.txt` and Drive outputs are independently audited.

Recovery finalization is interruption-safe and idempotent. READY and its final manifests are verified before Stage 11 is durably committed, and the original NOT_READY marker is deleted only after that checkpoint verifies. If interruption occurs after the checkpoint, the next invocation performs cleanup only; if the transaction is already complete, it exits without rewriting outputs.
