# Stage 3M v1.0.0 Validation Checklist

## Offline source validation

- [ ] Python compilation and AST parsing pass for executable, launcher, Drive-audit module, and validator.
- [ ] Pyflakes passes.
- [ ] Notebook JSON parses as nbformat 4.
- [ ] Code-manifest hashes and sizes match.
- [ ] ZIP CRC and byte-for-byte member checks pass.
- [ ] Frozen constants, A3-only seed panel, dimensions, loss, temperature, and momentum match.
- [ ] Stage 2.6M and Stage 3M parameter count, state keys, tensor shapes, forward shapes, embedding normalization, and deterministic evaluation match.
- [ ] Correct synthetic A3 checkpoint is accepted; A0/A1/A2, wrong seed/hash/config/architecture/loss, and corrupt state are rejected.
- [ ] All five deterministic selection hierarchy cases pass under shuffled row order.
- [ ] P0 representation geometry passes with 98 observed classes and aborts with 97; sparse non-dense P2/P3 class panels pass with finite observed-class geometry.
- [ ] Representation sampling reruns reproduce identical selected positions and identical position/global-index SHA-256 values.
- [ ] Five strict-access kinds are rejected and a pristine production guard remains all-zero.
- [ ] Partial/stale embedding stores are invalidated and complete hash/row-current stores are reusable.
- [ ] No invalid `%,d` format exists and representative runtime formatting passes.
- [ ] READY gate fails when any one required gate is false.
- [ ] Stage 10 is stale when READY is absent or the final manifest is absent/corrupt, and current only when every final product/hash is valid.
- [ ] `--preflight` maps to Stages 01–03 only and cannot evaluate, select, export, execute Stage 10, or create READY.

- [ ] `--drive-audit` cannot instantiate the execution pipeline, run Stage 04+, train, select/export, or create READY.
- [ ] Missing READY/hash/checkpoint/checkpoint-metadata evidence fails; all five structured strict-counter keys are mandatory.
- [ ] P0 with 97 classes fails, P2/P3 missing identities pass, rounded report prose passes, and a scientific ranking disagreement fails.

Run:

```bash
python validate_stage3m_v1_0_0.py
python Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.py --synthetic-validation
```

## Canonical Colab validation

- [ ] Pull the reviewed `codex/stage3m` commit and install `requirements_stage3m.txt`.
- [ ] `--drive-audit` writes five audit artifacts, reports `STAGE3M_DRIVE_AUDIT_PASS`, and creates no READY marker.
- [ ] `--preflight` verifies the real benchmark, Stage 2M, Stage 2.6M READY/artifact, all three A3 checkpoints, and architecture equivalence.
- [ ] Full `--resume` run completes P0–P3 metrics and representation diagnostics for every seed.
- [ ] Selection is determined only by the predeclared hierarchy.
- [ ] Canonical weights are element-identical to the selected Stage 2.6M checkpoint.
- [ ] Publication artifacts are generated from measured results.
- [ ] Stage 2.6M files and checkpoint hashes are unchanged.
- [ ] All five strict counters are zero and forbidden artifacts are absent.
- [ ] `MANYTX_STAGE3M_READY.txt` exists and `MANYTX_STAGE3M_NOT_READY.txt` does not.

Until every canonical Colab box passes, report Stage 3M as implemented but not scientifically frozen.
