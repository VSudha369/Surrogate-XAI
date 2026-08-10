# Surrogate-XAI

Research code for **Ultra-Low Latency Surrogate-Assisted Explainable AI for Zero-Day Threat Detection in Non-Stationary Physical Layer Communications**.

## Project status

The repository is organized around two clearly separated research branches:

- **Active ManyTx zero-day branch** — WiSig ManyTx benchmark engineering and diagnostics for transmitter-centric zero-day evaluation under receiver/day/equalization shift.
- **Legacy RadioML branch** — earlier AMC-focused diagnostics and representation-ablation code retained only for provenance and auxiliary low-SNR robustness work.

Current active progression:

| Stage | Status | Canonical artifact |
|---|---|---|
| Stage 1B | Frozen | `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3` |
| Stage 2M | Frozen / READY | Diagnostics v1.0.5 |
| Stage 2.6M | Design frozen; implementation pending | Controlled CE / SupCon / Prototype ablation |
| Stage 3M+ | Not yet canonical | Pending Stage 2.6M selection |

## Repository layout

```text
src/
  manytx/
    stage1b_benchmark/       # Canonical ManyTx benchmark-engineering source
    stage2m_diagnostics/     # Canonical ManyTx scientific diagnostics source
legacy/
  radioml/
    stage2_5_diagnostics/    # Superseded RadioML diagnostics
    stage2_6_representation/ # Superseded RadioML representation ablation
notebooks/
  manytx/stage2m/            # Thin Colab launcher notebook
docs/
  PROJECT_STATUS.md
  TEST_ACCESS_POLICY.md
  REPRODUCIBILITY.md
  ARTIFACT_REGISTRY.md
  STAGE2_6M_DESIGN.md
```

## Scientific data policy

Datasets, benchmark HDF5 files, checkpoints, generated embeddings, final-test predictions, and large experiment outputs are intentionally **not stored in GitHub**. Canonical artifacts are identified by hashes and reproduced/validated against external storage.

The strict zero-day test partition is protected throughout model development. Training, representation selection, threshold selection, and diagnostics must not use strict-zero-day signals or labels before final model selection is frozen.

## Canonical hashes

- WiSig ManyTx source SHA-256: `a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e`
- Stage 1B benchmark SHA-256: `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`
- Canonical Stage 2M script SHA-256: `46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba`
- Stage 2M artifact hash-manifest SHA-256: `0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a`

## Active Stage 2M conclusion

Stage 2M completed all safety gates with zero strict-test reads and recommended:

```text
PROCEED_STAGE_2_6M_WITH_CAUTION
```

The next active experiment is a **single-backbone controlled representation ablation** using CE, CE+SupCon, CE+Prototype, and CE+SupCon+Prototype. The older RadioML Stage 2.6 code in `legacy/` must not be used as the active ManyTx Stage 2.6M pipeline.

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and the stage-specific manifests/checklists. All canonical stages use explicit versioning, SHA-256 provenance, deterministic seeds where practical, resume-safe checkpoints, and frozen-test discipline.

## Contributing

Please keep active and legacy branches separated, never commit datasets/checkpoints/test predictions, and use pull requests for scientifically meaningful changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
