# Stage 4M Stage-03 local-SSD activation hotfix

This additive runtime hotfix is layered on the validated Stage 4M scientific executable from commit `d301b88404fe99bf6efd531cedd45e28eecf8f4d`.

## Purpose

The base local-sharding implementation activates the verified `/content/wisig_stage4m_local_v1_0_0` cache at Stage 04. Stage 03 also performs substantial signal inference while preparing clean teacher caches, so it must use the same local shards. This hotfix moves the runtime activation gate to Stage 03 without changing the scientific configuration.

## Safety invariants

- Base executable SHA-256 is fixed to `9f1889f2cd5efa2d38776f7524c52ce6989379c7717501ae9a9239949f8550d0`.
- Scientific configuration SHA-256 remains `78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80`.
- Stage 03+ ordinary loaders require `sharded_local`.
- Teacher-target caches are reusable only when they bind the local-cache identity, aggregate content hash, and runtime I/O policy.
- Drive reads are allowed for one-time shard staging, but training/evaluation signal hot-path reads from Drive must remain zero.
- Strict/final-test protections, K0-K3 objectives, seeds, optimizer, augmentation, AMP overflow policy, P0 selection, and READY semantics are inherited unchanged from the validated base.

## Canonical Colab order

1. run the hotfix validator;
2. hotfix preflight;
3. stage authorized local shards;
4. verify local shards;
5. canonical `--run --resume` through the hotfix launcher;
6. monitor Drive from a separate notebook/runtime with `Stage4M_Live_Drive_Monitor_v1_0_0.py`.

Do not execute the base launcher for the canonical Stage 4M run after this hotfix is adopted.
