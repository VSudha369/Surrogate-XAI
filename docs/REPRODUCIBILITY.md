# Reproducibility

## Canonical provenance

Every active stage must record:

- source/benchmark SHA-256;
- code/script SHA-256;
- configuration SHA-256;
- seed values;
- package/runtime versions;
- split/mapping identities;
- completion/READY markers;
- output hash manifests.

## Data location

Data are not versioned in GitHub. Reproduction requires the externally stored WiSig ManyTx source and the frozen Stage 1B benchmark. Verify hashes before running any later stage.

## Execution order

1. Stage 1B benchmark engineering — already frozen; do not regenerate unless intentionally creating a new benchmark version.
2. Stage 2M diagnostics — already frozen; use as evidence/provenance for Stage 2.6M.
3. Stage 2.6M controlled representation ablation — next active implementation.

## Code validation

The repository bootstrap validated canonical Python sources with `python -m py_compile` and validated committed notebook JSON. No common GitHub/AWS/Google/OpenAI credential token patterns were detected in the canonical stage sources during bootstrap.
