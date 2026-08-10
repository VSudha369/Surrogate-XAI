# Contributing

## Branch and review policy

Use a dedicated branch and pull request for all non-trivial changes. Keep commits focused and describe the scientific reason for each change.

## Scientific invariants

1. Never modify frozen Stage 1B benchmark artifacts in place.
2. Never use strict-zero-day test signals/labels for training, model selection, threshold selection, representation diagnostics, or hyperparameter tuning.
3. Keep calibration-unknown results explicitly labeled as diagnostic, not final zero-day performance.
4. Do not silently change splits, mappings, seeds, preprocessing, or benchmark hashes.
5. Preserve active ManyTx and legacy RadioML code as separate branches/directories.

## Repository hygiene

Do not commit datasets, HDF5 benchmarks, checkpoints, generated embeddings, predictions, large reports, or archives. Store hashes/manifests and reproduction instructions instead.

## Validation before PR

At minimum:

- compile modified Python files;
- validate notebook JSON when notebooks change;
- scan for accidental secrets/credentials;
- verify benchmark/test-access guards for scientific-stage changes;
- update provenance/status documentation when a canonical version changes.
