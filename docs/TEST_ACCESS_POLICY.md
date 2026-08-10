# Test Access Policy

The strict zero-day partition is reserved for final evaluation only.

## Forbidden before final selection lock

- reading strict-zero-day I/Q samples;
- reading strict-zero-day labels for modeling or diagnostics;
- generating strict-zero-day embeddings or novelty scores;
- tuning model architecture, losses, hyperparameters, thresholds, or XAI using strict-zero-day data;
- reporting strict-zero-day performance during development.

## Allowed during development

- Train Known for fitting/training;
- P0/P1/P2/P3 known validation for model selection and domain robustness;
- Calibration Unknown for explicitly labeled diagnostic analyses only, subject to stage protocol;
- verifying strict-test file existence/count/hash without loading strict sample indices/signals.

Final test access requires a frozen selection manifest/checkpoint/threshold policy.
