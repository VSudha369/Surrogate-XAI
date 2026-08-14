# Stage 4M v1.0.0 output schema

All canonical output is contained by `<canonical-root>/06_surrogate_kd`.

- `configs/`: frozen runtime/scientific configuration.
- `manifests/STAGE4M_PREDECESSOR_LOCK.json`: hashes only the approved Stage 1B–3.5M continuity evidence.
- `manifests/STAGE3M_STAGE4M_TEACHER_EQUIVALENCE.json`: independent P0-P3 teacher equivalence.
- `manifests/STUDENT_ARCHITECTURE_FREEZE.json`: complete layer table and compression gate.
- `manifests/KD_OBJECTIVE_POLICY.json`: exact T=4 objective formulas and coefficients.
- `manifests/TRAINING_TARGET_POLICY.json`: shared single-augmentation, online sample-matched KD, K0 no-teacher-forward, and clean prototype-source contract.
- `manifests/AMP_RUNTIME_SAFETY_POLICY.json`: fixed CUDA-AMP overflow backoff/skip policy, 32-consecutive-overflow ceiling, epoch scheduler scope, norm-5 clipping, and explicit pre-hotfix checkpoint restart policy.
- `manifests/STAGE4M_LOCAL_IO_POLICY.json`: frozen runtime-only Drive/local/output routing policy.
- `manifests/STAGE4M_LOCAL_DATA_CACHE_MANIFEST.json`: executable-bound shard inventory, source identities, per-shard hashes, and coverage evidence.
- `manifests/STAGE4M_LOCAL_DATA_IDENTITY.json`: stable cache-content identity used by resumable checkpoints.
- `manifests/STAGE4M_DATA_STAGING_STATUS.json`: atomic `INITIALIZING/COPYING/VERIFYING/READY/FAILED` staging state.
- `manifests/STAGE4M_LIVE_PROGRESS.json`, `STAGE4M_HEARTBEAT.json`: atomic execution status and 25-batch heartbeat.
- `manifests/STAGE4M_IO_COUNTERS.json`: local/Drive training and evaluation read evidence.
- `manifests/STAGE4M_INTERRUPTED.json`: safe last-completed-epoch recovery evidence after `KeyboardInterrupt`.
- `checkpoints/K0..K3/seed_<seed>/`: exact-resume `latest.pt`, P0-selected `best.pt`, `history.csv`, and `epoch_status.json` persisted after every completed epoch.
- `manifests/CANONICAL_SURROGATE_SELECTION.json`: P0-only evidence and deterministic ranking.
- `manifests/CANONICAL_SURROGATE_SELECTION_LOCK.json`: immutable canonical model identity.
- `checkpoints/canonical/`: training checkpoint, deploy checkpoint without auxiliary components, and state-dict.
- `tables/KD_SEED_RESULTS.csv`, `KD_ARM_SUMMARY.csv`: three-seed P0 evidence.
- `tables/KNOWN_DOMAIN_FIDELITY.csv`, `REPRESENTATION_FIDELITY.csv`: post-selection P1-P3 diagnostics.
- `tables/ZD_CALIBRATED_DIAGNOSTIC.csv`: separately labelled Calibration Unknown diagnostics.
- `performance/`: compression and preliminary latency; neither affects selection.
- `publication/`: workbook and PDF generated from measured Stage 4M outputs.
- `manifests/STAGE4M_FINAL_STATUS.json`, `STAGE4M_HASH_MANIFEST.json`, and `MANYTX_STAGE4M_READY.txt`: final transaction products.

Authorized signal shards and local teacher caches live under `/content/wisig_stage4m_local_v1_0_0`, never contain strict rows, and are scientifically disposable. The ordinary cache contains only Train Known and P0-P3. Calibration Unknown uses a separately manifested post-lock namespace. Clean Train Known teacher-cache data are prototype-only and never sample-level KD targets; P0-P3 caches are unaugmented evaluation/fidelity evidence.

Each `STAGE_<NN>_CHECKPOINT.json` binds current pipeline/configuration, predecessor, architecture, objective, training-target policy, AMP runtime-safety policy, teacher, benchmark, stage-specific selection lock, every required input SHA-256, and every output SHA-256 plus byte size. Stage 04+ checkpoints additionally bind the runtime I/O policy, stable local-cache identity, and aggregate logical cache hash. Training checkpoints persist model/auxiliary/optimizer/scheduler/scaler/RNG/loader state, exposure SHA, P0 best state, early-stop state, and complete AMP accounting.
