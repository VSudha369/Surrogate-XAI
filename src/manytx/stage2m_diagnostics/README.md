# Stage 2M — WiSig ManyTx Scientific Benchmark Diagnostics v1.0.5

Version 1.0.5 is an implementation-ready, CPU-first diagnostics and zero-day learnability audit for the frozen `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`. It performs no deep teacher training, benchmark mutation, final-test evaluation, architecture search, surrogate training, or threshold tuning on strict-zero-day identities.

## Deliverables

- `Stage2M_WiSig_ManyTx_Scientific_Diagnostics_v1_0_5.py` — standalone ten-stage pipeline.
- `Stage2M_WiSig_ManyTx_Scientific_Diagnostics_v1_0_5.ipynb` — one-cell Colab notebook.
- `Stage2M_Colab_Launcher_v1_0_5.py` — subprocess launcher that prevents Jupyter `-f kernel.json` arguments from reaching the pipeline.
- `SCIENTIFIC_PROTOCOL.md`, `OUTPUT_SCHEMA.md`, and `VALIDATION_CHECKLIST.md` — scientific and engineering contracts.
- `CODE_MANIFEST.json` and `requirements.txt` — code provenance and dependencies.

The implementation was prepared from the supplied master prompt and reviewed against the conventions in `Surrogate_XA..I.ipynb`. No benchmark results are included or invented; they are generated only when the canonical benchmark is present and its SHA-256 matches.

## Fastest Colab use

1. Upload the standalone script and launcher to the project area in Google Drive.
2. Set `RF_PROJECT_ROOT` if the project is not at `/content/drive/MyDrive/colab files /Surrogate-XAI/project_root`.
3. Run the single cell in the supplied notebook, or paste the launcher into one Colab cell.

The launcher mounts Drive only when needed, locates the script and canonical benchmark, verifies the benchmark SHA, installs only missing packages, and starts the standalone program with `subprocess.check_call`.

## Command-line use

```bash
python Stage2M_WiSig_ManyTx_Scientific_Diagnostics_v1_0_5.py \
  --project-root "/content/drive/MyDrive/colab files /Surrogate-XAI/project_root" \
  --copy-local
```

Useful deterministic controls:

```text
--start-stage 1 --end-stage 10
--chunk-size 4096
--max-fit-samples 120000
--max-domain-per-tx 300
--max-cluster-samples 5000
--bootstrap-replicates 500
--linear-epochs 2
--force
```

`--force` invalidates validated checkpoints and rebuilds outputs. Without it, a stage resumes only when the benchmark hash, configuration hash, required files, and their hashes match its atomic checkpoint.

## Immutable input contract

The program requires:

```text
<project_root>/MANYTX_ZERO_DAY_BRANCH_v1.0.3/
└── 01_benchmark_engineering/
    ├── benchmark/WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5
    ├── train_known_indices.npy and the five other allowed split index files
    ├── strict_zero_day_test_indices.npy
    └── strict_zero_day_shift_test_indices.npy
```

It verifies the canonical HDF5 SHA-256 exactly:

```text
9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9
```

Stage 1B provenance documents must retain the expected source SHA and the documentary record of 5,307 extreme-amplitude observations. Stage 1B did not store their per-sample membership in the normalized HDF5, so v1.0.5 never invents or imputes that membership. It separately reads `signal_integrity_row_audit.csv` by canonical row order, parses only allowed rows, and fits a one-sided robust log-RMS rule on Train Known only. Strict rows remain opaque binary lines and are never decoded or CSV-parsed.

Exact frozen mappings are validated at `mappings/transmitter_id_to_index.json` (98 known transmitters), `mappings/receiver_id_to_index.json` (18 receivers), and `mappings/day_id_to_index.json` (4 days), including contiguous bijective index values.

Train Known must contain exactly 98 identities, and Calibration Unknown must contain exactly 22 identities disjoint from Train Known. P0/P1/P2/P3 are validated as nonempty subsets of Train Known rather than forced to contain all 98 identities. Thus P2's observed 97/98 coverage is reported explicitly instead of treated as benchmark corruption. Protocol-specific macro metrics exclude identities with no observations; fixed-98-class-frame metrics are also emitted separately. Missing-domain centroids are recorded as unavailable, never computed from a synthetic zero vector.

## Strict-test protection

The two strict index arrays are never loaded, memory-mapped, or used for overlap computation. Their existence and SHA are checked as files; their counts are verified only from frozen Stage 1B manifests. Every HDF5 signal, label/metadata, and feature read requires an allowed `SplitRef` capability. `StrictTestGuard` rejects strict or unknown roles and records three counters that must all remain zero. The raw-RMS audit additionally records `strict_or_other_forbidden_rows_parsed = 0` and READY verifies it.

Checkpoints bind the benchmark SHA, normalized configuration SHA, Stage 2M version, standalone script SHA, and every output hash. Any prior-version checkpoint, configuration, or unversioned partial output is explicitly recorded as incompatible and forces a Stage 01 rebuild; starting v1.0.5 from a later stage in that condition aborts.

## Output boundary

All generated artifacts are confined to:

```text
<project_root>/MANYTX_ZERO_DAY_BRANCH_v1.0.3/02_benchmark_diagnostics/
```

The optional local benchmark copy goes to Colab scratch, is hash-verified, opened read-only, and is not a scientific output. Stage 1B is never written.

## Computational profile

I/Q tensors are read in deterministic chunks and never loaded wholesale. Full feature banks are written as NumPy memory-mapped arrays. Expensive pairwise analyses use persisted, deterministic transmitter-stratified samples. CPU is sufficient; GPU is not used. Runtime and output size depend mainly on Drive I/O, CPU count, and compression. Local scratch is strongly recommended in Colab.

## Status semantics

`MANYTX_STAGE2M_READY.txt` is emitted only after all 20 factual safety/completeness gates pass. Model accuracy or AUROC is never a READY gate. Weak learnability is valid evidence. A critical exception removes any stale READY marker and writes `MANYTX_STAGE2M_NOT_READY.txt` plus a structured final-status manifest.

Calibration-unknown outputs always state:

```text
CALIBRATION-UNKNOWN DIAGNOSTIC RESULT — NOT FINAL ZERO-DAY TEST PERFORMANCE
```
