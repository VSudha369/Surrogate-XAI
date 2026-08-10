# Stage 1B — WiSig ManyTx Zero-Day Benchmark Engineering

Canonical source: `Stage1B_WiSig_ManyTx_ZeroDay_Benchmark_v1_0_3.py`

- Stage version: `Stage 1B v1.0.3`
- Benchmark: `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`
- Canonical source SHA-256: `2bb03d9fe8f5663f7baeddd334968bf2135e1472827e13b843fcce710ff77581`
- Source ManyTx SHA-256: `a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e`
- Frozen benchmark SHA-256: `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`
- Status: **FROZEN / READ ONLY**

Stage 1B constructs the leakage-safe ManyTx benchmark only. It does not train a classifier, surrogate, zero-day detector, or XAI system. It preserves the native 256-sample I/Q observation length, creates deterministic known/calibration-unknown/strict-zero-day transmitter partitions, defines P0–P5 domain protocols, freezes mappings/splits/provenance, and enforces final-test-only access.

The frozen output is external to GitHub because datasets/HDF5 artifacts are intentionally excluded from source control.
