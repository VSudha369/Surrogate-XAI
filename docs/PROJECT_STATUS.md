# Project Status

## Active ManyTx branch

### Stage 1B — Benchmark Engineering

- Canonical version: `WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3`
- Status: **FROZEN / READ ONLY**
- Samples: 1,020,643
- Tx: 150 total = 98 known + 22 calibration unknown + 30 strict zero-day
- Rx: 18
- Capture dates: 4
- Signal: 2 x 256 I/Q
- Benchmark SHA-256: `9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9`

### Stage 2M — Scientific Diagnostics

- Canonical executed version: `1.0.5`
- Status: **MANYTX_STAGE2M_READY / FROZEN**
- Script SHA-256: `46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba`
- Strict-zero-day signals/labels/features accessed: 0 / 0 / 0
- Recommendation: `PROCEED_STAGE_2_6M_WITH_CAUTION`

Key evidence: handcrafted Tx separability is weak; receiver and combined domain shift are measurable; equalization state is a strong nuisance variable; calibration-unknown separation in handcrafted space is near chance.

### Stage 2.6M — Controlled Representation Learning

- Status: **DESIGN FROZEN; CANONICAL CODE NOT YET PUBLISHED**
- Planned arms: CE, CE+SupCon, CE+Prototype, CE+SupCon+Prototype
- One common backbone and identical exposure/budget across arms
- Strict zero-day remains inaccessible

## Legacy RadioML branch

The retained Stage 2.5 and Stage 2.6 implementations are historical AMC/RadioML experiments. They are not the active ManyTx Stage 2M/2.6M pipeline and must not be used for current ManyTx model selection.
