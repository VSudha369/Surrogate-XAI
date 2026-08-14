# Stage 4M v1.0.0 validation checklist

- [x] Read-only Drive predecessor audit completed before code generation.
- [x] Benchmark, Stage 2M, Stage 2.6M, Stage 3M, teacher, and Stage 3.5M metadata hashes fixed.
- [x] Stage 3.5M metadata allowlist and strict-result denylist enforced.
- [x] One half-width student; deployed parameters ≤40% of teacher; 64-D native embedding; 98 logits.
- [x] K0-K3 formulas, T=4, seeds, optimizer, scheduler, and training budget frozen.
- [x] Teacher eval/frozen/detached and excluded from optimizer.
- [x] K1-K3 teacher/student receive the identical once-augmented tensor; K0 performs no teacher training forward.
- [x] Clean Train Known cache rejected as sample-level KD; retained only for K3 class prototypes.
- [x] Train Known-only training; P0-only selection; P1-P3 reporting-only.
- [x] Calibration Unknown blocked until immutable selection lock.
- [x] Exact resume binds RNG, loader, model, optimizer, scheduler, objective, training-target policy, architecture, teacher, benchmark, predecessor, selection, all input hashes, and output hashes/sizes.
- [x] Teacher caches allow only Train Known and P0-P3.
- [x] Compression and minimum P0 fidelity gates block READY.
- [x] Stage-12 final transaction is interruption-safe, idempotent, and protected from later failed invocations.
- [x] Preflight cannot train, cache targets, access Calibration Unknown/strict data, or create READY.
- [x] Validator includes the A-O KD matching, dependency-mutation, provenance hydration, READY protection, and preflight-scope matrices.
- [x] Frozen Stage 3.5M, Stage 3M, and Stage 2.6M validators rerun after packaging.
