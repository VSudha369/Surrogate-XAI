# Stage 2.6M Design Status

**Stage 2.6M — WiSig ManyTx Controlled Representation Learning & Domain-Robust Embedding Ablation**

Status: **design frozen; canonical implementation pending**.

The current ManyTx Stage 2.6M must not reuse the archived RadioML Stage 2.6 implementation as the active pipeline.

## Controlled arms

- A0 — CE
- A1 — CE + supervised contrastive loss
- A2 — CE + prototype compactness loss
- A3 — CE + supervised contrastive + prototype loss

All arms must share the same backbone, input (`2 x 256`), 98 known Tx classes, sampler, optimizer/scheduler family, seed panel, training budget, validation protocols, augmentation policy, and evaluation code.

## Selection evidence

Primary known-validation evidence comes from P0/P1/P2/P3. P2/P3 must additionally report fixed 98-class-frame metrics because source coverage omits some Tx identities in those domains. Calibration Unknown is secondary diagnostic evidence. Strict zero-day remains completely inaccessible.

## Stage 2M motivation

Stage 2M found weak handcrafted transmitter separability, measurable receiver/combined shift, very strong equalization-domain separation, and near-chance calibration-unknown novelty separation. Stage 2.6M therefore tests whether learned discriminative/compact embeddings address these bottlenecks without architecture search.
