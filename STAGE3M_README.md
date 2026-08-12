# Stage 3M v1.0.0 — WiSig ManyTx Canonical Teacher Freeze

Stage 3M promotes and independently verifies the three frozen Stage 2.6M A3 checkpoints, selects one seed using known validation only, and exports an immutable downstream teacher. The default and only canonical source is `stage2_6m_promote`; no model training is implemented.

## Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
rm -rf /content/Surrogate-XAI
git clone --branch codex/stage3m --single-branch https://github.com/VSudha369/Surrogate-XAI.git /content/Surrogate-XAI
cd /content/Surrogate-XAI
python -m pip install -q -r requirements_stage3m.txt
python validate_stage3m_v1_0_0.py
python -u Stage3M_Colab_Launcher_v1_0_0.py --drive-audit
python -u Stage3M_Colab_Launcher_v1_0_0.py --preflight
python -u Stage3M_Colab_Launcher_v1_0_0.py --teacher-source stage2_6m_promote --resume
```

Set `WISIG_BRANCH_ROOT` only if Drive contains multiple matching branch roots. The value must end in `MANYTX_ZERO_DAY_BRANCH_v1.0.3`. The launcher accepts paths containing spaces.

Preflight runs Stages 01–03 and cannot evaluate candidates, select/export a teacher, execute Stage 10, or create READY. The full command uses hash-bound resume and evaluates the real Drive benchmark. It neither resets nor modifies Stage 2.6M.

Representation geometry is explicitly observed-class geometry. P2/P3 may contain fewer than 98 transmitter identities; coverage and deterministic sampling hashes are persisted per seed/protocol. P0 must contain all 98 classes because P0 Fisher ratio is the frozen fourth-level selection tie-break. Stage 10 is transaction-safe: the final hash manifest and READY marker verify before its atomic completion checkpoint is written.

Drive audit recursively inventories the frozen Stage 1B, Stage 2M, and Stage 2.6M trees; verifies real report/table schemas, all five strict counters, known-domain class coverage, A3 checkpoint provenance on CPU, and Stage 2.6M A3 embedding stores; and writes only `04_canonical_teacher/audit/`. It cannot evaluate signals, select/export a teacher, or create READY. Preflight requires this audit to pass before Stages 01-03.

The Drive audit measured P0=98 classes, P1=98, P2=97 (missing class 72), and P3=93 (missing 50, 52, 58, 71, 72); every observed class has at least two samples. These are predecessor facts, not hard-coded substitutes for Stage 3M measurement.

See [STAGE3M_SCIENTIFIC_PROTOCOL.md](STAGE3M_SCIENTIFIC_PROTOCOL.md), [STAGE3M_OUTPUT_SCHEMA.md](STAGE3M_OUTPUT_SCHEMA.md), and [STAGE3M_VALIDATION_CHECKLIST.md](STAGE3M_VALIDATION_CHECKLIST.md).
