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
python -u Stage3M_Colab_Launcher_v1_0_0.py --preflight
python -u Stage3M_Colab_Launcher_v1_0_0.py --teacher-source stage2_6m_promote --resume
```

Set `WISIG_BRANCH_ROOT` only if Drive contains multiple matching branch roots. The value must end in `MANYTX_ZERO_DAY_BRANCH_v1.0.3`. The launcher accepts paths containing spaces.

Preflight runs Stages 01–03 and cannot create READY. The full command uses hash-bound resume and evaluates the real Drive benchmark. It neither resets nor modifies Stage 2.6M.

See [STAGE3M_SCIENTIFIC_PROTOCOL.md](STAGE3M_SCIENTIFIC_PROTOCOL.md), [STAGE3M_OUTPUT_SCHEMA.md](STAGE3M_OUTPUT_SCHEMA.md), and [STAGE3M_VALIDATION_CHECKLIST.md](STAGE3M_VALIDATION_CHECKLIST.md).
