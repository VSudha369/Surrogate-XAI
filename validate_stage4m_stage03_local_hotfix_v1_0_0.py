#!/usr/bin/env python3
"""Regression validator for the Stage 4M Stage-03 local-SSD activation hotfix."""
from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

EXPECTED_BASE_SHA256 = "9f1889f2cd5efa2d38776f7524c52ce6989379c7717501ae9a9239949f8550d0"
EXPECTED_BASE_VALIDATOR_SHA256 = "d118737c91a45c1159f8cc5f1337ea5b37d994e6df5c2d03330ae6d3ecb71bf6"
EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"
BASE = "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.py"
BASE_VALIDATOR = "validate_stage4m_v1_0_0.py"
HOTFIX = "Stage4M_Stage03_Local_Activation_Hotfix_v1_0_0.py"
LAUNCHER = "Stage4M_Colab_Stage03_Local_Hotfix_Launcher_v1_0_0.py"
MONITOR = "Stage4M_Live_Drive_Monitor_v1_0_0.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--skip-base-validator", action="store_true")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    base = root / BASE
    base_validator = root / BASE_VALIDATOR
    hotfix = root / HOTFIX
    launcher = root / LAUNCHER
    monitor = root / MONITOR

    tests = []
    def check(name: str, condition: bool) -> None:
        tests.append((name, bool(condition)))
        if not condition:
            raise AssertionError(name)

    for path in (base, base_validator, hotfix, launcher, monitor):
        check(f"exists:{path.name}", path.is_file())
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        check(f"ast:{path.name}", True)

    check("base executable frozen", sha256(base) == EXPECTED_BASE_SHA256)
    check("base validator frozen", sha256(base_validator) == EXPECTED_BASE_VALIDATOR_SHA256)

    source = hotfix.read_text(encoding="utf-8")
    launch_source = launcher.read_text(encoding="utf-8")
    monitor_source = monitor.read_text(encoding="utf-8")

    check("base SHA bound", EXPECTED_BASE_SHA256 in source)
    check("scientific config SHA bound", EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 in source)
    check("Stage03 activation threshold", "if stage >= 3 and not self.local_cache_active" in source)
    check("Stage03 local loader guard", "Stage {stage:02d} requires verified local shards" in source)
    check("non-local Stage03 loader abort", "Drive signal hot path forbidden" in source)
    check("teacher cache local provenance", '\"signal_source_backend\": \"sharded_local\"' in source)
    check("teacher cache binds identity", '\"local_cache_identity_sha256\"' in source)
    check("teacher cache binds aggregate", '\"local_cache_aggregate_sha256\"' in source)
    check("teacher cache binds runtime policy", '\"runtime_io_policy_sha256\"' in source)
    check("evaluation Drive counter exists", '\"evaluation_signal_reads_drive\"' in source)
    check("training Drive counter gate", '\"training_signal_reads_drive\"' in source)
    check("evaluation Drive counter gate", "Canonical evaluation used the Drive signal hot path" in source)
    check("staging Drive explicitly allowed", '\"staging_drive_signal_reads_allowed\": True' in source)
    check("launcher calls hotfix", HOTFIX in launch_source)
    check("launcher supports staging", '\"--stage-local-data\"' in launch_source)
    check("launcher supports verification", '\"--verify-local-data\"' in launch_source)
    check("launcher supports run", '\"--run\"' in launch_source)
    check("monitor reads progress", "STAGE4M_LIVE_PROGRESS.json" in monitor_source)
    check("monitor reads heartbeat", "STAGE4M_HEARTBEAT.json" in monitor_source)
    check("monitor reads IO counters", "STAGE4M_IO_COUNTERS.json" in monitor_source)
    check("monitor does not use nvidia-smi", "nvidia-smi" not in monitor_source)

    if not args.skip_base_validator:
        subprocess.check_call([sys.executable, "-u", str(base_validator)])
    subprocess.check_call([sys.executable, "-u", str(hotfix), "--synthetic-validation"])

    passed = sum(ok for _, ok in tests)
    print(f"STAGE4M_STAGE03_LOCAL_HOTFIX_VALIDATION_PASS ({passed}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
