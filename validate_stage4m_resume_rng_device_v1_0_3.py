#!/usr/bin/env python3
"""Regression validator for Stage 4M v1.0.3 RNG-safe resume recovery."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

PARENT = "Stage4M_Resume_Scientific_Current_Hotfix_v1_0_2.py"
RECOVERY = "Stage4M_Resume_RNG_Device_Hotfix_v1_0_3.py"
LAUNCHER = "Stage4M_Colab_Resume_RNG_Device_Launcher_v1_0_3.py"
EXPECTED_PARENT_GIT_BLOB_SHA1 = "b83b34aa0a7ef3b6989f49d8139727fca890c67c"
EXPECTED_CONFIG_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage4m_rng_recovery_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    parent, recovery, launcher = root / PARENT, root / RECOVERY, root / LAUNCHER
    tests = []

    def check(name: str, condition: bool) -> None:
        tests.append((name, bool(condition)))
        if not condition:
            raise AssertionError(name)

    for path in (parent, recovery, launcher):
        check(f"exists:{path.name}", path.is_file())
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        check(f"ast:{path.name}", True)

    source = recovery.read_text(encoding="utf-8")
    launch = launcher.read_text(encoding="utf-8")
    check("exact v1.0.2 parent git blob bound", EXPECTED_PARENT_GIT_BLOB_SHA1 in source)
    check("scientific config bound", EXPECTED_CONFIG_SHA256 in source)
    check("CPU-only RNG normalization", '.to(device="cpu", dtype=base.torch.uint8).contiguous()' in source)
    check("CPU RNG restored after normalization", 'base.torch.set_rng_state(_rng_byte_tensor(payload["torch_cpu"], "CPU"))' in source)
    check("CUDA RNG states normalized", 'base.torch.cuda.set_rng_state_all(normalized)' in source)
    check("CUDA device-count guard", "Saved CUDA RNG device count mismatch" in source)
    check("saved RNG dtype guarded", "RNG state dtype is not uint8" in source)
    check("checkpoint bytes not rewritten", '"checkpoint_mutated": False' in source)
    check("RNG bytes unchanged audit", '"rng_bytes_mutated": False' in source)
    check("science unchanged audit", '"scientific_changes": False' in source)
    check("base restore_rng monkeypatched only", "base.restore_rng = restore_rng_cpu_safe" in source)
    check("launcher points v1.0.3", RECOVERY in launch)
    check("launcher supports preflight", '"--preflight"' in launch)
    check("launcher supports cache verification", '"--verify-local-data"' in launch)
    check("launcher preserves resume default", "resume=True is active" in launch)

    module = load_module(recovery)
    check("scientific configuration unchanged", module.base.Stage4Config().configuration_sha256() == EXPECTED_CONFIG_SHA256)
    fixture = module.base.torch.arange(32, dtype=module.base.torch.uint8)
    normalized = module._rng_byte_tensor(fixture, "fixture")
    check("normalized RNG remains exact bytes", module.base.torch.equal(fixture.cpu(), normalized))
    check("normalized RNG is CPU", normalized.device.type == "cpu")
    check("normalized RNG is uint8", normalized.dtype == module.base.torch.uint8)

    wrong_dtype_rejected = False
    try:
        module._rng_byte_tensor(module.base.torch.arange(4, dtype=module.base.torch.int64), "bad")
    except module.base.ScientificAbort:
        wrong_dtype_rejected = True
    check("wrong RNG dtype rejected", wrong_dtype_rejected)

    nontensor_rejected = False
    try:
        module._rng_byte_tensor([1, 2, 3], "bad")
    except module.base.ScientificAbort:
        nontensor_rejected = True
    check("non-tensor RNG rejected", nontensor_rejected)

    print(f"STAGE4M_RESUME_RNG_DEVICE_VALIDATION_PASS ({sum(ok for _, ok in tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
