#!/usr/bin/env python3
"""Single-strip Google Colab launcher for Stage 2M v1.0.5."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_SHA = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
SCRIPT_NAME = "Stage2M_WiSig_ManyTx_Scientific_Diagnostics_v1_0_5.py"
IMPORTS = {
    "numpy": "numpy>=1.24,<3", "pandas": "pandas>=2.0,<3", "scipy": "scipy>=1.10,<2",
    "sklearn": "scikit-learn>=1.3,<2", "h5py": "h5py>=3.9,<4", "matplotlib": "matplotlib>=3.7,<4",
    "openpyxl": "openpyxl>=3.1,<4", "pyarrow": "pyarrow>=12,<25", "psutil": "psutil>=5.9,<8", "tabulate": "tabulate>=0.9,<1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mount_drive_if_needed() -> None:
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        print("[PASS] Google Drive already mounted")
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This launcher expects Google Colab or an already accessible project root") from exc
    drive.mount("/content/drive")


def locate_script() -> Path:
    explicit = os.environ.get("RF_STAGE2M_SCRIPT")
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path.cwd() / SCRIPT_NAME, Path("/content") / SCRIPT_NAME]
    base = Path("/content/drive/MyDrive")
    if base.exists():
        preferred = base / "colab files " / "Surrogate-XAI"
        candidates += [preferred / SCRIPT_NAME, preferred / "Stage2M_WiSig_ManyTx_Scientific_Diagnostics_v1_0_5" / SCRIPT_NAME]
        candidates += list(preferred.glob(f"**/{SCRIPT_NAME}")) if preferred.exists() else []
    matches = [path.resolve() for path in candidates if path and path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {SCRIPT_NAME}. Set RF_STAGE2M_SCRIPT to its absolute Drive path.")
    return matches[0]


def locate_project_root() -> Path:
    explicit = os.environ.get("RF_PROJECT_ROOT")
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path("/content/drive/MyDrive/colab files /Surrogate-XAI/project_root"), Path("/content/drive/MyDrive/Surrogate-XAI/project_root")]
    for root in candidates:
        benchmark = root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3/01_benchmark_engineering/benchmark/WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
        if benchmark.is_file():
            return root.resolve()
    raise FileNotFoundError("Canonical v1.0.3 benchmark not found. Set RF_PROJECT_ROOT to the project_root directory.")


def install_missing() -> None:
    missing = [requirement for module, requirement in IMPORTS.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing dependencies:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("[PASS] All required dependencies already installed")


def main() -> None:
    mount_drive_if_needed()
    script = locate_script()
    project_root = locate_project_root()
    benchmark = project_root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3/01_benchmark_engineering/benchmark/WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
    observed = sha256_file(benchmark)
    if observed != EXPECTED_SHA:
        raise RuntimeError(f"ABORT_STAGE_2M\nBENCHMARK_HASH_MISMATCH\nExpected: {EXPECTED_SHA}\nObserved: {observed}")
    print(f"[PASS] Canonical benchmark SHA-256 verified: {observed}")
    install_missing()
    command = [sys.executable, str(script), "--project-root", str(project_root), "--copy-local"]
    print("Launching Stage 2M in a subprocess so Jupyter kernel arguments cannot reach argparse")
    subprocess.check_call(command)
    output = project_root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3/02_benchmark_diagnostics"
    print(f"Stage 2M output: {output}")
    ready, not_ready = output / "MANYTX_STAGE2M_READY.txt", output / "MANYTX_STAGE2M_NOT_READY.txt"
    if not_ready.exists() and not ready.exists():
        raise RuntimeError(not_ready.read_text(encoding="utf-8"))
    if not ready.exists():
        raise RuntimeError("Stage 2M ended without READY or NOT_READY marker")
    status = json.loads((output / "manifests/STAGE2M_FINAL_STATUS.json").read_text(encoding="utf-8"))
    print(ready.read_text(encoding="utf-8"))
    print("Final Stage 2.6M recommendation:", status["recommendation"])


if __name__ == "__main__":
    main()
