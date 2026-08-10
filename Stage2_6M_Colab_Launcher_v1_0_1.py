#!/usr/bin/env python3
"""Single-strip Google Colab launcher for Stage 2.6M v1.0.1.

Copy this entire file into one Colab cell.  It never uses ``%run`` and passes
only explicit arguments to the standalone pipeline process.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_VERSION = "1.0.5"
EXPECTED_STAGE2M_SCRIPT_SHA256 = "46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba"
EXPECTED_STAGE2M_HASH_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
SCRIPT_NAME = "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_1.py"
BRANCH_NAME = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
REQUIRED = {
    "h5py": "h5py>=3.10",
    "matplotlib": "matplotlib>=3.8",
    "numpy": "numpy>=1.26",
    "openpyxl": "openpyxl>=3.1",
    "pandas": "pandas>=2.1",
    "scipy": "scipy>=1.11",
    "sklearn": "scikit-learn>=1.4",
    "tabulate": "tabulate>=0.9",
    "torch": "torch>=2.1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mount_drive() -> None:
    if Path("/content").is_dir():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive", force_remount=False)
        except ImportError:
            pass


def find_one(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} below {root}; found {len(matches)}")
    return matches[0]


def find_one_directory(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one directory {name} below {root}; found {len(matches)}")
    return matches[0]


def install_missing() -> None:
    missing = [requirement for module, requirement in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])


def verify_stage2m(stage2m: Path) -> None:
    if not stage2m.is_dir():
        raise RuntimeError(f"Frozen Stage 2M directory missing: {stage2m}")
    status_path = stage2m / "manifests" / "STAGE2M_FINAL_STATUS.json"
    hash_manifest_path = stage2m / "manifests" / "HASH_MANIFEST.json"
    if not status_path.is_file():
        raise RuntimeError(f"Stage 2M structured final status missing: {status_path}")
    if not hash_manifest_path.is_file():
        raise RuntimeError(f"Stage 2M artifact hash manifest missing: {hash_manifest_path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stage 2M structured final status is invalid JSON: {status_path}") from exc
    if not isinstance(status, dict):
        raise RuntimeError("Stage 2M structured final status must be a JSON object")
    required_status = {
        "status": "MANYTX_STAGE2M_READY",
        "stage_version": EXPECTED_STAGE2M_VERSION,
        "script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "recommendation": "PROCEED_STAGE_2_6M_WITH_CAUTION",
        "failed_gates": [],
        "final_test_model_evaluation_performed": False,
        "final_test_threshold_selection_performed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": status.get(key, "<missing>")}
        for key, expected in required_status.items()
        if status.get(key, "<missing>") != expected or type(status.get(key, "<missing>")) is not type(expected)
    }
    strict_guard = status.get("strict_test_guard")
    if not isinstance(strict_guard, dict):
        mismatches["strict_test_guard"] = {"expected": "JSON object", "actual": type(strict_guard).__name__}
    else:
        required_guard = {
            "strict_index_arrays_loaded": False,
            "strict_test_signal_reads": 0,
            "strict_test_label_reads": 0,
            "strict_test_feature_reads": 0,
        }
        for key, expected in required_guard.items():
            actual = strict_guard.get(key, "<missing>")
            if actual != expected or type(actual) is not type(expected):
                mismatches[f"strict_test_guard.{key}"] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Stage 2M structured final status contract failed: {json.dumps(mismatches, sort_keys=True)}")
    actual_manifest_sha = sha256_file(hash_manifest_path)
    if actual_manifest_sha != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256:
        raise RuntimeError(
            "Stage 2M HASH_MANIFEST.json SHA-256 mismatch: "
            f"expected {EXPECTED_STAGE2M_HASH_MANIFEST_SHA256}, actual {actual_manifest_sha}"
        )
    print("[PASS] Stage 2M structured final status and HASH_MANIFEST SHA-256 verified")


def main() -> None:
    print("=" * 100)
    print("STAGE 2.6M — COLAB LAUNCHER v1.0.1")
    print("=" * 100)
    mount_drive()
    search_root = Path(os.environ.get("WISIG_SEARCH_ROOT", "/content/drive/MyDrive"))
    branch_override = os.environ.get("WISIG_BRANCH_ROOT", "")
    branch_root = Path(branch_override).expanduser().resolve() if branch_override else find_one_directory(search_root, BRANCH_NAME)
    if not branch_root.is_dir():
        raise RuntimeError(f"Canonical branch root missing: {branch_root}")
    script_override = os.environ.get("WISIG_STAGE2_6M_SCRIPT", "")
    script = Path(script_override).expanduser().resolve() if script_override else find_one(search_root, SCRIPT_NAME)
    benchmark = branch_root / "01_benchmark_engineering" / "benchmark" / "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
    if not benchmark.is_file():
        raise RuntimeError(f"Canonical benchmark missing: {benchmark}")
    actual_sha = sha256_file(benchmark)
    print("Benchmark SHA-256:", actual_sha)
    if actual_sha != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("ABORT_STAGE_2_6M — BENCHMARK_HASH_MISMATCH")
    verify_stage2m(branch_root / "02_benchmark_diagnostics")
    install_missing()
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU:", props.name)
        print("GPU memory GiB:", round(props.total_memory / 2**30, 2))
    else:
        print("GPU: unavailable; execution will be CPU-only and is not recommended for the full profile")
    profile = os.environ.get("WISIG_STAGE2_6M_PROFILE", "full")
    command = [
        sys.executable,
        str(script),
        "--branch-root",
        str(branch_root),
        "--profile",
        profile,
    ]
    subprocess.check_call(command)
    output = branch_root / "03_representation_ablation"
    ready = output / "MANYTX_STAGE2_6M_READY.txt"
    not_ready = output / "MANYTX_STAGE2_6M_NOT_READY.txt"
    if not_ready.exists():
        raise RuntimeError(not_ready.read_text(encoding="utf-8"))
    if profile == "full" and not ready.exists():
        raise RuntimeError(f"READY marker missing after successful process exit: {ready}")
    objective_path = output / "manifests" / "CANONICAL_STAGE3M_OBJECTIVE.json"
    if objective_path.is_file():
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        print("Selected Stage 3M objective:", objective["decision"])
    print("Stage 2.6M output:", output)


if __name__ == "__main__":
    main()
