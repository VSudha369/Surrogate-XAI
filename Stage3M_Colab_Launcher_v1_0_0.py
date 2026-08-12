#!/usr/bin/env python3
"""Colab launcher for Stage 3M v1.0.0 canonical teacher promotion."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_VERSION = "1.0.5"
EXPECTED_STAGE2M_SCRIPT_SHA256 = "46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba"
EXPECTED_STAGE2M_HASH_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_DECISION = "SELECT_CE_SUPCON_PROTOTYPE"
BRANCH_NAME = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
SCRIPT_NAME = "Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.py"
STRICT_KEYS = (
    "strict_zero_day_signal_read_violations", "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations", "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def drive_ready() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def mount_drive() -> None:
    if drive_ready():
        print("[PASS] Google Drive is already mounted and accessible")
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Mount Google Drive in the parent Colab cell before launching Stage 3M") from exc
    if not drive_ready():
        raise RuntimeError("Google Drive mount is unavailable")


def locate_branch_root() -> Path:
    explicit = os.environ.get("WISIG_BRANCH_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.name != BRANCH_NAME:
            raise RuntimeError(f"WISIG_BRANCH_ROOT must end in {BRANCH_NAME}")
        return candidate
    roots = [path for path in Path("/content/drive/MyDrive").rglob(BRANCH_NAME) if path.is_dir()]
    valid = [path.resolve() for path in roots if (path / "03_representation_ablation" / "MANYTX_STAGE2_6M_READY.txt").is_file()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected exactly one READY canonical branch root; found {valid}")
    return valid[0]


def verify_stage26(branch_root: Path) -> None:
    stage26 = branch_root / "03_representation_ablation"
    ready = stage26 / "MANYTX_STAGE2_6M_READY.txt"
    hash_manifest = stage26 / "manifests" / "HASH_MANIFEST.json"
    status_path = stage26 / "manifests" / "STAGE2_6M_FINAL_STATUS.json"
    if not all(path.is_file() for path in (ready, hash_manifest, status_path)):
        raise RuntimeError("Stage 2.6M frozen READY artifacts are incomplete")
    lines = ready.read_text(encoding="utf-8").splitlines()
    values = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    if lines[0] != "MANYTX_STAGE2_6M_READY" or values.get("decision") != EXPECTED_DECISION:
        raise RuntimeError("Stage 2.6M READY decision mismatch")
    if values.get("artifact_sha256") != EXPECTED_STAGE26_ARTIFACT_SHA256 or sha256_file(hash_manifest) != EXPECTED_STAGE26_ARTIFACT_SHA256:
        raise RuntimeError("Stage 2.6M artifact SHA mismatch")
    if any(values.get(key) != "0" for key in STRICT_KEYS):
        raise RuntimeError("Stage 2.6M strict-zero-day counters are not all zero")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("decision") != EXPECTED_DECISION or status.get("selected_arm") != "A3":
        raise RuntimeError("Stage 2.6M final status does not freeze A3")
    if any(status.get("strict_zero_day_violation_counters", {}).values()):
        raise RuntimeError("Stage 2.6M structured strict counters are non-zero")
    print("[PASS] Stage 2.6M READY, artifact SHA, A3 decision, and strict counters verified")


def verify_stage2m(branch_root: Path) -> None:
    stage2m = branch_root / "02_benchmark_diagnostics"
    status_path = stage2m / "manifests" / "STAGE2M_FINAL_STATUS.json"
    hash_manifest = stage2m / "manifests" / "HASH_MANIFEST.json"
    if not status_path.is_file() or not hash_manifest.is_file():
        raise RuntimeError("Frozen Stage 2M status or hash manifest is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required = {
        "status": "MANYTX_STAGE2M_READY",
        "stage_version": EXPECTED_STAGE2M_VERSION,
        "script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "recommendation": "PROCEED_STAGE_2_6M_WITH_CAUTION",
        "failed_gates": [],
        "final_test_model_evaluation_performed": False,
        "final_test_threshold_selection_performed": False,
    }
    mismatches = {key: {"expected": expected, "actual": status.get(key)} for key, expected in required.items() if status.get(key) != expected}
    strict_guard = status.get("strict_test_guard", {})
    for key, expected in {
        "strict_index_arrays_loaded": False,
        "strict_test_signal_reads": 0,
        "strict_test_label_reads": 0,
        "strict_test_feature_reads": 0,
    }.items():
        if strict_guard.get(key) != expected:
            mismatches[f"strict_test_guard.{key}"] = {"expected": expected, "actual": strict_guard.get(key)}
    if mismatches:
        raise RuntimeError(f"Stage 2M structured final status contract failed: {json.dumps(mismatches, sort_keys=True)}")
    if sha256_file(hash_manifest) != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256:
        raise RuntimeError("Stage 2M HASH_MANIFEST.json SHA-256 mismatch")
    print("[PASS] Stage 2M structured final status and HASH_MANIFEST SHA-256 verified")


def main() -> None:
    print("=" * 100)
    print("STAGE 3M — COLAB LAUNCHER v1.0.0")
    print("=" * 100)
    mount_drive()
    branch_root = locate_branch_root()
    repository = Path(os.environ.get("WISIG_REPOSITORY_ROOT", Path(__file__).resolve().parent)).resolve()
    script = Path(os.environ.get("WISIG_STAGE3M_SCRIPT", repository / SCRIPT_NAME)).resolve()
    benchmark = branch_root / "01_benchmark_engineering" / "benchmark" / "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
    if not script.is_file():
        raise RuntimeError(f"Stage 3M executable missing: {script}")
    if not benchmark.is_file() or sha256_file(benchmark) != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("Canonical benchmark missing or SHA mismatch")
    verify_stage2m(branch_root)
    verify_stage26(branch_root)
    print("Branch root:", branch_root)
    print("Pipeline script:", script)
    print("Benchmark SHA-256:", EXPECTED_BENCHMARK_SHA256)
    try:
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print("GPU:", properties.name)
            print("GPU memory GiB:", round(properties.total_memory / 2**30, 2))
        else:
            print("GPU: unavailable; CPU evaluation will be slow")
    except ImportError:
        print("GPU: torch unavailable before dependency setup")
    command = [sys.executable, "-u", str(script), "--branch-root", str(branch_root), "--repository-root", str(repository)]
    command.extend(sys.argv[1:])
    environment = dict(os.environ); environment["PYTHONUNBUFFERED"] = "1"
    subprocess.check_call(command, env=environment)
    if "--preflight" not in sys.argv[1:]:
        ready = branch_root / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt"
        if not ready.is_file():
            raise RuntimeError(f"Stage 3M READY marker missing after successful exit: {ready}")


if __name__ == "__main__":
    main()
