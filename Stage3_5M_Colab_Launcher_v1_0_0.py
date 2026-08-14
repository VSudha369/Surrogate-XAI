#!/usr/bin/env python3
"""Colab launcher for Stage 3.5M v1.0.0 open-set detection."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


BRANCH_NAME = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
SCRIPT_NAME = "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.py"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE3M_HASH_MANIFEST_SHA256 = "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0"
EXPECTED_TEACHER_SHA256 = "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4"
EXPECTED_TEACHER_STATE_SHA256 = "7d6c6ff609fb86618ae7b92bcd55b0c8a440ed2769561d4de9b4485802e639d7"
STRICT_KEYS = (
    "strict_zero_day_signal_read_violations",
    "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations",
    "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mount_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        print("[PASS] Google Drive is already mounted and accessible")
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Mount Google Drive in the parent Colab cell before launching Stage 3.5M") from exc
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Google Drive mount is unavailable")


def locate_branch_root() -> Path:
    explicit = os.environ.get("WISIG_BRANCH_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.name != BRANCH_NAME:
            raise RuntimeError(f"WISIG_BRANCH_ROOT must end in {BRANCH_NAME}")
        return candidate
    candidates = [
        path.resolve() for path in Path("/content/drive/MyDrive").rglob(BRANCH_NAME)
        if path.is_dir()
        and all((path / name).is_dir() for name in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation", "04_canonical_teacher"))
        and (path / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Stage-3M-READY canonical branch root; found {candidates}")
    return candidates[0]


def parse_ready(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values = {"marker": lines[0] if lines else ""}
    values.update(dict(line.split("=", 1) for line in lines[1:] if "=" in line))
    return values


def verify_stage3m(branch_root: Path) -> None:
    stage3m = branch_root / "04_canonical_teacher"
    ready_path = stage3m / "MANYTX_STAGE3M_READY.txt"
    manifest = stage3m / "manifests" / "STAGE3M_HASH_MANIFEST.json"
    status_path = stage3m / "manifests" / "STAGE3M_FINAL_STATUS.json"
    teacher = stage3m / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"
    state = stage3m / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt"
    if not all(path.is_file() for path in (ready_path, manifest, status_path, teacher, state)):
        raise RuntimeError("Frozen Stage 3M artifacts are incomplete")
    if sha256_file(manifest) != EXPECTED_STAGE3M_HASH_MANIFEST_SHA256:
        raise RuntimeError("Stage 3M final hash-manifest SHA mismatch")
    if sha256_file(teacher) != EXPECTED_TEACHER_SHA256 or sha256_file(state) != EXPECTED_TEACHER_STATE_SHA256:
        raise RuntimeError("Canonical teacher SHA mismatch")
    ready = parse_ready(ready_path)
    required = {
        "marker": "MANYTX_STAGE3M_READY", "teacher_version": "1.0", "objective": "CE_SUPCON_PROTOTYPE",
        "source_arm": "A3", "selected_seed": "123", "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
        "canonical_teacher_sha256": EXPECTED_TEACHER_SHA256,
        "final_zero_day_evaluation_performed": "NO", "surrogate_training_performed": "NO",
        "xai_performed": "NO", "next_stage": "STAGE_3_5M",
    }
    mismatches = {key: {"expected": expected, "actual": ready.get(key)} for key, expected in required.items() if ready.get(key) != expected}
    for key in STRICT_KEYS:
        if ready.get(key) != "0":
            mismatches[key] = {"expected": "0", "actual": ready.get(key)}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "MANYTX_STAGE3M_READY" or int(status.get("selected_seed", -1)) != 123 or any(status.get("strict_zero_day_counters", {}).values()):
        mismatches["STAGE3M_FINAL_STATUS"] = "not frozen seed 123 with zero strict counters"
    if mismatches:
        raise RuntimeError(f"Stage 3M predecessor contract mismatch: {json.dumps(mismatches, sort_keys=True)}")
    print("[PASS] Stage 3M READY, seed-123 teacher, hashes, and strict counters verified")


def main() -> None:
    print("=" * 100)
    print("STAGE 3.5M — COLAB LAUNCHER v1.0.0")
    print("=" * 100)
    mount_drive()
    branch_root = locate_branch_root()
    repository = Path(os.environ.get("WISIG_REPOSITORY_ROOT", Path(__file__).resolve().parent)).resolve()
    script = Path(os.environ.get("WISIG_STAGE3_5M_SCRIPT", repository / SCRIPT_NAME)).resolve()
    benchmark = branch_root / "01_benchmark_engineering" / "benchmark" / "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
    if not script.is_file():
        raise RuntimeError(f"Stage 3.5M executable missing: {script}")
    if not benchmark.is_file() or sha256_file(benchmark) != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("Canonical benchmark missing or SHA mismatch")
    verify_stage3m(branch_root)
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
            print("GPU: unavailable; CPU inference will be slow")
    except ImportError:
        print("GPU: torch unavailable before dependency setup")
    command = [sys.executable, "-u", str(script), "--branch-root", str(branch_root), "--repository-root", str(repository)]
    command.extend(sys.argv[1:])
    environment = dict(os.environ); environment["PYTHONUNBUFFERED"] = "1"
    subprocess.check_call(command, env=environment)
    if "--preflight" not in sys.argv[1:]:
        ready = branch_root / "05_zero_day_open_set" / "MANYTX_STAGE3_5M_READY.txt"
        if not ready.is_file():
            raise RuntimeError(f"Stage 3.5M READY marker missing after successful full exit: {ready}")


if __name__ == "__main__":
    main()
