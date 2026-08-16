#!/usr/bin/env python3
"""Colab launcher for Stage 4M v1.0.2 restart-safe recovery."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
RECOVERY_EXECUTABLE = "Stage4M_Resume_Scientific_Current_Hotfix_v1_0_2.py"


def mount_drive() -> None:
    mount = Path("/content/drive/MyDrive")
    if mount.is_dir() and any(mount.iterdir()):
        print("[PASS] Google Drive is already mounted and accessible")
        return
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None or not hasattr(shell, "kernel"):
            raise RuntimeError("no live parent Colab kernel")
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:
        raise RuntimeError("Mount Google Drive in the parent Colab cell before launching Stage 4M") from exc
    if not mount.is_dir():
        raise RuntimeError("Google Drive mount did not become accessible")


def valid_root(path: Path) -> bool:
    required = (
        "01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation",
        "04_canonical_teacher", "05_zero_day_open_set",
    )
    return (
        all((path / name).is_dir() for name in required)
        and (path / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt").is_file()
        and (path / "05_zero_day_open_set" / "MANYTX_STAGE3_5M_READY.txt").is_file()
    )


def discover(explicit: Optional[str]) -> Path:
    requested = explicit or os.environ.get("WISIG_BRANCH_ROOT")
    candidates = (
        [Path(requested).expanduser().resolve()]
        if requested
        else [path.resolve() for path in Path("/content/drive/MyDrive").rglob(CANONICAL_BRANCH) if path.is_dir()]
    )
    valid = [path for path in candidates if valid_root(path)]
    if not valid:
        raise RuntimeError("No valid canonical branch root found; verify the mounted Google account")
    if len(valid) != 1:
        raise RuntimeError("Multiple canonical branch roots found:\n" + "\n".join(map(str, valid)))
    return valid[0]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root")
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--config")
    parser.add_argument("--profile", choices=("full", "pilot"), default="full")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--stage-local-data", action="store_true")
    modes.add_argument("--verify-local-data", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-start", type=int, default=1)
    parser.add_argument("--stage-end", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    print("=" * 100)
    print("STAGE 4M — SCIENTIFIC-CURRENT RESUME LAUNCHER v1.0.2")
    print("=" * 100)
    mount_drive()
    root = discover(args.branch_root)
    repository = Path(args.repository_root).expanduser().resolve()
    pipeline = repository / RECOVERY_EXECUTABLE
    if not pipeline.is_file():
        raise RuntimeError(f"Stage 4M v1.0.2 recovery executable missing: {pipeline}")
    print(f"Branch root: {root}")
    print(f"Recovery executable: {pipeline}")
    torch = __import__("torch")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory GiB: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")
    else:
        print("GPU: NOT AVAILABLE")

    command = [
        sys.executable, "-u", str(pipeline),
        "--branch-root", str(root),
        "--repository-root", str(repository),
        "--profile", args.profile,
    ]
    if args.config:
        command.extend(("--config", args.config))
    if args.preflight:
        command.append("--preflight")
    elif args.stage_local_data:
        command.append("--stage-local-data")
    elif args.verify_local_data:
        command.append("--verify-local-data")
    else:
        command.extend(("--run", "--stage-start", str(args.stage_start), "--stage-end", str(args.stage_end)))
        if args.resume:
            print("[PASS] Resume requested; base Stage 4M resume=True is active")

    environment = os.environ.copy()
    environment["WISIG_BRANCH_ROOT"] = str(root)
    environment["PYTHONUNBUFFERED"] = "1"
    print("Executing:", " ".join(command))
    subprocess.check_call(command, env=environment)


if __name__ == "__main__":
    main()
