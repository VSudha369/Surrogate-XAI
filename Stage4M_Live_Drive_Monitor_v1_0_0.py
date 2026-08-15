#!/usr/bin/env python3
"""Read-only Drive monitor for Stage 4M. Safe to run in a separate Colab runtime."""
from __future__ import annotations

import json
import time
from pathlib import Path

CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
REFRESH_SECONDS = 20


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except Exception as exc:
        return {"read_error": repr(exc)}


def find_root() -> Path:
    roots = [
        p for p in Path("/content/drive/MyDrive").rglob(CANONICAL_BRANCH)
        if p.is_dir() and (p / "06_surrogate_kd").is_dir()
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one Stage 4M root, found: {roots}")
    return roots[0] / "06_surrogate_kd"


def compact(label: str, payload) -> None:
    print(f"\n{label}")
    print("-" * 100)
    if payload is None:
        print("[waiting]")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> None:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass

    root = find_root()
    manifests = root / "manifests"
    files = {
        "DATA STAGING": manifests / "STAGE4M_DATA_STAGING_STATUS.json",
        "LIVE PROGRESS": manifests / "STAGE4M_LIVE_PROGRESS.json",
        "HEARTBEAT": manifests / "STAGE4M_HEARTBEAT.json",
        "I/O COUNTERS": manifests / "STAGE4M_IO_COUNTERS.json",
        "INTERRUPTION": manifests / "STAGE4M_INTERRUPTED.json",
        "FAILURE": manifests / "STAGE4M_FAILURE.json",
    }

    while True:
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass
        print("=" * 100)
        print("STAGE 4M LIVE DRIVE MONITOR — READ ONLY")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Drive output: {root}")
        print("=" * 100)
        for label, path in files.items():
            compact(label, load_json(path))

        print("\nPIPELINE MARKERS")
        print("-" * 100)
        print("READY     :", (root / "MANYTX_STAGE4M_READY.txt").is_file())
        print("NOT_READY :", (root / "MANYTX_STAGE4M_NOT_READY.txt").is_file())
        print(f"\nRefresh: every {REFRESH_SECONDS} seconds. This monitor never inspects the training GPU/process.")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
