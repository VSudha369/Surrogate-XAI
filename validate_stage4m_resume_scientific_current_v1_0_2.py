#!/usr/bin/env python3
"""Static regression validator for Stage 4M v1.0.2 recovery hardening."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

PARENT = "Stage4M_Resume_Stable_Predecessor_Hotfix_v1_0_1.py"
RECOVERY = "Stage4M_Resume_Scientific_Current_Hotfix_v1_0_2.py"
LAUNCHER = "Stage4M_Colab_Resume_Scientific_Current_Launcher_v1_0_2.py"
EXPECTED_PARENT_SHA256 = "c40bec579a189393dfc11935596e93b8f4eb4e2e1d4bfa967e54eccdf4c5b26b"
EXPECTED_CONFIG_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    check("parent v1.0.1 frozen", sha256(parent) == EXPECTED_PARENT_SHA256)
    source = recovery.read_text(encoding="utf-8")
    launch = launcher.read_text(encoding="utf-8")
    check("parent SHA bound", EXPECTED_PARENT_SHA256 in source)
    check("scientific config bound", EXPECTED_CONFIG_SHA256 in source)
    check("no recursive generic stage_current migration gate",
          "all(self.stage_current(stage) for stage in (1, 2, 3))" not in source)
    check("scientific stage checker present", "stages_01_03_scientifically_current" in source)
    check("stage2 shared-equivalence exception documented",
          "Stage 02's teacher-equivalence output is intentionally rewritten" in source)
    check("stage3 outputs hash checked",
          'bound_file_rows_current(stage3.get("outputs", []), require_size=True)' in source)
    check("stage inputs hash checked",
          'bound_file_rows_current(payload.get("inputs", []), require_size=False)' in source)
    check("teacher caches checked", 'for partition in ("train_known", "p0", "p1", "p2", "p3")' in source)
    check("strict counters checked", "any(self.guard.counters().values())" in source)
    check("known legacy executables restricted", "known_legacy_executables" in source)
    check("migration mismatch fields restricted",
          '{"executable_sha256", "predecessor_lock_sha256"}' in source)
    check("parent cache exact compatibility", "EXPECTED_PARENT_RECOVERY_SHA256" in source)
    check("cache compatibility full verifier", "super().verify_local_data_cache" in source)
    check("cache compatibility audit", "STAGE4M_LOCAL_CACHE_COMPATIBILITY_v1_0_2.json" in source)
    check("science unchanged audit", '"scientific_changes": False' in source)
    check("launcher points v1.0.2", RECOVERY in launch)
    check("launcher supports preflight", '"--preflight"' in launch)
    check("launcher supports staging", '"--stage-local-data"' in launch)
    check("launcher supports verify", '"--verify-local-data"' in launch)
    check("launcher preserves default resume", "resume=True is active" in launch)

    print(f"STAGE4M_RESUME_SCIENTIFIC_CURRENT_VALIDATION_PASS ({sum(ok for _, ok in tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
