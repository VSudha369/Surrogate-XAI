#!/usr/bin/env python3
"""Static regression validator for Stage 4M v1.0.1 restart-safe resume hotfix."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

PARENT = "Stage4M_Stage03_Local_Activation_Hotfix_v1_0_0.py"
RECOVERY = "Stage4M_Resume_Stable_Predecessor_Hotfix_v1_0_1.py"
LAUNCHER = "Stage4M_Colab_Resume_Stable_Launcher_v1_0_1.py"
EXPECTED_PARENT_SHA256 = "18e6307485deab52da038d57a28e99db4647bec87dd18d3687a1dad7f2a32876"
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

    check("parent hotfix frozen", sha256(parent) == EXPECTED_PARENT_SHA256)
    source = recovery.read_text(encoding="utf-8")
    launch = launcher.read_text(encoding="utf-8")
    check("parent SHA bound", EXPECTED_PARENT_SHA256 in source)
    check("scientific config bound", EXPECTED_CONFIG_SHA256 in source)
    check("stable predecessor schema", "STAGE4M_STABLE_PREDECESSOR_SCIENCE_IDENTITY_v1" in source)
    check("volatile preflight excluded", "Deliberately excludes Stage4M preflight timestamps" in source)
    check("known parent only", "legacy_parent" in source and "EXPECTED_PARENT_HOTFIX_SHA256" in source)
    check("migration fields restricted", '{"executable_sha256", "predecessor_lock_sha256"}' in source)
    check("stages 01-03 reverified", "all(self.stage_current(stage) for stage in (1, 2, 3))" in source)
    check("migration audit", "STAGE4M_RESUME_PROVENANCE_MIGRATION.json" in source)
    check("scientific changes false", '\"scientific_changes\": False' in source)
    check("selection changes false", '\"selection_changes\": False' in source)
    check("strict access false", '\"strict_access_permitted\": False' in source)
    check("launcher points recovery", RECOVERY in launch)
    check("launcher preserves default resume", "base Stage 4M resume=True is already active" in launch)
    check("launcher does not pass unsupported resume", 'command.append("--resume")' not in launch)
    check("launcher supports staging", '"--stage-local-data"' in launch)
    check("launcher supports verification", '"--verify-local-data"' in launch)
    check("launcher supports preflight", '"--preflight"' in launch)

    print(f"STAGE4M_RESUME_STABLE_VALIDATION_PASS ({sum(ok for _, ok in tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
