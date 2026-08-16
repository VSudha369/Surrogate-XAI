#!/usr/bin/env python3
"""Stage 4M v1.0.2 restart recovery: scientifically-current stage checks.

This runtime-only layer extends v1.0.1. It fixes two recovery-only issues:
1) Stage 03 legitimately updates the shared teacher-equivalence artifact created
   in Stage 02, so generic recursive stage_current(2) is self-invalidating.
2) A verified local SSD cache created by the exact v1.0.1 recovery executable
   may be reused after this runtime-only v1.0.2 patch when every other cache
   invariant and all shard hashes remain valid.

No scientific objective, architecture, data authorization, teacher, optimizer,
AMP policy, selection rule, metric, or strict-data guard is changed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PARENT_FILENAME = "Stage4M_Resume_Stable_Predecessor_Hotfix_v1_0_1.py"
EXPECTED_PARENT_RECOVERY_SHA256 = "c40bec579a189393dfc11935596e93b8f4eb4e2e1d4bfa967e54eccdf4c5b26b"
EXPECTED_STAGE03_LOCAL_HOTFIX_SHA256 = "18e6307485deab52da038d57a28e99db4647bec87dd18d3687a1dad7f2a32876"
EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"
CACHE_COMPATIBILITY_AUDIT = "STAGE4M_LOCAL_CACHE_COMPATIBILITY_v1_0_2.json"


def _load_parent() -> Any:
    path = Path(__file__).resolve().with_name(PARENT_FILENAME)
    spec = importlib.util.spec_from_file_location("stage4m_resume_parent_v101", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage 4M v1.0.1 recovery executable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.base.sha256_file(path) != EXPECTED_PARENT_RECOVERY_SHA256:
        raise RuntimeError("Stage 4M v1.0.1 recovery hash mismatch; refuse v1.0.2")
    if module.base.Stage4Config().configuration_sha256() != EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256:
        raise RuntimeError("Stage 4M scientific configuration changed; refuse v1.0.2")
    return module


parent = _load_parent()
base = parent.base


class Stage4Pipeline(parent.Stage4Pipeline):
    """Recovery-only hardening with explicit scientific stage-current checks."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.script_sha = base.sha256_file(Path(__file__).resolve())

    def verify_local_data_cache(self, require_existing: bool = True) -> bool:
        """Accept exact v1.0.1 cache provenance only after full base verification.

        The base verifier checks every shard file hash, logical content hash,
        authorized metadata mapping, benchmark/config/runtime policy, and strict
        counters. We temporarily present the exact known parent executable SHA
        only for that verification because executable identity is the sole
        runtime field changed by this patch.
        """
        local_manifest = self.local_cache_root / "cache_manifest.json"
        if not local_manifest.is_file():
            return super().verify_local_data_cache(require_existing=require_existing)
        try:
            payload = json.loads(local_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return super().verify_local_data_cache(require_existing=require_existing)

        if payload.get("executable_sha256") != EXPECTED_PARENT_RECOVERY_SHA256:
            return super().verify_local_data_cache(require_existing=require_existing)

        current_script_sha = self.script_sha
        try:
            self.script_sha = EXPECTED_PARENT_RECOVERY_SHA256
            valid = super().verify_local_data_cache(require_existing=require_existing)
        finally:
            self.script_sha = current_script_sha

        if not valid:
            return False

        audit_path = self.output / "manifests" / CACHE_COMPATIBILITY_AUDIT
        base.atomic_json(audit_path, {
            "status": "PASS",
            "compatibility_scope": "RUNTIME_EXECUTABLE_IDENTITY_ONLY",
            "scientific_changes": False,
            "parent_cache_executable_sha256": EXPECTED_PARENT_RECOVERY_SHA256,
            "current_executable_sha256": self.script_sha,
            "scientific_configuration_sha256": self.config.configuration_sha256(),
            "runtime_io_policy_sha256": self.runtime_io_policy_sha,
            "local_cache_identity_sha256": self.local_cache_identity_sha,
            "local_cache_aggregate_sha256": self.local_cache_aggregate_sha,
            "strict_counters": self.guard.counters(),
            "verified_at": base.utc_now(),
        }, self.output)
        return True

    def stages_01_03_scientifically_current(self) -> bool:
        """Verify Stage 01-03 science without recursive generic stage_current()."""
        try:
            self.hydrate_provenance()
            if any(self.guard.counters().values()):
                return False

            preflight_path = self.output / "manifests" / "STAGE4M_PREFLIGHT.json"
            if not preflight_path.is_file():
                return False
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            if (
                preflight.get("status") != "STAGE4M_PREFLIGHT_PASS"
                or preflight.get("executable_sha256") != self.script_sha
                or preflight.get("configuration_sha256") != self.config.configuration_sha256()
            ):
                return False

            lock_path = self.output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json"
            if not lock_path.is_file():
                return False
            current_lock_sha = base.sha256_file(lock_path)
            if self.predecessor_lock_sha != current_lock_sha:
                return False

            stable_sha = self.stable_predecessor_science_sha()
            if not stable_sha:
                return False

            stage_payloads = {}
            for stage in (1, 2, 3):
                path = self.stage_checkpoint(stage)
                if not path.is_file():
                    return False
                payload = json.loads(path.read_text(encoding="utf-8"))
                stage_payloads[stage] = payload
                common = {
                    "stage": stage,
                    "status": "PASS",
                    "pipeline_version": base.PIPELINE_VERSION,
                    "configuration_sha256": self.config.configuration_sha256(),
                    "executable_sha256": self.script_sha,
                    "teacher_sha256": base.EXPECTED_TEACHER_SHA256,
                    "benchmark_sha256": base.EXPECTED_BENCHMARK_SHA256,
                    "predecessor_lock_sha256": current_lock_sha,
                }
                if any(payload.get(key) != value for key, value in common.items()):
                    return False
                if not base.bound_file_rows_current(payload.get("inputs", []), require_size=False):
                    return False

            stage1, stage2, stage3 = stage_payloads[1], stage_payloads[2], stage_payloads[3]
            if not base.bound_file_rows_current(stage1.get("outputs", []), require_size=True):
                return False

            expected_policy_fields = {
                "architecture_freeze_sha256": self.architecture_freeze_sha,
                "objective_policy_sha256": self.objective_policy_sha,
                "training_target_policy_sha256": self.training_target_policy_sha,
                "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
            }
            for payload in (stage2, stage3):
                if any(payload.get(key) != value for key, value in expected_policy_fields.items()):
                    return False

            # Stage 02's teacher-equivalence output is intentionally rewritten
            # by Stage 03 with partition-level equivalence evidence. Therefore
            # do not require the old Stage 02 output-file hash here.
            equivalence = self.output / "manifests" / "STAGE3M_STAGE4M_TEACHER_EQUIVALENCE.json"
            if not equivalence.is_file():
                return False
            eq = json.loads(equivalence.read_text(encoding="utf-8"))
            if eq.get("status") != "PASS" or eq.get("teacher_sha256") != base.EXPECTED_TEACHER_SHA256:
                return False

            if not base.bound_file_rows_current(stage3.get("outputs", []), require_size=True):
                return False

            for partition in ("train_known", "p0", "p1", "p2", "p3"):
                if not self.cache_current(partition):
                    return False

            return True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def validate_training_checkpoint(self, payload: Mapping[str, Any], arm: str, seed: int) -> None:
        expected = self.checkpoint_validation_fields(arm, seed)
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if not mismatches:
            return

        allowed_migration_fields = {"executable_sha256", "predecessor_lock_sha256"}
        known_legacy_executables = {
            EXPECTED_STAGE03_LOCAL_HOTFIX_SHA256,
            EXPECTED_PARENT_RECOVERY_SHA256,
        }
        legacy_known = payload.get("executable_sha256") in known_legacy_executables
        only_runtime_predecessor_migration = (
            legacy_known
            and set(mismatches).issubset(allowed_migration_fields)
            and "executable_sha256" in mismatches
            and all(
                payload.get(key) == value
                for key, value in expected.items()
                if key not in allowed_migration_fields
            )
        )
        if only_runtime_predecessor_migration:
            if not self.stages_01_03_scientifically_current():
                raise base.ScientificAbort(
                    f"RESUME_MIGRATION_REQUIRES_SCIENTIFICALLY_CURRENT_STAGES_01_03 {arm}/seed={seed}"
                )
            self._write_resume_migration_audit(payload, arm, seed, mismatches)
            self.logger.warning(
                "Accepted v1.0.2 runtime-only checkpoint provenance migration | "
                "arm=%s seed=%s epoch=%s | mismatches=%s",
                arm, seed, payload.get("epoch"), mismatches,
            )
            return

        raise base.ScientificAbort(f"STALE_STAGE4M_CHECKPOINT {arm}/seed={seed}: {mismatches}")


parent.base.Stage4Pipeline = Stage4Pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    return parent.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
