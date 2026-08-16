#!/usr/bin/env python3
"""Stage 4M v1.0.1 runtime-only resume provenance hardening.

Extends the validated Stage-03 local-SSD hotfix. Scientific objectives,
architecture, data partitions, selection rules, AMP policy, and strict guards
are unchanged. The only purpose is to make completed training checkpoints
portable across Colab runtime restarts that regenerate volatile preflight /
predecessor-lock files after the immutable predecessor artifacts are reverified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PARENT_FILENAME = "Stage4M_Stage03_Local_Activation_Hotfix_v1_0_0.py"
EXPECTED_PARENT_HOTFIX_SHA256 = "18e6307485deab52da038d57a28e99db4647bec87dd18d3687a1dad7f2a32876"
EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"
MIGRATION_AUDIT_FILENAME = "STAGE4M_RESUME_PROVENANCE_MIGRATION.json"


def _load_parent() -> Any:
    path = Path(__file__).resolve().with_name(PARENT_FILENAME)
    spec = importlib.util.spec_from_file_location("stage4m_stage03_local_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage 4M parent hotfix: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.base.sha256_file(path) != EXPECTED_PARENT_HOTFIX_SHA256:
        raise RuntimeError("Stage 4M parent hotfix hash mismatch; refuse resume migration")
    if module.base.Stage4Config().configuration_sha256() != EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256:
        raise RuntimeError("Stage 4M scientific configuration changed; refuse resume migration")
    return module


parent = _load_parent()
base = parent.base


class Stage4Pipeline(parent.Stage4Pipeline):
    """Preserve valid checkpoints across volatile predecessor-lock regeneration."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.script_sha = base.sha256_file(Path(__file__).resolve())
        self._stable_predecessor_science_sha: Optional[str] = None

    def stable_predecessor_science_payload(self) -> Mapping[str, Any]:
        """Return only immutable scientific predecessor identities.

        Deliberately excludes Stage4M preflight timestamps, local-cache manifest
        timestamps, and Stage4M predecessor-lock file bytes. Those runtime
        artifacts are re-created after a Colab restart and must not invalidate
        completed scientific epochs.
        """
        stage35 = self.stage35_root
        paths = {
            "stage2m_hash_manifest": self.root / "02_benchmark_diagnostics" / "manifests" / "HASH_MANIFEST.json",
            "stage26_hash_manifest": self.root / "03_representation_ablation" / "manifests" / "HASH_MANIFEST.json",
            "stage3_ready": self.stage3_root / "MANYTX_STAGE3M_READY.txt",
            "stage3_hash_manifest": self.stage3_root / "manifests" / "STAGE3M_HASH_MANIFEST.json",
            "teacher_checkpoint": self.teacher_path,
            "teacher_state_dict": self.stage3_root / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt",
            "benchmark": self.root / "01_benchmark_engineering" / "benchmark" / f"{base.CANONICAL_BENCHMARK}.h5",
            "stage35_ready": stage35 / "MANYTX_STAGE3_5M_READY.txt",
            "stage35_final": stage35 / "manifests" / "STAGE3_5M_FINAL_STATUS.json",
            "stage35_hash_manifest": stage35 / "manifests" / "STAGE3_5M_HASH_MANIFEST.json",
            "stage35_recovery": stage35 / "manifests" / "POST_LOCK_RECOVERY_MANIFEST.json",
            "stage35_strict_lock": stage35 / "manifests" / "STRICT_ZERO_DAY_EVALUATION_LOCK.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise base.ScientificAbort(f"Stable predecessor identity missing frozen inputs: {missing}")
        payload = {
            "schema": "STAGE4M_STABLE_PREDECESSOR_SCIENCE_IDENTITY_v1",
            "benchmark_sha256": base.EXPECTED_BENCHMARK_SHA256,
            "teacher_sha256": base.EXPECTED_TEACHER_SHA256,
            "teacher_state_sha256": base.EXPECTED_TEACHER_STATE_SHA256,
            "stage2m_expected_sha256": base.EXPECTED_STAGE2M_MANIFEST_SHA256,
            "stage26_expected_sha256": base.EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_expected_sha256": base.EXPECTED_STAGE3M_MANIFEST_SHA256,
            "files": {name: base.sha256_file(path) for name, path in sorted(paths.items())},
        }
        self.verify_stage35_metadata()
        if payload["files"]["benchmark"] != base.EXPECTED_BENCHMARK_SHA256:
            raise base.ScientificAbort("Stable predecessor benchmark hash mismatch")
        if payload["files"]["teacher_checkpoint"] != base.EXPECTED_TEACHER_SHA256:
            raise base.ScientificAbort("Stable predecessor teacher hash mismatch")
        if payload["files"]["teacher_state_dict"] != base.EXPECTED_TEACHER_STATE_SHA256:
            raise base.ScientificAbort("Stable predecessor teacher-state hash mismatch")
        return payload

    def stable_predecessor_science_sha(self) -> str:
        if self._stable_predecessor_science_sha is None:
            self._stable_predecessor_science_sha = base.sha256_object(self.stable_predecessor_science_payload())
        return self._stable_predecessor_science_sha

    def checkpoint_validation_fields(self, arm: str, seed: int) -> Mapping[str, Any]:
        fields = dict(super().checkpoint_validation_fields(arm, seed))
        fields["predecessor_lock_sha256"] = self.stable_predecessor_science_sha()
        return fields

    def _write_resume_migration_audit(
        self, payload: Mapping[str, Any], arm: str, seed: int, mismatches: Sequence[str]
    ) -> None:
        path = self.output / "manifests" / MIGRATION_AUDIT_FILENAME
        previous = []
        if path.is_file():
            try:
                previous = list(json.loads(path.read_text(encoding="utf-8")).get("migrations", []))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                previous = []
        row = {
            "arm": arm,
            "seed": int(seed),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "legacy_executable_sha256": payload.get("executable_sha256"),
            "legacy_predecessor_lock_sha256": payload.get("predecessor_lock_sha256"),
            "new_executable_sha256": self.script_sha,
            "stable_predecessor_science_sha256": self.stable_predecessor_science_sha(),
            "mismatches_migrated": list(mismatches),
            "scientific_configuration_sha256": self.config.configuration_sha256(),
            "benchmark_sha256": base.EXPECTED_BENCHMARK_SHA256,
            "teacher_sha256": base.EXPECTED_TEACHER_SHA256,
            "strict_counters": self.guard.counters(),
            "migrated_at": base.utc_now(),
        }
        key = (row["arm"], row["seed"], row["checkpoint_epoch"], row["legacy_predecessor_lock_sha256"])
        deduped = [existing for existing in previous if (
            existing.get("arm"), int(existing.get("seed", -1)), int(existing.get("checkpoint_epoch", -1)),
            existing.get("legacy_predecessor_lock_sha256")
        ) != key]
        deduped.append(row)
        base.atomic_json(path, {
            "status": "RUNTIME_PROVENANCE_MIGRATION_ONLY",
            "scientific_changes": False,
            "selection_changes": False,
            "strict_access_permitted": False,
            "migration_policy": "KNOWN_PARENT_EXECUTABLE_PLUS_ONLY_EXECUTABLE_AND_PREDECESSOR_IDENTITY_MISMATCH",
            "migrations": deduped,
        }, self.output)

    def validate_training_checkpoint(self, payload: Mapping[str, Any], arm: str, seed: int) -> None:
        expected = self.checkpoint_validation_fields(arm, seed)
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if not mismatches:
            return

        allowed_migration_fields = {"executable_sha256", "predecessor_lock_sha256"}
        legacy_parent = payload.get("executable_sha256") == EXPECTED_PARENT_HOTFIX_SHA256
        only_runtime_predecessor_migration = (
            legacy_parent
            and set(mismatches).issubset(allowed_migration_fields)
            and "executable_sha256" in mismatches
            and all(payload.get(key) == value for key, value in expected.items() if key not in allowed_migration_fields)
        )
        if only_runtime_predecessor_migration:
            if not all(self.stage_current(stage) for stage in (1, 2, 3)):
                raise base.ScientificAbort(
                    f"RESUME_MIGRATION_REQUIRES_CURRENT_STAGES_01_03 {arm}/seed={seed}"
                )
            self._write_resume_migration_audit(payload, arm, seed, mismatches)
            self.logger.warning(
                "Accepted runtime-only checkpoint provenance migration | arm=%s seed=%s epoch=%s | mismatches=%s",
                arm, seed, payload.get("epoch"), mismatches,
            )
            return

        raise base.ScientificAbort(f"STALE_STAGE4M_CHECKPOINT {arm}/seed={seed}: {mismatches}")


parent.base.Stage4Pipeline = Stage4Pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    return parent.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
