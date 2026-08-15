#!/usr/bin/env python3
"""Stage 4M v1.0.0 runtime hotfix: activate verified local shards before Stage 03.

This runtime-only layer preserves the validated Stage 4M scientific executable
at commit d301b884... and overrides only local-I/O activation, Stage 03+ loader
source enforcement, disposable teacher-cache provenance, and Drive I/O gates.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

BASE_FILENAME = "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.py"
EXPECTED_BASE_EXECUTABLE_SHA256 = "9f1889f2cd5efa2d38776f7524c52ce6989379c7717501ae9a9239949f8550d0"
EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name(BASE_FILENAME)
    spec = importlib.util.spec_from_file_location("stage4m_d301b_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage 4M base executable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.sha256_file(path) != EXPECTED_BASE_EXECUTABLE_SHA256:
        raise RuntimeError("Stage 4M base executable hash mismatch; refuse runtime hotfix")
    if module.Stage4Config().configuration_sha256() != EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256:
        raise RuntimeError("Stage 4M scientific configuration changed; refuse runtime hotfix")
    return module


base = _load_base()


class Stage4Pipeline(base.Stage4Pipeline):
    """Runtime-only Stage 03 local-shard activation hardening."""

    def __init__(self, config: Any):
        super().__init__(config)
        # Bind all new scientific-runtime artifacts to this exact hotfix executable.
        self.script_sha = base.sha256_file(Path(__file__).resolve())
        self.io_counters.setdefault("evaluation_signal_reads_drive", 0)

    def persist_io_counters(self) -> Path:
        path = self.output / "manifests" / "STAGE4M_IO_COUNTERS.json"
        drive_hot_path_zero = (
            int(self.io_counters.get("training_signal_reads_drive", 0)) == 0
            and int(self.io_counters.get("evaluation_signal_reads_drive", 0)) == 0
        )
        base.atomic_json(path, {
            "status": "PASS" if drive_hot_path_zero else "FAIL",
            "pipeline_version": base.PIPELINE_VERSION,
            "counters": dict(self.io_counters),
            "staging_drive_signal_reads_allowed": True,
            "ordinary_training_drive_signal_reads_allowed": False,
            "ordinary_evaluation_drive_signal_reads_allowed": False,
            "strict_counters": self.guard.counters(),
            "updated_at": base.utc_now(),
        }, self.output)
        return path

    def build_loader(
        self, partition: str, batches: Optional[Sequence[Sequence[int]]] = None, seed: int = 0
    ) -> Tuple[Any, Any]:
        if partition not in (*base.AUTHORIZED_LOCAL_PARTITIONS, "calibration_unknown"):
            self.guard.reject("signal", f"forbidden DataLoader partition: {partition}")

        benchmark = self.ensure_context()
        stage = int(self.execution_context.get("stage") or 0)

        if partition == "calibration_unknown":
            self.ensure_calibration_local_cache()
        elif stage >= 3:
            if not self.local_cache_active:
                raise base.ScientificAbort(
                    f"Stage {stage:02d} requires verified local shards before loading {partition}"
                )
            if benchmark.backend_for(partition) != "sharded_local":
                raise base.ScientificAbort(
                    f"Stage {stage:02d} Drive signal hot path forbidden for {partition}"
                )
        elif self.local_cache_active and benchmark.backend_for(partition) != "sharded_local":
            raise base.ScientificAbort(f"Local cache active but {partition} is not local-sharded")

        dataset = self.stage26.WiSigH5Dataset(benchmark, partition, self.guard)
        compatibility = self.stage26.Stage26Config(
            branch_root=str(self.root), output_dir=str(self.root / "03_representation_ablation"),
            batch_size=self.config.batch_size, eval_batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers, prefetch_factor=self.config.prefetch_factor,
        )
        loader = self.stage26.build_loader(dataset, compatibility, batches=batches, seed=seed)
        if stage >= 3 and partition != "calibration_unknown" and getattr(dataset, "backend", None) != "sharded_local":
            dataset.close()
            raise base.ScientificAbort(
                f"Stage {stage:02d} DataLoader resolved non-local backend for {partition}"
            )
        return dataset, loader

    def cache_current(self, partition: str) -> bool:
        if not super().cache_current(partition):
            return False
        manifest = self.cache_dir(partition) / "store_manifest.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        return (
            payload.get("signal_source_backend") == "sharded_local"
            and payload.get("local_cache_identity_sha256") == self.local_cache_identity_sha
            and payload.get("local_cache_aggregate_sha256") == self.local_cache_aggregate_sha
            and payload.get("runtime_io_policy_sha256") == self.runtime_io_policy_sha
        )

    def prepare_teacher_cache(self, partition: str) -> Path:
        store = super().prepare_teacher_cache(partition)
        manifest = store / "store_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {
            "signal_source_backend": "sharded_local",
            "local_cache_identity_sha256": self.local_cache_identity_sha,
            "local_cache_aggregate_sha256": self.local_cache_aggregate_sha,
            "runtime_io_policy_sha256": self.runtime_io_policy_sha,
            "drive_signal_hot_path_used": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            if not self.local_cache_active or self.ensure_context().backend_for(partition) != "sharded_local":
                raise base.ScientificAbort(
                    f"Teacher cache {partition} cannot be provenance-bound to local shards"
                )
            payload.update(expected)
            base.atomic_json(manifest, payload, store)
        return store

    def run(self) -> None:
        self.hydrate_provenance()
        stages = {
            1: self.stage_01, 2: self.stage_02, 3: self.stage_03, 4: self.stage_04,
            5: self.stage_05, 6: self.stage_06, 7: self.stage_07, 8: self.stage_08,
            9: self.stage_09, 10: self.stage_10, 11: self.stage_11, 12: self.stage_12,
        }
        for stage in range(self.config.stage_start, self.config.stage_end + 1):
            self.hydrate_provenance()
            self.execution_context.update({"stage": stage, "arm": None, "seed": None, "epoch": None})
            if stage >= 3 and not self.local_cache_active:
                self.ensure_local_data_cache()
                self.hydrate_provenance()
            if self.config.resume and self.stage_current(stage):
                self.write_live_progress("STAGE_REUSED")
                print(f"[REUSE] Stage {stage:02d} — hash-current")
                continue
            self.write_live_progress(f"STAGE_{stage:02d}_STARTED")
            stages[stage]()
            self.write_live_progress(f"STAGE_{stage:02d}_PASS")
            print(f"[PASS] Stage {stage:02d}")

    def stage_12(self) -> None:
        if int(self.io_counters.get("training_signal_reads_drive", 0)) != 0:
            raise base.ScientificAbort("Canonical training used the Drive signal hot path")
        if int(self.io_counters.get("evaluation_signal_reads_drive", 0)) != 0:
            raise base.ScientificAbort("Canonical evaluation used the Drive signal hot path")
        super().stage_12()


# Make the validated base CLI construct the hardened runtime pipeline.
base.Stage4Pipeline = Stage4Pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
