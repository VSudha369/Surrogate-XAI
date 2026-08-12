#!/usr/bin/env python3
"""Executable benchmark-independent validation for Stage 2.6M v1.0.2."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"
PERFORMANCE = ROOT / "stage2_6m_performance_v1_0_2.py"
LAUNCHER = ROOT / "Stage2_6M_Colab_Launcher_v1_0_2.py"
PREDECESSOR = ROOT / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_1.py"


def source_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
    return result


def static_validation() -> None:
    main_source = MAIN.read_text(encoding="utf-8")
    performance_source = PERFORMANCE.read_text(encoding="utf-8")
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    main_tree = ast.parse(main_source)
    predecessor_source = PREDECESSOR.read_text(encoding="utf-8")
    predecessor_tree = ast.parse(predecessor_source)
    ast.parse(performance_source)
    ast.parse(launcher_source)
    assignments = source_assignments(main_tree)
    assert ast.literal_eval(assignments["PIPELINE_VERSION"]) == "1.0.2"
    assert ast.literal_eval(assignments["EXPECTED_BENCHMARK_SHA256"]) == "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
    assert ast.literal_eval(assignments["MAX_CONSECUTIVE_AMP_OVERFLOWS"]) == 32
    arms = ast.literal_eval(assignments["ARM_DEFINITIONS"])
    assert arms == {
        "A0": {"name": "CE", "supcon_weight": 0.0, "prototype_weight": 0.0},
        "A1": {"name": "CE + SupCon", "supcon_weight": 0.1, "prototype_weight": 0.0},
        "A2": {"name": "CE + Prototype", "supcon_weight": 0.0, "prototype_weight": 0.1},
        "A3": {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1},
    }
    config = next(node for node in main_tree.body if isinstance(node, ast.ClassDef) and node.name == "Stage26Config")
    config_source = ast.get_source_segment(main_source, config) or ""
    for invariant in (
        "seeds: Tuple[int, ...] = (42, 123, 2026)",
        "max_epochs: int = 40",
        "minimum_epochs: int = 12",
        "batch_size: int = 256",
        "samples_per_tx: int = 4",
        "embedding_dim: int = 128",
        "num_classes: int = 98",
        "temperature: float = 0.07",
        "prototype_momentum: float = 0.95",
        "validation_every: int = 1",
    ):
        assert invariant in config_source, invariant
    augmentation = next(node for node in main_tree.body if isinstance(node, ast.ClassDef) and node.name == "RFAugmentation")
    augmentation_source = ast.get_source_segment(main_source, augmentation) or ""
    assert "vectorized_circular_shift(x, shifts)" in augmentation_source
    assert "torch.stack([torch.roll" not in augmentation_source
    assert "__getitems__" in performance_source
    assert "np.unique(requested, return_inverse=True)" in performance_source
    assert "Calibration Unknown shards are forbidden before Stage 08" in performance_source
    assert "strict_zero_day_rows_read\": 0" in performance_source
    assert "INSUFFICIENT_LOCAL_DISK" in performance_source
    assert ".copying" in performance_source and "os.replace(temporary, destination)" in performance_source
    assert "Local cache/shards must remain under /content on Colab" in performance_source
    assert "command.extend(sys.argv[1:])" in launcher_source
    compatible_v102_sha = "3a7f795a07163a590f1b24d66ba9cc1574de1e6966bc87157886e8668a79d5d1"
    assert compatible_v102_sha in main_source and compatible_v102_sha in launcher_source
    stage04_checkpoint_sha = "f5af2c7a364a6303c62f3c5875ea0b1dabeb9a6974dd46d40b9a758ae1ac09da"
    assert stage04_checkpoint_sha in main_source and stage04_checkpoint_sha in launcher_source
    assert "labels.detach().to(device=logits_f.device, dtype=torch.long, non_blocking=True)" in main_source
    assert "checkpoint_script_sha_is_compatible(payload[\"script_sha\"], script_sha)" in main_source
    pipeline = next(node for node in main_tree.body if isinstance(node, ast.ClassDef) and node.name == "Stage26Pipeline")
    pipeline_source = ast.get_source_segment(main_source, pipeline) or ""
    assert "def _ensure_partition_shard_for_benchmark(" in pipeline_source
    internal_method = next(
        node for node in pipeline.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_ensure_partition_shard_for_benchmark"
    )
    internal_source = ast.get_source_segment(main_source, internal_method) or ""
    assert "ensure_context(" not in internal_source
    ensure_known_method = next(
        node for node in pipeline.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "ensure_known_runtime_shards"
    )
    ensure_known_source = ast.get_source_segment(main_source, ensure_known_method) or ""
    assert "for partition in" not in ensure_known_source
    assert 'benchmark.backend_for(partition)' in main_source
    for timing_name in (
        "sampled_gpu_forward_ms", "sampled_gpu_objective_ms", "sampled_gpu_backward_ms",
        "sampled_gpu_optimizer_ms", "sampled_gpu_total_ms",
    ):
        assert timing_name in main_source
    for class_name in ("ConvNormAct", "ResidualTemporalBlock", "WiSigRepresentationNet", "EMAPrototypeBank"):
        current = next(node for node in main_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        predecessor = next(node for node in predecessor_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        assert ast.dump(current, include_attributes=False) == ast.dump(predecessor, include_attributes=False), class_name
    print("STATIC_AST_AND_SCIENTIFIC_INVARIANTS_PASS")


@dataclass
class Metadata:
    indices: np.ndarray
    labels: np.ndarray
    receiver: np.ndarray
    day: np.ndarray
    equalized: np.ndarray


class Guard:
    def __init__(self, allowed: np.ndarray):
        self.allowed = np.asarray(allowed, dtype=np.int64)

    def authorize_rows(self, partition: str, indices: np.ndarray, operation: str) -> None:
        assert partition == "train_known"
        assert np.isin(indices, self.allowed).all(), operation


def load_main_module():
    module_name = "stage2_6m_v1_0_2_integration_fixture"
    spec = importlib.util.spec_from_file_location(module_name, MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_storage_context_validation(root: Path) -> None:
    """Exercise the actual Stage26Pipeline context/shard call graph without WiSig data."""
    module = load_main_module()
    zero_counters = {
        "strict_test_signal_reads": 0,
        "strict_test_label_reads": 0,
        "strict_test_embedding_reads": 0,
        "strict_test_metric_reads": 0,
        "strict_test_threshold_reads": 0,
    }

    class FixtureGuard:
        strict_file_audit = [{"status": "fixture_verified"}]

        def counters(self):
            return dict(zero_counters)

        def assert_zero(self):
            assert not any(self.counters().values())

    class FixtureCacheManager:
        ensure_calls = 0

        def __init__(self, cache_root, performance_root):
            self.cache_root = Path(cache_root)

        def ensure(self, source, expected_sha):
            type(self).ensure_calls += 1
            return Path(source), {"status": "VERIFIED", "sha256": expected_sha}

    architecture_before = module.architecture_signature(module.WiSigRepresentationNet(98, 128, 0.10))
    fixed_batches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    exposure_before = module.batch_exposure_sha256(fixed_batches)

    cases = (
        ("A", "auto", "single_local", 0),
        ("B", "auto", "sharded_local", 1),
        ("C", "single_drive", "single_drive", 0),
        ("D", "sharded_local", "sharded_local", 1),
    )
    for name, requested_mode, selected_backend, expected_builds in cases:
        case_root = root / f"context_case_{name.lower()}"
        branch_root = case_root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
        output_root = branch_root / "03_representation_ablation"
        performance_root = output_root / "performance"
        benchmark_path = branch_root / "01_benchmark_engineering" / "benchmark" / "canonical.h5"
        index_path = branch_root / "01_benchmark_engineering" / "splits" / "train_known_indices.npy"
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        performance_root.mkdir(parents=True, exist_ok=True)
        benchmark_path.write_bytes(b"opaque-fixture")
        np.save(index_path, np.asarray([0, 1, 2, 3], dtype=np.int64), allow_pickle=False)

        config = module.Stage26Config(
            branch_root=str(branch_root),
            benchmark_path=str(benchmark_path),
            stage2m_dir=str(branch_root / "02_benchmark_diagnostics"),
            output_dir=str(output_root),
            storage_mode=requested_mode,
            local_cache_root=str(case_root / "cache"),
            performance_preflight=False,
        )
        scientific_sha_before = config.configuration_sha256()
        if requested_mode == "auto":
            (performance_root / "PERFORMANCE_PREFLIGHT_STATUS.json").write_text(
                json.dumps({
                    "canonical_benchmark_sha256": module.EXPECTED_BENCHMARK_SHA256,
                    "storage_selection": {"selected_backend": selected_backend},
                    "runtime_settings": {"storage_backend": selected_backend},
                }),
                encoding="utf-8",
            )

        partitions = {
            partition: Metadata(
                indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
                labels=np.asarray([0, 1, 2, 3], dtype=np.int16),
                receiver=np.asarray(["r0"] * 4, dtype=object),
                day=np.asarray(["d0"] * 4, dtype=object),
                equalized=np.asarray([0, 1, 0, 1], dtype=np.int8),
            )
            for partition in ("train_known", "p0", "p1", "p2", "p3", "calibration_unknown")
        }

        def resolved_fixture(_config, _guard, runtime_path):
            initial = "single_drive" if requested_mode == "single_drive" else "single_local"
            benchmark = module.ResolvedBenchmark(
                h5_path=Path(runtime_path),
                signal_key="signals",
                signal_orientation="channels_first",
                total_samples=4,
                metadata_fields={},
                partitions=partitions,
                transmitter_mapping={},
                schema_rows=[],
                canonical_h5_path=benchmark_path,
                runtime_backend=initial,
            )
            module.apply_partition_backend_policy(benchmark, initial)
            return benchmark

        class FixtureShardManager:
            def __init__(self):
                self.build_calls = 0

            def build(self, source, partition, metadata, frozen_indices, shard_count, guard, **kwargs):
                self.build_calls += 1
                assert partition == "train_known"
                manifest = case_root / "cache" / "authorized_train_known_manifest.json"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({
                    "partition": partition,
                    "benchmark_sha256": module.EXPECTED_BENCHMARK_SHA256,
                    "strict_zero_day_rows_read": 0,
                }), encoding="utf-8")
                return manifest

        shard_manager = FixtureShardManager()
        pipeline = module.Stage26Pipeline.__new__(module.Stage26Pipeline)
        pipeline.config = config
        pipeline.stage2m_provenance = None
        pipeline.guard = FixtureGuard()
        pipeline.benchmark = None
        pipeline.performance_root = performance_root
        pipeline.cache_report = {}
        pipeline.storage_selection = {}
        pipeline.performance_preflight_status = {}
        pipeline.logger = logging.getLogger(f"stage26-fixture-{name}")
        pipeline._shard_manager = lambda benchmark: shard_manager

        original_verify = module.verify_stage2m
        original_resolve = module.resolve_benchmark
        original_cache = module.LocalCacheManager
        original_discover = module.discover_partition_index_files
        FixtureCacheManager.ensure_calls = 0
        try:
            module.verify_stage2m = lambda _config: {"status": "PASS"}
            module.resolve_benchmark = resolved_fixture
            module.LocalCacheManager = FixtureCacheManager
            module.discover_partition_index_files = lambda _root: {"train_known": index_path}
            benchmark = pipeline.ensure_context()
            same_benchmark = pipeline.ensure_context()
        finally:
            module.verify_stage2m = original_verify
            module.resolve_benchmark = original_resolve
            module.LocalCacheManager = original_cache
            module.discover_partition_index_files = original_discover

        assert same_benchmark is benchmark
        assert benchmark.backend_for("train_known") == selected_backend
        non_training_expected = "single_drive" if selected_backend == "single_drive" else "single_local"
        for partition in ("p0", "p1", "p2", "p3", "calibration_unknown"):
            assert benchmark.backend_for(partition) == non_training_expected
            assert partition not in benchmark.shard_manifests
        assert shard_manager.build_calls == expected_builds
        assert ("train_known" in benchmark.shard_manifests) == (expected_builds == 1)
        assert FixtureCacheManager.ensure_calls == (0 if requested_mode == "single_drive" else 1)
        assert config.configuration_sha256() == scientific_sha_before
        assert module.EXPECTED_BENCHMARK_SHA256 == "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
        assert pipeline.guard.counters() == zero_counters
        forbidden_tokens = ("strict_zero_day", "zero_day_shift_test", "strict_test")
        assert not any(token in path.name.lower() for path in case_root.rglob("*") for token in forbidden_tokens)
        assert module.architecture_signature(module.WiSigRepresentationNet(98, 128, 0.10)) == architecture_before
        assert module.batch_exposure_sha256(fixed_batches) == exposure_before

    print("SHARDED_FULL_RUN_CONTEXT_INITIALIZATION_PASS")
    print("STORAGE_CONTEXT_CASES_A_B_C_D_PASS")


def evaluation_checkpoint_compatibility_validation(root: Path) -> None:
    import torch

    module = load_main_module()
    branch_root = root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
    output_root = branch_root / "03_representation_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    config = module.Stage26Config(
        branch_root=str(branch_root),
        output_dir=str(output_root),
        device="cpu",
    )
    run = module.create_training_runs(config, 42, torch.device("cpu"))["A0"]
    payload = module.checkpoint_payload(
        run,
        config,
        epoch=40,
        benchmark_sha=module.EXPECTED_BENCHMARK_SHA256,
        script_sha="f5af2c7a364a6303c62f3c5875ea0b1dabeb9a6974dd46d40b9a758ae1ac09da",
        exposure_sha="fixture-exposure",
        group_state={"epochs_without_any_improvement": 0},
    )
    checkpoint = module.checkpoint_dir(config, "A0", 42) / "best_selection.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    model, loaded, checkpoint_sha = module.load_trained_model(
        config,
        "A0",
        42,
        torch.device("cpu"),
        module.EXPECTED_BENCHMARK_SHA256,
        "current-patched-script-sha",
    )
    assert loaded["script_sha"] == payload["script_sha"]
    assert checkpoint_sha == module.sha256_file(checkpoint)
    assert module.architecture_signature(model) == payload["architecture_signature"]
    print("EVALUATION_CHECKPOINT_COMPATIBLE_SCRIPT_SHA_PASS")


def runtime_validation() -> None:
    try:
        import h5py
        import torch
        from torch.utils.data import DataLoader
        from stage2_6m_performance_v1_0_2 import (
            AuthorizedShardManager,
            BatchedSignalDataset,
            LocalCacheManager,
            PerformanceAbort,
            gpu_snapshot,
            identity_collate,
            memory_snapshot,
            reset_seed_state,
            sha256_file,
            validate_shard_equivalence,
            validate_vectorized_shift,
        )
    except ImportError as exc:
        raise RuntimeError("Runtime validation requires h5py, torch, and psutil; run it in Colab") from exc
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "canonical.h5"
        count = 53
        signals = np.arange(count * 2 * 256, dtype=np.float32).reshape(count, 2, 256)
        with h5py.File(source, "w") as handle:
            handle.create_dataset("signals", data=signals)
        expected_sha = sha256_file(source)
        cache_root = root / "cache"
        performance_root = root / "output" / "performance"
        local, report = LocalCacheManager(cache_root, performance_root).ensure(source, expected_sha)
        assert report["status"] == "VERIFIED" and sha256_file(local) == expected_sha
        local.write_bytes(b"corrupt")
        local, report = LocalCacheManager(cache_root, performance_root).ensure(source, expected_sha)
        assert sha256_file(local) == expected_sha and not report["reused_verified_copy"]
        partial = local.with_name(local.name + ".copying")
        partial.write_bytes(b"partial")
        local, _ = LocalCacheManager(cache_root, performance_root).ensure(source, expected_sha)
        assert sha256_file(local) == expected_sha and not partial.exists()
        indices = np.asarray([50, 1, 17, 3, 42, 9, 25, 31, 7, 11, 13, 15, 19, 21, 23, 27, 29], dtype=np.int64)
        metadata = Metadata(
            indices=indices,
            labels=np.arange(len(indices), dtype=np.int16) % 5,
            receiver=np.asarray([f"rx{i % 3}" for i in range(len(indices))], dtype=object),
            day=np.asarray([f"day{i % 2}" for i in range(len(indices))], dtype=object),
            equalized=np.arange(len(indices), dtype=np.int8) % 2,
        )
        index_path = root / "train_known_indices.npy"
        np.save(index_path, indices, allow_pickle=False)
        manager = AuthorizedShardManager(cache_root, expected_sha, "signals", "channels_first")
        manifest = manager.build(local, "train_known", metadata, index_path, 5, Guard(indices))
        single = BatchedSignalDataset(local, "signals", "channels_first", "train_known", metadata, "single_local")
        sharded = BatchedSignalDataset(local, "signals", "channels_first", "train_known", metadata, "sharded_local", manifest)
        requested = [0, 4, 1, 4, 16, 2, 0]
        first = single.__getitems__(requested)
        second = sharded.__getitems__(requested)
        for key in first:
            assert torch.equal(first[key], second[key]), key
        validate_shard_equivalence(single, sharded, manifest)
        loader = DataLoader(single, batch_size=4, num_workers=0, collate_fn=identity_collate)
        assert next(iter(loader))["x"].shape == (4, 2, 256)
        single.close()
        worker_loader = DataLoader(
            single,
            batch_size=4,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=identity_collate,
        )
        assert next(iter(worker_loader))["x"].shape == (4, 2, 256)
        del worker_loader
        try:
            manager.build(local, "strict_zero_day", metadata, index_path, 5, Guard(indices))
        except PerformanceAbort:
            pass
        else:
            raise AssertionError("Strict shard generation was not rejected")
        validate_vectorized_shift(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accumulator = load_main_module().StreamingClassificationMetrics(num_classes=3)
        metric_logits = torch.tensor([[4.0, 1.0, -1.0], [0.0, 3.0, 1.0]], device=metric_device)
        metric_labels = torch.tensor([0, 1], device="cpu")
        accumulator.update(metric_logits, metric_labels)
        metric_summary, _ = accumulator.finish()
        assert metric_summary["accuracy"] == 1.0 and accumulator.count == 2
        print("STREAMING_METRICS_CROSS_DEVICE_LABEL_ALIGNMENT_PASS")
        host_snapshot = memory_snapshot()
        device_snapshot = gpu_snapshot()
        assert host_snapshot["cpu_utilization_sampling_semantics"] in {
            "interval_since_previous_snapshot_not_epoch_average", "unavailable"
        }
        assert device_snapshot["gpu_utilization_sampling_semantics"] == "instantaneous_nvidia_smi_sample_not_epoch_average"
        assert "torch_peak_allocated_bytes" in device_snapshot and "available_ram_bytes" in host_snapshot
        single.close()
        sharded.close()
        shard_payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert shard_payload["strict_zero_day_rows_read"] == 0
        assert shard_payload["total_authorized_rows"] == len(indices)
        output_root = root / "03_representation_ablation"
        unrelated = root / "02_benchmark_diagnostics" / "keep.txt"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("keep", encoding="utf-8")
        for arm in ("A0", "A1", "A2", "A3"):
            seed_dir = output_root / "checkpoints" / arm / "seed_42"
            seed_dir.mkdir(parents=True)
            (seed_dir / "last.pt").write_bytes(b"test")
        history = output_root / "logs" / "training_history_seed_42.csv"
        history.parent.mkdir(parents=True)
        history.write_text("epoch\n2\n", encoding="utf-8")
        removed = reset_seed_state(output_root, 42)
        assert len(removed) == 5 and unrelated.read_text(encoding="utf-8") == "keep"
        assert (output_root / "performance" / "RESET_SEED_42_STATUS.json").is_file()
        pipeline_storage_context_validation(root)
        evaluation_checkpoint_compatibility_validation(root)
    print("BENCHMARK_INDEPENDENT_RUNTIME_EQUIVALENCE_PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    static_validation()
    if not args.static_only:
        runtime_validation()
    print("STAGE2_6M_V1_0_2_CODE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
