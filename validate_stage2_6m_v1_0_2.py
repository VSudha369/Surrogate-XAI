#!/usr/bin/env python3
"""Executable benchmark-independent validation for Stage 2.6M v1.0.2."""
from __future__ import annotations

import argparse
import ast
import json
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
            identity_collate,
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
