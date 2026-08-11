#!/usr/bin/env python3
"""Execution-only performance engine for Stage 2.6M v1.0.2.

This module never discovers scientific partitions.  Callers must supply a named
authorized partition and its already-verified frozen global-index array.  The
only whole-benchmark operation is an opaque byte copy followed by SHA-256.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import psutil
except ImportError:  # pragma: no cover - launcher installs the declared dependency
    psutil = None


SCHEMA_VERSION = "stage2_6m_runtime_storage_v1"
AUTHORIZED_PARTITIONS = frozenset({"train_known", "p0", "p1", "p2", "p3", "calibration_unknown"})
KNOWN_RUNTIME_PARTITIONS = frozenset({"train_known", "p0", "p1", "p2", "p3"})
FORBIDDEN_TOKENS = ("strict_zero_day", "zero_day_shift_test", "strict_test")
DEFAULT_CACHE_ROOT = Path("/content/wisig_stage2_6m_cache")
DEFAULT_SHARD_HANDLE_LIMIT = 8


class PerformanceAbort(RuntimeError):
    """Execution-engine integrity failure that forbids scientific use."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def assert_safe_runtime_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PerformanceAbort(f"Runtime path escapes cache root: {resolved}")
    token = str(resolved).lower()
    if any(forbidden in token for forbidden in FORBIDDEN_TOKENS):
        raise PerformanceAbort(f"Strict-zero-day-like runtime path is forbidden: {resolved}")
    return resolved


def assert_local_cache_root(cache_root: Path) -> Path:
    resolved = assert_safe_runtime_path(cache_root, cache_root)
    content_root = Path("/content")
    if content_root.is_dir() and content_root.resolve() not in (resolved, *resolved.parents):
        raise PerformanceAbort(f"Local cache/shards must remain under /content on Colab: {resolved}")
    return resolved


def percentile(values: Sequence[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value)) if values else math.nan


def memory_snapshot() -> Dict[str, float]:
    if psutil is None:
        return {
            "process_cpu_percent": math.nan,
            "system_cpu_percent": math.nan,
            "rss_bytes": math.nan,
            "available_ram_bytes": math.nan,
        }
    process = psutil.Process()
    vm = psutil.virtual_memory()
    return {
        "process_cpu_percent": float(process.cpu_percent(interval=None)),
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "rss_bytes": float(process.memory_info().rss),
        "available_ram_bytes": float(vm.available),
    }


def gpu_snapshot() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "gpu_utilization_percent": math.nan,
        "gpu_memory_used_bytes": math.nan,
        "torch_memory_allocated_bytes": 0,
        "torch_memory_reserved_bytes": 0,
        "torch_peak_allocated_bytes": 0,
        "torch_peak_reserved_bytes": 0,
    }
    if torch.cuda.is_available():
        result.update({
            "torch_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
        try:
            text = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).strip().splitlines()[0]
            utilization, memory_mib = [float(item.strip()) for item in text.split(",")[:2]]
            result["gpu_utilization_percent"] = utilization
            result["gpu_memory_used_bytes"] = int(memory_mib * 2**20)
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            pass
    return result


class LocalCacheManager:
    """Creates a verified opaque local copy without parsing any HDF5 row."""

    def __init__(self, cache_root: Path, performance_root: Path):
        self.cache_root = assert_local_cache_root(cache_root)
        self.performance_root = performance_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def ensure(self, source: Path, expected_sha256: str) -> Tuple[Path, Dict[str, Any]]:
        source = source.resolve()
        if not source.is_file():
            raise PerformanceAbort(f"Canonical Drive benchmark missing: {source}")
        source_bytes = source.stat().st_size
        source_verify_start = time.perf_counter()
        source_sha = sha256_file(source)
        source_verify_duration = time.perf_counter() - source_verify_start
        if source_sha != expected_sha256:
            raise PerformanceAbort(f"Canonical Drive benchmark SHA mismatch: {source_sha}")
        destination = assert_safe_runtime_path(self.cache_root / source.name, self.cache_root)
        temporary = assert_safe_runtime_path(destination.with_name(destination.name + ".copying"), self.cache_root)
        if temporary.exists():
            temporary.unlink()
        free_before = shutil.disk_usage(self.cache_root).free
        copy_duration = 0.0
        if destination.is_file():
            verify_start = time.perf_counter()
            local_sha = sha256_file(destination)
            verification_duration = time.perf_counter() - verify_start
            if local_sha != expected_sha256 or destination.stat().st_size != source_bytes:
                destination.unlink()
            else:
                report = self._report(
                    source,
                    destination,
                    source_bytes,
                    source_sha,
                    local_sha,
                    copy_duration,
                    verification_duration,
                    source_verify_duration,
                    free_before,
                    shutil.disk_usage(self.cache_root).free,
                    reused=True,
                )
                self._persist(report)
                print("LOCAL_CANONICAL_BENCHMARK_VERIFIED")
                return destination, report
        required = source_bytes + max(512 * 2**20, source_bytes // 10)
        if free_before < required:
            raise PerformanceAbort(
                "INSUFFICIENT_LOCAL_DISK: local canonical cache requires "
                f"{required:,} bytes including safety headroom; available={free_before:,}. "
                "Use --storage-mode single_drive only as an explicit controlled fallback."
            )
        started = time.perf_counter()
        try:
            with source.open("rb") as source_handle, temporary.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=16 * 1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            copy_duration = time.perf_counter() - started
            verify_start = time.perf_counter()
            local_sha = sha256_file(temporary)
            verification_duration = time.perf_counter() - verify_start
            if local_sha != expected_sha256 or temporary.stat().st_size != source_bytes:
                raise PerformanceAbort("Local canonical HDF5 copy failed SHA/byte verification")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        report = self._report(
            source,
            destination,
            source_bytes,
            source_sha,
            local_sha,
            copy_duration,
            verification_duration,
            source_verify_duration,
            free_before,
            shutil.disk_usage(self.cache_root).free,
            reused=False,
        )
        self._persist(report)
        print("LOCAL_CANONICAL_BENCHMARK_VERIFIED")
        return destination, report

    @staticmethod
    def _report(
        source: Path,
        destination: Path,
        file_bytes: int,
        source_sha: str,
        local_sha: str,
        copy_duration: float,
        verification_duration: float,
        source_verification_duration: float,
        free_before: int,
        free_after: int,
        reused: bool,
    ) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "VERIFIED",
            "drive_source_path": str(source),
            "local_destination": str(destination),
            "file_bytes": int(file_bytes),
            "source_sha256": source_sha,
            "local_sha256": local_sha,
            "copy_duration_seconds": copy_duration,
            "effective_copy_mb_per_second": (file_bytes / 2**20 / copy_duration) if copy_duration > 0 else None,
            "verification_duration_seconds": verification_duration,
            "source_verification_duration_seconds": source_verification_duration,
            "local_free_space_before": int(free_before),
            "local_free_space_after": int(free_after),
            "reused_verified_copy": bool(reused),
            "opaque_filesystem_copy_only": True,
            "hdf5_rows_parsed_during_copy": 0,
            "generated_at": utc_now(),
        }

    def _persist(self, report: Mapping[str, Any]) -> None:
        atomic_json(self.cache_root / "LOCAL_CACHE_MANIFEST.json", report)
        atomic_json(self.performance_root / "local_cache_report.json", report)


def read_h5_rows_with_duplicates(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    requested = np.asarray(indices, dtype=np.int64).reshape(-1)
    if not len(requested):
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    unique, inverse = np.unique(requested, return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse]


def _metadata_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


class AuthorizedShardManager:
    """Builds only caller-authorized named partitions beneath disposable /content."""

    def __init__(self, cache_root: Path, source_sha256: str, signal_key: str, orientation: str):
        self.cache_root = assert_local_cache_root(cache_root)
        self.shards_root = assert_safe_runtime_path(self.cache_root / "shards_v1", self.cache_root)
        self.source_sha256 = source_sha256
        self.signal_key = signal_key
        self.orientation = orientation

    def build(
        self,
        source_h5: Path,
        partition: str,
        partition_data: Any,
        partition_index_path: Path,
        requested_shards: int,
        guard: Any,
        *,
        allow_calibration: bool = False,
    ) -> Path:
        self._authorize_partition(partition, allow_calibration)
        indices = np.asarray(partition_data.indices, dtype=np.int64)
        guard.authorize_rows(partition, indices, "authorized local shard generation")
        index_sha = sha256_file(partition_index_path)
        partition_root = assert_safe_runtime_path(self.shards_root / partition, self.cache_root)
        manifest_path = partition_root / "LOCAL_SHARD_MANIFEST.json"
        if manifest_path.is_file() and self.verify(manifest_path, partition, index_sha, len(indices)):
            return manifest_path
        building = assert_safe_runtime_path(self.shards_root / f".{partition}.building", self.cache_root)
        if building.exists():
            shutil.rmtree(building)
        building.mkdir(parents=True, exist_ok=False)
        incomplete = building / "INCOMPLETE"
        incomplete.write_text("incomplete\n", encoding="utf-8")
        actual_shards = max(1, min(int(requested_shards), len(indices)))
        position_groups = [group for group in np.array_split(np.arange(len(indices), dtype=np.int64), actual_shards) if len(group)]
        entries: List[Dict[str, Any]] = []
        started = time.perf_counter()
        signal_bytes = 0
        try:
            with h5py.File(source_h5, "r", swmr=True) as source:
                signal_dataset = source[self.signal_key]
                for shard_id, positions in enumerate(position_groups):
                    global_indices = indices[positions]
                    values = read_h5_rows_with_duplicates(signal_dataset, global_indices)
                    if self.orientation == "channels_last":
                        values = values.transpose(0, 2, 1)
                    values = np.ascontiguousarray(values)
                    signal_bytes += values.nbytes
                    filename = f"{partition}_shard_{shard_id:04d}.h5"
                    temporary = building / (filename + ".tmp")
                    final = building / filename
                    with h5py.File(temporary, "w") as shard:
                        shard.create_dataset("signals", data=values, compression=None)
                        shard.create_dataset("global_indices", data=global_indices.astype(np.int64, copy=False))
                        shard.create_dataset("labels", data=np.asarray(partition_data.labels[positions], dtype=np.int16))
                        shard.create_dataset("receiver", data=_metadata_strings(partition_data.receiver[positions]), dtype=h5py.string_dtype("utf-8"))
                        shard.create_dataset("day", data=_metadata_strings(partition_data.day[positions]), dtype=h5py.string_dtype("utf-8"))
                        shard.create_dataset("equalized", data=np.asarray(partition_data.equalized[positions], dtype=np.int8))
                        shard.attrs["schema_version"] = SCHEMA_VERSION
                        shard.attrs["partition"] = partition
                        shard.attrs["source_benchmark_sha256"] = self.source_sha256
                        shard.attrs["source_partition_index_sha256"] = index_sha
                        shard.flush()
                    os.replace(temporary, final)
                    entries.append({
                        "filename": filename,
                        "sha256": sha256_file(final),
                        "row_count": int(len(positions)),
                        "global_index_minimum": int(global_indices.min()),
                        "global_index_maximum": int(global_indices.max()),
                        "ordered_global_indices_sha256": sha256_array(global_indices.astype(np.int64, copy=False)),
                    })
            duration = time.perf_counter() - started
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "source_canonical_benchmark_sha256": self.source_sha256,
                "partition": partition,
                "frozen_partition_index_file": str(partition_index_path.resolve()),
                "partition_index_sha256": index_sha,
                "total_authorized_rows": int(len(indices)),
                "requested_shard_count": int(requested_shards),
                "actual_shard_count": len(entries),
                "rows_per_shard": [entry["row_count"] for entry in entries],
                "shards": entries,
                "ordered_global_indices_sha256": sha256_array(indices),
                "signal_shape": [2, 256],
                "signal_dtype": str(values.dtype),
                "build_duration_seconds": duration,
                "build_throughput_mb_per_second": signal_bytes / 2**20 / max(duration, 1e-12),
                "total_local_bytes": int(sum((building / entry["filename"]).stat().st_size for entry in entries)),
                "strict_zero_day_rows_read": 0,
                "authorization_source": "caller-supplied frozen authorized partition indices",
                "generated_at": utc_now(),
            }
            atomic_json(building / "LOCAL_SHARD_MANIFEST.json", manifest)
            incomplete.unlink()
            if partition_root.exists():
                shutil.rmtree(partition_root)
            os.replace(building, partition_root)
        finally:
            if building.exists():
                shutil.rmtree(building)
        if not self.verify(manifest_path, partition, index_sha, len(indices)):
            raise PerformanceAbort("ABORT_STAGE_2_6M — LOCAL_SHARD_EQUIVALENCE_FAILED")
        return manifest_path

    @staticmethod
    def _authorize_partition(partition: str, allow_calibration: bool) -> None:
        normalized = partition.lower()
        if normalized not in AUTHORIZED_PARTITIONS or any(token in normalized for token in FORBIDDEN_TOKENS):
            raise PerformanceAbort(f"Unauthorized shard partition: {partition}")
        if normalized == "calibration_unknown" and not allow_calibration:
            raise PerformanceAbort("Calibration Unknown shards are forbidden before Stage 08")

    def verify(self, manifest_path: Path, partition: str, index_sha: str, count: int) -> bool:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("partition") != partition
                or payload.get("source_canonical_benchmark_sha256") != self.source_sha256
                or payload.get("partition_index_sha256") != index_sha
                or int(payload.get("total_authorized_rows", -1)) != int(count)
                or (manifest_path.parent / "INCOMPLETE").exists()
            ):
                return False
            return all(
                (manifest_path.parent / entry["filename"]).is_file()
                and sha256_file(manifest_path.parent / entry["filename"]) == entry["sha256"]
                for entry in payload["shards"]
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False


def identity_collate(batch: Any) -> Any:
    """DataLoader collate for Dataset.__getitems__ returning a complete batch."""

    return batch


class BatchedSignalDataset(Dataset):
    """Worker-local batched HDF5 reader preserving duplicates and request order."""

    def __init__(
        self,
        h5_path: Path,
        signal_key: str,
        orientation: str,
        partition: str,
        metadata: Any,
        backend: str,
        shard_manifest: Optional[Path] = None,
        shard_handle_limit: int = DEFAULT_SHARD_HANDLE_LIMIT,
    ):
        if partition not in AUTHORIZED_PARTITIONS:
            raise PerformanceAbort(f"Unauthorized dataset partition: {partition}")
        self.h5_path = Path(h5_path)
        self.signal_key = signal_key
        self.orientation = orientation
        self.partition = partition
        self.metadata = metadata
        self.backend = backend
        self.shard_manifest = Path(shard_manifest) if shard_manifest else None
        self.shard_handle_limit = int(shard_handle_limit)
        self._h5: Optional[h5py.File] = None
        self._signals: Optional[h5py.Dataset] = None
        self._shard_handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._mapping_global = np.empty(0, dtype=np.int64)
        self._mapping_shard = np.empty(0, dtype=np.int32)
        self._mapping_local = np.empty(0, dtype=np.int32)
        self._shard_paths: List[Path] = []
        self.last_batch_read_seconds = 0.0
        if backend == "sharded_local":
            if not self.shard_manifest or not self.shard_manifest.is_file():
                raise PerformanceAbort(f"Shard manifest missing for {partition}")
            self._load_mapping()
        elif backend not in {"single_local", "single_drive"}:
            raise PerformanceAbort(f"Unsupported runtime storage backend: {backend}")

    def _load_mapping(self) -> None:
        assert self.shard_manifest is not None
        payload = json.loads(self.shard_manifest.read_text(encoding="utf-8"))
        globals_parts: List[np.ndarray] = []
        shard_parts: List[np.ndarray] = []
        local_parts: List[np.ndarray] = []
        self._shard_paths = []
        for shard_id, entry in enumerate(payload["shards"]):
            path = self.shard_manifest.parent / entry["filename"]
            self._shard_paths.append(path)
            with h5py.File(path, "r") as handle:
                global_indices = np.asarray(handle["global_indices"], dtype=np.int64)
            globals_parts.append(global_indices)
            shard_parts.append(np.full(len(global_indices), shard_id, dtype=np.int32))
            local_parts.append(np.arange(len(global_indices), dtype=np.int32))
        globals_all = np.concatenate(globals_parts)
        order = np.argsort(globals_all, kind="stable")
        self._mapping_global = globals_all[order]
        self._mapping_shard = np.concatenate(shard_parts)[order]
        self._mapping_local = np.concatenate(local_parts)[order]
        if len(np.unique(self._mapping_global)) != len(self._mapping_global):
            raise PerformanceAbort(f"Duplicate global index in {self.partition} shard mapping")

    def __len__(self) -> int:
        return len(self.metadata.indices)

    def _open_single(self) -> h5py.Dataset:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._signals = self._h5[self.signal_key]
        assert self._signals is not None
        return self._signals

    def _shard(self, shard_id: int) -> h5py.File:
        if shard_id in self._shard_handles:
            handle = self._shard_handles.pop(shard_id)
            self._shard_handles[shard_id] = handle
            return handle
        handle = h5py.File(self._shard_paths[shard_id], "r")
        self._shard_handles[shard_id] = handle
        while len(self._shard_handles) > self.shard_handle_limit:
            _, evicted = self._shard_handles.popitem(last=False)
            evicted.close()
        return handle

    def _read_sharded(self, global_indices: np.ndarray) -> np.ndarray:
        lookup = np.searchsorted(self._mapping_global, global_indices)
        valid = lookup < len(self._mapping_global)
        if not valid.all() or not np.array_equal(self._mapping_global[lookup], global_indices):
            raise PerformanceAbort(f"Requested row is absent from authorized {self.partition} shards")
        shard_ids = self._mapping_shard[lookup]
        local_rows = self._mapping_local[lookup]
        output: Optional[np.ndarray] = None
        for shard_id in np.unique(shard_ids):
            requested_positions = np.flatnonzero(shard_ids == shard_id)
            rows = local_rows[requested_positions].astype(np.int64)
            values = read_h5_rows_with_duplicates(self._shard(int(shard_id))["signals"], rows)
            if output is None:
                output = np.empty((len(global_indices), *values.shape[1:]), dtype=values.dtype)
            output[requested_positions] = values
        if output is None:
            raise PerformanceAbort("Empty sharded batch request")
        return output

    def __getitems__(self, positions: Sequence[int]) -> Dict[str, torch.Tensor]:
        started = time.perf_counter()
        requested_positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        global_indices = np.asarray(self.metadata.indices[requested_positions], dtype=np.int64)
        if self.backend == "sharded_local":
            values = self._read_sharded(global_indices)
        else:
            values = read_h5_rows_with_duplicates(self._open_single(), global_indices)
            if self.orientation == "channels_last":
                values = values.transpose(0, 2, 1)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.shape[1:] != (2, 256) or not np.isfinite(values).all():
            raise PerformanceAbort(f"Invalid signal batch for {self.partition}: {values.shape}")
        self.last_batch_read_seconds = time.perf_counter() - started
        return {
            "x": torch.from_numpy(values),
            "y": torch.from_numpy(np.asarray(self.metadata.labels[requested_positions], dtype=np.int64)),
            "position": torch.from_numpy(requested_positions.copy()),
            "global_index": torch.from_numpy(global_indices.copy()),
            "equalized": torch.from_numpy(np.asarray(self.metadata.equalized[requested_positions], dtype=np.int64)),
        }

    def __getitem__(self, position: int) -> Dict[str, torch.Tensor]:
        batch = self.__getitems__([position])
        return {key: value[0] for key, value in batch.items()}

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._signals = None
        for handle in self._shard_handles.values():
            handle.close()
        self._shard_handles.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_signals"] = None
        state["_shard_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def vectorized_circular_shift(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or shifts.shape != (x.shape[0],):
        raise ValueError("Expected x=[B,C,T] and shifts=[B]")
    length = x.shape[-1]
    source = (torch.arange(length, device=x.device).view(1, 1, length) - shifts.view(-1, 1, 1)) % length
    return torch.gather(x, -1, source.expand(-1, x.shape[1], -1))


def reference_circular_shift(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.roll(x[index], int(shifts[index].item()), dims=-1) for index in range(x.shape[0])])


def validate_vectorized_shift(device: torch.device) -> Dict[str, Any]:
    x = torch.arange(8 * 2 * 37, dtype=torch.float32, device=device).reshape(8, 2, 37).requires_grad_(True)
    shifts = torch.tensor([-4, -1, 0, 1, 4, 11, -15, 37], device=device)
    optimized = vectorized_circular_shift(x, shifts)
    reference = reference_circular_shift(x, shifts)
    if not torch.equal(optimized, reference) or not torch.isfinite(optimized).all():
        raise PerformanceAbort("Vectorized circular-shift equivalence failed")
    optimized.sum().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise PerformanceAbort("Vectorized circular shift is not gradient-compatible")
    print("VECTORIZED_CIRCULAR_SHIFT_EQUIVALENCE_PASS")
    return {"status": "PASS", "exact_tensor_equality": True, "gradient_compatible": True}


def compare_batches(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> None:
    for key in ("x", "y", "position", "global_index", "equalized"):
        if not torch.equal(first[key], second[key]):
            raise PerformanceAbort(f"Storage backend batch mismatch: {key}")


def deterministic_equivalence_positions(metadata: Any, shard_rows: Sequence[int], limit: int = 512) -> np.ndarray:
    count = len(metadata.indices)
    positions = {0, min(1, count - 1), max(0, count - 2), count - 1}
    for boundary in shard_rows:
        if 0 <= boundary < count:
            positions.update({max(0, boundary - 1), boundary, min(count - 1, boundary + 1)})
    rng = np.random.default_rng(2_602_000_001)
    positions.update(rng.choice(count, size=min(limit, count), replace=False).tolist())
    combos: Dict[Tuple[str, str, int], int] = {}
    for position, combo in enumerate(zip(metadata.receiver, metadata.day, metadata.equalized)):
        key = (str(combo[0]), str(combo[1]), int(combo[2]))
        combos.setdefault(key, position)
        if len(combos) >= 128:
            break
    positions.update(combos.values())
    return np.asarray(sorted(positions), dtype=np.int64)


def validate_shard_equivalence(single: BatchedSignalDataset, sharded: BatchedSignalDataset, manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored: Dict[str, List[np.ndarray]] = {key: [] for key in ("global_indices", "labels", "receiver", "day", "equalized")}
    for entry in manifest["shards"]:
        with h5py.File(manifest_path.parent / entry["filename"], "r") as handle:
            for key in stored:
                stored[key].append(np.asarray(handle[key]))
    global_indices = np.concatenate(stored["global_indices"]).astype(np.int64, copy=False)
    labels = np.concatenate(stored["labels"]).astype(np.int16, copy=False)
    receiver = np.concatenate(stored["receiver"])
    day = np.concatenate(stored["day"])
    equalized = np.concatenate(stored["equalized"]).astype(np.int8, copy=False)
    receiver_text = np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in receiver])
    day_text = np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in day])
    if not (
        np.array_equal(global_indices, np.asarray(single.metadata.indices, dtype=np.int64))
        and np.array_equal(labels, np.asarray(single.metadata.labels, dtype=np.int16))
        and np.array_equal(receiver_text, np.asarray([str(value) for value in single.metadata.receiver]))
        and np.array_equal(day_text, np.asarray([str(value) for value in single.metadata.day]))
        and np.array_equal(equalized, np.asarray(single.metadata.equalized, dtype=np.int8))
    ):
        raise PerformanceAbort("ABORT_STAGE_2_6M — LOCAL_SHARD_EQUIVALENCE_FAILED: metadata mismatch")
    boundaries = np.cumsum(np.asarray(manifest["rows_per_shard"], dtype=np.int64))[:-1].tolist()
    positions = deterministic_equivalence_positions(single.metadata, boundaries)
    requested = np.concatenate((positions, positions[: min(32, len(positions))][::-1]))
    compare_batches(single.__getitems__(requested), sharded.__getitems__(requested))
    print("AUTHORIZED_SHARD_EQUIVALENCE_PASS")
    return {
        "status": "PASS",
        "tested_rows": int(len(requested)),
        "duplicates_tested": True,
        "arbitrary_order_tested": True,
        "shard_boundaries_tested": True,
        "bitwise_signal_equality": True,
        "metadata_equality": True,
    }


def benchmark_dataset(dataset: BatchedSignalDataset, batches: Sequence[Sequence[int]], warmup: int, measured: int) -> Dict[str, Any]:
    selected = list(batches[: warmup + measured])
    if len(selected) <= warmup:
        raise PerformanceAbort("Insufficient deterministic batches for storage benchmark")
    for batch in selected[:warmup]:
        dataset.__getitems__(batch)
    latencies: List[float] = []
    batch_rates: List[float] = []
    samples = 0
    start_snapshot = memory_snapshot()
    started = time.perf_counter()
    for batch in selected[warmup:]:
        begin = time.perf_counter()
        result = dataset.__getitems__(batch)
        latency = time.perf_counter() - begin
        latencies.append(latency)
        batch_samples = int(result["x"].shape[0])
        batch_rates.append(batch_samples / max(latency, 1e-12))
        samples += batch_samples
    duration = time.perf_counter() - started
    end_snapshot = memory_snapshot()
    return {
        "backend": dataset.backend,
        "measured_batches": len(latencies),
        "samples": samples,
        "duration_seconds": duration,
        "batches_per_second": len(latencies) / duration,
        "samples_per_second": samples / duration,
        "median_samples_per_second": statistics.median(batch_rates),
        "mean_batch_fetch_seconds": statistics.fmean(latencies),
        "median_batch_fetch_seconds": statistics.median(latencies),
        "p95_batch_fetch_seconds": percentile(latencies, 95),
        "p99_batch_fetch_seconds": percentile(latencies, 99),
        "bytes_per_second": samples * 2 * 256 * 4 / duration,
        "cpu_start": start_snapshot,
        "cpu_end": end_snapshot,
        "file_open_count": len(dataset._shard_handles) if dataset.backend == "sharded_local" else 1,
        "shard_grouping_enabled": dataset.backend == "sharded_local",
    }


def select_storage_backend(single: Mapping[str, Any], sharded: Mapping[str, Any], build_cost: float) -> Dict[str, Any]:
    single_rate = float(single["median_samples_per_second"])
    shard_rate = float(sharded["median_samples_per_second"])
    difference = 100.0 * (shard_rate - single_rate) / single_rate
    p95_ok = float(sharded["p95_batch_fetch_seconds"]) <= 1.10 * float(single["p95_batch_fetch_seconds"])
    p95_reduction = 100.0 * (
        float(single["p95_batch_fetch_seconds"]) - float(sharded["p95_batch_fetch_seconds"])
    ) / float(single["p95_batch_fetch_seconds"])
    selected = "sharded_local" if difference >= 10.0 and p95_ok else "single_local"
    rationale = (
        "Sharded local selected: verified throughput improvement >=10% and P95 latency within 10%."
        if selected == "sharded_local"
        else "Single local selected: sharded backend did not satisfy both the 10% throughput and P95 latency gates."
    )
    return {
        "selected_backend": selected,
        "single_local_samples_per_second": single_rate,
        "sharded_local_samples_per_second": shard_rate,
        "selection_rate_statistic": "median per-batch samples/second",
        "percentage_difference": difference,
        "single_local_p95_seconds": float(single["p95_batch_fetch_seconds"]),
        "sharded_local_p95_seconds": float(sharded["p95_batch_fetch_seconds"]),
        "p95_latency_reduction_percent": p95_reduction,
        "shard_build_cost_seconds": float(build_cost),
        "selection_rationale": rationale,
        "generated_at": utc_now(),
    }


def dataloader_autotune(
    dataset: BatchedSignalDataset,
    batches: Sequence[Sequence[int]],
    worker_init_fn: Any,
    seed: int,
) -> Dict[str, Any]:
    cpu_count = max(1, os.cpu_count() or 1)
    worker_candidates = sorted({0, min(2, cpu_count), min(4, cpu_count)})
    tests: List[Dict[str, Any]] = []
    benchmark_batches = list(batches[: min(120, len(batches))])
    for workers in worker_candidates:
        prefetch_candidates = [None] if workers == 0 else [2, 4]
        for prefetch in prefetch_candidates:
            dataset.close()
            generator = torch.Generator().manual_seed(seed)
            kwargs: Dict[str, Any] = {
                "dataset": dataset,
                "batch_sampler": _StaticBatchSampler(benchmark_batches),
                "num_workers": workers,
                "pin_memory": torch.cuda.is_available(),
                "worker_init_fn": worker_init_fn,
                "generator": generator,
                "collate_fn": identity_collate,
            }
            if workers:
                kwargs.update({"persistent_workers": True, "prefetch_factor": prefetch})
            started = time.perf_counter()
            latencies: List[float] = []
            samples = 0
            try:
                iterator = iter(DataLoader(**kwargs))
                for _ in benchmark_batches:
                    begin = time.perf_counter()
                    batch = next(iterator)
                    latencies.append(time.perf_counter() - begin)
                    samples += int(batch["x"].shape[0])
                duration = time.perf_counter() - started
                tests.append({
                    "num_workers": workers,
                    "prefetch_factor": prefetch,
                    "samples_per_second": samples / duration,
                    "median_latency_seconds": statistics.median(latencies),
                    "p95_latency_seconds": percentile(latencies, 95),
                    "rss_bytes": memory_snapshot()["rss_bytes"],
                    "status": "PASS",
                })
            except Exception as exc:
                tests.append({
                    "num_workers": workers,
                    "prefetch_factor": prefetch,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    passed = [test for test in tests if test["status"] == "PASS"]
    if not passed:
        raise PerformanceAbort("All DataLoader autotune candidates failed")
    selected = max(passed, key=lambda item: (item["samples_per_second"], -item["num_workers"]))
    return {
        "cpu_core_count": cpu_count,
        "tested_configurations": tests,
        "selected_num_workers": int(selected["num_workers"]),
        "selected_prefetch_factor": int(selected["prefetch_factor"] or 2),
        "selected_pin_memory": bool(torch.cuda.is_available()),
        "selected_persistent_workers": bool(selected["num_workers"] > 0),
        "generated_at": utc_now(),
    }


class _StaticBatchSampler:
    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(map(int, batch)) for batch in batches]

    def __iter__(self) -> Iterable[List[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def write_performance_summary(performance_root: Path, payload: Mapping[str, Any]) -> None:
    performance_root.mkdir(parents=True, exist_ok=True)
    atomic_json(performance_root / "performance_summary.json", payload)
    storage = payload.get("storage_selection", {})
    text = "\n".join([
        "# Stage 2.6M v1.0.2 performance summary",
        "",
        "Structured prior-run baseline telemetry was unavailable; comparison status: `BASELINE_NOT_MEASURED`.",
        "Current results therefore use the deterministic single-local versus sharded-local A/B microbenchmark.",
        "",
        "| Metric | Single local | Sharded local | Difference % |",
        "|---|---:|---:|---:|",
        f"| Samples/s | {storage.get('single_local_samples_per_second', math.nan):.3f} | {storage.get('sharded_local_samples_per_second', math.nan):.3f} | {storage.get('percentage_difference', math.nan):.3f}% |",
        f"| P95 latency (s) | {storage.get('single_local_p95_seconds', math.nan):.6f} | {storage.get('sharded_local_p95_seconds', math.nan):.6f} | {storage.get('p95_latency_reduction_percent', math.nan):.3f}% reduction |",
        "",
        f"Selected backend: `{storage.get('selected_backend', 'unavailable')}`.",
        "",
        "GPU utilization percentage-point and relative changes are reported only when both structured baseline and optimized telemetry exist; no values are invented.",
    ])
    (performance_root / "performance_summary.md").write_text(text + "\n", encoding="utf-8")


def reset_seed_state(output_root: Path, seed: int) -> List[str]:
    if seed not in {42, 123, 2026}:
        raise PerformanceAbort(f"Reset is restricted to the frozen seed panel; received {seed}")
    root = output_root.resolve()
    if root.name != "03_representation_ablation":
        raise PerformanceAbort(f"Refusing reset outside 03_representation_ablation: {root}")
    removed: List[str] = []
    for arm in ("A0", "A1", "A2", "A3"):
        target = (root / "checkpoints" / arm / f"seed_{seed}").resolve()
        if root not in target.parents:
            raise PerformanceAbort(f"Reset target escapes output root: {target}")
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    history = (root / "logs" / f"training_history_seed_{seed}.csv").resolve()
    if root not in history.parents:
        raise PerformanceAbort(f"Reset target escapes output root: {history}")
    if history.exists():
        history.unlink()
        removed.append(str(history))
    atomic_json(
        root / "performance" / f"RESET_SEED_{seed}_STATUS.json",
        {
            "status": "RESET_COMPLETE",
            "seed": seed,
            "removed_paths": removed,
            "scope": "four arm checkpoint directories plus one seed training-history CSV only",
            "upstream_artifacts_modified": False,
            "generated_at": utc_now(),
        },
    )
    return removed
