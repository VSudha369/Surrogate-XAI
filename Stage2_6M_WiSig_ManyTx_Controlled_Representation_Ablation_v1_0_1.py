#!/usr/bin/env python3
"""Stage 2.6M — WiSig ManyTx controlled representation-learning ablation.

Version 1.0.1.  This executable implements the ten-stage, four-arm, three-seed
experiment defined by the Stage 2.6M scientific protocol.  It is intentionally
restricted to Train Known fitting, P0–P3 known validation, and frozen-model
Calibration Unknown diagnostics.  Strict zero-day signal, label, embedding,
metric, and threshold access is forbidden by construction and audited at exit.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader, Dataset, Sampler


PIPELINE_VERSION = "1.0.1"
CANONICAL_BENCHMARK = "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3"
CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_VERSION = "1.0.5"
EXPECTED_STAGE2M_SCRIPT_SHA256 = "46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba"
EXPECTED_STAGE2M_HASH_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
EXPECTED_SIGNAL_SHAPE = (2, 256)
EXPECTED_TOTAL_SAMPLES = 1_020_643
EXPECTED_KNOWN_CLASSES = 98
EXPECTED_CAL_UNKNOWN_CLASSES = 22
EXPECTED_PARTITION_COUNTS: Dict[str, int] = {
    "train_known": 388_139,
    "p0": 68_495,
    "p1": 153_529,
    "p2": 27_088,
    "p3": 8_992,
    "calibration_unknown": 158_400,
}
STRICT_INDEX_EXPECTATIONS = {
    "strict_zero_day_test_indices.npy": 216_000,
    "strict_zero_day_shift_test_indices.npy": 3_000,
}
STRICT_FORBIDDEN_ARTIFACTS = (
    "strict_zero_day_embeddings.npy",
    "strict_zero_day_predictions.csv",
    "strict_zero_day_auroc.csv",
    "strict_zero_day_scores.csv",
)
PARTITION_DISPLAY = {
    "train_known": "Train Known",
    "p0": "P0",
    "p1": "P1",
    "p2": "P2",
    "p3": "P3",
    "calibration_unknown": "Calibration Unknown",
}
ARM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "A0": {"name": "CE", "supcon_weight": 0.0, "prototype_weight": 0.0},
    "A1": {"name": "CE + SupCon", "supcon_weight": 0.1, "prototype_weight": 0.0},
    "A2": {"name": "CE + Prototype", "supcon_weight": 0.0, "prototype_weight": 0.1},
    "A3": {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1},
}
ARM_DECISIONS = {
    "A0": "SELECT_CE",
    "A1": "SELECT_CE_SUPCON",
    "A2": "SELECT_CE_PROTOTYPE",
    "A3": "SELECT_CE_SUPCON_PROTOTYPE",
}
REQUIRED_OUTPUT_DIRS = (
    "configs",
    "checkpoints",
    "logs",
    "manifests",
    "embeddings",
    "metrics",
    "statistics",
    "tables",
    "figures",
    "reports",
    "publication",
    "cache",
)


class ScientificAbort(RuntimeError):
    """A hard scientific-contract failure that prevents a READY result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def natural_key(value: Any) -> Tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value)))


def json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def assert_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ScientificAbort(f"OUTPUT_PATH_ESCAPE: {resolved} is outside {root_resolved}")
    return resolved


def atomic_write_text(path: Path, text: str, output_root: Optional[Path] = None) -> None:
    if output_root is not None:
        assert_within(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any, output_root: Optional[Path] = None) -> None:
    atomic_write_text(path, json.dumps(json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", output_root)


def atomic_write_csv(path: Path, frame: pd.DataFrame, output_root: Optional[Path] = None) -> None:
    if output_root is not None:
        assert_within(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_states() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_states(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


@dataclass
class Stage26Config:
    branch_root: str = field(default_factory=lambda: os.environ.get(
        "WISIG_BRANCH_ROOT",
        "/content/drive/MyDrive/colab files /Surrogate-XAI/project_root/MANYTX_ZERO_DAY_BRANCH_v1.0.3",
    ))
    benchmark_path: str = ""
    stage2m_dir: str = ""
    output_dir: str = ""
    profile: str = "full"
    seeds: Tuple[int, ...] = (42, 123, 2026)
    max_epochs: int = 40
    minimum_epochs: int = 12
    early_stopping_patience: int = 10
    batch_size: int = 256
    samples_per_tx: int = 4
    eval_batch_size: int = 1024
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    embedding_dim: int = 128
    num_classes: int = 98
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    prototype_momentum: float = 0.95
    label_smoothing: float = 0.0
    dropout: float = 0.10
    augmentation_enabled: bool = True
    phase_rotation_radians: float = 0.12
    amplitude_jitter: float = 0.05
    awgn_std: float = 0.01
    maximum_circular_shift: int = 4
    amp_enabled: bool = True
    deterministic_algorithms: bool = True
    validation_every: int = 1
    cluster_sample_per_class: int = 100
    domain_sample_per_tx: int = 100
    covariance_fit_limit: int = 100_000
    bootstrap_iterations: int = 1000
    domain_cv_folds: int = 5
    practical_f1_delta: float = 0.002
    noninferiority_f1_delta: float = 0.001
    resume: bool = True
    stage_start: int = 1
    stage_end: int = 10
    device: str = "auto"

    @property
    def branch_root_path(self) -> Path:
        return Path(self.branch_root).expanduser().resolve()

    @property
    def benchmark_path_resolved(self) -> Path:
        if self.benchmark_path:
            return Path(self.benchmark_path).expanduser().resolve()
        return self.branch_root_path / "01_benchmark_engineering" / "benchmark" / f"{CANONICAL_BENCHMARK}.h5"

    @property
    def stage2m_path(self) -> Path:
        return Path(self.stage2m_dir).expanduser().resolve() if self.stage2m_dir else self.branch_root_path / "02_benchmark_diagnostics"

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else self.branch_root_path / "03_representation_ablation"

    def validate(self) -> None:
        if self.profile not in {"full", "pilot"}:
            raise ValueError("profile must be 'full' or 'pilot'")
        if tuple(self.seeds) != (42, 123, 2026):
            raise ScientificAbort("The frozen seed panel must be exactly 42, 123, 2026")
        if self.num_classes != EXPECTED_KNOWN_CLASSES or self.embedding_dim != 128:
            raise ScientificAbort("Stage 2.6M freezes 98 classes and a 128-dimensional embedding")
        if self.batch_size <= 0 or self.samples_per_tx <= 1 or self.batch_size % self.samples_per_tx:
            raise ValueError("batch_size must be divisible by samples_per_tx, which must exceed one")
        if not (1 <= self.stage_start <= self.stage_end <= 10):
            raise ValueError("stage range must be within 1..10")
        if self.profile == "pilot":
            self.max_epochs = min(self.max_epochs, 3)
            self.minimum_epochs = min(self.minimum_epochs, 1)
            self.early_stopping_patience = min(self.early_stopping_patience, 2)
            self.bootstrap_iterations = min(self.bootstrap_iterations, 100)
            self.cluster_sample_per_class = min(self.cluster_sample_per_class, 20)
            self.domain_sample_per_tx = min(self.domain_sample_per_tx, 20)
            self.covariance_fit_limit = min(self.covariance_fit_limit, 20_000)
        output = self.output_root
        if output.parent.resolve() != self.branch_root_path.resolve():
            raise ScientificAbort("Output must be the branch-root 03_representation_ablation directory")
        if output.name != "03_representation_ablation":
            raise ScientificAbort("Output directory name must be 03_representation_ablation")

    def frozen_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["benchmark_path_resolved"] = str(self.benchmark_path_resolved)
        payload["stage2m_path"] = str(self.stage2m_path)
        payload["output_root"] = str(self.output_root)
        return payload

    def configuration_sha256(self) -> str:
        excluded = {"stage_start", "stage_end", "resume", "device", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor"}
        payload = {k: v for k, v in self.frozen_payload().items() if k not in excluded}
        return sha256_object(payload)


def setup_logger(config: Stage26Config) -> logging.Logger:
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage2_6m")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(config.output_root / "logs" / "stage2_6m.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def startup_banner() -> str:
    return "\n".join([
        "=" * 100,
        "STAGE 2.6M — WISIG MANYTX CONTROLLED REPRESENTATION LEARNING & DOMAIN-ROBUST EMBEDDING ABLATION",
        "=" * 100,
        f"CANONICAL BENCHMARK: {CANONICAL_BENCHMARK}",
        "BENCHMARK MODE: READ ONLY",
        "STAGE 2M DIAGNOSTICS: VERIFIED / READ ONLY",
        "TRAINING DATA: TRAIN KNOWN ONLY",
        "KNOWN VALIDATION: P0 / P1 / P2 / P3",
        "CALIBRATION UNKNOWN: DIAGNOSTIC ONLY",
        "STRICT ZERO-DAY SIGNAL ACCESS: FORBIDDEN",
        "ARCHITECTURE SEARCH: DISABLED",
        "SURROGATE TRAINING: DISABLED",
        "FINAL ZERO-DAY EVALUATION: DISABLED",
        "XAI: DISABLED",
        "=" * 100,
    ])


def flatten_json_records(value: Any, trail: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from flatten_json_records(child, trail + (str(key),))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from flatten_json_records(child, trail + (str(idx),))
    else:
        yield trail, value


def iter_json_containers(value: Any) -> Iterator[Any]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from iter_json_containers(child)
    elif isinstance(value, list):
        yield value
        for child in value:
            yield from iter_json_containers(child)


class StrictZeroDayGuard:
    """Authorizes only named non-final partitions and records policy violations."""

    def __init__(self, branch_root: Path, output_root: Path):
        self.branch_root = branch_root.resolve()
        self.output_root = output_root.resolve()
        self.strict_test_signal_reads = 0
        self.strict_test_label_reads = 0
        self.strict_test_embedding_reads = 0
        self.strict_test_metric_reads = 0
        self.strict_test_threshold_reads = 0
        self._allowed_indices: Dict[str, np.ndarray] = {}
        self.strict_file_audit: List[Dict[str, Any]] = []

    @staticmethod
    def is_strict_path(path: Path) -> bool:
        token = normalize_token(path.name)
        return "strict_zero_day" in token or "zero_day_shift_test" in token

    def forbid_data_path(self, path: Path, operation: str) -> None:
        if self.is_strict_path(path):
            raise ScientificAbort(f"STRICT_ZERO_DAY_ACCESS_VIOLATION: {operation}: {path}")

    def register_allowed_indices(self, partition: str, indices: np.ndarray) -> None:
        if partition not in EXPECTED_PARTITION_COUNTS:
            raise ScientificAbort(f"Unapproved partition registration: {partition}")
        array = np.asarray(indices, dtype=np.int64)
        if array.ndim != 1 or len(np.unique(array)) != len(array):
            raise ScientificAbort(f"Invalid or duplicate indices in {partition}")
        self._allowed_indices[partition] = array

    def authorize_rows(self, partition: str, indices: np.ndarray, operation: str) -> None:
        if partition not in self._allowed_indices:
            raise ScientificAbort(f"Partition has not been authorized: {partition}")
        expected = self._allowed_indices[partition]
        requested = np.asarray(indices, dtype=np.int64)
        if not np.isin(requested, expected, assume_unique=False).all():
            raise ScientificAbort(f"STRICT_ZERO_DAY_ACCESS_VIOLATION: unauthorized rows in {operation}")

    def verify_strict_files_from_manifests(self) -> List[Dict[str, Any]]:
        engineering = self.branch_root / "01_benchmark_engineering"
        json_files = sorted(engineering.rglob("*.json"))
        manifests: List[Tuple[Path, Any]] = []
        for path in json_files:
            try:
                manifests.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        for filename, expected_count in STRICT_INDEX_EXPECTATIONS.items():
            matches = [p for p in engineering.rglob(filename) if p.is_file()]
            if len(matches) != 1:
                raise ScientificAbort(f"Expected exactly one frozen {filename}; found {len(matches)}")
            path = matches[0]
            actual_sha = sha256_file(path)
            declared_count: Optional[int] = None
            declared_sha: Optional[str] = None
            evidence_files: set[str] = set()
            name_token = normalize_token(filename)
            stem_token = normalize_token(Path(filename).stem)
            for manifest_path, payload in manifests:
                for container in iter_json_containers(payload):
                    serialized = normalize_token(json.dumps(json_ready(container), sort_keys=True))
                    if stem_token not in serialized and name_token not in serialized:
                        continue
                    evidence_files.add(str(manifest_path))
                    for trail, value in flatten_json_records(container):
                        joined = normalize_token("_".join(trail))
                        if isinstance(value, (int, float)) and any(t in joined for t in ("count", "samples", "size", "length")):
                            if int(value) == expected_count:
                                declared_count = int(value)
                        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()) and any(t in joined for t in ("sha", "hash", "digest")):
                            declared_sha = value.lower()
            if declared_count != expected_count:
                raise ScientificAbort(f"Frozen manifest count missing or wrong for {filename}")
            if declared_sha is not None and declared_sha != actual_sha:
                raise ScientificAbort(f"Frozen manifest SHA mismatch for {filename}")
            self.strict_file_audit.append({
                "path": str(path),
                "exists": True,
                "count_from_manifest": declared_count,
                "expected_count": expected_count,
                "sha256": actual_sha,
                "declared_sha256": declared_sha,
                "manifest_evidence": sorted(evidence_files),
                "loaded_into_memory": False,
            })
        return self.strict_file_audit

    def scan_output(self) -> None:
        if not self.output_root.exists():
            return
        for path in self.output_root.rglob("*"):
            if not path.is_file():
                continue
            token = normalize_token(path.name)
            if token in {normalize_token(x) for x in STRICT_FORBIDDEN_ARTIFACTS} or (
                "strict_zero_day" in token and path.name not in {"STRICT_TEST_GUARD.json"}
            ):
                raise ScientificAbort(f"Forbidden strict-test artifact exists: {path}")

    def counters(self) -> Dict[str, int]:
        return {
            "strict_test_signal_reads": self.strict_test_signal_reads,
            "strict_test_label_reads": self.strict_test_label_reads,
            "strict_test_embedding_reads": self.strict_test_embedding_reads,
            "strict_test_metric_reads": self.strict_test_metric_reads,
            "strict_test_threshold_reads": self.strict_test_threshold_reads,
        }

    def assert_zero(self) -> None:
        nonzero = {k: v for k, v in self.counters().items() if v != 0}
        if nonzero:
            raise ScientificAbort(f"STRICT_ZERO_DAY_ACCESS_VIOLATION: {nonzero}")

    def manifest(self) -> Dict[str, Any]:
        return {
            "policy": "STRICT ZERO-DAY SIGNAL/LABEL/EMBEDDING/METRIC/THRESHOLD ACCESS FORBIDDEN",
            "enforcement": [
                "authorized partition allowlist",
                "strict-path prohibition",
                "frozen authorized index arrays only",
                "output artifact scan",
                "static strict-artifact guard",
            ],
            "allowed_partitions": sorted(self._allowed_indices),
            "strict_files": self.strict_file_audit,
            "counter_semantics": "Violation counters raised by the structural guard; not independent instrumentation of every HDF5 operation.",
            "violation_counters": self.counters(),
            "all_violation_counters_zero": all(v == 0 for v in self.counters().values()),
            "generated_at": utc_now(),
        }


@dataclass
class MetadataField:
    dataset_path: str
    compound_field: Optional[str] = None


@dataclass
class PartitionData:
    name: str
    indices: np.ndarray
    transmitter_raw: np.ndarray
    labels: np.ndarray
    receiver: np.ndarray
    day: np.ndarray
    equalized: np.ndarray


@dataclass
class ResolvedBenchmark:
    h5_path: Path
    signal_key: str
    signal_orientation: str
    total_samples: int
    metadata_fields: Dict[str, MetadataField]
    partitions: Dict[str, PartitionData]
    transmitter_mapping: Dict[str, int]
    schema_rows: List[Dict[str, Any]]


def decode_string_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.array([x.decode("utf-8", errors="strict") for x in array.reshape(-1)], dtype=object)
    if array.dtype.kind == "O":
        out = []
        for value in array.reshape(-1):
            out.append(value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value))
        return np.array(out, dtype=object)
    return array.reshape(-1).astype(str).astype(object)


def normalize_equalization(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int8)
    true_tokens = {"1", "true", "yes", "equalized", "eq", "with_equalization"}
    false_tokens = {"0", "false", "no", "unequalized", "raw", "without_equalization", "not_equalized"}
    for i, value in enumerate(values):
        token = normalize_token(value)
        if token in true_tokens:
            result[i] = 1
        elif token in false_tokens:
            result[i] = 0
        else:
            try:
                numeric = int(float(str(value)))
            except ValueError as exc:
                raise ScientificAbort(f"Unrecognized equalization state: {value!r}") from exc
            if numeric not in (0, 1):
                raise ScientificAbort(f"Equalization state must be binary: {value!r}")
            result[i] = numeric
    return result


def read_h5_rows(dataset: h5py.Dataset, indices: np.ndarray, field_name: Optional[str] = None) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    if len(sorted_indices) and np.any(np.diff(sorted_indices) == 0):
        raise ScientificAbort("Duplicate HDF5 row request is not permitted")
    if field_name is None:
        data = dataset[sorted_indices]
    else:
        data = dataset.fields(field_name)[sorted_indices]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(data)[inverse]


def h5_schema(h5: h5py.File) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append({
                "path": name,
                "shape": tuple(int(v) for v in obj.shape),
                "dtype": str(obj.dtype),
                "compound_fields": list(obj.dtype.names or ()),
            })

    h5.visititems(visitor)
    return rows


def choose_signal_dataset(h5: h5py.File, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, str, int]:
    candidates: List[Tuple[int, str, str, int]] = []
    preferred = {"x": 0, "signals": 0, "signal": 1, "iq": 1, "waveforms": 2, "data": 3}
    for row in rows:
        shape = tuple(row["shape"])
        if len(shape) != 3:
            continue
        orientation = "channels_first" if shape[1:] == (2, 256) else "channels_last" if shape[1:] == (256, 2) else ""
        if not orientation:
            continue
        base = normalize_token(Path(str(row["path"])).name)
        score = preferred.get(base, 10)
        candidates.append((score, str(row["path"]), orientation, int(shape[0])))
    if not candidates:
        raise ScientificAbort("No HDF5 signal dataset with [N,2,256] or [N,256,2] shape was found")
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, path, orientation, total = candidates[0]
    if total != EXPECTED_TOTAL_SAMPLES:
        raise ScientificAbort(f"Signal dataset sample count {total} != {EXPECTED_TOTAL_SAMPLES}")
    return path, orientation, total


def choose_metadata_field(
    h5: h5py.File,
    rows: Sequence[Mapping[str, Any]],
    total: int,
    aliases: Sequence[str],
) -> MetadataField:
    aliases_norm = [normalize_token(x) for x in aliases]
    candidates: List[Tuple[int, MetadataField]] = []
    for row in rows:
        shape = tuple(row["shape"])
        if not shape or int(shape[0]) != total:
            continue
        path = str(row["path"])
        base = normalize_token(Path(path).name)
        if base in aliases_norm:
            candidates.append((aliases_norm.index(base), MetadataField(path)))
        for compound_name in row.get("compound_fields", []):
            token = normalize_token(compound_name)
            if token in aliases_norm:
                candidates.append((aliases_norm.index(token), MetadataField(path, str(compound_name))))
    if not candidates:
        raise ScientificAbort(f"Required metadata field not found; aliases={aliases}")
    candidates.sort(key=lambda x: (x[0], x[1].dataset_path, x[1].compound_field or ""))
    return candidates[0][1]


PARTITION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train_known": ("train_known", "known_train", "train_known_indices"),
    "p0": ("p0", "validation_known", "p0_known_validation", "known_validation_p0", "p0_indices"),
    "p1": ("p1", "p1_cross_day_validation", "cross_day_validation", "p1_indices"),
    "p2": ("p2", "p2_cross_receiver_validation", "cross_receiver_validation", "p2_indices"),
    "p3": ("p3", "p3_cross_day_receiver_validation", "cross_day_receiver_validation", "p3_indices"),
    "calibration_unknown": ("calibration_unknown", "cal_unknown", "calibration_unknown_indices"),
}


def discover_partition_index_files(branch_root: Path) -> Dict[str, Path]:
    engineering = branch_root / "01_benchmark_engineering"
    result: Dict[str, Path] = {}
    for path in sorted(engineering.rglob("*.npy")):
        if StrictZeroDayGuard.is_strict_path(path):
            continue
        token = normalize_token(path.stem)
        token_without_indices = re.sub(r"(_global)?_indices$", "", token)
        for partition, aliases in PARTITION_ALIASES.items():
            normalized_aliases = {normalize_token(a) for a in aliases}
            normalized_aliases |= {re.sub(r"(_global)?_indices$", "", a) for a in normalized_aliases}
            if token in normalized_aliases or token_without_indices in normalized_aliases:
                if partition in result and result[partition].resolve() != path.resolve():
                    raise ScientificAbort(f"Multiple index files match {partition}: {result[partition]} and {path}")
                result[partition] = path
    return result


def resolve_partition_indices(
    branch_root: Path,
    total: int,
    guard: StrictZeroDayGuard,
) -> Dict[str, np.ndarray]:
    index_files = discover_partition_index_files(branch_root)
    partitions: Dict[str, np.ndarray] = {}
    for partition, path in index_files.items():
        guard.forbid_data_path(path, "partition index load")
        values = np.load(path, allow_pickle=False)
        partitions[partition] = np.asarray(values, dtype=np.int64).reshape(-1)
    missing = set(EXPECTED_PARTITION_COUNTS) - set(partitions)
    if missing:
        searched_root = branch_root / "01_benchmark_engineering"
        raise ScientificAbort(
            "Missing required frozen authorized partition index arrays: "
            f"{sorted(missing)}. Searched beneath {searched_root}. "
            "Stage 2.6M consumes the six frozen Stage 1B split arrays and never reconstructs splits from HDF5 metadata."
        )
    for partition, expected_count in EXPECTED_PARTITION_COUNTS.items():
        values = np.asarray(partitions[partition], dtype=np.int64)
        if len(values) != expected_count:
            raise ScientificAbort(f"{partition} count {len(values):,} != frozen count {expected_count:,}")
        if len(np.unique(values)) != len(values) or (len(values) and (values.min() < 0 or values.max() >= total)):
            raise ScientificAbort(f"{partition} indices are invalid")
        guard.register_allowed_indices(partition, values)
    owners: Dict[int, str] = {}
    for partition, values in partitions.items():
        for index in values:
            key = int(index)
            if key in owners:
                raise ScientificAbort(f"Allowed partitions overlap at global index {key}: {owners[key]} and {partition}")
            owners[key] = partition
    return partitions


def resolve_benchmark(config: Stage26Config, guard: StrictZeroDayGuard) -> ResolvedBenchmark:
    h5_path = config.benchmark_path_resolved
    if not h5_path.is_file():
        raise ScientificAbort(f"Canonical benchmark missing: {h5_path}")
    actual_sha = sha256_file(h5_path)
    if actual_sha != EXPECTED_BENCHMARK_SHA256:
        raise ScientificAbort("ABORT_STAGE_2_6M\nBENCHMARK_HASH_MISMATCH")
    with h5py.File(h5_path, "r", swmr=True) as handle:
        rows = h5_schema(handle)
        signal_key, orientation, total = choose_signal_dataset(handle, rows)
        fields = {
            "transmitter": choose_metadata_field(handle, rows, total, ("transmitter_id", "tx_id", "transmitter", "tx")),
            "receiver": choose_metadata_field(handle, rows, total, ("receiver_id", "rx_id", "receiver", "rx")),
            "day": choose_metadata_field(handle, rows, total, ("day_id", "capture_date", "capture_day", "date", "day")),
            "equalized": choose_metadata_field(handle, rows, total, ("equalization_state", "equalized", "equalization", "eq_state")),
        }
        raw_indices = resolve_partition_indices(config.branch_root_path, total, guard)
        raw_meta: Dict[str, Dict[str, np.ndarray]] = {}
        for partition, indices in raw_indices.items():
            guard.authorize_rows(partition, indices, "metadata read")
            raw_meta[partition] = {}
            for name, field_ref in fields.items():
                values = read_h5_rows(handle[field_ref.dataset_path], indices, field_ref.compound_field)
                raw_meta[partition][name] = decode_string_array(values)
    train_tx = sorted(set(str(v) for v in raw_meta["train_known"]["transmitter"]), key=natural_key)
    if len(train_tx) != EXPECTED_KNOWN_CLASSES:
        raise ScientificAbort(f"Train Known contains {len(train_tx)} transmitter identities, expected 98")
    tx_mapping = {tx: idx for idx, tx in enumerate(train_tx)}
    partitions: Dict[str, PartitionData] = {}
    for name, indices in raw_indices.items():
        tx_raw = np.array([str(v) for v in raw_meta[name]["transmitter"]], dtype=object)
        if name == "calibration_unknown":
            unknown_tx = set(tx_raw)
            if unknown_tx & set(tx_mapping):
                raise ScientificAbort("Calibration Unknown overlaps Train Known transmitter identities")
            labels = np.full(len(indices), -1, dtype=np.int16)
        else:
            absent = sorted(set(tx_raw) - set(tx_mapping), key=natural_key)
            if absent:
                raise ScientificAbort(f"Known protocol {name} has unknown transmitters: {absent}")
            labels = np.array([tx_mapping[str(v)] for v in tx_raw], dtype=np.int16)
        partitions[name] = PartitionData(
            name=name,
            indices=np.asarray(indices, dtype=np.int64),
            transmitter_raw=tx_raw,
            labels=labels,
            receiver=np.asarray(raw_meta[name]["receiver"], dtype=object),
            day=np.asarray(raw_meta[name]["day"], dtype=object),
            equalized=normalize_equalization(raw_meta[name]["equalized"]),
        )
    calibration_tx_count = len(set(partitions["calibration_unknown"].transmitter_raw))
    if calibration_tx_count != EXPECTED_CAL_UNKNOWN_CLASSES:
        raise ScientificAbort(f"Calibration Unknown contains {calibration_tx_count} transmitters, expected 22")
    return ResolvedBenchmark(
        h5_path=h5_path,
        signal_key=signal_key,
        signal_orientation=orientation,
        total_samples=total,
        metadata_fields=fields,
        partitions=partitions,
        transmitter_mapping=tx_mapping,
        schema_rows=rows,
    )


class WiSigH5Dataset(Dataset):
    """Worker-safe, read-only HDF5 view over one explicitly authorized partition."""

    def __init__(self, benchmark: ResolvedBenchmark, partition: str, guard: StrictZeroDayGuard):
        if partition not in benchmark.partitions:
            raise KeyError(partition)
        self.h5_path = benchmark.h5_path
        self.signal_key = benchmark.signal_key
        self.orientation = benchmark.signal_orientation
        self.partition = partition
        self.metadata = benchmark.partitions[partition]
        guard.authorize_rows(partition, self.metadata.indices, "dataset construction")
        self._h5: Optional[h5py.File] = None
        self._signals: Optional[h5py.Dataset] = None

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._signals = self._h5[self.signal_key]

    def __len__(self) -> int:
        return len(self.metadata.indices)

    def __getitem__(self, position: int) -> Dict[str, torch.Tensor]:
        self._open()
        assert self._signals is not None
        global_index = int(self.metadata.indices[position])
        value = np.asarray(self._signals[global_index], dtype=np.float32)
        if self.orientation == "channels_last":
            value = value.T
        if value.shape != EXPECTED_SIGNAL_SHAPE:
            raise ScientificAbort(f"Signal row {global_index} has shape {value.shape}, expected {EXPECTED_SIGNAL_SHAPE}")
        if not np.isfinite(value).all():
            raise ScientificAbort(f"Non-finite signal values at global index {global_index}")
        return {
            "x": torch.from_numpy(np.ascontiguousarray(value)),
            "y": torch.tensor(int(self.metadata.labels[position]), dtype=torch.long),
            "position": torch.tensor(int(position), dtype=torch.long),
            "global_index": torch.tensor(global_index, dtype=torch.long),
            "equalized": torch.tensor(int(self.metadata.equalized[position]), dtype=torch.long),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._signals = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_signals"] = None
        return state

    def __del__(self) -> None:
        self.close()


def worker_init_fn(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DomainBalancedTxSampler(Sampler[List[int]]):
    """Deterministic Tx-primary batches diversified by Rx × day × equalization cell."""

    def __init__(
        self,
        labels: np.ndarray,
        receiver: np.ndarray,
        day: np.ndarray,
        equalized: np.ndarray,
        batch_size: int,
        samples_per_tx: int,
        seed: int,
        epoch: int,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.receiver = np.asarray(receiver, dtype=object)
        self.day = np.asarray(day, dtype=object)
        self.equalized = np.asarray(equalized, dtype=np.int8)
        self.batch_size = int(batch_size)
        self.samples_per_tx = int(samples_per_tx)
        self.classes_per_batch = self.batch_size // self.samples_per_tx
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.steps = math.ceil(len(self.labels) / self.batch_size)
        self.class_cells: Dict[int, Dict[Tuple[str, str, int], np.ndarray]] = {}
        for label in sorted(np.unique(self.labels).tolist()):
            positions = np.flatnonzero(self.labels == label)
            cells: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
            for pos in positions:
                cells[(str(self.receiver[pos]), str(self.day[pos]), int(self.equalized[pos]))].append(int(pos))
            self.class_cells[int(label)] = {
                key: np.asarray(values, dtype=np.int64) for key, values in sorted(cells.items(), key=lambda kv: str(kv[0]))
            }
        if len(self.class_cells) != EXPECTED_KNOWN_CLASSES:
            raise ScientificAbort(f"Sampler received {len(self.class_cells)} classes instead of 98")

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + 104_729 * self.epoch)
        classes = np.asarray(sorted(self.class_cells), dtype=np.int64)
        class_order = rng.permutation(classes).tolist()
        class_cursor = 0
        cell_orders: Dict[int, List[Tuple[str, str, int]]] = {}
        cell_cursors: Dict[int, int] = {}
        sample_orders: Dict[Tuple[int, Tuple[str, str, int]], np.ndarray] = {}
        sample_cursors: Dict[Tuple[int, Tuple[str, str, int]], int] = {}
        for label, cells in self.class_cells.items():
            keys = list(cells)
            rng.shuffle(keys)
            cell_orders[label] = keys
            cell_cursors[label] = 0
            for key, values in cells.items():
                sample_orders[(label, key)] = rng.permutation(values)
                sample_cursors[(label, key)] = 0

        def next_class() -> int:
            nonlocal class_cursor, class_order
            if class_cursor >= len(class_order):
                class_order = rng.permutation(classes).tolist()
                class_cursor = 0
            value = int(class_order[class_cursor])
            class_cursor += 1
            return value

        def next_position(label: int) -> int:
            keys = cell_orders[label]
            cell_cursor = cell_cursors[label]
            key = keys[cell_cursor % len(keys)]
            cell_cursors[label] = cell_cursor + 1
            compound = (label, key)
            order = sample_orders[compound]
            cursor = sample_cursors[compound]
            if cursor >= len(order):
                order = rng.permutation(self.class_cells[label][key])
                sample_orders[compound] = order
                cursor = 0
            value = int(order[cursor])
            sample_cursors[compound] = cursor + 1
            return value

        for _ in range(self.steps):
            chosen_classes: List[int] = []
            while len(chosen_classes) < self.classes_per_batch:
                candidate = next_class()
                if candidate not in chosen_classes:
                    chosen_classes.append(candidate)
            batch: List[int] = []
            for label in chosen_classes:
                for _sample in range(self.samples_per_tx):
                    batch.append(next_position(label))
            if len(batch) != self.batch_size:
                raise ScientificAbort("Sampler produced an incorrectly sized batch")
            yield batch

    def manifest(self) -> Dict[str, Any]:
        domain_counts = []
        for label, cells in self.class_cells.items():
            positions = np.flatnonzero(self.labels == label)
            domain_counts.append({
                "class_index": label,
                "samples": int(len(positions)),
                "receivers": int(len(set(self.receiver[positions]))),
                "days": int(len(set(self.day[positions]))),
                "equalization_states": int(len(set(self.equalized[positions].tolist()))),
                "domain_cells": int(len(cells)),
            })
        return {
            "policy": "Tx-primary deterministic sampling with within-Tx Rx/day/equalization-cell cycling",
            "batch_size": self.batch_size,
            "samples_per_tx": self.samples_per_tx,
            "classes_per_batch": self.classes_per_batch,
            "steps_per_epoch": self.steps,
            "seed": self.seed,
            "epoch": self.epoch,
            "per_class": domain_counts,
        }


class StaticBatchSampler(Sampler[List[int]]):
    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(map(int, batch)) for batch in batches]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def batch_exposure_sha256(batches: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.asarray(batch, dtype=np.int64).tobytes())
    return digest.hexdigest()


def build_loader(
    dataset: WiSigH5Dataset,
    config: Stage26Config,
    *,
    batches: Optional[Sequence[Sequence[int]]] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    common: Dict[str, Any] = {
        "dataset": dataset,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    if config.num_workers > 0:
        common["persistent_workers"] = config.persistent_workers
        common["prefetch_factor"] = config.prefetch_factor
    if batches is not None:
        common["batch_sampler"] = StaticBatchSampler(batches)
    else:
        common.update({"batch_size": config.eval_batch_size, "shuffle": shuffle, "drop_last": False})
    return DataLoader(**common)


class RFAugmentation:
    def __init__(self, config: Stage26Config):
        self.config = config

    def __call__(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.config.augmentation_enabled:
            return x
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        phase = (torch.rand(batch, 1, 1, device=device, generator=generator, dtype=dtype) * 2 - 1) * self.config.phase_rotation_radians
        cosine, sine = torch.cos(phase), torch.sin(phase)
        i, q = x[:, 0:1], x[:, 1:2]
        x = torch.cat((i * cosine - q * sine, i * sine + q * cosine), dim=1)
        scale = 1 + (torch.rand(batch, 1, 1, device=device, generator=generator, dtype=dtype) * 2 - 1) * self.config.amplitude_jitter
        x = x * scale
        if self.config.awgn_std > 0:
            rms = x.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8).to(dtype)
            noise = torch.randn(x.shape, device=device, generator=generator, dtype=dtype)
            x = x + noise * (self.config.awgn_std * rms)
        maximum = self.config.maximum_circular_shift
        if maximum > 0:
            shifts = torch.randint(-maximum, maximum + 1, (batch,), device=device, generator=generator)
            x = torch.stack([torch.roll(x[j], int(shifts[j].item()), dims=-1) for j in range(batch)], dim=0)
        return x


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 5, stride=stride, dilation=dilation)
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class WiSigRepresentationNet(nn.Module):
    """Common compact RF temporal backbone shared unchanged by A0–A3."""

    def __init__(self, num_classes: int = 98, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.iq_mixer = nn.Sequential(
            nn.Conv1d(2, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ConvNormAct(32, 64, 7, stride=2),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(64, 64, dilation=1, dropout=dropout),
            ResidualTemporalBlock(64, 128, stride=2, dilation=1, dropout=dropout),
            ResidualTemporalBlock(128, 128, dilation=2, dropout=dropout),
            ResidualTemporalBlock(128, 256, stride=2, dilation=1, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.temporal(self.iq_mixer(x))
        pooled = torch.cat((features.mean(dim=-1), features.float().var(dim=-1, unbiased=False).add(1e-8).sqrt().to(features.dtype)), dim=1)
        embedding_raw = self.projection(pooled)
        embedding_normalized = F.normalize(embedding_raw.float(), dim=1, eps=1e-8)
        logits = self.classifier(embedding_raw)
        return {
            "logits": logits,
            "embedding_raw": embedding_raw,
            "embedding_normalized": embedding_normalized,
        }


def architecture_signature(model: nn.Module) -> str:
    payload = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()]
    return sha256_object(payload)


class EMAPrototypeBank(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, momentum: float):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, embedding_dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("updates", torch.zeros(num_classes, dtype=torch.long))

    def targets(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = self.prototypes[labels].detach().clone()
        for label in labels.unique():
            k = int(label.item())
            if not bool(self.initialized[k]):
                centroid = F.normalize(z[labels == label].detach().mean(dim=0, keepdim=True), dim=1).squeeze(0)
                targets[labels == label] = centroid
        return targets

    def loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = self.targets(z.float(), labels)
        return (z.float() - target.float()).square().sum(dim=1).mean()

    @torch.no_grad()
    def update(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        for label in labels.unique():
            k = int(label.item())
            centroid = F.normalize(z[labels == label].float().mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if bool(self.initialized[k]):
                value = self.momentum * self.prototypes[k] + (1 - self.momentum) * centroid
                self.prototypes[k] = F.normalize(value.unsqueeze(0), dim=1).squeeze(0)
            else:
                self.prototypes[k] = centroid
                self.initialized[k] = True
            self.updates[k] += 1


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z.float(), dim=1, eps=1e-8)
    logits = torch.matmul(z, z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(labels), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    denominator_mask = ~identity
    exp_logits = torch.exp(logits) * denominator_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    mean_log_prob = (log_prob * positives).sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_prob[valid].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite supervised contrastive loss")
    return loss


def objective_loss(
    arm: str,
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    prototypes: EMAPrototypeBank,
    config: Stage26Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    definition = ARM_DEFINITIONS[arm]
    logits = outputs["logits"].float()
    z = outputs["embedding_normalized"].float()
    ce = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    supcon = z.sum() * 0.0
    proto = z.sum() * 0.0
    if definition["supcon_weight"] > 0:
        supcon = supervised_contrastive_loss(z, labels, config.temperature)
    if definition["prototype_weight"] > 0:
        proto = prototypes.loss(z, labels)
    total = ce + definition["supcon_weight"] * supcon + definition["prototype_weight"] * proto
    components = {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "supcon": float(supcon.detach().cpu()),
        "prototype": float(proto.detach().cpu()),
    }
    return total, components


class StreamingClassificationMetrics:
    def __init__(self, num_classes: int = 98, ece_bins: int = 15):
        self.num_classes = num_classes
        self.ece_bins = ece_bins
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.count = 0
        self.correct = 0
        self.top5_correct = 0
        self.ce_sum = 0.0
        self.ece_count = np.zeros(ece_bins, dtype=np.int64)
        self.ece_conf = np.zeros(ece_bins, dtype=np.float64)
        self.ece_correct = np.zeros(ece_bins, dtype=np.float64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits_f = logits.detach().float()
        labels_l = labels.detach().long()
        probabilities = torch.softmax(logits_f, dim=1)
        predictions = logits_f.argmax(dim=1)
        top5 = logits_f.topk(k=min(5, logits_f.shape[1]), dim=1).indices
        self.ce_sum += float(F.cross_entropy(logits_f, labels_l, reduction="sum").cpu())
        self.count += int(len(labels_l))
        self.correct += int((predictions == labels_l).sum().cpu())
        self.top5_correct += int(top5.eq(labels_l[:, None]).any(dim=1).sum().cpu())
        y = labels_l.cpu().numpy()
        pred = predictions.cpu().numpy()
        np.add.at(self.confusion, (y, pred), 1)
        confidence = probabilities.max(dim=1).values.cpu().numpy()
        correctness = (pred == y).astype(np.float64)
        bins = np.minimum((confidence * self.ece_bins).astype(np.int64), self.ece_bins - 1)
        for bin_index in range(self.ece_bins):
            mask = bins == bin_index
            if mask.any():
                self.ece_count[bin_index] += int(mask.sum())
                self.ece_conf[bin_index] += float(confidence[mask].sum())
                self.ece_correct[bin_index] += float(correctness[mask].sum())

    def finish(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        if self.count == 0:
            raise ScientificAbort("Cannot finalize empty classification metrics")
        true_support = self.confusion.sum(axis=1).astype(np.float64)
        predicted_support = self.confusion.sum(axis=0).astype(np.float64)
        true_positive = np.diag(self.confusion).astype(np.float64)
        recall = np.divide(true_positive, true_support, out=np.zeros_like(true_positive), where=true_support > 0)
        precision = np.divide(true_positive, predicted_support, out=np.zeros_like(true_positive), where=predicted_support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)
        observed = true_support > 0
        ece = 0.0
        for idx in range(self.ece_bins):
            if self.ece_count[idx]:
                avg_conf = self.ece_conf[idx] / self.ece_count[idx]
                avg_acc = self.ece_correct[idx] / self.ece_count[idx]
                ece += self.ece_count[idx] / self.count * abs(avg_acc - avg_conf)
        metrics = {
            "samples": float(self.count),
            "accuracy": self.correct / self.count,
            "top5_accuracy": self.top5_correct / self.count,
            "cross_entropy": self.ce_sum / self.count,
            "ece": float(ece),
            "observed_class_macro_f1": float(f1[observed].mean()),
            "fixed98_macro_f1": float(f1.mean()),
            "observed_class_balanced_accuracy": float(recall[observed].mean()),
            "fixed98_balanced_accuracy": float(recall.mean()),
            "observed_classes": float(observed.sum()),
        }
        per_class = pd.DataFrame({
            "class_index": np.arange(self.num_classes),
            "support": true_support.astype(np.int64),
            "predicted": predicted_support.astype(np.int64),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        return metrics, per_class


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    model.eval()
    accumulator = StreamingClassificationMetrics(EXPECTED_KNOWN_CLASSES)
    use_amp = amp_enabled and device.type == "cuda"
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(x)
        accumulator.update(outputs["logits"], y)
    metrics, per_class = accumulator.finish()
    return metrics, per_class, accumulator.confusion


def resolve_device(config: Stage26Config) -> torch.device:
    if config.device != "auto":
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ScientificAbort("CUDA was requested but is unavailable")
    return device


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(path: Path, payload: Any, output_root: Path) -> None:
    assert_within(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@dataclass
class TrainingRun:
    arm: str
    seed: int
    model: WiSigRepresentationNet
    prototypes: EMAPrototypeBank
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: Any
    best_known: float = -math.inf
    best_selection: Tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)
    best_metrics: Dict[str, Any] = field(default_factory=dict)


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def create_training_runs(config: Stage26Config, seed: int, device: torch.device) -> Dict[str, TrainingRun]:
    seed_everything(seed, config.deterministic_algorithms)
    base_model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout)
    base_state = copy.deepcopy(base_model.state_dict())
    runs: Dict[str, TrainingRun] = {}
    for arm in ARM_DEFINITIONS:
        model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout)
        model.load_state_dict(base_state, strict=True)
        model.to(device)
        prototypes = EMAPrototypeBank(config.num_classes, config.embedding_dim, config.prototype_momentum).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.01)
        scaler = make_grad_scaler(config.amp_enabled and device.type == "cuda")
        runs[arm] = TrainingRun(arm, seed, model, prototypes, optimizer, scheduler, scaler)
    signatures = {arm: architecture_signature(run.model) for arm, run in runs.items()}
    if len(set(signatures.values())) != 1:
        raise ScientificAbort(f"Architecture mismatch between arms: {signatures}")
    initial_state_hashes = {arm: sha256_object([(k, v.detach().cpu().numpy().tobytes().hex()) for k, v in run.model.state_dict().items()]) for arm, run in runs.items()}
    if len(set(initial_state_hashes.values())) != 1:
        raise ScientificAbort("Same-seed arm initialization states differ")
    return runs


def checkpoint_dir(config: Stage26Config, arm: str, seed: int) -> Path:
    return config.output_root / "checkpoints" / arm / f"seed_{seed}"


def checkpoint_payload(
    run: TrainingRun,
    config: Stage26Config,
    epoch: int,
    benchmark_sha: str,
    script_sha: str,
    exposure_sha: str,
    group_state: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model_state": run.model.state_dict(),
        "optimizer_state": run.optimizer.state_dict(),
        "scheduler_state": run.scheduler.state_dict(),
        "scaler_state": run.scaler.state_dict(),
        "epoch": int(epoch),
        "arm": run.arm,
        "seed": run.seed,
        "best_metrics": run.best_metrics,
        "best_known": run.best_known,
        "best_selection": run.best_selection,
        "config": config.frozen_payload(),
        "configuration_sha": config.configuration_sha256(),
        "benchmark_sha": benchmark_sha,
        "stage2m_sha": EXPECTED_STAGE2M_SCRIPT_SHA256,
        "script_sha": script_sha,
        "rng_states": capture_rng_states(),
        "prototype_state": run.prototypes.state_dict(),
        "sampler_state": {"epoch": epoch, "exposure_sha256": exposure_sha},
        "architecture_signature": architecture_signature(run.model),
        "loss_coefficients": ARM_DEFINITIONS[run.arm],
        "group_state": dict(group_state),
    }


def validate_checkpoint(
    payload: Mapping[str, Any],
    run: TrainingRun,
    config: Stage26Config,
    benchmark_sha: str,
    script_sha: str,
) -> None:
    checks = {
        "benchmark SHA": payload.get("benchmark_sha") == benchmark_sha,
        "Stage 2M SHA": payload.get("stage2m_sha") == EXPECTED_STAGE2M_SCRIPT_SHA256,
        "script SHA": payload.get("script_sha") == script_sha,
        "configuration SHA": payload.get("configuration_sha") == config.configuration_sha256(),
        "arm": payload.get("arm") == run.arm,
        "seed": int(payload.get("seed", -1)) == run.seed,
        "architecture signature": payload.get("architecture_signature") == architecture_signature(run.model),
        "loss coefficients": payload.get("loss_coefficients") == ARM_DEFINITIONS[run.arm],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ScientificAbort(f"INVALIDATE_AND_RESTART_ARM: {run.arm}/seed={run.seed}; mismatches={failed}")


def resume_training_runs(
    runs: Dict[str, TrainingRun],
    config: Stage26Config,
    benchmark_sha: str,
    script_sha: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, Dict[str, Any]]:
    if not config.resume:
        return 0, {"stale_epochs": 0, "epochs_without_any_improvement": 0}
    paths = {arm: checkpoint_dir(config, arm, run.seed) / "last.pt" for arm, run in runs.items()}
    existing = {arm: path.is_file() for arm, path in paths.items()}
    if not any(existing.values()):
        return 0, {"stale_epochs": 0, "epochs_without_any_improvement": 0}
    if not all(existing.values()):
        raise ScientificAbort(f"INVALIDATE_AND_RESTART_ARM: incomplete synchronized seed checkpoint set {existing}")
    epochs = set()
    group_states = []
    for arm, run in runs.items():
        payload = safe_torch_load(paths[arm], map_location=device)
        validate_checkpoint(payload, run, config, benchmark_sha, script_sha)
        run.model.load_state_dict(payload["model_state"], strict=True)
        run.optimizer.load_state_dict(payload["optimizer_state"])
        run.scheduler.load_state_dict(payload["scheduler_state"])
        run.scaler.load_state_dict(payload["scaler_state"])
        run.prototypes.load_state_dict(payload["prototype_state"], strict=True)
        run.best_metrics = dict(payload.get("best_metrics", {}))
        run.best_known = float(payload.get("best_known", -math.inf))
        run.best_selection = tuple(payload.get("best_selection", (-math.inf, -math.inf, -math.inf)))  # type: ignore[assignment]
        epochs.add(int(payload["epoch"]))
        group_states.append(dict(payload.get("group_state", {})))
    if len(epochs) != 1 or len({sha256_object(x) for x in group_states}) != 1:
        raise ScientificAbort("INVALIDATE_AND_RESTART_ARM: synchronized arm checkpoints disagree")
    completed_epoch = epochs.pop()
    logger.info("Resuming synchronized seed %s after epoch %d", next(iter(runs.values())).seed, completed_epoch)
    return completed_epoch, group_states[0]


def train_one_epoch(
    run: TrainingRun,
    loader: DataLoader,
    augmentation: RFAugmentation,
    config: Stage26Config,
    device: torch.device,
    augmentation_seed: int,
) -> Dict[str, float]:
    run.model.train()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(augmentation_seed)
    sums = defaultdict(float)
    samples = 0
    successful_steps = 0
    nonfinite_events = 0
    equalized_counts = np.zeros(2, dtype=np.int64)
    use_amp = config.amp_enabled and device.type == "cuda"
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        eq = batch["equalized"].cpu().numpy()
        equalized_counts += np.bincount(eq, minlength=2)[:2]
        x = augmentation(x, generator)
        run.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = run.model(x)
        loss, components = objective_loss(run.arm, outputs, y, run.prototypes, config)
        if not torch.isfinite(loss):
            nonfinite_events += 1
            if nonfinite_events >= 2:
                raise ScientificAbort(f"Persistent non-finite loss for {run.arm}, seed {run.seed}")
            continue
        previous_scale = float(run.scaler.get_scale())
        run.scaler.scale(loss).backward()
        run.scaler.unscale_(run.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), max_norm=5.0)
        if not torch.isfinite(gradient_norm):
            nonfinite_events += 1
            run.optimizer.zero_grad(set_to_none=True)
            if nonfinite_events >= 2:
                raise ScientificAbort(f"Persistent non-finite gradients for {run.arm}, seed {run.seed}")
            continue
        run.scaler.step(run.optimizer)
        run.scaler.update()
        current_scale = float(run.scaler.get_scale())
        step_succeeded = not use_amp or current_scale >= previous_scale
        if step_succeeded:
            successful_steps += 1
            if ARM_DEFINITIONS[run.arm]["prototype_weight"] > 0:
                run.prototypes.update(outputs["embedding_normalized"].detach(), y)
        batch_n = len(y)
        samples += batch_n
        for key, value in components.items():
            sums[key] += value * batch_n
        sums["gradient_norm"] += float(gradient_norm.detach().cpu()) * batch_n
    if samples == 0 or successful_steps == 0:
        raise ScientificAbort(f"No successful optimizer steps for {run.arm}, seed {run.seed}")
    run.scheduler.step()
    result = {key: value / samples for key, value in sums.items()}
    result.update({
        "samples": float(samples),
        "successful_optimizer_steps": float(successful_steps),
        "nonfinite_events": float(nonfinite_events),
        "equalized_0": float(equalized_counts[0]),
        "equalized_1": float(equalized_counts[1]),
        "learning_rate": float(run.optimizer.param_groups[0]["lr"]),
    })
    return result


def train_seed_group(
    config: Stage26Config,
    benchmark: ResolvedBenchmark,
    guard: StrictZeroDayGuard,
    seed: int,
    device: torch.device,
    logger: logging.Logger,
    benchmark_sha: str,
    script_sha: str,
) -> pd.DataFrame:
    train_dataset = WiSigH5Dataset(benchmark, "train_known", guard)
    validation_datasets = {name: WiSigH5Dataset(benchmark, name, guard) for name in ("p0", "p1", "p2", "p3")}
    validation_loaders = {
        name: build_loader(dataset, config, seed=seed + 10_000 + index)
        for index, (name, dataset) in enumerate(validation_datasets.items())
    }
    runs = create_training_runs(config, seed, device)
    start_epoch, group_state = resume_training_runs(runs, config, benchmark_sha, script_sha, device, logger)
    stale_epochs = int(group_state.get("epochs_without_any_improvement", 0))
    history_path = config.output_root / "logs" / f"training_history_seed_{seed}.csv"
    if history_path.is_file() and start_epoch:
        history_rows = pd.read_csv(history_path).to_dict("records")
        history_rows = [row for row in history_rows if int(row["epoch"]) <= start_epoch]
    else:
        history_rows: List[Dict[str, Any]] = []
    augmentation = RFAugmentation(config)
    for epoch in range(start_epoch + 1, config.max_epochs + 1):
        sampler = DomainBalancedTxSampler(
            train_dataset.metadata.labels,
            train_dataset.metadata.receiver,
            train_dataset.metadata.day,
            train_dataset.metadata.equalized,
            config.batch_size,
            config.samples_per_tx,
            seed,
            epoch,
        )
        batches = list(iter(sampler))
        exposure_sha = batch_exposure_sha256(batches)
        any_improvement = False
        epoch_rows: List[Dict[str, Any]] = []
        for arm, run in runs.items():
            stochastic_seed = seed * 1_000_003 + epoch
            seed_everything(stochastic_seed, config.deterministic_algorithms)
            loader = build_loader(train_dataset, config, batches=batches, seed=seed + epoch * 100)
            train_metrics = train_one_epoch(
                run,
                loader,
                augmentation,
                config,
                device,
                augmentation_seed=stochastic_seed,
            )
            protocol_metrics: Dict[str, Dict[str, float]] = {}
            if epoch % config.validation_every == 0 or epoch == config.max_epochs:
                for protocol, validation_loader in validation_loaders.items():
                    metrics, _, _ = evaluate_classifier(run.model, validation_loader, device, config.amp_enabled)
                    protocol_metrics[protocol] = metrics
            if len(protocol_metrics) != 4:
                raise ScientificAbort("Every checkpointing epoch must evaluate P0, P1, P2, and P3")
            known_mean = float(np.mean([protocol_metrics[p]["fixed98_macro_f1"] for p in ("p0", "p1", "p2", "p3")]))
            known_worst = float(min(protocol_metrics[p]["fixed98_macro_f1"] for p in ("p0", "p1", "p2", "p3")))
            selection_tuple = (known_mean, known_worst, protocol_metrics["p0"]["fixed98_macro_f1"])
            row: Dict[str, Any] = {
                "arm": arm,
                "arm_name": ARM_DEFINITIONS[arm]["name"],
                "seed": seed,
                "epoch": epoch,
                "exposure_sha256": exposure_sha,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "known_macro_fixed_mean": known_mean,
                "known_macro_fixed_worst": known_worst,
            }
            for protocol, values in protocol_metrics.items():
                for key, value in values.items():
                    row[f"{protocol}_{key}"] = value
            epoch_rows.append(row)
            run.best_metrics = {"protocols": protocol_metrics, "known_macro_fixed_mean": known_mean, "epoch": epoch}
            is_known_best = known_mean > run.best_known + 1e-12
            is_selection_best = selection_tuple > run.best_selection
            if is_known_best:
                run.best_known = known_mean
                any_improvement = True
            if is_selection_best:
                run.best_selection = selection_tuple
                any_improvement = True
            provisional_group_state = {"epochs_without_any_improvement": 0 if any_improvement else stale_epochs + 1}
            payload = checkpoint_payload(run, config, epoch, benchmark_sha, script_sha, exposure_sha, provisional_group_state)
            base = checkpoint_dir(config, arm, seed)
            atomic_torch_save(base / "last.pt", payload, config.output_root)
            if is_known_best:
                atomic_torch_save(base / "best_known_macro_f1.pt", payload, config.output_root)
            if is_selection_best:
                atomic_torch_save(base / "best_selection.pt", payload, config.output_root)
            logger.info(
                "Seed %d epoch %d %s | train loss %.5f | fixed-98 P0–P3 mean %.5f | exposure %s",
                seed,
                epoch,
                arm,
                train_metrics["loss"],
                known_mean,
                exposure_sha[:12],
            )
        if len({row["exposure_sha256"] for row in epoch_rows}) != 1:
            raise ScientificAbort("Same-epoch sampler exposure differs between arms")
        common_group_state = {"epochs_without_any_improvement": 0 if any_improvement else stale_epochs + 1}
        for arm, run in runs.items():
            synchronized_payload = checkpoint_payload(
                run,
                config,
                epoch,
                benchmark_sha,
                script_sha,
                exposure_sha,
                common_group_state,
            )
            atomic_torch_save(checkpoint_dir(config, arm, seed) / "last.pt", synchronized_payload, config.output_root)
        history_rows.extend(epoch_rows)
        atomic_write_csv(history_path, pd.DataFrame(history_rows), config.output_root)
        stale_epochs = 0 if any_improvement else stale_epochs + 1
        if epoch >= config.minimum_epochs and stale_epochs >= config.early_stopping_patience:
            logger.info("Synchronized group early stop for seed %d at epoch %d", seed, epoch)
            break
    for dataset in validation_datasets.values():
        dataset.close()
    train_dataset.close()
    return pd.DataFrame(history_rows)


def verify_stage2m(config: Stage26Config) -> Dict[str, Any]:
    root = config.stage2m_path
    if not root.is_dir():
        raise ScientificAbort(f"Stage 2M diagnostics directory missing: {root}")
    status_path = root / "manifests" / "STAGE2M_FINAL_STATUS.json"
    hash_manifest_path = root / "manifests" / "HASH_MANIFEST.json"
    if not status_path.is_file():
        raise ScientificAbort(f"Stage 2M structured final status missing: {status_path}")
    if not hash_manifest_path.is_file():
        raise ScientificAbort(f"Stage 2M artifact hash manifest missing: {hash_manifest_path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificAbort(f"Stage 2M structured final status is invalid JSON: {status_path}") from exc
    if not isinstance(status, Mapping):
        raise ScientificAbort("Stage 2M structured final status must be a JSON object")
    required_status = {
        "status": "MANYTX_STAGE2M_READY",
        "stage_version": EXPECTED_STAGE2M_VERSION,
        "script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "recommendation": "PROCEED_STAGE_2_6M_WITH_CAUTION",
        "failed_gates": [],
        "final_test_model_evaluation_performed": False,
        "final_test_threshold_selection_performed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": status.get(key, "<missing>")}
        for key, expected in required_status.items()
        if status.get(key, "<missing>") != expected or type(status.get(key, "<missing>")) is not type(expected)
    }
    strict_guard = status.get("strict_test_guard")
    if not isinstance(strict_guard, Mapping):
        mismatches["strict_test_guard"] = {"expected": "JSON object", "actual": type(strict_guard).__name__}
    else:
        required_guard = {
            "strict_index_arrays_loaded": False,
            "strict_test_signal_reads": 0,
            "strict_test_label_reads": 0,
            "strict_test_feature_reads": 0,
        }
        for key, expected in required_guard.items():
            actual = strict_guard.get(key, "<missing>")
            if actual != expected or type(actual) is not type(expected):
                mismatches[f"strict_test_guard.{key}"] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ScientificAbort(f"Stage 2M structured final status contract failed: {json.dumps(mismatches, sort_keys=True)}")
    actual_manifest_sha = sha256_file(hash_manifest_path)
    if actual_manifest_sha != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256:
        raise ScientificAbort(
            "Stage 2M HASH_MANIFEST.json SHA-256 mismatch: "
            f"expected {EXPECTED_STAGE2M_HASH_MANIFEST_SHA256}, actual {actual_manifest_sha}"
        )
    return {
        "directory": str(root),
        "executed_version": EXPECTED_STAGE2M_VERSION,
        "structured_final_status": str(status_path),
        "structured_final_status_sha256": sha256_file(status_path),
        "status": status["status"],
        "recommendation": status["recommendation"],
        "canonical_script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
        "hash_manifest": str(hash_manifest_path),
        "hash_manifest_sha256": EXPECTED_STAGE2M_HASH_MANIFEST_SHA256,
        "strict_test_guard": dict(strict_guard),
        "read_only": True,
    }


def atomic_numpy_save(path: Path, array: np.ndarray, output_root: Path) -> None:
    assert_within(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def fixed_unicode(values: np.ndarray) -> np.ndarray:
    text = [str(v) for v in values]
    width = max(1, max((len(v) for v in text), default=1))
    return np.asarray(text, dtype=f"<U{width}")


def trained_checkpoint_path(config: Stage26Config, arm: str, seed: int) -> Path:
    path = checkpoint_dir(config, arm, seed) / "best_selection.pt"
    if not path.is_file():
        raise ScientificAbort(f"Missing best-selection checkpoint: {path}")
    return path


def load_trained_model(
    config: Stage26Config,
    arm: str,
    seed: int,
    device: torch.device,
    benchmark_sha: str,
    script_sha: str,
) -> Tuple[WiSigRepresentationNet, Mapping[str, Any], str]:
    path = trained_checkpoint_path(config, arm, seed)
    payload = safe_torch_load(path, map_location=device)
    required = {
        "model_state",
        "arm",
        "seed",
        "benchmark_sha",
        "stage2m_sha",
        "script_sha",
        "configuration_sha",
        "architecture_signature",
        "loss_coefficients",
    }
    missing = required - set(payload)
    if missing:
        raise ScientificAbort(f"Checkpoint {path} is missing fields: {sorted(missing)}")
    model = WiSigRepresentationNet(config.num_classes, config.embedding_dim, config.dropout).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    checks = [
        payload["arm"] == arm,
        int(payload["seed"]) == seed,
        payload["benchmark_sha"] == benchmark_sha,
        payload["stage2m_sha"] == EXPECTED_STAGE2M_SCRIPT_SHA256,
        payload["script_sha"] == script_sha,
        payload["configuration_sha"] == config.configuration_sha256(),
        payload["architecture_signature"] == architecture_signature(model),
        payload["loss_coefficients"] == ARM_DEFINITIONS[arm],
    ]
    if not all(checks):
        raise ScientificAbort(f"INVALIDATE_AND_RESTART_ARM: evaluation checkpoint mismatch for {arm}/seed={seed}")
    model.eval()
    return model, payload, sha256_file(path)


def embedding_store_dir(config: Stage26Config, arm: str, seed: int, partition: str) -> Path:
    return config.output_root / "embeddings" / arm / f"seed_{seed}" / partition


def store_is_current(store: Path, checkpoint_sha: str, expected_rows: int) -> bool:
    manifest_path = store / "store_manifest.json"
    required = ("embedding_normalized.npy", "logits.npy", "labels.npy", "global_indices.npy", "receiver.npy", "day.npy", "equalized.npy")
    if (store / "INCOMPLETE").exists() or not manifest_path.is_file() or not all((store / name).is_file() for name in required):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return payload.get("checkpoint_sha256") == checkpoint_sha and int(payload.get("rows", -1)) == expected_rows and payload.get("complete") is True


@torch.no_grad()
def extract_embedding_store(
    config: Stage26Config,
    benchmark: ResolvedBenchmark,
    guard: StrictZeroDayGuard,
    arm: str,
    seed: int,
    partition: str,
    model: WiSigRepresentationNet,
    checkpoint_sha: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[np.ndarray]]:
    metadata = benchmark.partitions[partition]
    store = embedding_store_dir(config, arm, seed, partition)
    if store_is_current(store, checkpoint_sha, len(metadata.indices)):
        logger.info("Reusing current embedding store %s/%s/seed=%d", partition, arm, seed)
        if partition in {"p0", "p1", "p2", "p3", "train_known"}:
            logits = np.load(store / "logits.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(store / "labels.npy", mmap_mode="r", allow_pickle=False)
            accumulator = StreamingClassificationMetrics(EXPECTED_KNOWN_CLASSES)
            for start in range(0, len(labels), config.eval_batch_size):
                end = min(start + config.eval_batch_size, len(labels))
                accumulator.update(torch.from_numpy(np.asarray(logits[start:end], dtype=np.float32)), torch.from_numpy(np.asarray(labels[start:end], dtype=np.int64)))
            return (*accumulator.finish(), accumulator.confusion)
        return None, None, None
    assert_within(store, config.output_root)
    store.mkdir(parents=True, exist_ok=True)
    incomplete = store / "INCOMPLETE"
    atomic_write_text(incomplete, utc_now() + "\n", config.output_root)
    dataset = WiSigH5Dataset(benchmark, partition, guard)
    loader = build_loader(dataset, config, seed=seed + 77_777)
    count = len(dataset)
    partial_embedding = store / "embedding_normalized.partial.npy"
    partial_logits = store / "logits.partial.npy"
    embeddings = np.lib.format.open_memmap(partial_embedding, mode="w+", dtype=np.float16, shape=(count, config.embedding_dim))
    logits_memmap = np.lib.format.open_memmap(partial_logits, mode="w+", dtype=np.float16, shape=(count, config.num_classes))
    accumulator = StreamingClassificationMetrics(EXPECTED_KNOWN_CLASSES) if partition != "calibration_unknown" else None
    cursor = 0
    use_amp = config.amp_enabled and device.type == "cuda"
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(x)
        batch_n = len(x)
        embeddings[cursor:cursor + batch_n] = outputs["embedding_normalized"].float().cpu().numpy().astype(np.float16)
        logits_memmap[cursor:cursor + batch_n] = outputs["logits"].float().cpu().numpy().astype(np.float16)
        if accumulator is not None:
            accumulator.update(outputs["logits"], batch["y"])
        cursor += batch_n
    embeddings.flush()
    logits_memmap.flush()
    del embeddings, logits_memmap
    if cursor != count:
        raise ScientificAbort(f"Embedding extraction wrote {cursor} rows, expected {count}")
    os.replace(partial_embedding, store / "embedding_normalized.npy")
    os.replace(partial_logits, store / "logits.npy")
    atomic_numpy_save(store / "labels.npy", metadata.labels.astype(np.int16), config.output_root)
    atomic_numpy_save(store / "global_indices.npy", metadata.indices.astype(np.int64), config.output_root)
    atomic_numpy_save(store / "receiver.npy", fixed_unicode(metadata.receiver), config.output_root)
    atomic_numpy_save(store / "day.npy", fixed_unicode(metadata.day), config.output_root)
    atomic_numpy_save(store / "equalized.npy", metadata.equalized.astype(np.int8), config.output_root)
    manifest = {
        "complete": True,
        "arm": arm,
        "seed": seed,
        "partition": partition,
        "rows": count,
        "embedding_dimension": config.embedding_dim,
        "logit_dimension": config.num_classes,
        "embedding_dtype": "float16",
        "checkpoint_sha256": checkpoint_sha,
        "strict_zero_day": False,
        "created_at": utc_now(),
    }
    atomic_write_json(store / "store_manifest.json", manifest, config.output_root)
    if incomplete.exists():
        incomplete.unlink()
    dataset.close()
    logger.info("Created embedding store %s/%s/seed=%d with %,d rows", partition, arm, seed, count)
    if accumulator is None:
        return None, None, None
    metrics, per_class = accumulator.finish()
    return metrics, per_class, accumulator.confusion


def load_embedding_store(config: Stage26Config, arm: str, seed: int, partition: str) -> Dict[str, np.ndarray]:
    store = embedding_store_dir(config, arm, seed, partition)
    if not store_is_current(store, json.loads((store / "store_manifest.json").read_text(encoding="utf-8"))["checkpoint_sha256"], EXPECTED_PARTITION_COUNTS[partition]):
        raise ScientificAbort(f"Incomplete embedding store: {store}")
    return {
        "embedding": np.load(store / "embedding_normalized.npy", mmap_mode="r", allow_pickle=False),
        "logits": np.load(store / "logits.npy", mmap_mode="r", allow_pickle=False),
        "labels": np.load(store / "labels.npy", mmap_mode="r", allow_pickle=False),
        "global_indices": np.load(store / "global_indices.npy", mmap_mode="r", allow_pickle=False),
        "receiver": np.load(store / "receiver.npy", mmap_mode="r", allow_pickle=False),
        "day": np.load(store / "day.npy", mmap_mode="r", allow_pickle=False),
        "equalized": np.load(store / "equalized.npy", mmap_mode="r", allow_pickle=False),
    }


def compute_class_geometry(embedding: np.ndarray, labels: np.ndarray, num_classes: int = 98) -> Dict[str, Any]:
    dimension = int(embedding.shape[1])
    sums = np.zeros((num_classes, dimension), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for start in range(0, len(labels), 8192):
        end = min(start + 8192, len(labels))
        x = np.asarray(embedding[start:end], dtype=np.float32)
        y = np.asarray(labels[start:end], dtype=np.int64)
        for label in np.unique(y):
            mask = y == label
            sums[label] += x[mask].sum(axis=0, dtype=np.float64)
            counts[label] += int(mask.sum())
    centroids = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    centroid_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = np.divide(centroids, centroid_norm, out=np.zeros_like(centroids), where=centroid_norm > 0)
    squared_radius_sum = np.zeros(num_classes, dtype=np.float64)
    radii: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(num_classes)]
    margin_values: List[np.ndarray] = []
    for start in range(0, len(labels), 4096):
        end = min(start + 4096, len(labels))
        x = np.asarray(embedding[start:end], dtype=np.float32)
        y = np.asarray(labels[start:end], dtype=np.int64)
        distance_sq = np.maximum(
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(centroids * centroids, axis=1)[None, :]
            - 2 * x @ centroids.T,
            0.0,
        )
        own = np.sqrt(distance_sq[np.arange(len(y)), y])
        distance_sq[np.arange(len(y)), y] = np.inf
        wrong = np.sqrt(distance_sq.min(axis=1))
        margin_values.append((wrong - own).astype(np.float32))
        for label in np.unique(y):
            values = own[y == label]
            squared_radius_sum[label] += float(np.square(values).sum())
            radii[label] = np.concatenate((radii[label], values.astype(np.float32)))
    present = counts > 0
    global_centroid = np.average(centroids[present], axis=0, weights=counts[present])
    within_variance = float(squared_radius_sum.sum() / counts.sum())
    between_variance = float(np.average(np.sum((centroids[present] - global_centroid) ** 2, axis=1), weights=counts[present]))
    present_centroids = centroids[present]
    centroid_distances = np.sqrt(np.maximum(
        np.sum(present_centroids**2, axis=1, keepdims=True)
        + np.sum(present_centroids**2, axis=1)[None, :]
        - 2 * present_centroids @ present_centroids.T,
        0.0,
    ))
    off_diagonal = ~np.eye(len(present_centroids), dtype=bool)
    centroid_cosine = normalized_centroids[present] @ normalized_centroids[present].T
    all_radii = np.concatenate([radii[k] for k in np.flatnonzero(present)])
    all_margins = np.concatenate(margin_values)
    return {
        "centroids": centroids.astype(np.float32),
        "counts": counts,
        "class_mean_radius": np.array([float(r.mean()) if len(r) else np.nan for r in radii]),
        "within_class_variance": within_variance,
        "between_class_variance": between_variance,
        "fisher_ratio": between_variance / max(within_variance, 1e-12),
        "mean_intra_class_radius": float(all_radii.mean()),
        "median_intra_class_radius": float(np.median(all_radii)),
        "mean_inter_centroid_distance": float(centroid_distances[off_diagonal].mean()),
        "median_inter_centroid_distance": float(np.median(centroid_distances[off_diagonal])),
        "mean_nearest_centroid_margin": float(all_margins.mean()),
        "median_nearest_centroid_margin": float(np.median(all_margins)),
        "mean_cosine_centroid_similarity": float(centroid_cosine[off_diagonal].mean()),
        "median_cosine_centroid_similarity": float(np.median(centroid_cosine[off_diagonal])),
    }


def deterministic_stratified_positions(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels).tolist()):
        positions = np.flatnonzero(np.asarray(labels) == label)
        take = min(per_class, len(positions))
        selected.append(np.sort(rng.choice(positions, size=take, replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def representation_metrics_for_store(
    config: Stage26Config,
    arm: str,
    seed: int,
    partition: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    store = load_embedding_store(config, arm, seed, partition)
    geometry = compute_class_geometry(store["embedding"], store["labels"])
    protocol_seed = seed + {"train_known": 0, "p0": 100, "p1": 200, "p2": 300, "p3": 400}[partition]
    positions = deterministic_stratified_positions(store["labels"], config.cluster_sample_per_class, protocol_seed)
    sample_indices_path = config.output_root / "cache" / "cluster_samples" / f"{partition}_seed_{seed}.npy"
    sampled_global = np.asarray(store["global_indices"][positions], dtype=np.int64)
    if sample_indices_path.is_file():
        prior = np.load(sample_indices_path, allow_pickle=False)
        if not np.array_equal(prior, sampled_global):
            raise ScientificAbort(f"Deterministic cluster sample changed: {sample_indices_path}")
    else:
        atomic_numpy_save(sample_indices_path, sampled_global, config.output_root)
    x = np.asarray(store["embedding"][positions], dtype=np.float32)
    y = np.asarray(store["labels"][positions], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ScientificAbort(f"Cluster sample for {partition} has fewer than two classes")
    cluster = {
        "silhouette": float(silhouette_score(x, y, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(x, y)),
        "calinski_harabasz": float(calinski_harabasz_score(x, y)),
        "cluster_sample_size": int(len(x)),
        "cluster_sample_indices_path": str(sample_indices_path),
        "cluster_sample_indices_sha256": sha256_file(sample_indices_path),
    }
    scalar_geometry = {k: v for k, v in geometry.items() if k not in {"centroids", "counts", "class_mean_radius"}}
    return {**scalar_geometry, **cluster}, geometry


def balanced_protocol_pair(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    per_tx: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    common = sorted(set(np.asarray(left["labels"]).tolist()) & set(np.asarray(right["labels"]).tolist()))
    for label in common:
        lpos = np.flatnonzero(np.asarray(left["labels"]) == label)
        rpos = np.flatnonzero(np.asarray(right["labels"]) == label)
        take = min(per_tx, len(lpos), len(rpos))
        if take < 2:
            continue
        lsel = rng.choice(lpos, take, replace=False)
        rsel = rng.choice(rpos, take, replace=False)
        x_parts.extend((np.asarray(left["embedding"][lsel], dtype=np.float32), np.asarray(right["embedding"][rsel], dtype=np.float32)))
        y_parts.extend((np.zeros(take, dtype=np.int8), np.ones(take, dtype=np.int8)))
    if not x_parts:
        raise ScientificAbort("No transmitter-balanced samples for domain classifier")
    return np.concatenate(x_parts), np.concatenate(y_parts)


def balanced_equalization_pair(store: Mapping[str, np.ndarray], per_tx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    labels = np.asarray(store["labels"])
    equalized = np.asarray(store["equalized"])
    for label in sorted(np.unique(labels).tolist()):
        zero = np.flatnonzero((labels == label) & (equalized == 0))
        one = np.flatnonzero((labels == label) & (equalized == 1))
        take = min(per_tx, len(zero), len(one))
        if take < 2:
            continue
        zsel = rng.choice(zero, take, replace=False)
        osel = rng.choice(one, take, replace=False)
        x_parts.extend((np.asarray(store["embedding"][zsel], dtype=np.float32), np.asarray(store["embedding"][osel], dtype=np.float32)))
        y_parts.extend((np.zeros(take, dtype=np.int8), np.ones(take, dtype=np.int8)))
    if not x_parts:
        raise ScientificAbort("No Tx-balanced equalization samples for domain classifier")
    return np.concatenate(x_parts), np.concatenate(y_parts)


def domain_classifier_auc(x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> Dict[str, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y)):
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=3000,
            tol=1e-4,
            class_weight="balanced",
            random_state=seed + fold,
        )
        classifier.fit(x[train_index], y[train_index])
        scores = classifier.predict_proba(x[test_index])[:, 1]
        aucs.append(float(roc_auc_score(y[test_index], scores)))
    return {
        "auroc": float(np.mean(aucs)),
        "auroc_std_folds": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "samples": float(len(y)),
        "folds": float(folds),
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_novelty_geometry(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: Stage26Config,
    seed: int,
) -> Dict[str, np.ndarray]:
    geometry = compute_class_geometry(embedding, labels)
    centroids = np.asarray(geometry["centroids"], dtype=np.float64)
    normalized_centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    sample_per_class = max(1, config.covariance_fit_limit // EXPECTED_KNOWN_CLASSES)
    positions = deterministic_stratified_positions(labels, sample_per_class, seed + 901)
    x = np.asarray(embedding[positions], dtype=np.float64)
    y = np.asarray(labels[positions], dtype=np.int64)
    residuals = x - centroids[y]
    covariance = LedoitWolf(assume_centered=True).fit(residuals)
    return {
        "centroids": centroids,
        "normalized_centroids": normalized_centroids,
        "precision": np.asarray(covariance.precision_, dtype=np.float64),
        "shrinkage": np.array([float(covariance.shrinkage_)], dtype=np.float64),
        "covariance_fit_positions": positions,
    }


def compute_novelty_scores(
    embedding: np.ndarray,
    logits: np.ndarray,
    geometry: Mapping[str, np.ndarray],
    chunk_size: int = 4096,
) -> Dict[str, np.ndarray]:
    centroids = np.asarray(geometry["centroids"], dtype=np.float64)
    normalized_centroids = np.asarray(geometry["normalized_centroids"], dtype=np.float64)
    precision = np.asarray(geometry["precision"], dtype=np.float64)
    centroid_sq = np.sum(centroids**2, axis=1)
    centroid_p = centroids @ precision
    centroid_p_sq = np.sum(centroid_p * centroids, axis=1)
    outputs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for start in range(0, len(embedding), chunk_size):
        end = min(start + chunk_size, len(embedding))
        x = np.asarray(embedding[start:end], dtype=np.float64)
        logit = np.asarray(logits[start:end], dtype=np.float64)
        euclidean_sq = np.maximum(np.sum(x * x, axis=1, keepdims=True) + centroid_sq[None, :] - 2 * x @ centroids.T, 0.0)
        cosine_distance = 1 - x @ normalized_centroids.T
        xp = x @ precision
        mahalanobis_sq = np.maximum(np.sum(xp * x, axis=1, keepdims=True) + centroid_p_sq[None, :] - 2 * xp @ centroids.T, 0.0)
        probabilities = stable_softmax(logit)
        logsumexp = logit.max(axis=1) + np.log(np.exp(logit - logit.max(axis=1, keepdims=True)).sum(axis=1))
        outputs["nearest_prototype_euclidean"].append(np.sqrt(euclidean_sq.min(axis=1)).astype(np.float32))
        outputs["cosine_prototype_distance"].append(cosine_distance.min(axis=1).astype(np.float32))
        outputs["shrinkage_mahalanobis"].append(np.sqrt(mahalanobis_sq.min(axis=1)).astype(np.float32))
        outputs["energy"].append((-logsumexp).astype(np.float32))
        outputs["one_minus_msp"].append((1 - probabilities.max(axis=1)).astype(np.float32))
    return {key: np.concatenate(values) for key, values in outputs.items()}


def cohen_d(known: np.ndarray, unknown: np.ndarray) -> float:
    n1, n2 = len(known), len(unknown)
    pooled = math.sqrt(((n1 - 1) * known.var(ddof=1) + (n2 - 1) * unknown.var(ddof=1)) / max(n1 + n2 - 2, 1))
    return float((unknown.mean() - known.mean()) / max(pooled, 1e-12))


def robust_overlap(known: np.ndarray, unknown: np.ndarray, bins: int = 100) -> float:
    combined = np.concatenate((known, unknown))
    low, high = np.quantile(combined, [0.005, 0.995])
    if high <= low:
        return 1.0
    left, edges = np.histogram(np.clip(known, low, high), bins=bins, range=(low, high), density=True)
    right, _ = np.histogram(np.clip(unknown, low, high), bins=edges, density=True)
    width = np.diff(edges)
    return float(np.sum(np.minimum(left, right) * width))


def bootstrap_auc_ci(
    known: np.ndarray,
    unknown: np.ndarray,
    iterations: int,
    seed: int,
    cap_per_group: int = 5000,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    if len(known) > cap_per_group:
        known = known[rng.choice(len(known), cap_per_group, replace=False)]
    if len(unknown) > cap_per_group:
        unknown = unknown[rng.choice(len(unknown), cap_per_group, replace=False)]
    y = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8)))
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        ks = known[rng.integers(0, len(known), len(known))]
        us = unknown[rng.integers(0, len(unknown), len(unknown))]
        values[iteration] = roc_auc_score(y, np.concatenate((ks, us)))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), int(len(y))


def summarize_novelty_score(
    known: np.ndarray,
    unknown: np.ndarray,
    config: Stage26Config,
    seed: int,
) -> Dict[str, float]:
    y = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8)))
    scores = np.concatenate((known, unknown))
    auc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))
    low, high, bootstrap_n = bootstrap_auc_ci(known, unknown, config.bootstrap_iterations, seed)
    return {
        "auroc": auc,
        "auprc": auprc,
        "cohen_d": cohen_d(known, unknown),
        "robust_overlap": robust_overlap(known, unknown),
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "bootstrap_samples": float(bootstrap_n),
        "known_samples": float(len(known)),
        "unknown_samples": float(len(unknown)),
    }


def domain_match_keys(store: Mapping[str, np.ndarray], fields: Sequence[str]) -> np.ndarray:
    arrays = [np.asarray(store[field]).astype(str) for field in fields]
    return np.array(["|".join(values) for values in zip(*arrays)], dtype=object)


def calibration_domain_matched_rows(
    known_store: Mapping[str, np.ndarray],
    unknown_store: Mapping[str, np.ndarray],
    known_scores: Mapping[str, np.ndarray],
    unknown_scores: Mapping[str, np.ndarray],
    arm: str,
    seed: int,
) -> List[Dict[str, Any]]:
    definitions = {
        "same_day": ("day",),
        "same_receiver": ("receiver",),
        "same_equalization": ("equalized",),
        "day_receiver": ("day", "receiver"),
        "day_receiver_equalization": ("day", "receiver", "equalized"),
    }
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed + 44_001)
    for analysis, fields in definitions.items():
        known_keys = domain_match_keys(known_store, fields)
        unknown_keys = domain_match_keys(unknown_store, fields)
        common = sorted(set(known_keys) & set(unknown_keys))
        for score_name in known_scores:
            stratum_aucs = []
            total_balanced = 0
            for key in common:
                kp = np.flatnonzero(known_keys == key)
                up = np.flatnonzero(unknown_keys == key)
                take = min(len(kp), len(up), 5000)
                if take < 10:
                    continue
                ksel = rng.choice(kp, take, replace=False)
                usel = rng.choice(up, take, replace=False)
                y = np.concatenate((np.zeros(take, dtype=np.int8), np.ones(take, dtype=np.int8)))
                score = np.concatenate((known_scores[score_name][ksel], unknown_scores[score_name][usel]))
                auc = float(roc_auc_score(y, score))
                stratum_aucs.append(auc)
                total_balanced += 2 * take
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "analysis": analysis,
                    "fields": "+".join(fields),
                    "score": score_name,
                    "row_type": "stratum",
                    "stratum": key,
                    "auroc": auc,
                    "samples": 2 * take,
                })
            rows.append({
                "arm": arm,
                "seed": seed,
                "analysis": analysis,
                "fields": "+".join(fields),
                "score": score_name,
                "row_type": "macro_aggregate",
                "stratum": "ALL_MATCHED_STRATA",
                "auroc": float(np.mean(stratum_aucs)) if stratum_aucs else np.nan,
                "samples": total_balanced,
                "strata": len(stratum_aucs),
            })
    return rows


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2 or float(array.std(ddof=1)) == 0:
        return mean, mean, mean
    sem = stats.sem(array)
    interval = stats.t.interval(confidence, len(array) - 1, loc=mean, scale=sem)
    return mean, float(interval[0]), float(interval[1])


def paired_bootstrap_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float]:
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        boot[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def paired_cohen_dz(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    std = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    return float(difference.mean() / std) if std > 0 else 0.0


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    m = len(values)
    for rank_position in range(m - 1, -1, -1):
        index = order[rank_position]
        rank = rank_position + 1
        running = min(running, values[index] * m / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0, 1)


def atomic_savez(path: Path, output_root: Path, **arrays: np.ndarray) -> None:
    assert_within(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_figure_pair(fig: plt.Figure, base: Path, output_root: Path) -> None:
    assert_within(base, output_root)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_arm_summary(frame: pd.DataFrame, value: str, extra_groups: Sequence[str] = ()) -> pd.DataFrame:
    groups = ["arm", *extra_groups]
    return frame.groupby(groups, dropna=False)[value].agg(["mean", "std", "median", "count"]).reset_index()


class Stage26Pipeline:
    def __init__(self, config: Stage26Config):
        self.config = config
        self.config.validate()
        for name in REQUIRED_OUTPUT_DIRS:
            assert_within(self.config.output_root / name, self.config.output_root).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(config)
        self.guard = StrictZeroDayGuard(config.branch_root_path, config.output_root)
        self.benchmark: Optional[ResolvedBenchmark] = None
        self.stage2m_provenance: Optional[Dict[str, Any]] = None
        script_candidate = Path(__file__).resolve()
        self.script_sha = sha256_file(script_candidate)
        self.benchmark_sha = EXPECTED_BENCHMARK_SHA256
        self.device = resolve_device(config)

    def ensure_context(self) -> ResolvedBenchmark:
        if self.stage2m_provenance is None:
            self.stage2m_provenance = verify_stage2m(self.config)
        if not self.guard.strict_file_audit:
            self.guard.verify_strict_files_from_manifests()
        if self.benchmark is None:
            self.benchmark = resolve_benchmark(self.config, self.guard)
        return self.benchmark

    def stage_manifest_path(self, stage_number: int) -> Path:
        return self.config.output_root / "manifests" / f"stage_{stage_number:02d}_manifest.json"

    def complete_stage(self, stage_number: int, name: str, outputs: Sequence[Path], extra: Optional[Mapping[str, Any]] = None) -> None:
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise ScientificAbort(f"Stage {stage_number:02d} did not create required outputs: {missing}")
        manifest = {
            "stage": stage_number,
            "name": name,
            "status": "PASS",
            "pipeline_version": PIPELINE_VERSION,
            "benchmark_sha256": self.benchmark_sha,
            "stage2m_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "script_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(),
            "outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
            "completed_at": utc_now(),
            **dict(extra or {}),
        }
        atomic_write_json(self.stage_manifest_path(stage_number), manifest, self.config.output_root)
        print(f"[PASS] Stage {stage_number:02d} — {name}")

    def require_prior_stages(self, stage_number: int) -> None:
        for prior in range(1, stage_number):
            path = self.stage_manifest_path(prior)
            if not path.is_file():
                raise ScientificAbort(f"Prior stage manifest missing: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or payload.get("configuration_sha256") != self.config.configuration_sha256():
                raise ScientificAbort(f"Prior stage manifest is stale or failed: {path}")

    def stage_01(self) -> None:
        benchmark = self.ensure_context()
        if self.config.branch_root_path.name != CANONICAL_BRANCH:
            raise ScientificAbort(f"Branch root must resolve to {CANONICAL_BRANCH}")
        provenance = {
            "pipeline": f"Stage 2.6M v{PIPELINE_VERSION}",
            "canonical_branch": CANONICAL_BRANCH,
            "benchmark": {
                "path": str(benchmark.h5_path),
                "sha256": self.benchmark_sha,
                "expected_sha256": EXPECTED_BENCHMARK_SHA256,
                "read_only": True,
                "signal_dataset": benchmark.signal_key,
                "signal_orientation": benchmark.signal_orientation,
                "total_samples": benchmark.total_samples,
            },
            "stage2m": self.stage2m_provenance,
            "partition_counts": {name: len(part.indices) for name, part in benchmark.partitions.items()},
            "known_transmitter_mapping": benchmark.transmitter_mapping,
            "strict_zero_day_policy": "authorized split-array allowlist plus strict-path/output guards; strict arrays never loaded",
            "generated_at": utc_now(),
        }
        provenance_path = self.config.output_root / "manifests" / "STAGE2_6M_INPUT_PROVENANCE.json"
        guard_path = self.config.output_root / "manifests" / "STRICT_TEST_GUARD.json"
        config_path = self.config.output_root / "manifests" / "STAGE2_6M_CONFIG.json"
        schema_path = self.config.output_root / "tables" / "benchmark_h5_schema.csv"
        report_path = self.config.output_root / "reports" / "input_verification_report.md"
        atomic_write_json(provenance_path, provenance, self.config.output_root)
        atomic_write_json(guard_path, self.guard.manifest(), self.config.output_root)
        atomic_write_json(config_path, self.config.frozen_payload(), self.config.output_root)
        atomic_write_csv(schema_path, pd.DataFrame(benchmark.schema_rows), self.config.output_root)
        report = f"""# Stage 2.6M input verification

Measured fact: the canonical benchmark exists at `{benchmark.h5_path}` and its SHA-256 equals `{self.benchmark_sha}`.

Measured fact: Stage 2M executed version {EXPECTED_STAGE2M_VERSION}, declared canonical script SHA, benchmark SHA, READY status, compatible proceed recommendation, final-test prohibitions, and zero strict-access fields were verified from its structured final status. The artifact HASH_MANIFEST SHA was verified independently.

Measured fact: the benchmark resolver found `{benchmark.signal_key}` with orientation `{benchmark.signal_orientation}` and {benchmark.total_samples:,} rows. All six authorized partition counts match the frozen protocol.

Measured fact: strict zero-day index files were checked only as opaque frozen files. Their arrays were not loaded; all strict-test violation counters are zero. Structural enforcement is provided by the authorized partition allowlist, strict-path prohibition, frozen-index-only resolver, and output scan.

Scientific interpretation: Stage 2.6M may proceed without altering Stage 1B or Stage 2M and without exposing final zero-day information.
"""
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(1, "Frozen Input Verification", [report_path, provenance_path, guard_path, config_path, schema_path])

    def stage_02(self) -> None:
        benchmark = self.ensure_context()
        train = benchmark.partitions["train_known"]
        sampler_a = DomainBalancedTxSampler(train.labels, train.receiver, train.day, train.equalized, self.config.batch_size, self.config.samples_per_tx, 42, 1)
        sampler_b = DomainBalancedTxSampler(train.labels, train.receiver, train.day, train.equalized, self.config.batch_size, self.config.samples_per_tx, 42, 1)
        batches_a = list(iter(sampler_a))
        batches_b = list(iter(sampler_b))
        if batch_exposure_sha256(batches_a) != batch_exposure_sha256(batches_b):
            raise ScientificAbort("DomainBalancedTxSampler is not deterministic")
        audit_rows = []
        for batch_index, batch in enumerate(batches_a):
            positions = np.asarray(batch, dtype=np.int64)
            eq = train.equalized[positions]
            labels = train.labels[positions]
            cells = set(zip(train.transmitter_raw[positions], train.receiver[positions], train.day[positions], train.equalized[positions]))
            audit_rows.append({
                "batch": batch_index,
                "samples": len(positions),
                "classes": len(np.unique(labels)),
                "equalized_0": int((eq == 0).sum()),
                "equalized_1": int((eq == 1).sum()),
                "unique_tx_rx_day_eq_cells": len(cells),
            })
        audit = pd.DataFrame(audit_rows)
        if audit[["equalized_0", "equalized_1"]].sum().min() == 0:
            raise ScientificAbort("Sampler failed to expose both equalization states")
        table_path = self.config.output_root / "tables" / "sampler_balance_summary.csv"
        report_path = self.config.output_root / "reports" / "domain_sampler_report.md"
        manifest_path = self.config.output_root / "manifests" / "SAMPLER_MANIFEST.json"
        atomic_write_csv(table_path, audit, self.config.output_root)
        sampler_manifest = sampler_a.manifest()
        sampler_manifest["deterministic_exposure_sha256"] = batch_exposure_sha256(batches_a)
        sampler_manifest["same_sampler_all_arms"] = True
        atomic_write_json(manifest_path, sampler_manifest, self.config.output_root)
        per_class = pd.DataFrame(sampler_manifest["per_class"])
        report = f"""# Domain-balanced sampler verification

Measured fact: all {len(per_class)} Train Known classes are represented. The deterministic epoch contains {len(batches_a):,} batches of {self.config.batch_size} samples, {self.config.samples_per_tx} samples per transmitter, and both equalization states.

Measured fact: regenerating seed 42 / epoch 1 produced the same exposure SHA-256 `{batch_exposure_sha256(batches_a)}`.

Scientific interpretation: transmitter identity remains the primary sampling variable while receiver, capture day, and equalization cells are cycled within transmitter wherever source coverage permits.

Equalized sample totals: 0 = {int(audit.equalized_0.sum()):,}; 1 = {int(audit.equalized_1.sum()):,}.
"""
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(2, "Dataset & Domain Sampler Verification", [report_path, table_path, manifest_path])

    def stage_03(self) -> None:
        benchmark = self.ensure_context()
        seed = 42
        runs = create_training_runs(self.config, seed, self.device)
        train_dataset = WiSigH5Dataset(benchmark, "train_known", self.guard)
        train = benchmark.partitions["train_known"]
        sampler = DomainBalancedTxSampler(train.labels, train.receiver, train.day, train.equalized, self.config.batch_size, self.config.samples_per_tx, seed, 0)
        real_loader = build_loader(train_dataset, self.config, batches=[next(iter(sampler))], seed=seed)
        real_batch = next(iter(real_loader))
        real_x = real_batch["x"].to(self.device)
        real_y = real_batch["y"].to(self.device)
        synthetic_x = torch.randn(self.config.batch_size, 2, 256, device=self.device)
        synthetic_y = torch.arange(self.config.batch_size, device=self.device) % self.config.num_classes
        rows = []
        for arm, run in runs.items():
            checks = {}
            for batch_name, x, y in (("synthetic", synthetic_x, synthetic_y), ("real", real_x, real_y)):
                run.model.zero_grad(set_to_none=True)
                outputs = run.model(x)
                if outputs["embedding_raw"].shape != (len(x), self.config.embedding_dim):
                    raise ScientificAbort("Embedding raw shape mismatch")
                norms = outputs["embedding_normalized"].norm(dim=1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
                    raise ScientificAbort("Normalized embedding has non-unit norms")
                loss, _ = objective_loss(arm, outputs, y, run.prototypes, self.config)
                loss.backward()
                finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in run.model.parameters())
                if not torch.isfinite(loss) or not finite:
                    raise ScientificAbort(f"Non-finite {batch_name} loss/gradient for {arm}")
                checks[f"{batch_name}_loss"] = float(loss.detach().cpu())
                checks[f"{batch_name}_finite_gradients"] = bool(finite)
            rows.append({
                "arm": arm,
                "objective": ARM_DEFINITIONS[arm]["name"],
                "parameter_count": sum(p.numel() for p in run.model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in run.model.parameters() if p.requires_grad),
                "embedding_dim": run.model.embedding_dim,
                "classifier_classes": run.model.num_classes,
                "architecture_signature": architecture_signature(run.model),
                **checks,
            })
        frame = pd.DataFrame(rows)
        if frame.parameter_count.nunique() != 1 or frame.architecture_signature.nunique() != 1:
            raise ScientificAbort("Arm parameter equivalence failed")
        table_path = self.config.output_root / "tables" / "arm_parameter_equivalence.csv"
        report_path = self.config.output_root / "reports" / "model_and_loss_validation.md"
        model_manifest = self.config.output_root / "manifests" / "MODEL_ARCHITECTURE_MANIFEST.json"
        loss_manifest = self.config.output_root / "manifests" / "LOSS_MANIFEST.json"
        atomic_write_csv(table_path, frame, self.config.output_root)
        atomic_write_json(model_manifest, {
            "model": "WiSigRepresentationNet",
            "input_shape": ["B", 2, 256],
            "embedding_dim": self.config.embedding_dim,
            "num_classes": self.config.num_classes,
            "parameter_count": int(frame.parameter_count.iloc[0]),
            "architecture_signature": frame.architecture_signature.iloc[0],
            "outputs": ["logits", "embedding_raw", "embedding_normalized"],
            "identical_across_arms": True,
        }, self.config.output_root)
        atomic_write_json(loss_manifest, {
            "arms": ARM_DEFINITIONS,
            "temperature": self.config.temperature,
            "prototype_method": "EMA normalized class prototypes",
            "prototype_momentum": self.config.prototype_momentum,
            "prototype_separation_term": False,
            "additional_auxiliary_objectives": [],
        }, self.config.output_root)
        report = f"""# Common backbone and objective validation

Measured fact: A0–A3 each contain {int(frame.parameter_count.iloc[0]):,} parameters and share architecture signature `{frame.architecture_signature.iloc[0]}`.

Measured fact: synthetic [B,2,256] and authorized real Train Known mini-batches produced finite losses and gradients for CE, CE+SupCon, CE+Prototype, and CE+SupCon+Prototype.

Scientific interpretation: loss composition is the only major intentional variable in the controlled ablation.
"""
        atomic_write_text(report_path, report, self.config.output_root)
        train_dataset.close()
        del runs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.complete_stage(3, "Common Backbone & Objective Validation", [report_path, table_path, model_manifest, loss_manifest])

    def stage_04(self) -> None:
        benchmark = self.ensure_context()
        all_history = []
        for seed in self.config.seeds:
            history = train_seed_group(
                self.config,
                benchmark,
                self.guard,
                seed,
                self.device,
                self.logger,
                self.benchmark_sha,
                self.script_sha,
            )
            all_history.append(history)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        frame = pd.concat(all_history, ignore_index=True)
        for (seed, epoch), group in frame.groupby(["seed", "epoch"]):
            if len(group) != 4 or group.exposure_sha256.nunique() != 1:
                raise ScientificAbort(f"Training exposure mismatch for seed={seed}, epoch={epoch}")
        final_epochs = frame.groupby(["seed", "arm"]).epoch.max().unstack()
        if not (final_epochs.nunique(axis=1) == 1).all():
            raise ScientificAbort("Synchronized early stopping produced unequal arm budgets")
        history_path = self.config.output_root / "tables" / "training_history_summary.csv"
        exposure_path = self.config.output_root / "tables" / "training_exposure_equivalence.csv"
        report_path = self.config.output_root / "reports" / "training_report.md"
        checkpoint_manifest_path = self.config.output_root / "manifests" / "CHECKPOINT_MANIFEST.json"
        seed_manifest_path = self.config.output_root / "manifests" / "SEED_MANIFEST.json"
        atomic_write_csv(history_path, frame, self.config.output_root)
        exposure = frame.groupby(["seed", "epoch"]).agg(
            arm_count=("arm", "nunique"),
            exposure_sha256=("exposure_sha256", "first"),
            samples_per_arm=("train_samples", "first"),
        ).reset_index()
        atomic_write_csv(exposure_path, exposure, self.config.output_root)
        checkpoint_rows = []
        for arm in ARM_DEFINITIONS:
            for seed in self.config.seeds:
                for filename in ("last.pt", "best_known_macro_f1.pt", "best_selection.pt"):
                    path = checkpoint_dir(self.config, arm, seed) / filename
                    if not path.is_file():
                        raise ScientificAbort(f"Required checkpoint missing: {path}")
                    checkpoint_rows.append({"arm": arm, "seed": seed, "type": filename, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        atomic_write_json(checkpoint_manifest_path, {"checkpoints": checkpoint_rows, "resume_safety": "all frozen signatures enforced"}, self.config.output_root)
        atomic_write_json(seed_manifest_path, {"seeds": list(self.config.seeds), "same_panel_all_arms": True, "synchronized_epoch_budget": final_epochs.reset_index().to_dict("records")}, self.config.output_root)
        report = f"""# Controlled multi-seed training

Measured fact: four controlled arms completed for each frozen seed: {', '.join(map(str, self.config.seeds))}.

Measured fact: within every seed and epoch, all arms consumed the same ordered global training exposure and the same augmentation seed. Group-level early stopping kept the completed epoch budget equal across arms.

Measured fact: last, best-known-macro-F1, and best-selection checkpoints exist for every arm/seed and contain benchmark, Stage 2M, script, configuration, architecture, loss, RNG, prototype, and sampler state.
"""
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(4, "Controlled Multi-Seed Training", [history_path, exposure_path, report_path, checkpoint_manifest_path, seed_manifest_path])

    def stage_05(self) -> None:
        benchmark = self.ensure_context()
        metric_rows: List[Dict[str, Any]] = []
        per_class_rows: List[pd.DataFrame] = []
        confusion_dir = self.config.output_root / "metrics" / "confusion_matrices"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        for arm in ARM_DEFINITIONS:
            for seed in self.config.seeds:
                model, _, checkpoint_sha = load_trained_model(self.config, arm, seed, self.device, self.benchmark_sha, self.script_sha)
                for partition in ("train_known", "p0", "p1", "p2", "p3"):
                    metrics, per_class, confusion = extract_embedding_store(
                        self.config,
                        benchmark,
                        self.guard,
                        arm,
                        seed,
                        partition,
                        model,
                        checkpoint_sha,
                        self.device,
                        self.logger,
                    )
                    assert metrics is not None and per_class is not None and confusion is not None
                    metric_rows.append({"arm": arm, "arm_name": ARM_DEFINITIONS[arm]["name"], "seed": seed, "protocol": partition, **metrics})
                    enriched = per_class.assign(arm=arm, seed=seed, protocol=partition)
                    per_class_rows.append(enriched)
                    atomic_numpy_save(confusion_dir / f"{arm}_seed_{seed}_{partition}.npy", confusion.astype(np.int64), self.config.output_root)
                del model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        metrics_frame = pd.DataFrame(metric_rows)
        known = metrics_frame[metrics_frame.protocol.isin(["p0", "p1", "p2", "p3"])].copy()
        known_path = self.config.output_root / "tables" / "known_protocol_results.csv"
        per_class_path = self.config.output_root / "tables" / "known_protocol_per_class.csv"
        fixed_path = self.config.output_root / "tables" / "fixed98_macro_results.csv"
        degradation_path = self.config.output_root / "tables" / "protocol_degradation.csv"
        report_path = self.config.output_root / "reports" / "known_tx_evaluation_report.md"
        atomic_write_csv(known_path, known, self.config.output_root)
        atomic_write_csv(per_class_path, pd.concat(per_class_rows, ignore_index=True), self.config.output_root)
        fixed_columns = ["arm", "arm_name", "seed", "protocol", "observed_classes", "observed_class_macro_f1", "fixed98_macro_f1", "observed_class_balanced_accuracy", "fixed98_balanced_accuracy"]
        atomic_write_csv(fixed_path, known[fixed_columns], self.config.output_root)
        degradation_rows = []
        for (arm, seed), group in known.groupby(["arm", "seed"]):
            p0 = float(group.loc[group.protocol == "p0", "fixed98_macro_f1"].iloc[0])
            for protocol in ("p1", "p2", "p3"):
                value = float(group.loc[group.protocol == protocol, "fixed98_macro_f1"].iloc[0])
                degradation_rows.append({"arm": arm, "seed": seed, "protocol": protocol, "p0_fixed98_macro_f1": p0, "protocol_fixed98_macro_f1": value, "absolute_degradation": p0 - value, "relative_degradation": (p0 - value) / max(p0, 1e-12)})
        atomic_write_csv(degradation_path, pd.DataFrame(degradation_rows), self.config.output_root)
        summary = grouped_arm_summary(known, "fixed98_macro_f1", ("protocol",))
        report = "# Known-transmitter P0–P3 evaluation\n\nMeasured fact: every arm/seed was evaluated with a frozen forward pass on P0, P1, P2, and P3. No evaluation statistics were fitted back into the model.\n\nMeasured fact: P2 and P3 include both observed-class and fixed-98 macro-F1/balanced-accuracy results; missing identities receive zero in the fixed frame.\n\n" + summary.to_markdown(index=False) + "\n"
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(5, "Known-Tx P0–P3 Evaluation", [known_path, per_class_path, fixed_path, degradation_path, report_path])

    def stage_06(self) -> None:
        self.ensure_context()
        rows = []
        for arm in ARM_DEFINITIONS:
            for seed in self.config.seeds:
                for partition in ("train_known", "p0", "p1", "p2", "p3"):
                    metrics, _ = representation_metrics_for_store(self.config, arm, seed, partition)
                    rows.append({"arm": arm, "arm_name": ARM_DEFINITIONS[arm]["name"], "seed": seed, "protocol": partition, **metrics})
        frame = pd.DataFrame(rows)
        table_path = self.config.output_root / "tables" / "embedding_separability.csv"
        report_path = self.config.output_root / "reports" / "embedding_separability_report.md"
        atomic_write_csv(table_path, frame, self.config.output_root)
        summary = grouped_arm_summary(frame[frame.protocol == "p0"], "fisher_ratio")
        report = "# Embedding separability evaluation\n\nMeasured fact: full-store within/between-class geometry and deterministic stratified cluster metrics were computed for Train Known and P0–P3. Exact cluster sample global indices are persisted under `cache/cluster_samples`.\n\nP0 Fisher-ratio seed summary:\n\n" + summary.to_markdown(index=False) + "\n\nScientific interpretation: larger Fisher ratio, inter-centroid distance, and centroid margin together with smaller intra-class radius indicate more transmitter-discriminative geometry.\n"
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(6, "Embedding Separability Evaluation", [table_path, report_path])

    def stage_07(self) -> None:
        self.ensure_context()
        drift_rows: List[Dict[str, Any]] = []
        leakage_rows: List[Dict[str, Any]] = []
        for arm in ARM_DEFINITIONS:
            for seed in self.config.seeds:
                train_store = load_embedding_store(self.config, arm, seed, "train_known")
                train_geometry = compute_class_geometry(train_store["embedding"], train_store["labels"])
                train_centroids = np.asarray(train_geometry["centroids"], dtype=np.float64)
                train_radii = np.asarray(train_geometry["class_mean_radius"], dtype=np.float64)
                for protocol in ("p0", "p1", "p2", "p3"):
                    protocol_store = load_embedding_store(self.config, arm, seed, protocol)
                    protocol_geometry = compute_class_geometry(protocol_store["embedding"], protocol_store["labels"])
                    protocol_centroids = np.asarray(protocol_geometry["centroids"], dtype=np.float64)
                    counts = np.asarray(protocol_geometry["counts"])
                    values = []
                    for label in np.flatnonzero(counts > 0):
                        euclidean = float(np.linalg.norm(protocol_centroids[label] - train_centroids[label]))
                        cosine = float(1 - np.dot(protocol_centroids[label], train_centroids[label]) / max(np.linalg.norm(protocol_centroids[label]) * np.linalg.norm(train_centroids[label]), 1e-12))
                        normalized = euclidean / max(float(train_radii[label]), 1e-12)
                        row = {"arm": arm, "seed": seed, "protocol": protocol, "row_type": "per_tx", "class_index": int(label), "euclidean_drift": euclidean, "cosine_drift": cosine, "drift_over_train_radius": normalized}
                        drift_rows.append(row)
                        values.append(row)
                    for metric in ("euclidean_drift", "cosine_drift", "drift_over_train_radius"):
                        array = np.array([row[metric] for row in values], dtype=np.float64)
                        mean, ci_low, ci_high = mean_ci(array)
                        drift_rows.append({
                            "arm": arm,
                            "seed": seed,
                            "protocol": protocol,
                            "row_type": "summary",
                            "class_index": -1,
                            "metric": metric,
                            "mean": mean,
                            "median": float(np.median(array)),
                            "q1": float(np.quantile(array, 0.25)),
                            "q3": float(np.quantile(array, 0.75)),
                            "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                        })
                protocol_pairs = (("p0_vs_p1", "p0", "p1"), ("p0_vs_p2", "p0", "p2"), ("p0_vs_p3", "p0", "p3"))
                for comparison, left_name, right_name in protocol_pairs:
                    left = load_embedding_store(self.config, arm, seed, left_name)
                    right = load_embedding_store(self.config, arm, seed, right_name)
                    x, y = balanced_protocol_pair(left, right, self.config.domain_sample_per_tx, seed + len(leakage_rows))
                    values = domain_classifier_auc(x, y, self.config.domain_cv_folds, seed)
                    leakage_rows.append({"arm": arm, "seed": seed, "comparison": comparison, "nuisance": {"p0_vs_p1": "day", "p0_vs_p2": "receiver", "p0_vs_p3": "day_receiver"}[comparison], **values})
                xeq, yeq = balanced_equalization_pair(train_store, self.config.domain_sample_per_tx, seed + 808)
                values = domain_classifier_auc(xeq, yeq, self.config.domain_cv_folds, seed + 808)
                leakage_rows.append({"arm": arm, "seed": seed, "comparison": "equalized_0_vs_1", "nuisance": "equalization", **values})
        drift = pd.DataFrame(drift_rows)
        leakage = pd.DataFrame(leakage_rows)
        drift_path = self.config.output_root / "tables" / "centroid_drift.csv"
        leakage_path = self.config.output_root / "tables" / "domain_leakage_auc.csv"
        report_path = self.config.output_root / "reports" / "domain_robustness_report.md"
        atomic_write_csv(drift_path, drift, self.config.output_root)
        atomic_write_csv(leakage_path, leakage, self.config.output_root)
        summary = grouped_arm_summary(leakage, "auroc", ("comparison",))
        report = "# Domain robustness evaluation\n\nMeasured fact: Train-to-P0/P1/P2/P3 class-centroid drift is reported per transmitter and as mean, median, IQR, and 95% CI summaries.\n\nMeasured fact: frozen embeddings were evaluated with transmitter-balanced SGD log-loss domain classifiers for day, receiver, combined day+receiver, and equalization comparisons.\n\n" + summary.to_markdown(index=False) + "\n\nScientific interpretation: AUROC nearer 0.5 is desirable only when known-transmitter discrimination is retained; no domain-adversarial objective was optimized.\n"
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(7, "Domain Robustness Evaluation", [drift_path, leakage_path, report_path])

    def stage_08(self) -> None:
        benchmark = self.ensure_context()
        summary_rows: List[Dict[str, Any]] = []
        matched_rows: List[Dict[str, Any]] = []
        plot_samples: List[pd.DataFrame] = []
        for arm in ARM_DEFINITIONS:
            for seed in self.config.seeds:
                model, _, checkpoint_sha = load_trained_model(self.config, arm, seed, self.device, self.benchmark_sha, self.script_sha)
                extract_embedding_store(self.config, benchmark, self.guard, arm, seed, "calibration_unknown", model, checkpoint_sha, self.device, self.logger)
                del model
                train_store = load_embedding_store(self.config, arm, seed, "train_known")
                known_store = load_embedding_store(self.config, arm, seed, "p0")
                unknown_store = load_embedding_store(self.config, arm, seed, "calibration_unknown")
                geometry = fit_novelty_geometry(train_store["embedding"], train_store["labels"], self.config, seed)
                fit_global_indices = np.asarray(train_store["global_indices"][geometry["covariance_fit_positions"]], dtype=np.int64)
                fit_indices_path = self.config.output_root / "cache" / "novelty_geometry" / f"{arm}_seed_{seed}_train_fit_indices.npy"
                atomic_numpy_save(fit_indices_path, fit_global_indices, self.config.output_root)
                known_scores = compute_novelty_scores(known_store["embedding"], known_store["logits"], geometry)
                unknown_scores = compute_novelty_scores(unknown_store["embedding"], unknown_store["logits"], geometry)
                raw_path = self.config.output_root / "metrics" / "calibration_raw" / f"{arm}_seed_{seed}.npz"
                atomic_savez(raw_path, self.config.output_root, **{f"known_{k}": v for k, v in known_scores.items()}, **{f"unknown_{k}": v for k, v in unknown_scores.items()})
                for score_index, score_name in enumerate(known_scores):
                    values = summarize_novelty_score(known_scores[score_name], unknown_scores[score_name], self.config, seed + score_index * 101)
                    summary_rows.append({
                        "arm": arm,
                        "arm_name": ARM_DEFINITIONS[arm]["name"],
                        "seed": seed,
                        "known_reference": "P0",
                        "unknown_partition": "Calibration Unknown",
                        "score": score_name,
                        "fit_partition": "Train Known only",
                        "threshold_tuned": False,
                        "shrinkage": float(geometry["shrinkage"][0]),
                        **values,
                    })
                    rng = np.random.default_rng(seed + score_index)
                    k_take = min(1000, len(known_scores[score_name]))
                    u_take = min(1000, len(unknown_scores[score_name]))
                    plot_samples.append(pd.DataFrame({"arm": arm, "seed": seed, "score": score_name, "group": "Known P0", "value": known_scores[score_name][rng.choice(len(known_scores[score_name]), k_take, replace=False)]}))
                    plot_samples.append(pd.DataFrame({"arm": arm, "seed": seed, "score": score_name, "group": "Calibration Unknown", "value": unknown_scores[score_name][rng.choice(len(unknown_scores[score_name]), u_take, replace=False)]}))
                matched_rows.extend(calibration_domain_matched_rows(known_store, unknown_store, known_scores, unknown_scores, arm, seed))
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        summary = pd.DataFrame(summary_rows)
        matched = pd.DataFrame(matched_rows)
        summary_path = self.config.output_root / "tables" / "calibration_unknown_scores.csv"
        matched_path = self.config.output_root / "tables" / "calibration_unknown_domain_matched.csv"
        sample_path = self.config.output_root / "metrics" / "calibration_score_plot_samples.csv"
        report_path = self.config.output_root / "reports" / "calibration_unknown_diagnostic_report.md"
        atomic_write_csv(summary_path, summary, self.config.output_root)
        atomic_write_csv(matched_path, matched, self.config.output_root)
        atomic_write_csv(sample_path, pd.concat(plot_samples, ignore_index=True), self.config.output_root)
        report_summary = grouped_arm_summary(summary, "auroc", ("score",))
        report = "# Calibration-unknown diagnostic evaluation\n\nCalibration Unknown is used only after model freezing. All centroids and shrinkage covariance geometry were fitted exclusively from Train Known embeddings; no threshold was selected.\n\nMeasured fact: five required novelty scores report AUROC, AUPRC, Cohen's d, robust overlap, and stratified-bootstrap 95% AUROC CI. Domain-matched analyses cover day, receiver, equalization, day+receiver, and day+receiver+equalization strata.\n\n" + report_summary.to_markdown(index=False) + "\n\nScientific interpretation: these results are a zero-day precursor diagnostic, not final zero-day performance.\n"
        atomic_write_text(report_path, report, self.config.output_root)
        self.complete_stage(8, "Calibration-Unknown Diagnostic Evaluation", [summary_path, matched_path, sample_path, report_path])

    def stage_09(self) -> None:
        known = pd.read_csv(self.config.output_root / "tables" / "known_protocol_results.csv")
        separation = pd.read_csv(self.config.output_root / "tables" / "embedding_separability.csv")
        leakage = pd.read_csv(self.config.output_root / "tables" / "domain_leakage_auc.csv")
        calibration = pd.read_csv(self.config.output_root / "tables" / "calibration_unknown_scores.csv")
        known_run = known[known.protocol.isin(["p0", "p1", "p2", "p3"])].groupby(["arm", "seed"], as_index=False).fixed98_macro_f1.mean().rename(columns={"fixed98_macro_f1": "classification"})
        sep_run = separation[separation.protocol == "p0"][["arm", "seed", "fisher_ratio", "silhouette"]].rename(columns={"fisher_ratio": "separation"})
        leakage = leakage.assign(leakage_deviation=(leakage.auroc - 0.5).abs())
        domain_run = leakage.groupby(["arm", "seed"], as_index=False).leakage_deviation.mean()
        domain_run["domain_robustness"] = -domain_run.leakage_deviation
        calibration_run = calibration.groupby(["arm", "seed"], as_index=False).auroc.mean().rename(columns={"auroc": "calibration_unknown"})
        seed_summary = known_run.merge(sep_run, on=["arm", "seed"], validate="one_to_one").merge(domain_run[["arm", "seed", "domain_robustness", "leakage_deviation"]], on=["arm", "seed"], validate="one_to_one").merge(calibration_run, on=["arm", "seed"], validate="one_to_one")
        matrix_rows = []
        for arm, group in seed_summary.groupby("arm"):
            row: Dict[str, Any] = {"arm": arm, "arm_name": ARM_DEFINITIONS[arm]["name"]}
            for component in ("classification", "separation", "silhouette", "domain_robustness", "leakage_deviation", "calibration_unknown"):
                values = group.sort_values("seed")[component].to_numpy(dtype=np.float64)
                mean, low, high = mean_ci(values)
                row.update({
                    f"{component}_mean": mean,
                    f"{component}_std": float(values.std(ddof=1)),
                    f"{component}_median": float(np.median(values)),
                    f"{component}_ci95_low": low,
                    f"{component}_ci95_high": high,
                })
            matrix_rows.append(row)
        matrix = pd.DataFrame(matrix_rows).sort_values(
            ["classification_mean", "separation_mean", "domain_robustness_mean", "calibration_unknown_mean"],
            ascending=False,
        ).reset_index(drop=True)
        comparisons = (("A1", "A0"), ("A2", "A0"), ("A3", "A0"), ("A3", "A1"), ("A3", "A2"))
        component_directions = {
            "classification": "higher_is_better",
            "separation": "higher_is_better",
            "domain_robustness": "higher_is_better_negative_leakage_deviation",
            "calibration_unknown": "higher_is_better_secondary_only",
        }
        test_rows = []
        effect_rows = []
        for component_index, (component, direction) in enumerate(component_directions.items()):
            for comparison_index, (candidate_arm, reference_arm) in enumerate(comparisons):
                candidate = seed_summary[seed_summary.arm == candidate_arm].sort_values("seed")[component].to_numpy(dtype=np.float64)
                reference = seed_summary[seed_summary.arm == reference_arm].sort_values("seed")[component].to_numpy(dtype=np.float64)
                difference, ci_low, ci_high = paired_bootstrap_difference(candidate, reference, self.config.bootstrap_iterations, 42 + component_index * 100 + comparison_index)
                try:
                    if np.allclose(candidate, reference):
                        statistic, p_value = 0.0, 1.0
                    else:
                        statistic, p_value = stats.wilcoxon(candidate, reference, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    statistic, p_value = np.nan, 1.0
                test_rows.append({
                    "component": component,
                    "direction": direction,
                    "comparison": f"{candidate_arm}_vs_{reference_arm}",
                    "candidate": candidate_arm,
                    "reference": reference_arm,
                    "paired_difference_mean": difference,
                    "paired_bootstrap_ci95_low": ci_low,
                    "paired_bootstrap_ci95_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "pairs": len(candidate),
                })
                effect_rows.append({
                    "component": component,
                    "comparison": f"{candidate_arm}_vs_{reference_arm}",
                    "paired_cohen_dz": paired_cohen_dz(candidate, reference),
                    "mean_difference": difference,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                })
        tests = pd.DataFrame(test_rows)
        tests["bh_fdr_q_value"] = bh_fdr(tests.wilcoxon_p_value.fillna(1.0).to_numpy())
        tests["fdr_significant_0_05"] = tests.bh_fdr_q_value < 0.05
        effects = pd.DataFrame(effect_rows)
        top_arm = str(matrix.iloc[0].arm)
        top_seed = seed_summary[seed_summary.arm == top_arm].sort_values("seed")
        ce_seed = seed_summary[seed_summary.arm == "A0"].sort_values("seed")
        gain, gain_low, gain_high = paired_bootstrap_difference(
            top_seed.classification.to_numpy(),
            ce_seed.classification.to_numpy(),
            self.config.bootstrap_iterations,
            26_000,
        )
        if top_arm == "A0":
            decision = "SELECT_CE"
            selected_arm: Optional[str] = "A0"
            rationale = "CE ranked first on the primary known-validation criterion; auxiliary objectives did not justify replacement."
        else:
            top_sep = float(matrix.loc[matrix.arm == top_arm, "separation_mean"].iloc[0])
            ce_sep = float(matrix.loc[matrix.arm == "A0", "separation_mean"].iloc[0])
            if gain >= self.config.practical_f1_delta and gain_low >= -self.config.noninferiority_f1_delta and top_sep >= ce_sep:
                decision = ARM_DECISIONS[top_arm]
                selected_arm = top_arm
                rationale = (
                    f"{top_arm} led the fixed-frame P0–P3 macro-F1 hierarchy by {gain:.6f} versus CE "
                    f"(paired bootstrap 95% CI {gain_low:.6f} to {gain_high:.6f}) without a P0 Fisher-ratio regression."
                )
            else:
                decision = "NO_OBJECTIVE_CLEARLY_SUPERIOR"
                selected_arm = None
                rationale = (
                    f"{top_arm} ranked first lexicographically, but its CE-relative primary gain {gain:.6f} "
                    f"(95% CI {gain_low:.6f} to {gain_high:.6f}) did not satisfy the predeclared practical/noninferiority evidence rule."
                )
        decision_payload = {
            "decision": decision,
            "selected_arm": selected_arm,
            "ranking_top_arm": top_arm,
            "primary_metric": "macro-average fixed-98 macro-F1 across P0/P1/P2/P3",
            "hierarchy": ["known classification", "embedding separation", "domain robustness", "calibration-unknown diagnostics"],
            "calibration_unknown_used_for_primary_selection": False,
            "ce_relative_primary_gain": gain,
            "ce_relative_primary_gain_ci95": [gain_low, gain_high],
            "rationale": rationale,
            "generated_at": utc_now(),
        }
        seed_path = self.config.output_root / "tables" / "seed_summary.csv"
        tests_path = self.config.output_root / "tables" / "paired_statistical_tests.csv"
        effects_path = self.config.output_root / "tables" / "effect_sizes.csv"
        matrix_path = self.config.output_root / "tables" / "objective_selection_matrix.csv"
        decision_path = self.config.output_root / "statistics" / "selection_decision.json"
        statistics_manifest = self.config.output_root / "manifests" / "STATISTICS_MANIFEST.json"
        statistics_report = self.config.output_root / "reports" / "statistical_comparison_report.md"
        selection_report = self.config.output_root / "reports" / "objective_selection_report.md"
        atomic_write_csv(seed_path, seed_summary, self.config.output_root)
        atomic_write_csv(tests_path, tests, self.config.output_root)
        atomic_write_csv(effects_path, effects, self.config.output_root)
        atomic_write_csv(matrix_path, matrix, self.config.output_root)
        atomic_write_json(decision_path, decision_payload, self.config.output_root)
        atomic_write_json(statistics_manifest, {
            "paired_seed_panel": list(self.config.seeds),
            "paired_bootstrap_iterations": self.config.bootstrap_iterations,
            "wilcoxon": "two-sided paired Wilcoxon signed-rank",
            "multiple_testing": "Benjamini-Hochberg FDR across declared component comparisons",
            "effect_size": "paired Cohen dz",
            "primary_comparisons": [f"{a}_vs_{b}" for a, b in comparisons],
            "selection_rule": decision_payload,
        }, self.config.output_root)
        statistical_text = "# Statistical comparison\n\nMeasured fact: all comparisons are paired by the frozen seeds 42, 123, and 2026. Paired bootstrap confidence intervals, Wilcoxon signed-rank tests, paired Cohen dz, and BH-FDR are reported without suppressing null findings.\n\n" + tests.to_markdown(index=False) + "\n"
        selection_text = f"""# Objective selection

Measured fact: `{top_arm}` ranked first under the declared lexicographic hierarchy.

Statistical inference: its primary CE-relative paired mean difference is {gain:.6f}, with paired-bootstrap 95% CI [{gain_low:.6f}, {gain_high:.6f}].

Scientific interpretation: {rationale}

Stage 3M recommendation: **{decision}**.

Calibration Unknown remained secondary and the strict zero-day final test was unavailable.
"""
        atomic_write_text(statistics_report, statistical_text, self.config.output_root)
        atomic_write_text(selection_report, selection_text, self.config.output_root)
        self.complete_stage(9, "Statistical Comparison & Objective Selection", [seed_path, tests_path, effects_path, matrix_path, decision_path, statistics_manifest, statistics_report, selection_report])

    def generate_figures(self, decision: Mapping[str, Any]) -> pd.DataFrame:
        plt.style.use("seaborn-v0_8-whitegrid")
        figures_root = self.config.output_root / "figures"
        history = pd.read_csv(self.config.output_root / "tables" / "training_history_summary.csv")
        known = pd.read_csv(self.config.output_root / "tables" / "known_protocol_results.csv")
        degradation = pd.read_csv(self.config.output_root / "tables" / "protocol_degradation.csv")
        separation = pd.read_csv(self.config.output_root / "tables" / "embedding_separability.csv")
        leakage = pd.read_csv(self.config.output_root / "tables" / "domain_leakage_auc.csv")
        drift = pd.read_csv(self.config.output_root / "tables" / "centroid_drift.csv")
        calibration_samples = pd.read_csv(self.config.output_root / "metrics" / "calibration_score_plot_samples.csv")
        matrix = pd.read_csv(self.config.output_root / "tables" / "objective_selection_matrix.csv")

        def line_by_arm(frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str, name: str) -> None:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for arm, group in frame.groupby("arm"):
                summary = group.groupby(x)[y].agg(["mean", "std"]).reset_index()
                ax.plot(summary[x], summary["mean"], marker="o", linewidth=2, label=f"{arm}: {ARM_DEFINITIONS[arm]['name']}")
                ax.fill_between(summary[x], summary["mean"] - summary["std"].fillna(0), summary["mean"] + summary["std"].fillna(0), alpha=0.15)
            ax.set(title=title, xlabel=x.replace("_", " ").title(), ylabel=ylabel)
            ax.legend(fontsize=8)
            save_figure_pair(fig, figures_root / name, self.config.output_root)

        def bar_by_arm(frame: pd.DataFrame, category: str, value: str, title: str, ylabel: str, name: str) -> None:
            summary = frame.groupby(["arm", category])[value].mean().unstack(category)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            summary.plot(kind="bar", ax=ax)
            ax.set(title=title, xlabel="Objective arm", ylabel=ylabel)
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title=category)
            save_figure_pair(fig, figures_root / name, self.config.output_root)

        line_by_arm(history, "epoch", "train_loss", "Training loss by objective arm", "Objective loss", "training_loss_by_arm")
        line_by_arm(history, "epoch", "known_macro_fixed_mean", "Known-validation fixed-98 macro-F1", "Macro-average fixed-98 macro-F1", "validation_macro_f1_by_arm")
        bar_by_arm(known, "protocol", "fixed98_macro_f1", "P0–P3 fixed-frame macro-F1", "Fixed-98 macro-F1", "P0_P1_P2_P3_macro_f1")
        fixed_long = known.melt(id_vars=["arm", "seed", "protocol"], value_vars=["observed_class_macro_f1", "fixed98_macro_f1"], var_name="frame", value_name="macro_f1")
        bar_by_arm(fixed_long[fixed_long.protocol.isin(["p2", "p3"])], "frame", "macro_f1", "Observed vs fixed-98 macro-F1 on incomplete protocols", "Macro-F1", "fixed98_macro_f1")
        bar_by_arm(degradation, "protocol", "absolute_degradation", "Protocol degradation relative to P0", "P0 minus protocol fixed-98 macro-F1", "protocol_degradation")
        p0_sep = separation[separation.protocol == "p0"]
        bar_by_arm(p0_sep, "protocol", "silhouette", "P0 embedding silhouette", "Silhouette", "silhouette_by_arm")
        distance_long = p0_sep.melt(id_vars=["arm", "seed"], value_vars=["mean_intra_class_radius", "mean_inter_centroid_distance"], var_name="distance", value_name="value")
        bar_by_arm(distance_long, "distance", "value", "P0 intra-class radius and inter-centroid distance", "Distance", "intra_inter_distance")
        bar_by_arm(p0_sep, "protocol", "mean_nearest_centroid_margin", "P0 nearest-centroid margin", "Margin", "centroid_margin")
        bar_by_arm(leakage[leakage.comparison == "p0_vs_p2"], "comparison", "auroc", "Receiver-domain leakage", "Domain AUROC", "receiver_domain_auc")
        bar_by_arm(leakage[leakage.comparison == "equalized_0_vs_1"], "comparison", "auroc", "Equalization-domain leakage", "Domain AUROC", "equalization_domain_auc")
        drift_summary = drift[(drift.row_type == "summary") & (drift.metric == "euclidean_drift")]
        bar_by_arm(drift_summary, "protocol", "mean", "Train-to-protocol transmitter-centroid drift", "Mean Euclidean drift", "centroid_drift")

        plot_arm = decision.get("selected_arm") or decision["ranking_top_arm"]
        p0_store = load_embedding_store(self.config, str(plot_arm), 42, "p0")
        positions = deterministic_stratified_positions(p0_store["labels"], 30, 26_600)
        x = np.asarray(p0_store["embedding"][positions], dtype=np.float32)
        reduced = PCA(n_components=2, random_state=42).fit_transform(x)
        labels = np.asarray(p0_store["labels"][positions])
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="turbo", s=7, alpha=0.65)
        ax.set(title=f"PCA of P0 embeddings by transmitter — {plot_arm}", xlabel="PC1", ylabel="PC2")
        fig.colorbar(scatter, ax=ax, label="Known transmitter class")
        save_figure_pair(fig, figures_root / "PCA_embeddings_by_tx", self.config.output_root)

        receiver = np.asarray(p0_store["receiver"][positions]).astype(str)
        equalized = np.asarray(p0_store["equalized"][positions])
        receiver_codes, receiver_names = pd.factorize(receiver, sort=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        sc0 = axes[0].scatter(reduced[:, 0], reduced[:, 1], c=receiver_codes, cmap="tab20", s=7, alpha=0.65)
        axes[0].set(title="P0 embeddings by receiver", xlabel="PC1", ylabel="PC2")
        fig.colorbar(sc0, ax=axes[0], label=f"Receiver code ({len(receiver_names)} receivers)")
        sc1 = axes[1].scatter(reduced[:, 0], reduced[:, 1], c=equalized, cmap="coolwarm", s=7, alpha=0.65, vmin=0, vmax=1)
        axes[1].set(title="P0 embeddings by equalization", xlabel="PC1", ylabel="PC2")
        fig.colorbar(sc1, ax=axes[1], ticks=[0, 1], label="Equalization state")
        save_figure_pair(fig, figures_root / "PCA_embeddings_by_domain", self.config.output_root)

        selected_scores = calibration_samples[(calibration_samples.arm == plot_arm) & (calibration_samples.score == "nearest_prototype_euclidean")]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for group, values in selected_scores.groupby("group"):
            ax.hist(values.value, bins=60, density=True, alpha=0.45, label=group)
        ax.set(title=f"Calibration-unknown score distribution — {plot_arm}", xlabel="Nearest-prototype Euclidean distance", ylabel="Density")
        ax.legend()
        save_figure_pair(fig, figures_root / "calibration_unknown_score_distributions", self.config.output_root)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(matrix.classification_mean, matrix.separation_mean, c=matrix.leakage_deviation_mean, cmap="viridis_r", s=120)
        for _, row in matrix.iterrows():
            ax.annotate(row.arm, (row.classification_mean, row.separation_mean), xytext=(5, 5), textcoords="offset points")
        ax.set(title="Objective trade-off", xlabel="P0–P3 macro-average fixed-98 macro-F1", ylabel="P0 Fisher ratio")
        fig.colorbar(scatter, ax=ax, label="Mean |domain AUROC − 0.5| (lower is better)")
        save_figure_pair(fig, figures_root / "objective_tradeoff_plot", self.config.output_root)

        figure_rows = []
        for png in sorted(figures_root.glob("*.png")):
            pdf = png.with_suffix(".pdf")
            figure_rows.append({
                "figure": png.stem,
                "png": str(png),
                "png_sha256": sha256_file(png),
                "pdf": str(pdf),
                "pdf_sha256": sha256_file(pdf),
            })
        return pd.DataFrame(figure_rows)

    def generate_publication_artifacts(self, figure_manifest: pd.DataFrame, decision: Mapping[str, Any]) -> List[Path]:
        publication = self.config.output_root / "publication"
        workbook_path = publication / "Stage2_6M_summary.xlsx"
        latex_path = publication / "Stage2_6M_tables.tex"
        pdf_path = publication / "Stage2_6M_report.pdf"
        figure_manifest_path = publication / "figure_manifest.csv"
        tables = self.config.output_root / "tables"
        sheet_sources = {
            "Configuration": pd.DataFrame([self.config.frozen_payload()]),
            "Training": pd.read_csv(tables / "training_history_summary.csv"),
            "P0-P3": pd.read_csv(tables / "known_protocol_results.csv"),
            "Fixed98": pd.read_csv(tables / "fixed98_macro_results.csv"),
            "Separability": pd.read_csv(tables / "embedding_separability.csv"),
            "DomainRobustness": pd.read_csv(tables / "domain_leakage_auc.csv"),
            "CalibrationUnknown": pd.read_csv(tables / "calibration_unknown_scores.csv"),
            "Statistics": pd.read_csv(tables / "paired_statistical_tests.csv"),
            "Selection": pd.read_csv(tables / "objective_selection_matrix.csv"),
        }
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheet_sources.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.book[sheet_name]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                for column_cells in worksheet.columns:
                    width = min(42, max(10, max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2))
                    worksheet.column_dimensions[column_cells[0].column_letter].width = width
        latex_parts = ["% Stage 2.6M publication tables — generated, not hand-edited\n"]
        for name in ("arm_configuration", "fixed98_macro_results", "embedding_separability", "domain_leakage_auc", "objective_selection_matrix"):
            path = tables / f"{name}.csv"
            frame = pd.read_csv(path)
            latex_parts.append(f"\\subsection*{{{name.replace('_', ' ').title()}}}\n")
            latex_parts.append(frame.to_latex(index=False, longtable=len(frame) > 50, float_format=lambda x: f"{x:.6g}"))
        atomic_write_text(latex_path, "\n".join(latex_parts), self.config.output_root)
        atomic_write_csv(figure_manifest_path, figure_manifest, self.config.output_root)
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, "Stage 2.6M — WiSig ManyTx Controlled Representation Ablation", fontsize=16, weight="bold")
            summary = (
                f"Pipeline version: {PIPELINE_VERSION}\n"
                f"Benchmark SHA-256: {self.benchmark_sha}\n"
                f"Seed panel: {self.config.seeds}\n"
                f"Decision: {decision['decision']}\n\n"
                f"Measured fact\n{decision['ranking_top_arm']} ranked first under the declared lexicographic hierarchy.\n\n"
                f"Statistical inference\nCE-relative primary gain: {decision['ce_relative_primary_gain']:.6f}; "
                f"95% CI {decision['ce_relative_primary_gain_ci95']}.\n\n"
                f"Scientific interpretation\n{decision['rationale']}\n\n"
                f"Stage 3M recommendation\n{decision['decision']}\n\n"
                "Calibration Unknown is diagnostic only. Strict zero-day data were not accessed."
            )
            fig.text(0.08, 0.86, summary, fontsize=10, va="top", wrap=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for _, row in figure_manifest.iterrows():
                image = plt.imread(row.png)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.imshow(image)
                ax.axis("off")
                ax.set_title(row.figure.replace("_", " ").title(), fontsize=14)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        return [workbook_path, latex_path, pdf_path, figure_manifest_path]

    def stage_10(self) -> None:
        self.ensure_context()
        if self.config.profile != "full":
            raise ScientificAbort("MANYTX_STAGE2_6M_NOT_READY: pilot profile cannot create the canonical READY marker")
        for stage in range(1, 10):
            self.require_prior_stages(stage + 1)
        decision_path = self.config.output_root / "statistics" / "selection_decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        arm_config_path = self.config.output_root / "tables" / "arm_configuration.csv"
        arm_config = pd.DataFrame([
            {
                "arm": arm,
                "objective": values["name"],
                "ce_weight": 1.0,
                "supcon_weight": values["supcon_weight"],
                "prototype_weight": values["prototype_weight"],
                "temperature": self.config.temperature,
                "prototype_momentum": self.config.prototype_momentum,
                "embedding_dim": self.config.embedding_dim,
            }
            for arm, values in ARM_DEFINITIONS.items()
        ])
        atomic_write_csv(arm_config_path, arm_config, self.config.output_root)
        figure_manifest = self.generate_figures(decision)
        publication_paths = self.generate_publication_artifacts(figure_manifest, decision)
        architecture = json.loads((self.config.output_root / "manifests" / "MODEL_ARCHITECTURE_MANIFEST.json").read_text(encoding="utf-8"))
        selected_arm = decision.get("selected_arm")
        selected_loss = ARM_DEFINITIONS[selected_arm] if selected_arm else None
        canonical_objective = {
            "decision": decision["decision"],
            "selected_arm": selected_arm,
            "selected_objective": selected_loss["name"] if selected_loss else None,
            "loss_coefficients": selected_loss,
            "architecture_signature": architecture["architecture_signature"],
            "embedding_dimension": self.config.embedding_dim,
            "sampler_policy": "DomainBalancedTxSampler: Tx-primary, Rx/day/equalization-diverse",
            "augmentation_policy": {
                "enabled": self.config.augmentation_enabled,
                "phase_rotation_radians": self.config.phase_rotation_radians,
                "amplitude_jitter": self.config.amplitude_jitter,
                "awgn_std": self.config.awgn_std,
                "maximum_circular_shift": self.config.maximum_circular_shift,
            },
            "optimizer_policy": {"optimizer": "AdamW", "learning_rate": self.config.learning_rate, "weight_decay": self.config.weight_decay, "scheduler": "CosineAnnealingLR"},
            "training_budget": {"maximum_epochs": self.config.max_epochs, "minimum_epochs": self.config.minimum_epochs, "group_early_stopping_patience": self.config.early_stopping_patience},
            "seed_evidence": list(self.config.seeds),
            "selection_rationale": decision["rationale"],
            "calibration_unknown_role": "secondary diagnostic only",
            "strict_zero_day_role": "unavailable and not accessed",
            "generated_at": utc_now(),
        }
        canonical_path = self.config.output_root / "manifests" / "CANONICAL_STAGE3M_OBJECTIVE.json"
        atomic_write_json(canonical_path, canonical_objective, self.config.output_root)
        required_reports = [
            "input_verification_report.md",
            "domain_sampler_report.md",
            "model_and_loss_validation.md",
            "training_report.md",
            "known_tx_evaluation_report.md",
            "embedding_separability_report.md",
            "domain_robustness_report.md",
            "calibration_unknown_diagnostic_report.md",
            "statistical_comparison_report.md",
            "objective_selection_report.md",
        ]
        missing_reports = [name for name in required_reports if not (self.config.output_root / "reports" / name).is_file()]
        if missing_reports:
            raise ScientificAbort(f"Missing required reports: {missing_reports}")
        reviewer_path = self.config.output_root / "reports" / "reviewer_ready_stage2_6m_report.md"
        reviewer = f"""# Reviewer-ready Stage 2.6M report

## Measured fact

All four representation-objective arms were trained on Train Known only for seeds 42, 123, and 2026 with the same backbone, initialization, ordered sampler exposure, augmentation stream, optimizer family, synchronized epoch budget, and checkpoint rules. P0–P3 fixed-frame metrics, embedding geometry, nuisance-domain leakage, and frozen-model Calibration Unknown diagnostics were completed.

The lexicographic ranking leader was `{decision['ranking_top_arm']}`. Its CE-relative primary difference was {decision['ce_relative_primary_gain']:.6f}, with paired-bootstrap 95% CI {decision['ce_relative_primary_gain_ci95']}.

## Statistical inference

Declared paired comparisons use paired bootstrap intervals and Wilcoxon signed-rank tests across the frozen seed panel, with BH-FDR correction and paired Cohen dz effect sizes. Null and unresolved outcomes are retained in the tables.

## Scientific interpretation

{decision['rationale']}

Domain AUROC is interpreted jointly with Tx performance; a value nearer 0.5 is not treated as beneficial if transmitter discrimination degrades.

## Stage 3M recommendation

**{decision['decision']}**

        Calibration Unknown is not final zero-day evidence. Strict zero-day signals, labels, embeddings, metrics, and thresholds were never accessed.
"""
        atomic_write_text(reviewer_path, reviewer, self.config.output_root)
        self.guard.scan_output()
        self.guard.assert_zero()
        guard_path = self.config.output_root / "manifests" / "STRICT_TEST_GUARD.json"
        atomic_write_json(guard_path, self.guard.manifest(), self.config.output_root)
        required_tables = {
            "arm_configuration.csv",
            "arm_parameter_equivalence.csv",
            "training_history_summary.csv",
            "seed_summary.csv",
            "known_protocol_results.csv",
            "known_protocol_per_class.csv",
            "fixed98_macro_results.csv",
            "protocol_degradation.csv",
            "embedding_separability.csv",
            "centroid_drift.csv",
            "domain_leakage_auc.csv",
            "calibration_unknown_scores.csv",
            "calibration_unknown_domain_matched.csv",
            "paired_statistical_tests.csv",
            "effect_sizes.csv",
            "objective_selection_matrix.csv",
        }
        absent_tables = sorted(name for name in required_tables if not (self.config.output_root / "tables" / name).is_file())
        if absent_tables:
            raise ScientificAbort(f"Missing required tables: {absent_tables}")
        checkpoint_manifest = json.loads((self.config.output_root / "manifests" / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
        if len(checkpoint_manifest["checkpoints"]) != 36:
            raise ScientificAbort("Checkpoint gate requires 4 arms × 3 seeds × 3 checkpoint types")
        final_status_path = self.config.output_root / "manifests" / "STAGE2_6M_FINAL_STATUS.json"
        final_status = {
            "status": "MANYTX_STAGE2_6M_READY",
            "pipeline_version": PIPELINE_VERSION,
            "script_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(),
            "stage2m_script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "stage2m_artifact_manifest_sha256": EXPECTED_STAGE2M_HASH_MANIFEST_SHA256,
            "architecture_signature": architecture["architecture_signature"],
            "seed_panel": list(self.config.seeds),
            "profile": self.config.profile,
            "decision": decision["decision"],
            "selected_arm": selected_arm,
            "benchmark_sha256": self.benchmark_sha,
            "strict_zero_day_counter_semantics": "violation counters; structural prevention is the primary access-control evidence",
            "strict_zero_day_violation_counters": self.guard.counters(),
            "success_gates": {
                "benchmark_sha_correct": True,
                "stage2m_ready_verified": True,
                "upstream_read_only": True,
                "strict_arrays_never_loaded": True,
                "four_arms_three_seeds_completed": True,
                "same_architecture": True,
                "same_training_exposure": True,
                "p0_p3_and_fixed98_complete": True,
                "embedding_domain_calibration_statistics_complete": True,
                "publication_artifacts_complete": True,
                "stage3m_objective_manifest_complete": True,
                "forbidden_strict_artifacts_absent": True,
            },
            "generated_at": utc_now(),
        }
        atomic_write_json(final_status_path, final_status, self.config.output_root)
        stage10_outputs = [reviewer_path, canonical_path, arm_config_path, guard_path, final_status_path, *publication_paths]
        stage10_manifest = {
            "stage": 10,
            "name": "Freeze Recommendation for Stage 3M",
            "status": "PASS",
            "pipeline_version": PIPELINE_VERSION,
            "benchmark_sha256": self.benchmark_sha,
            "stage2m_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "script_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(),
            "decision": decision["decision"],
            "outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in stage10_outputs],
            "completed_at": utc_now(),
        }
        atomic_write_json(self.stage_manifest_path(10), stage10_manifest, self.config.output_root)
        file_manifest_path = self.config.output_root / "manifests" / "FILE_MANIFEST.json"
        hash_manifest_path = self.config.output_root / "manifests" / "HASH_MANIFEST.json"
        excluded = {file_manifest_path.resolve(), hash_manifest_path.resolve(), (self.config.output_root / "MANYTX_STAGE2_6M_READY.txt").resolve()}
        files = [path for path in sorted(self.config.output_root.rglob("*")) if path.is_file() and path.resolve() not in excluded and path.name != "MANYTX_STAGE2_6M_NOT_READY.txt"]
        file_manifest = [{"relative_path": str(path.relative_to(self.config.output_root)), "bytes": path.stat().st_size} for path in files]
        atomic_write_json(file_manifest_path, {"files": file_manifest, "count": len(file_manifest)}, self.config.output_root)
        files_with_manifest = files + [file_manifest_path]
        hash_rows = [{"relative_path": str(path.relative_to(self.config.output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files_with_manifest]
        atomic_write_json(hash_manifest_path, {"algorithm": "SHA-256", "files": hash_rows, "count": len(hash_rows)}, self.config.output_root)
        artifact_sha = sha256_file(hash_manifest_path)
        ready_path = self.config.output_root / "MANYTX_STAGE2_6M_READY.txt"
        not_ready_path = self.config.output_root / "MANYTX_STAGE2_6M_NOT_READY.txt"
        if not_ready_path.exists():
            not_ready_path.unlink()
        ready_text = (
            "MANYTX_STAGE2_6M_READY\n"
            f"decision={decision['decision']}\n"
            f"benchmark_sha256={self.benchmark_sha}\n"
            f"artifact_sha256={artifact_sha}\n"
            "strict_zero_day_signal_read_violations=0\nstrict_zero_day_label_read_violations=0\n"
            "strict_zero_day_embedding_read_violations=0\nstrict_zero_day_metric_read_violations=0\n"
            "strict_zero_day_threshold_read_violations=0\n"
        )
        atomic_write_text(ready_path, ready_text, self.config.output_root)
        print("[PASS] Stage 10 — Freeze Recommendation for Stage 3M")
        print("\nMANYTX_STAGE2_6M_READY")
        print(f"\nBenchmark SHA-256:\n{self.benchmark_sha}")
        print("\nStrict zero-day signal access violations: 0")
        print("Strict zero-day label access violations: 0")
        print("Strict zero-day embedding access violations: 0")
        print("Strict zero-day metric access violations: 0")
        print(f"\nSelected Stage 3M objective:\n{decision['decision']}")
        print(f"\nStage 2.6M output:\n{self.config.output_root}")
        print(f"\nArtifact SHA-256:\n{artifact_sha}")

    def run(self) -> None:
        print(startup_banner())
        self.logger.info("Device: %s", self.device)
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            self.logger.info("GPU: %s | memory %.2f GiB", properties.name, properties.total_memory / 2**30)
        stages = {
            1: self.stage_01,
            2: self.stage_02,
            3: self.stage_03,
            4: self.stage_04,
            5: self.stage_05,
            6: self.stage_06,
            7: self.stage_07,
            8: self.stage_08,
            9: self.stage_09,
            10: self.stage_10,
        }
        for stage_number in range(self.config.stage_start, self.config.stage_end + 1):
            if stage_number > 1:
                self.require_prior_stages(stage_number)
            stages[stage_number]()


def synthetic_validation(config: Stage26Config) -> None:
    """Dependency and finite-gradient validation that does not claim scientific results."""
    config.num_workers = 0
    config.batch_size = 196
    config.samples_per_tx = 2
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    runs = create_training_runs(config, 42, device)
    x = torch.randn(config.batch_size, 2, 256, device=device)
    labels = torch.arange(config.batch_size, device=device) % EXPECTED_KNOWN_CLASSES
    for arm, run in runs.items():
        outputs = run.model(x)
        loss, components = objective_loss(arm, outputs, labels, run.prototypes, config)
        loss.backward()
        if not math.isfinite(components["loss"]):
            raise ScientificAbort(f"Synthetic validation failed for {arm}")
        print(f"[PASS] {arm} {ARM_DEFINITIONS[arm]['name']} finite loss={components['loss']:.6f}")
    print("SYNTHETIC_CODE_VALIDATION_PASS — no benchmark results were produced")


def load_config_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration JSON must be an object")
    valid = {field.name for field in dataclasses.fields(Stage26Config)}
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {unknown}")
    if "seeds" in payload:
        payload["seeds"] = tuple(int(x) for x in payload["seeds"])
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2.6M WiSig ManyTx controlled representation-learning ablation")
    parser.add_argument("--config", type=Path, help="Optional JSON configuration override")
    parser.add_argument("--branch-root", type=str, help="Canonical MANYTX_ZERO_DAY_BRANCH_v1.0.3 path")
    parser.add_argument("--benchmark", type=str, help="Explicit canonical benchmark HDF5 path")
    parser.add_argument("--stage2m-dir", type=str, help="Explicit frozen Stage 2M diagnostics directory")
    parser.add_argument("--profile", choices=("full", "pilot"), help="Execution profile; only full may create READY")
    parser.add_argument("--stage-start", type=int, help="First internal stage (1..10)")
    parser.add_argument("--stage-end", type=int, help="Last internal stage (1..10)")
    parser.add_argument("--device", type=str, help="auto, cpu, cuda, or explicit torch device")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count")
    parser.add_argument("--no-resume", action="store_true", help="Disable exact checkpoint resume")
    parser.add_argument("--synthetic-validation", action="store_true", help="Run finite-gradient code validation without benchmark access")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Stage26Config:
    values: Dict[str, Any] = {}
    if args.config:
        values.update(load_config_file(args.config))
    cli_mapping = {
        "branch_root": args.branch_root,
        "benchmark_path": args.benchmark,
        "stage2m_dir": args.stage2m_dir,
        "profile": args.profile,
        "stage_start": args.stage_start,
        "stage_end": args.stage_end,
        "device": args.device,
        "num_workers": args.num_workers,
    }
    values.update({key: value for key, value in cli_mapping.items() if value is not None})
    if args.no_resume:
        values["resume"] = False
    return Stage26Config(**values)


def write_not_ready(config: Stage26Config, exc: BaseException) -> None:
    try:
        config.validate()
        config.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "MANYTX_STAGE2_6M_NOT_READY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "generated_at": utc_now(),
        }
        atomic_write_json(config.output_root / "manifests" / "STAGE2_6M_FINAL_STATUS.json", payload, config.output_root)
        atomic_write_text(config.output_root / "MANYTX_STAGE2_6M_NOT_READY.txt", f"MANYTX_STAGE2_6M_NOT_READY\n{type(exc).__name__}: {exc}\n", config.output_root)
        ready = config.output_root / "MANYTX_STAGE2_6M_READY.txt"
        if ready.exists():
            ready.unlink()
    except Exception:
        traceback.print_exc()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.synthetic_validation:
        synthetic_validation(config)
        return 0
    try:
        pipeline = Stage26Pipeline(config)
        pipeline.run()
        return 0
    except Exception as exc:
        print("MANYTX_STAGE2_6M_NOT_READY", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        write_not_ready(config, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
