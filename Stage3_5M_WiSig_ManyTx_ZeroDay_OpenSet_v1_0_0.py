#!/usr/bin/env python3
"""Stage 3.5M — frozen-teacher WiSig ManyTx open-set detection.

The executable fits deterministic post-hoc scorers using Train Known, freezes
thresholds using P0-P3 Known Validation only, and permits strict-zero-day signal
access for the first time in Stage 08 after a cryptographic evaluation lock has
been written.  The Stage 3M teacher is always inference-only and immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


PIPELINE_VERSION = "1.0.0"
CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
CANONICAL_BENCHMARK = "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE3M_HASH_MANIFEST_SHA256 = "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0"
EXPECTED_TEACHER_SHA256 = "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4"
EXPECTED_TEACHER_STATE_SHA256 = "7d6c6ff609fb86618ae7b92bcd55b0c8a440ed2769561d4de9b4485802e639d7"
EXPECTED_TEACHER_SEED = 123
EXPECTED_TEACHER_VERSION = "1.0"
EXPECTED_PARAMETER_COUNT = 849_634
EXPECTED_CLASSES = 98
EXPECTED_EMBEDDING_DIM = 128
KNOWN_VALIDATION = ("p0", "p1", "p2", "p3")
NON_STRICT_PARTITIONS = ("train_known", *KNOWN_VALIDATION, "calibration_unknown")
STRICT_PARTITIONS = {
    "strict_zero_day_test": ("strict_zero_day_test_indices.npy", 216_000),
    "strict_zero_day_shift_test": ("strict_zero_day_shift_test_indices.npy", 3_000),
}
SCORER_ORDER = (
    "S0_MSP",
    "S1_ENERGY",
    "S2_PROTOTYPE_COSINE",
    "S3_MAHALANOBIS",
    "S4_DIAG_GAUSSIAN_NLL",
)
SCORER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "S0_MSP": {"formula": "1 - max softmax probability", "direction": "higher_is_more_unknown", "fit": "none"},
    "S1_ENERGY": {"formula": "-T * logsumexp(logits / T)", "direction": "higher_is_more_unknown", "fit": "none"},
    "S2_PROTOTYPE_COSINE": {"formula": "1 - max normalized class-prototype cosine", "direction": "higher_is_more_unknown", "fit": "Train Known class means"},
    "S3_MAHALANOBIS": {"formula": "minimum regularized tied-covariance squared distance", "direction": "higher_is_more_unknown", "fit": "Train Known means and pooled covariance"},
    "S4_DIAG_GAUSSIAN_NLL": {"formula": "minimum class-conditional diagonal Gaussian negative log likelihood", "direction": "higher_is_more_unknown", "fit": "Train Known class means and diagonal variances"},
}
CANONICAL_SCORER_POLICY = "ALL_PREDECLARED"
CANONICAL_THRESHOLD_POLICY = "KNOWN_VALIDATION_TARGET_ACCEPTANCE_0.95_PER_SCORER"
STRICT_COUNTER_KEYS = (
    "strict_zero_day_signal_read_violations",
    "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations",
    "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations",
    "strict_zero_day_fit_violations",
)
REQUIRED_OUTPUT_DIRS = (
    "configs", "logs", "manifests", "metrics", "tables", "statistics",
    "figures", "reports", "publication", "scores", "thresholds", "performance",
)
FINAL_HASH_EXCLUSIONS = {
    "manifests/STAGE3_5M_HASH_MANIFEST.json": "A hash manifest cannot include itself.",
    "manifests/STAGE_11_CHECKPOINT.json": "The atomic completion checkpoint is written last.",
    "MANYTX_STAGE3_5M_READY.txt": "READY is written only after the final manifest verifies.",
    "MANYTX_STAGE3_5M_NOT_READY.txt": "READY and NOT_READY are mutually exclusive transaction markers.",
}
FINAL_REQUIRED = (
    "manifests/STAGE3_5M_FINAL_STATUS.json",
    "manifests/STAGE3_5M_HASH_MANIFEST.json",
    "manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json",
    "manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.sha256",
    "manifests/SCORER_POLICY_FREEZE.json",
    "manifests/PRE_STRICT_KNOWN_SCORE_BUNDLE.json",
    "manifests/FINAL_STRICT_SCORE_BUNDLE.json",
    "manifests/STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json",
    "thresholds/ZD_STRICT_THRESHOLDS.json",
    "tables/strict_open_set_metrics.csv",
    "statistics/strict_bootstrap_confidence_intervals.csv",
    "publication/Stage3_5M_report.pdf",
    "publication/Stage3_5M_tables.xlsx",
    "MANYTX_STAGE3_5M_READY.txt",
)


class ScientificAbort(RuntimeError):
    """Raised when a frozen scientific or leakage-control invariant fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_int64_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_strict_subset_relationship(
    main_indices: np.ndarray,
    shift_indices: np.ndarray,
    non_strict_indices: Optional[np.ndarray] = None,
    expected_main_rows: int = 216_000,
    expected_shift_rows: int = 3_000,
) -> Dict[str, Any]:
    """Validate the frozen contract: the shifted strict partition is nested in main."""
    main = np.asarray(main_indices, dtype=np.int64).reshape(-1)
    shift = np.asarray(shift_indices, dtype=np.int64).reshape(-1)
    main_unique = len(np.unique(main)) == len(main)
    shift_unique = len(np.unique(shift)) == len(shift)
    intersection = np.intersect1d(main, shift, assume_unique=main_unique and shift_unique)
    shift_outside = np.setdiff1d(shift, main, assume_unique=main_unique and shift_unique)
    main_outside = np.setdiff1d(main, shift, assume_unique=main_unique and shift_unique)
    non_strict_overlap = 0
    if non_strict_indices is not None:
        non_strict = np.asarray(non_strict_indices, dtype=np.int64).reshape(-1)
        non_strict_overlap = int(np.intersect1d(main, non_strict).size + np.intersect1d(shift, non_strict).size)
    audit = {
        "main_rows": int(len(main)), "shift_rows": int(len(shift)),
        "main_unique": main_unique, "shift_unique": shift_unique,
        "intersection_rows": int(len(intersection)),
        "shift_subset_of_main": bool(len(shift_outside) == 0),
        "shift_rows_outside_main": int(len(shift_outside)),
        "main_rows_outside_shift": int(len(main_outside)),
        "non_strict_overlap_rows": non_strict_overlap,
    }
    required = (
        len(main) == expected_main_rows and len(shift) == expected_shift_rows
        and main_unique and shift_unique and len(intersection) == len(shift)
        and len(shift_outside) == 0 and non_strict_overlap == 0
    )
    if not required:
        raise ScientificAbort(f"Frozen strict subset relationship mismatch: {audit}")
    return audit


def parse_ready_marker(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise ScientificAbort(f"Required READY marker missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ScientificAbort(f"Empty READY marker: {path}")
    result = {"marker": lines[0]}
    for line in lines[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def assert_within(path: Path, root: Path) -> Path:
    resolved, boundary = path.resolve(), root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ScientificAbort(f"Output escapes Stage 3.5M root: {resolved}")
    return resolved


def atomic_text(path: Path, text: str, root: Path) -> None:
    assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any], root: Path) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", root)


def atomic_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, root: Path, **arrays: np.ndarray) -> None:
    assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScientificAbort(f"Cannot import frozen implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in state.items():
        digest.update(key.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


class StrictZeroDayGuard:
    """Structural strict-partition gate with independent violation counters."""

    def __init__(self) -> None:
        self._counters = {key: 0 for key in STRICT_COUNTER_KEYS}
        self._allowed_indices: Dict[str, np.ndarray] = {}
        self.final_lock_active = False
        self.final_stage = 0

    @staticmethod
    def is_strict(path_or_partition: Any) -> bool:
        token = str(path_or_partition).lower()
        return "strict_zero_day" in token or "zero_day_shift_test" in token

    def reject(self, kind: str, reason: str) -> None:
        mapping = {
            "signal": STRICT_COUNTER_KEYS[0], "label": STRICT_COUNTER_KEYS[1],
            "embedding": STRICT_COUNTER_KEYS[2], "metric": STRICT_COUNTER_KEYS[3],
            "threshold": STRICT_COUNTER_KEYS[4], "fit": STRICT_COUNTER_KEYS[5],
        }
        if kind not in mapping:
            raise ValueError(kind)
        self._counters[mapping[kind]] += 1
        raise ScientificAbort(f"STRICT_ZERO_DAY_{kind.upper()}_ACCESS_VIOLATION: {reason}")

    def forbid_data_path(self, path: Path, operation: str) -> None:
        if self.is_strict(path):
            self.reject("signal", f"{operation}: {path}")

    def register_allowed_indices(self, partition: str, indices: np.ndarray) -> None:
        if partition not in NON_STRICT_PARTITIONS:
            self.reject("signal", f"unapproved partition registration: {partition}")
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ScientificAbort(f"Invalid authorized indices: {partition}")
        self._allowed_indices[partition] = values

    def authorize_rows(self, partition: str, indices: np.ndarray, operation: str) -> None:
        if partition not in self._allowed_indices:
            self.reject("signal", f"unregistered partition in {operation}: {partition}")
        requested = np.asarray(indices, dtype=np.int64)
        if not np.isin(requested, self._allowed_indices[partition]).all():
            self.reject("signal", f"unauthorized rows in {operation}: {partition}")

    def activate_final_lock(self, stage: int, lock_current: bool) -> None:
        if stage < 8 or not lock_current:
            self.reject("signal", "final evaluation lock is absent or stale")
        self.final_lock_active = True
        self.final_stage = stage

    def authorize_strict(self, kind: str, stage: int, lock_current: bool) -> None:
        if stage < 8 or not self.final_lock_active or not lock_current:
            self.reject(kind, f"strict access attempted before frozen Stage-08 lock (stage={stage})")

    def assert_fitting_allowed(self) -> None:
        if self.final_lock_active:
            self.reject("fit", "scorer fitting attempted after final evaluation lock")

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def assert_zero(self) -> None:
        if any(self._counters.values()):
            raise ScientificAbort(f"Strict-zero-day violation counters are non-zero: {self._counters}")


@dataclass
class Stage35Config:
    branch_root: str
    repository_root: str = ""
    benchmark_path: str = ""
    stage3m_dir: str = ""
    output_dir: str = ""
    local_cache_root: str = "/content/wisig_stage3_5m_cache"
    eval_batch_size: int = 1024
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    amp_enabled: bool = True
    device: str = "auto"
    energy_temperature: float = 1.0
    covariance_regularization: float = 0.001
    diagonal_variance_floor: float = 0.0001
    target_known_acceptance: float = 0.95
    alternate_known_quantile: float = 0.99
    calibrated_analysis: bool = False
    bootstrap_replicates: int = 1000
    bootstrap_max_per_group: int = 20_000
    random_seed: int = 3_500_001
    stage_start: int = 1
    stage_end: int = 11
    resume: bool = True
    preflight: bool = False

    @property
    def branch_root_path(self) -> Path:
        return Path(self.branch_root).expanduser().resolve()

    @property
    def repository_root_path(self) -> Path:
        return Path(self.repository_root).expanduser().resolve() if self.repository_root else Path(__file__).resolve().parent

    @property
    def benchmark_path_resolved(self) -> Path:
        return Path(self.benchmark_path).expanduser().resolve() if self.benchmark_path else self.branch_root_path / "01_benchmark_engineering" / "benchmark" / f"{CANONICAL_BENCHMARK}.h5"

    @property
    def stage3m_path(self) -> Path:
        return Path(self.stage3m_dir).expanduser().resolve() if self.stage3m_dir else self.branch_root_path / "04_canonical_teacher"

    @property
    def stage26_path(self) -> Path:
        return self.branch_root_path / "03_representation_ablation"

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else self.branch_root_path / "05_zero_day_open_set"

    def validate(self) -> None:
        if self.branch_root_path.name != CANONICAL_BRANCH:
            raise ScientificAbort(f"Branch root must end in {CANONICAL_BRANCH}")
        if self.output_root.parent != self.branch_root_path or self.output_root.name != "05_zero_day_open_set":
            raise ScientificAbort("Stage 3.5M output must be branch-root/05_zero_day_open_set")
        if not 1 <= self.stage_start <= self.stage_end <= 11:
            raise ValueError("Stage range must be within 1..11")
        if not 0.5 < self.target_known_acceptance < 1.0 or not self.target_known_acceptance < self.alternate_known_quantile < 1.0:
            raise ValueError("Invalid known-only threshold quantiles")
        if self.energy_temperature <= 0 or self.covariance_regularization <= 0 or self.diagonal_variance_floor <= 0:
            raise ValueError("Scorer regularization and temperature must be positive")
        if self.eval_batch_size <= 0 or self.num_workers < 0 or self.bootstrap_replicates < 1:
            raise ValueError("Invalid execution configuration")

    def scientific_payload(self) -> Dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_seed": EXPECTED_TEACHER_SEED,
            "teacher_version": EXPECTED_TEACHER_VERSION,
            "scorers": SCORER_DEFINITIONS,
            "canonical_scorer_policy": CANONICAL_SCORER_POLICY,
            "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY,
            "energy_temperature": self.energy_temperature,
            "covariance_regularization": self.covariance_regularization,
            "diagonal_variance_floor": self.diagonal_variance_floor,
            "target_known_acceptance": self.target_known_acceptance,
            "alternate_known_quantile": self.alternate_known_quantile,
            "calibrated_analysis": self.calibrated_analysis,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_max_per_group": self.bootstrap_max_per_group,
            "random_seed": self.random_seed,
        }

    def configuration_sha256(self) -> str:
        return sha256_object(self.scientific_payload())


def fit_statistics(embeddings: np.ndarray, labels: np.ndarray, covariance_regularization: float, diagonal_floor: float) -> Dict[str, np.ndarray]:
    """Fit deterministic known-only class and covariance statistics."""
    z = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or z.shape[1] != EXPECTED_EMBEDDING_DIM or y.shape != (len(z),):
        raise ScientificAbort("Invalid Train Known embedding/label shapes")
    if set(np.unique(y)) != set(range(EXPECTED_CLASSES)):
        raise ScientificAbort("Train Known must contain all 98 classes")
    counts = np.bincount(y, minlength=EXPECTED_CLASSES).astype(np.int64)
    sums = np.zeros((EXPECTED_CLASSES, EXPECTED_EMBEDDING_DIM), dtype=np.float64)
    second = z.T @ z
    np.add.at(sums, y, z)
    means = sums / counts[:, None]
    within = second - sum(counts[c] * np.outer(means[c], means[c]) for c in range(EXPECTED_CLASSES))
    covariance = within / max(len(z) - EXPECTED_CLASSES, 1)
    scale = max(float(np.trace(covariance) / EXPECTED_EMBEDDING_DIM), 1e-12)
    regularized = covariance + covariance_regularization * scale * np.eye(EXPECTED_EMBEDDING_DIM)
    precision = np.linalg.inv(regularized)
    diagonal = np.empty_like(means)
    for class_index in range(EXPECTED_CLASSES):
        residual = z[y == class_index] - means[class_index]
        diagonal[class_index] = np.maximum(np.mean(residual * residual, axis=0), diagonal_floor)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    prototypes = means / np.maximum(norms, 1e-12)
    if not all(np.isfinite(value).all() for value in (means, precision, diagonal, prototypes)):
        raise ScientificAbort("Non-finite known-only scorer fit")
    return {
        "counts": counts,
        "means": means.astype(np.float32),
        "prototypes": prototypes.astype(np.float32),
        "precision": precision.astype(np.float32),
        "diagonal_variance": diagonal.astype(np.float32),
    }


def fit_statistics_from_sufficient(
    counts: np.ndarray,
    sums: np.ndarray,
    second: np.ndarray,
    diagonal_sums: np.ndarray,
    covariance_regularization: float,
    diagonal_floor: float,
) -> Dict[str, np.ndarray]:
    if counts.shape != (EXPECTED_CLASSES,) or np.any(counts <= 0):
        raise ScientificAbort("Train Known sufficient statistics lack one or more classes")
    total = int(counts.sum())
    means = sums / counts[:, None]
    within = second - sum(counts[c] * np.outer(means[c], means[c]) for c in range(EXPECTED_CLASSES))
    covariance = within / max(total - EXPECTED_CLASSES, 1)
    scale = max(float(np.trace(covariance) / EXPECTED_EMBEDDING_DIM), 1e-12)
    precision = np.linalg.inv(covariance + covariance_regularization * scale * np.eye(EXPECTED_EMBEDDING_DIM))
    variances = diagonal_sums / counts[:, None] - means * means
    variances = np.maximum(variances, diagonal_floor)
    prototypes = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    outputs = {
        "counts": counts.astype(np.int64), "means": means.astype(np.float32),
        "prototypes": prototypes.astype(np.float32), "precision": precision.astype(np.float32),
        "diagonal_variance": variances.astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ScientificAbort("Non-finite known-only sufficient-statistic fit")
    return outputs


def score_outputs(logits: np.ndarray, embeddings: np.ndarray, fit: Mapping[str, np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """Return [N,5] scores; every column increases with unknownness."""
    l = np.asarray(logits, dtype=np.float64)
    z = np.asarray(embeddings, dtype=np.float64)
    msp_logits = l - l.max(axis=1, keepdims=True)
    probabilities = np.exp(msp_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    msp = 1.0 - probabilities.max(axis=1)
    raw = l / temperature
    energy = -temperature * (np.log(np.exp(raw - raw.max(axis=1, keepdims=True)).sum(axis=1)) + raw.max(axis=1))
    prototypes = np.asarray(fit["prototypes"], dtype=np.float64)
    cosine = 1.0 - np.max(z @ prototypes.T, axis=1)
    means = np.asarray(fit["means"], dtype=np.float64)
    precision = np.asarray(fit["precision"], dtype=np.float64)
    variances = np.asarray(fit["diagonal_variance"], dtype=np.float64)
    mahal = np.empty((len(z), EXPECTED_CLASSES), dtype=np.float64)
    diag_nll = np.empty_like(mahal)
    log_variance = np.log(variances)
    for class_index in range(EXPECTED_CLASSES):
        delta = z - means[class_index]
        mahal[:, class_index] = np.einsum("ni,ij,nj->n", delta, precision, delta, optimize=True)
        diag_nll[:, class_index] = 0.5 * np.sum(delta * delta / variances[class_index] + log_variance[class_index], axis=1)
    result = np.column_stack((msp, energy, cosine, mahal.min(axis=1), diag_nll.min(axis=1))).astype(np.float32)
    if result.shape != (len(z), len(SCORER_ORDER)) or not np.isfinite(result).all():
        raise ScientificAbort("Invalid open-set scorer output")
    return result


def freeze_known_thresholds(scores: Mapping[str, np.ndarray], target_acceptance: float, alternate_quantile: float) -> Dict[str, Any]:
    if set(scores) != set(SCORER_ORDER):
        raise ScientificAbort("Threshold fitting requires all predeclared scorers")
    thresholds: Dict[str, Any] = {}
    for scorer in SCORER_ORDER:
        values = np.asarray(scores[scorer], dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ScientificAbort(f"Invalid known-validation scores for {scorer}")
        canonical = quantile_higher(values, target_acceptance)
        alternate = quantile_higher(values, alternate_quantile)
        thresholds[scorer] = {
            "canonical_threshold": canonical,
            "canonical_rule": f"known score quantile {target_acceptance:.6f}",
            "target_known_acceptance": target_acceptance,
            "realized_known_acceptance": float(np.mean(values <= canonical)),
            "alternate_quantile_threshold": alternate,
            "alternate_quantile": alternate_quantile,
            "direction": "unknown_if_score_above_threshold",
            "source": "P0_P1_P2_P3_KNOWN_VALIDATION_ONLY",
        }
    return thresholds


def oscr_score(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray) -> float:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    correct = np.asarray(known_correct, dtype=bool)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    values = np.concatenate((known_scores, unknown_scores))
    unknown_marker = np.concatenate((np.zeros(len(known_scores), dtype=np.int8), np.ones(len(unknown_scores), dtype=np.int8)))
    correct_marker = np.concatenate((correct.astype(np.int8), np.zeros(len(unknown_scores), dtype=np.int8)))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative_unknown = np.cumsum(unknown_marker[order])
    cumulative_correct = np.cumsum(correct_marker[order])
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_values) != 0), len(sorted_values) - 1]
    fpr = np.r_[0.0, cumulative_unknown[group_ends] / len(unknown_scores)]
    ccr = np.r_[0.0, cumulative_correct[group_ends] / len(known_scores)]
    return float(np.trapezoid(ccr, fpr))


def open_set_metrics(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    correct = np.asarray(known_correct, dtype=bool)
    targets = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8)))
    values = np.concatenate((known, unknown))
    predicted = (values > threshold).astype(np.int8)
    fpr, tpr, _ = roc_curve(targets, values)
    candidates = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[candidates[0]]) if len(candidates) else 1.0
    detection_error = float(np.min(0.5 * (fpr + 1.0 - tpr)))
    return {
        "auroc": float(roc_auc_score(targets, values)),
        "auprc": float(average_precision_score(targets, values)),
        "unknown_f1": float(f1_score(targets, predicted, pos_label=1, zero_division=0)),
        "known_f1": float(f1_score(targets, predicted, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "fpr_at_95_tpr": fpr95,
        "detection_error": detection_error,
        "oscr": oscr_score(known, correct, unknown),
        "threshold": float(threshold),
        "known_acceptance_rate": float(np.mean(known <= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown > threshold)),
        "known_rows": float(len(known)),
        "unknown_rows": float(len(unknown)),
    }


def compare_inference_evidence(
    protocol: str,
    stage3_indices: np.ndarray,
    stage3_labels: np.ndarray,
    stage3_predictions: np.ndarray,
    stage35_indices: np.ndarray,
    stage35_labels: np.ndarray,
    stage35_predictions: np.ndarray,
    stage3_accuracy: float,
    stage3_fixed98_macro_f1: float,
    stage35_accuracy: float,
    stage35_fixed98_macro_f1: float,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    reference_indices = np.asarray(stage3_indices, dtype=np.int64)
    reference_labels = np.asarray(stage3_labels, dtype=np.int64)
    reference_predictions = np.asarray(stage3_predictions, dtype=np.int64)
    measured_indices = np.asarray(stage35_indices, dtype=np.int64)
    measured_labels = np.asarray(stage35_labels, dtype=np.int64)
    measured_predictions = np.asarray(stage35_predictions, dtype=np.int64)
    result: Dict[str, Any] = {
        "protocol": protocol, "rows": int(len(measured_indices)),
        "global_indices_equivalent": bool(np.array_equal(reference_indices, measured_indices)),
        "labels_equivalent": bool(np.array_equal(reference_labels, measured_labels)),
        "predictions_equivalent": bool(np.array_equal(reference_predictions, measured_predictions)),
        "stage3_accuracy": float(stage3_accuracy), "stage35_accuracy": float(stage35_accuracy),
        "accuracy_delta": float(stage35_accuracy - stage3_accuracy),
        "stage3_fixed98_macro_f1": float(stage3_fixed98_macro_f1),
        "stage35_fixed98_macro_f1": float(stage35_fixed98_macro_f1),
        "fixed98_macro_f1_delta": float(stage35_fixed98_macro_f1 - stage3_fixed98_macro_f1),
        "metric_tolerance": tolerance,
    }
    result["status"] = "PASS" if (
        result["global_indices_equivalent"] and result["labels_equivalent"] and result["predictions_equivalent"]
        and abs(result["accuracy_delta"]) <= tolerance and abs(result["fixed98_macro_f1_delta"]) <= tolerance
    ) else "FAIL"
    return result


def final_hash_manifest_current(output_root: Path) -> bool:
    path = output_root / "manifests" / "STAGE3_5M_HASH_MANIFEST.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = {row["relative_path"]: row for row in payload["files"]}
        excluded = set(FINAL_HASH_EXCLUSIONS)
        actual = {
            item.relative_to(output_root).as_posix(): item
            for item in output_root.rglob("*") if item.is_file() and item.relative_to(output_root).as_posix() not in excluded
        }
        return set(declared) == set(actual) and all(
            int(declared[name]["bytes"]) == item.stat().st_size and declared[name]["sha256"] == sha256_file(item)
            for name, item in actual.items()
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


class Stage35Pipeline:
    def __init__(self, config: Stage35Config):
        self.config = config
        config.validate()
        for name in REQUIRED_OUTPUT_DIRS:
            assert_within(config.output_root / name, config.output_root).mkdir(parents=True, exist_ok=True)
        self.script_sha = sha256_file(Path(__file__).resolve())
        self.guard = StrictZeroDayGuard()
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
        self.stage3m: Any = None
        self.stage26: Any = None
        self.benchmark: Any = None
        self.local_h5: Optional[Path] = None
        self.teacher: Optional[torch.nn.Module] = None
        self.teacher_payload: Dict[str, Any] = {}
        self.teacher_state_sha = ""
        self.logger = self._logger()

    def _logger(self) -> logging.Logger:
        logger = logging.getLogger(f"stage3-5m-{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter); logger.addHandler(stream)
        file_handler = logging.FileHandler(self.config.output_root / "logs" / "stage3_5m.log", encoding="utf-8"); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
        return logger

    def stage_manifest_path(self, stage: int) -> Path:
        return self.config.output_root / "manifests" / f"STAGE_{stage:02d}_CHECKPOINT.json"

    def _input_hashes(self, paths: Iterable[Path]) -> Dict[str, str]:
        return {str(path.resolve()): sha256_file(path) for path in paths}

    def stage_current(self, stage: int) -> bool:
        checkpoint = self.stage_manifest_path(stage)
        if not self.config.resume or not checkpoint.is_file():
            return False
        if stage >= 8 and not self.lock_current():
            return False
        if stage == 11:
            if not all((self.config.output_root / relative).is_file() for relative in FINAL_REQUIRED) or not final_hash_manifest_current(self.config.output_root):
                return False
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("pipeline_version") != PIPELINE_VERSION or payload.get("executable_sha256") != self.script_sha or payload.get("configuration_sha256") != self.config.configuration_sha256():
                return False
            for source, expected in payload.get("required_input_hashes", {}).items():
                path = Path(source)
                if not path.is_file() or sha256_file(path) != expected:
                    return False
            for row in payload.get("required_outputs", []):
                path = Path(row["path"])
                if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def complete_stage(self, stage: int, name: str, outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
        payload = {
            "stage": stage, "name": name, "status": "PASS", "pipeline_version": PIPELINE_VERSION,
            "executable_sha256": self.script_sha, "configuration_sha256": self.config.configuration_sha256(),
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_HASH_MANIFEST_SHA256, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "required_input_hashes": self._input_hashes(inputs),
            "required_outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
            "completed_at": utc_now(),
        }
        atomic_json(self.stage_manifest_path(stage), payload, self.config.output_root)
        print(f"[PASS] Stage {stage:02d} — {name}")

    def ensure_predecessor(self) -> None:
        if self.teacher_payload:
            return
        benchmark = self.config.benchmark_path_resolved
        stage3_ready_path = self.config.stage3m_path / "MANYTX_STAGE3M_READY.txt"
        stage3_hash_path = self.config.stage3m_path / "manifests" / "STAGE3M_HASH_MANIFEST.json"
        stage3_status_path = self.config.stage3m_path / "manifests" / "STAGE3M_FINAL_STATUS.json"
        freeze_path = self.config.stage3m_path / "manifests" / "TEACHER_FREEZE.json"
        teacher_path = self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"
        state_path = self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt"
        stage26_hash_path = self.config.stage26_path / "manifests" / "HASH_MANIFEST.json"
        required = (benchmark, stage3_ready_path, stage3_hash_path, stage3_status_path, freeze_path, teacher_path, state_path, stage26_hash_path)
        if not all(path.is_file() for path in required):
            raise ScientificAbort("Frozen Stage 3M predecessor artifacts are incomplete")
        if sha256_file(benchmark) != EXPECTED_BENCHMARK_SHA256:
            raise ScientificAbort("Canonical benchmark SHA mismatch")
        if sha256_file(stage3_hash_path) != EXPECTED_STAGE3M_HASH_MANIFEST_SHA256:
            raise ScientificAbort("Stage 3M final hash-manifest SHA mismatch")
        if sha256_file(stage26_hash_path) != EXPECTED_STAGE26_ARTIFACT_SHA256:
            raise ScientificAbort("Stage 2.6M artifact SHA mismatch")
        stage3_hash_manifest = json.loads(stage3_hash_path.read_text(encoding="utf-8"))
        rows = stage3_hash_manifest.get("files", [])
        if int(stage3_hash_manifest.get("count", -1)) != len(rows) or not rows:
            raise ScientificAbort("Stage 3M hash-manifest schema/count mismatch")
        for row in rows:
            artifact = self.config.stage3m_path / str(row["relative_path"])
            if not artifact.is_file() or artifact.stat().st_size != int(row["bytes"]) or sha256_file(artifact) != row["sha256"]:
                raise ScientificAbort(f"Stage 3M hash-manifest artifact mismatch: {row.get('relative_path')}")
        if sha256_file(teacher_path) != EXPECTED_TEACHER_SHA256 or sha256_file(state_path) != EXPECTED_TEACHER_STATE_SHA256:
            raise ScientificAbort("Frozen canonical teacher file SHA mismatch")
        ready = parse_ready_marker(stage3_ready_path)
        expected = {
            "marker": "MANYTX_STAGE3M_READY", "teacher_version": EXPECTED_TEACHER_VERSION,
            "objective": "CE_SUPCON_PROTOTYPE", "source_arm": "A3", "selected_seed": str(EXPECTED_TEACHER_SEED),
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "canonical_teacher_sha256": EXPECTED_TEACHER_SHA256, "final_zero_day_evaluation_performed": "NO",
            "surrogate_training_performed": "NO", "xai_performed": "NO", "next_stage": "STAGE_3_5M",
        }
        mismatches = {key: {"expected": value, "actual": ready.get(key)} for key, value in expected.items() if ready.get(key) != value}
        for key in STRICT_COUNTER_KEYS[:5]:
            if ready.get(key) != "0":
                mismatches[key] = {"expected": "0", "actual": ready.get(key)}
        if mismatches:
            raise ScientificAbort(f"Stage 3M READY contract mismatch: {mismatches}")
        status = json.loads(stage3_status_path.read_text(encoding="utf-8"))
        if status.get("status") != "MANYTX_STAGE3M_READY" or int(status.get("selected_seed", -1)) != EXPECTED_TEACHER_SEED:
            raise ScientificAbort("Stage 3M final status mismatch")
        if not status.get("gates") or not all(status["gates"].values()):
            raise ScientificAbort("Stage 3M final status contains a failed gate")
        if any(status.get("strict_zero_day_counters", {}).values()):
            raise ScientificAbort("Stage 3M strict-zero-day counters are non-zero")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("selected_seed") != EXPECTED_TEACHER_SEED or freeze.get("training_performed") is not False or freeze.get("source_export_state_dict_element_equivalent") is not True:
            raise ScientificAbort("Teacher freeze provenance mismatch")
        try:
            self.teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            self.teacher_payload = torch.load(teacher_path, map_location="cpu")
            state = torch.load(state_path, map_location="cpu")
        if not isinstance(self.teacher_payload, dict) or "model_state" not in self.teacher_payload:
            raise ScientificAbort("Canonical teacher payload is malformed")
        teacher_fields = {
            "teacher_version": EXPECTED_TEACHER_VERSION, "selected_seed": EXPECTED_TEACHER_SEED,
            "source_arm": "A3", "objective": "CE + SupCon + Prototype",
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT, "embedding_dimension": EXPECTED_EMBEDDING_DIM,
            "known_classes": EXPECTED_CLASSES, "training_performed": False,
        }
        failed_teacher_fields = [key for key, expected in teacher_fields.items() if self.teacher_payload.get(key) != expected]
        if failed_teacher_fields:
            raise ScientificAbort(f"Canonical teacher metadata mismatch: {failed_teacher_fields}")
        if list(state) != list(self.teacher_payload["model_state"]) or not all(torch.equal(state[key], self.teacher_payload["model_state"][key]) for key in state):
            raise ScientificAbort("Canonical teacher and state-dict exports differ")
        self.teacher_state_sha = state_tensor_sha256(state)

    def ensure_teacher(self) -> torch.nn.Module:
        self.ensure_predecessor()
        if self.teacher is not None:
            return self.teacher
        self.stage3m = load_module(self.config.repository_root_path / "Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.py", "stage3m_frozen_for_stage35")
        model = self.stage3m.WiSigRepresentationNet(EXPECTED_CLASSES, EXPECTED_EMBEDDING_DIM, 0.1)
        model.load_state_dict(self.teacher_payload["model_state"], strict=True)
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
            raise ScientificAbort("Canonical teacher parameter count mismatch")
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.teacher = model
        return model

    def ensure_benchmark(self) -> Any:
        self.ensure_predecessor()
        if self.benchmark is not None:
            return self.benchmark
        self.stage26 = load_module(self.config.repository_root_path / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py", "stage26_frozen_for_stage35")
        cache = self.stage26.LocalCacheManager(Path(self.config.local_cache_root), self.config.output_root / "performance")
        self.local_h5, report = cache.ensure(self.config.benchmark_path_resolved, EXPECTED_BENCHMARK_SHA256)
        atomic_json(self.config.output_root / "performance" / "LOCAL_CACHE_REPORT.json", report, self.config.output_root)
        self.benchmark = self.stage26.resolve_benchmark(self.config, self.guard, self.local_h5)
        return self.benchmark

    def loader(self, partition: str) -> Tuple[Any, DataLoader]:
        benchmark = self.ensure_benchmark()
        if partition not in NON_STRICT_PARTITIONS:
            self.guard.reject("signal", f"generic loader forbids {partition}")
        metadata = benchmark.partitions[partition]
        self.guard.authorize_rows(partition, metadata.indices, "Stage 3.5M non-strict loader")
        dataset = self.stage26.WiSigH5Dataset(benchmark, partition, self.guard)
        kwargs: Dict[str, Any] = {
            "dataset": dataset, "batch_size": self.config.eval_batch_size, "shuffle": False,
            "num_workers": self.config.num_workers, "pin_memory": self.config.pin_memory and self.device.type == "cuda",
            "collate_fn": self.stage26.identity_collate,
        }
        if self.config.num_workers:
            kwargs.update({"prefetch_factor": self.config.prefetch_factor, "persistent_workers": self.config.persistent_workers})
        return dataset, DataLoader(**kwargs)

    def fit_path(self) -> Path:
        return self.config.output_root / "scores" / "fitted" / "known_only_scorer_state.npz"

    def assert_prelock_mutation_allowed(self) -> None:
        lock, sidecar = self.lock_paths()
        if lock.exists() or sidecar.exists():
            self.guard.final_lock_active = True
            self.guard.reject("fit", "pre-strict scorer/score/threshold mutation attempted after final evaluation lock")

    def load_fit(self) -> Dict[str, np.ndarray]:
        path = self.fit_path()
        if not path.is_file():
            raise ScientificAbort("Known-only scorer fit is missing")
        with np.load(path, allow_pickle=False) as values:
            fit = {key: np.asarray(values[key]) for key in values.files}
        required = {"counts", "means", "prototypes", "precision", "diagonal_variance"}
        if set(fit) != required:
            raise ScientificAbort("Known-only scorer state schema mismatch")
        return fit

    def score_store(self, partition: str) -> Path:
        return self.config.output_root / "scores" / partition

    def score_store_current(
        self,
        partition: str,
        rows: int,
        fit_sha: str,
        expected_indices: np.ndarray,
        strict: bool = False,
        expected_lock_sha: Optional[str] = None,
    ) -> bool:
        store = self.score_store(partition)
        manifest = store / "store_manifest.json"
        required = ("scores.npy", "predictions.npy", "global_indices.npy") if strict else ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy")
        if (store / "INCOMPLETE").exists() or not manifest.is_file() or not all((store / name).is_file() for name in required):
            return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            expected_fields = {
                "pipeline_version": PIPELINE_VERSION,
                "executable_sha256": self.script_sha,
                "configuration_sha256": self.config.configuration_sha256(),
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
                "teacher_sha256": EXPECTED_TEACHER_SHA256,
                "fit_sha256": fit_sha,
                "scorer_order": list(SCORER_ORDER),
                "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
                "energy_temperature": self.config.energy_temperature,
                "partition": partition,
                "rows": rows,
                "strict": strict,
                "global_indices_sha256": sha256_int64_array(expected_indices),
            }
            if payload.get("complete") is not True or any(payload.get(key) != value for key, value in expected_fields.items()):
                return False
            if strict and (not expected_lock_sha or payload.get("evaluation_lock_sha256") != expected_lock_sha):
                return False
            shapes = {"scores.npy": (rows, len(SCORER_ORDER)), "predictions.npy": (rows,), "global_indices.npy": (rows,)}
            if not strict:
                shapes.update({"labels.npy": (rows,), "known_correct.npy": (rows,)})
            if not all(tuple(np.load(store / name, mmap_mode="r").shape) == shape for name, shape in shapes.items()):
                return False
            stored_indices = np.asarray(np.load(store / "global_indices.npy", mmap_mode="r"), dtype=np.int64)
            expected_indices_array = np.asarray(expected_indices, dtype=np.int64).reshape(-1)
            if not np.array_equal(stored_indices, expected_indices_array):
                return False
            declared = payload.get("files", {})
            return set(declared) == set(required) and all(
                int(declared[name]["bytes"]) == (store / name).stat().st_size
                and declared[name]["sha256"] == sha256_file(store / name)
                for name in required
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def extract_non_strict_scores(self, partition: str) -> Path:
        benchmark, model, fit = self.ensure_benchmark(), self.ensure_teacher(), self.load_fit()
        rows, fit_sha = len(benchmark.partitions[partition].indices), sha256_file(self.fit_path())
        store = self.score_store(partition)
        expected_indices = np.asarray(benchmark.partitions[partition].indices, dtype=np.int64)
        if self.score_store_current(partition, rows, fit_sha, expected_indices, strict=False):
            return store
        store.mkdir(parents=True, exist_ok=True)
        atomic_text(store / "INCOMPLETE", utc_now() + "\n", self.config.output_root)
        partial_scores = store / "scores.partial.npy"; partial_predictions = store / "predictions.partial.npy"
        scores = np.lib.format.open_memmap(partial_scores, mode="w+", dtype=np.float32, shape=(rows, len(SCORER_ORDER)))
        predictions = np.lib.format.open_memmap(partial_predictions, mode="w+", dtype=np.int16, shape=(rows,))
        dataset, loader = self.loader(partition)
        cursor = 0
        use_amp = self.config.amp_enabled and self.device.type == "cuda"
        with torch.inference_mode():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(x)
                logits = outputs["logits"].float().cpu().numpy()
                embeddings = outputs["embedding_normalized"].float().cpu().numpy()
                count = len(x)
                scores[cursor:cursor + count] = score_outputs(logits, embeddings, fit, self.config.energy_temperature)
                predictions[cursor:cursor + count] = logits.argmax(axis=1).astype(np.int16)
                cursor += count
        dataset.close(); scores.flush(); predictions.flush(); del scores, predictions
        if cursor != rows:
            raise ScientificAbort(f"Score extraction row mismatch for {partition}")
        os.replace(partial_scores, store / "scores.npy"); os.replace(partial_predictions, store / "predictions.npy")
        metadata = benchmark.partitions[partition]
        labels = metadata.labels.astype(np.int16)
        np.save(store / "labels.npy", labels, allow_pickle=False)
        np.save(store / "known_correct.npy", (np.load(store / "predictions.npy") == labels).astype(np.uint8), allow_pickle=False)
        np.save(store / "global_indices.npy", metadata.indices.astype(np.int64), allow_pickle=False)
        store_files = ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy")
        atomic_json(store / "store_manifest.json", {
            "complete": True, "pipeline_version": PIPELINE_VERSION, "executable_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(), "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "partition": partition, "rows": rows, "strict": False, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "fit_sha256": fit_sha, "scorer_order": list(SCORER_ORDER),
            "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS), "energy_temperature": self.config.energy_temperature,
            "global_indices_sha256": sha256_int64_array(metadata.indices),
            "labels_semantics": "KNOWN_CLASS_0_TO_97" if partition != "calibration_unknown" else "CALIBRATION_UNKNOWN_MINUS_ONE",
            "files": {name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in store_files},
            "generated_at": utc_now(),
        }, self.config.output_root)
        (store / "INCOMPLETE").unlink()
        return store

    def declared_strict_evidence(self) -> Dict[str, Any]:
        engineering = self.config.branch_root_path / "01_benchmark_engineering"
        manifests: List[Tuple[Path, Any]] = []
        for path in sorted(engineering.rglob("*.json")):
            try:
                manifests.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        evidence: Dict[str, Any] = {}
        stage26 = load_module(self.config.repository_root_path / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py", "stage26_strict_manifest_helpers")
        for partition, (filename, expected_count) in STRICT_PARTITIONS.items():
            matches = [path for path in engineering.rglob(filename) if path.is_file()]
            if len(matches) != 1:
                raise ScientificAbort(f"Expected exactly one sealed {filename}; found {matches}")
            hashes: set[str] = set(); counts: set[int] = set(); sources: set[str] = set()
            for manifest_path, payload in manifests:
                declared_hashes, declared_counts, _ = stage26.extract_bound_manifest_declarations(payload, filename)
                if declared_hashes or declared_counts:
                    hashes.update(str(value) for value in declared_hashes)
                    counts.update(int(value) for value in declared_counts)
                    sources.add(str(manifest_path))
            if expected_count not in counts or not hashes:
                raise ScientificAbort(f"Sealed manifest evidence incomplete for {filename}")
            evidence[partition] = {
                "path": str(matches[0].resolve()), "filename": filename, "expected_count": expected_count,
                "declared_sha256_candidates": sorted(hashes), "manifest_sources": sorted(sources),
                "file_content_loaded": False,
            }
        return evidence

    @staticmethod
    def _file_evidence(path: Path) -> Dict[str, Any]:
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}

    def known_bundle_path(self) -> Path:
        return self.config.output_root / "manifests" / "PRE_STRICT_KNOWN_SCORE_BUNDLE.json"

    def inference_equivalence_path(self) -> Path:
        return self.config.output_root / "manifests" / "STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json"

    def write_inference_equivalence(self) -> Tuple[Path, List[Path]]:
        stage3_table = self.config.stage3m_path / "tables" / "teacher_candidate_known_metrics.csv"
        if not stage3_table.is_file():
            raise ScientificAbort("Frozen Stage 3M known-metric table is missing")
        reference_table = pd.read_csv(stage3_table)
        stage35_table = pd.read_csv(self.config.output_root / "tables" / "closed_set_teacher_metrics.csv").set_index("partition")
        tolerance = 1e-10
        rows: List[Dict[str, Any]] = []
        inputs: List[Path] = [stage3_table, self.config.output_root / "tables" / "closed_set_teacher_metrics.csv"]
        overall_pass = True
        for protocol in KNOWN_VALIDATION:
            stage35_store = self.score_store(protocol)
            stage35_indices = np.asarray(np.load(stage35_store / "global_indices.npy", mmap_mode="r"), dtype=np.int64)
            stage35_labels = np.asarray(np.load(stage35_store / "labels.npy", mmap_mode="r"), dtype=np.int64)
            stage35_predictions = np.asarray(np.load(stage35_store / "predictions.npy", mmap_mode="r"), dtype=np.int64)
            stage3_store = self.config.stage3m_path / "embeddings" / f"seed_{EXPECTED_TEACHER_SEED}" / protocol
            primitives = {
                "global_indices": stage3_store / "global_indices.npy",
                "labels": stage3_store / "labels.npy",
                "logits": stage3_store / "logits.npy",
            }
            availability = {name: path.is_file() for name, path in primitives.items()}
            scalar_reference = reference_table[(reference_table.seed == EXPECTED_TEACHER_SEED) & (reference_table.protocol == protocol)]
            if len(scalar_reference) != 1:
                raise ScientificAbort(f"Frozen Stage 3M scalar evidence is not unique: {protocol}")
            scalar = scalar_reference.iloc[0]
            stage35_scalar = stage35_table.loc[protocol]
            scalar_values = {
                "stage3_accuracy": float(scalar.accuracy), "stage35_accuracy": float(stage35_scalar.accuracy),
                "stage3_fixed98_macro_f1": float(scalar.fixed98_macro_f1),
                "stage35_fixed98_macro_f1": float(stage35_scalar.fixed98_macro_f1),
            }
            metrics_equal = abs(scalar_values["stage35_accuracy"] - scalar_values["stage3_accuracy"]) <= tolerance and abs(
                scalar_values["stage35_fixed98_macro_f1"] - scalar_values["stage3_fixed98_macro_f1"]
            ) <= tolerance
            if all(availability.values()):
                inputs.extend(primitives.values())
                stage3_indices = np.asarray(np.load(primitives["global_indices"], mmap_mode="r"), dtype=np.int64)
                stage3_labels = np.asarray(np.load(primitives["labels"], mmap_mode="r"), dtype=np.int64)
                stage3_logits = np.asarray(np.load(primitives["logits"], mmap_mode="r"), dtype=np.float32)
                stage3_predictions = stage3_logits.argmax(axis=1).astype(np.int64)
                result = compare_inference_evidence(
                    protocol, stage3_indices, stage3_labels, stage3_predictions,
                    stage35_indices, stage35_labels, stage35_predictions,
                    scalar_values["stage3_accuracy"], scalar_values["stage3_fixed98_macro_f1"],
                    scalar_values["stage35_accuracy"], scalar_values["stage35_fixed98_macro_f1"], tolerance,
                )
                primitive_accuracy = float(np.mean(stage3_predictions == stage3_labels))
                primitive_f1 = float(f1_score(stage3_labels, stage3_predictions, labels=np.arange(EXPECTED_CLASSES), average="macro", zero_division=0))
                result.update({
                    "primitive_availability": availability,
                    "stage3_primitive_accuracy": primitive_accuracy,
                    "stage3_primitive_fixed98_macro_f1": primitive_f1,
                })
                if abs(primitive_accuracy - scalar_values["stage3_accuracy"]) > tolerance or abs(primitive_f1 - scalar_values["stage3_fixed98_macro_f1"]) > tolerance:
                    result["status"] = "FAIL"
            else:
                result = {
                    "protocol": protocol, "rows": int(len(stage35_indices)), "primitive_availability": availability,
                    **scalar_values,
                    "accuracy_delta": scalar_values["stage35_accuracy"] - scalar_values["stage3_accuracy"],
                    "fixed98_macro_f1_delta": scalar_values["stage35_fixed98_macro_f1"] - scalar_values["stage3_fixed98_macro_f1"],
                    "metric_tolerance": tolerance,
                }
                for name, available in availability.items():
                    result[f"{name}_equivalent"] = "NOT_AVAILABLE_IN_FROZEN_STAGE3M" if not available else "AVAILABLE_BUT_INCOMPLETE_PRIMITIVE_SET"
                result["status"] = "PASS_LIMITED_TO_FROZEN_SCALAR_EVIDENCE" if metrics_equal else "FAIL"
            overall_pass = overall_pass and result["status"].startswith("PASS")
            rows.append(result)
        output = self.inference_equivalence_path()
        atomic_json(output, {
            "status": "PASS" if overall_pass else "FAIL", "teacher_seed": EXPECTED_TEACHER_SEED,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "known_only": True, "strict_data_accessed": False,
            "protocols": rows, "generated_at": utc_now(),
        }, self.config.output_root)
        if not overall_pass:
            raise ScientificAbort(f"Stage 3M to Stage 3.5M inference equivalence failed: {rows}")
        return output, inputs

    def inference_equivalence_current(self, expected_sha: Optional[str] = None) -> bool:
        path = self.inference_equivalence_path()
        if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha):
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("status") == "PASS" and payload.get("teacher_seed") == EXPECTED_TEACHER_SEED and payload.get("teacher_sha256") == EXPECTED_TEACHER_SHA256 and payload.get("strict_data_accessed") is False
        except (OSError, json.JSONDecodeError):
            return False

    def write_known_score_bundle(self) -> Path:
        partitions: Dict[str, Any] = {}
        fit_sha = sha256_file(self.fit_path())
        for partition in KNOWN_VALIDATION:
            store = self.score_store(partition)
            indices = np.asarray(np.load(store / "global_indices.npy", mmap_mode="r"), dtype=np.int64)
            if not self.score_store_current(partition, len(indices), fit_sha, indices, strict=False):
                raise ScientificAbort(f"Cannot freeze stale known score store: {partition}")
            names = ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy", "store_manifest.json")
            partitions[partition] = {
                "partition": partition, "rows": len(indices),
                "store_manifest_sha256": sha256_file(store / "store_manifest.json"),
                "scores_sha256": sha256_file(store / "scores.npy"),
                "predictions_sha256": sha256_file(store / "predictions.npy"),
                "labels_sha256": sha256_file(store / "labels.npy"),
                "known_correct_sha256": sha256_file(store / "known_correct.npy"),
                "global_indices_sha256": sha256_file(store / "global_indices.npy"),
                "global_index_values_sha256": sha256_int64_array(indices),
                "teacher_sha256": EXPECTED_TEACHER_SHA256, "fit_sha256": fit_sha,
                "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
                "files": {name: self._file_evidence(store / name) for name in names},
            }
        threshold_path = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
        policy_path = self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json"
        payload: Dict[str, Any] = {
            "status": "FROZEN_BEFORE_STRICT_EVALUATION", "pipeline_version": PIPELINE_VERSION,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "scorer_fit_sha256": fit_sha, "zd_strict_thresholds_sha256": sha256_file(threshold_path),
            "scorer_policy_freeze_sha256": sha256_file(policy_path),
            "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
            "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
            "partitions": partitions, "strict_data_accessed": False, "generated_at": utc_now(),
        }
        canonical = dict(payload); canonical.pop("generated_at")
        payload["canonical_content_sha256"] = sha256_object(canonical)
        output = self.known_bundle_path(); atomic_json(output, payload, self.config.output_root)
        if not self.known_score_bundle_current(sha256_file(output)):
            raise ScientificAbort("Pre-strict known-score bundle failed immediate verification")
        return output

    def known_score_bundle_current(self, expected_sha: Optional[str] = None) -> bool:
        path = self.known_bundle_path()
        if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha):
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "status": "FROZEN_BEFORE_STRICT_EVALUATION", "pipeline_version": PIPELINE_VERSION,
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "teacher_sha256": EXPECTED_TEACHER_SHA256,
                "scorer_fit_sha256": sha256_file(self.fit_path()),
                "zd_strict_thresholds_sha256": sha256_file(self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"),
                "scorer_policy_freeze_sha256": sha256_file(self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json"),
                "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
                "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
                "strict_data_accessed": False,
            }
            if any(payload.get(key) != value for key, value in expected.items()) or set(payload.get("partitions", {})) != set(KNOWN_VALIDATION):
                return False
            canonical = dict(payload); canonical.pop("canonical_content_sha256", None); canonical.pop("generated_at", None)
            if payload.get("canonical_content_sha256") != sha256_object(canonical):
                return False
            for partition, record in payload["partitions"].items():
                if record.get("partition") != partition or record.get("teacher_sha256") != EXPECTED_TEACHER_SHA256 or record.get("fit_sha256") != expected["scorer_fit_sha256"] or record.get("configuration_sha256") != self.config.configuration_sha256() or record.get("executable_sha256") != self.script_sha:
                    return False
                files = record.get("files", {})
                required = {"scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy", "store_manifest.json"}
                if set(files) != required:
                    return False
                for name, evidence in files.items():
                    artifact = Path(evidence["path"])
                    if not artifact.is_file() or artifact.stat().st_size != int(evidence["bytes"]) or sha256_file(artifact) != evidence["sha256"]:
                        return False
                if any(record.get(key) != files[name]["sha256"] for key, name in {
                    "store_manifest_sha256": "store_manifest.json", "scores_sha256": "scores.npy",
                    "predictions_sha256": "predictions.npy", "labels_sha256": "labels.npy",
                    "known_correct_sha256": "known_correct.npy", "global_indices_sha256": "global_indices.npy",
                }.items()):
                    return False
                indices = np.asarray(np.load(Path(files["global_indices.npy"]["path"]), mmap_mode="r"), dtype=np.int64)
                if len(indices) != int(record.get("rows", -1)) or record.get("global_index_values_sha256") != sha256_int64_array(indices):
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def strict_bundle_path(self) -> Path:
        return self.config.output_root / "manifests" / "FINAL_STRICT_SCORE_BUNDLE.json"

    def write_strict_score_bundle(self, strict_indices: Mapping[str, np.ndarray], sealed_paths: Mapping[str, Path]) -> Path:
        lock_sha = sha256_file(self.lock_paths()[0]); fit_sha = sha256_file(self.fit_path())
        partitions: Dict[str, Any] = {}
        for partition in STRICT_PARTITIONS:
            indices = np.asarray(strict_indices[partition], dtype=np.int64)
            store = self.score_store(partition)
            if not self.score_store_current(partition, len(indices), fit_sha, indices, strict=True, expected_lock_sha=lock_sha):
                raise ScientificAbort(f"Cannot freeze stale strict score store: {partition}")
            names = ("scores.npy", "predictions.npy", "global_indices.npy", "store_manifest.json")
            partitions[partition] = {
                "partition": partition, "rows": len(indices), "sealed_index_path": str(sealed_paths[partition].resolve()),
                "sealed_index_sha256": sha256_file(sealed_paths[partition]),
                "global_indices_sha256": sha256_file(store / "global_indices.npy"),
                "global_index_values_sha256": sha256_int64_array(indices),
                "scores_sha256": sha256_file(store / "scores.npy"),
                "predictions_sha256": sha256_file(store / "predictions.npy"),
                "store_manifest_sha256": sha256_file(store / "store_manifest.json"),
                "teacher_sha256": EXPECTED_TEACHER_SHA256, "fit_sha256": fit_sha,
                "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
                "evaluation_lock_sha256": lock_sha,
                "files": {name: self._file_evidence(store / name) for name in names},
            }
        payload: Dict[str, Any] = {
            "status": "FROZEN_AFTER_STAGE08_EXTRACTION", "pipeline_version": PIPELINE_VERSION,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "scorer_fit_sha256": fit_sha, "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
            "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
            "evaluation_lock_sha256": lock_sha, "strict_labels_included": False,
            "partition_relationship": validate_strict_subset_relationship(
                strict_indices["strict_zero_day_test"], strict_indices["strict_zero_day_shift_test"],
            ),
            "partitions": partitions, "generated_at": utc_now(),
        }
        canonical = dict(payload); canonical.pop("generated_at")
        payload["canonical_content_sha256"] = sha256_object(canonical)
        output = self.strict_bundle_path(); atomic_json(output, payload, self.config.output_root)
        if not self.strict_score_bundle_current(sha256_file(output)):
            raise ScientificAbort("Final strict-score bundle failed immediate verification")
        return output

    def strict_score_bundle_current(self, expected_sha: Optional[str] = None) -> bool:
        path = self.strict_bundle_path()
        if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha) or not self.lock_current():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8")); lock_sha = sha256_file(self.lock_paths()[0]); fit_sha = sha256_file(self.fit_path())
            expected = {
                "status": "FROZEN_AFTER_STAGE08_EXTRACTION", "pipeline_version": PIPELINE_VERSION,
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "teacher_sha256": EXPECTED_TEACHER_SHA256,
                "scorer_fit_sha256": fit_sha, "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
                "configuration_sha256": self.config.configuration_sha256(), "executable_sha256": self.script_sha,
                "evaluation_lock_sha256": lock_sha, "strict_labels_included": False,
            }
            if any(payload.get(key) != value for key, value in expected.items()) or set(payload.get("partitions", {})) != set(STRICT_PARTITIONS):
                return False
            canonical = dict(payload); canonical.pop("canonical_content_sha256", None); canonical.pop("generated_at", None)
            if payload.get("canonical_content_sha256") != sha256_object(canonical):
                return False
            verified_indices: Dict[str, np.ndarray] = {}
            for partition, record in payload["partitions"].items():
                sealed = Path(record["sealed_index_path"])
                if not sealed.is_file() or sha256_file(sealed) != record.get("sealed_index_sha256"):
                    return False
                files = record.get("files", {}); required = {"scores.npy", "predictions.npy", "global_indices.npy", "store_manifest.json"}
                if set(files) != required:
                    return False
                for evidence in files.values():
                    artifact = Path(evidence["path"])
                    if not artifact.is_file() or artifact.stat().st_size != int(evidence["bytes"]) or sha256_file(artifact) != evidence["sha256"]:
                        return False
                indices = np.asarray(np.load(Path(files["global_indices.npy"]["path"]), mmap_mode="r"), dtype=np.int64)
                verified_indices[partition] = indices
                if len(indices) != int(record.get("rows", -1)) or record.get("global_index_values_sha256") != sha256_int64_array(indices):
                    return False
                if any(record.get(key) != files[name]["sha256"] for key, name in {
                    "global_indices_sha256": "global_indices.npy", "scores_sha256": "scores.npy",
                    "predictions_sha256": "predictions.npy", "store_manifest_sha256": "store_manifest.json",
                }.items()):
                    return False
                if record.get("teacher_sha256") != EXPECTED_TEACHER_SHA256 or record.get("fit_sha256") != fit_sha or record.get("configuration_sha256") != self.config.configuration_sha256() or record.get("executable_sha256") != self.script_sha or record.get("evaluation_lock_sha256") != lock_sha:
                    return False
            relationship = validate_strict_subset_relationship(
                verified_indices["strict_zero_day_test"], verified_indices["strict_zero_day_shift_test"],
            )
            if payload.get("partition_relationship") != relationship:
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def lock_paths(self) -> Tuple[Path, Path]:
        return (
            self.config.output_root / "manifests" / "STRICT_ZERO_DAY_EVALUATION_LOCK.json",
            self.config.output_root / "manifests" / "STRICT_ZERO_DAY_EVALUATION_LOCK.sha256",
        )

    def lock_current(self) -> bool:
        lock, sidecar = self.lock_paths()
        if not lock.is_file() or not sidecar.is_file():
            return False
        digest = sidecar.read_text(encoding="utf-8").strip()
        if digest != sha256_file(lock):
            return False
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            required = {
                "teacher_sha256": EXPECTED_TEACHER_SHA256,
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
                "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
                "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_HASH_MANIFEST_SHA256,
                "executable_sha256": self.script_sha,
                "configuration_sha256": self.config.configuration_sha256(),
                "canonical_scorer": CANONICAL_SCORER_POLICY,
                "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY,
                "scorer_definitions": SCORER_DEFINITIONS,
            }
            if any(payload.get(key) != value for key, value in required.items()):
                return False
            counters = payload.get("strict_violation_counters_at_lock")
            if not isinstance(counters, dict) or set(counters) != set(STRICT_COUNTER_KEYS) or any(counters[key] != 0 for key in STRICT_COUNTER_KEYS):
                return False
            if payload.get("post_lock_fitting_permitted") is not False or payload.get("post_lock_calibration_permitted") is not False:
                return False
            fit = self.fit_path()
            thresholds = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
            policy = self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json"
            known_bundle = self.known_bundle_path()
            equivalence = self.inference_equivalence_path()
            declaration = self.config.output_root / "manifests" / "STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json"
            bound = {
                "scorer_fit_sha256": fit,
                "threshold_manifest_sha256": thresholds,
                "policy_freeze_sha256": policy,
                "known_score_bundle_sha256": known_bundle,
                "stage3m_stage35_inference_equivalence_sha256": equivalence,
                "sealed_strict_declaration_sha256": declaration,
            }
            if any(not path.is_file() or payload.get(field) != sha256_file(path) for field, path in bound.items()):
                return False
            if not self.known_score_bundle_current(payload["known_score_bundle_sha256"]):
                return False
            if not self.inference_equivalence_current(payload["stage3m_stage35_inference_equivalence_sha256"]):
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def write_or_verify_lock(self) -> str:
        lock, sidecar = self.lock_paths()
        if lock.exists() or sidecar.exists():
            if not self.lock_current():
                raise ScientificAbort("Existing strict evaluation lock is incomplete or stale")
            return sha256_file(lock)
        self.guard.assert_zero()
        policy = self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json"
        thresholds = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
        fit = self.fit_path()
        known_bundle = self.known_bundle_path()
        equivalence = self.inference_equivalence_path()
        declaration = self.config.output_root / "manifests" / "STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json"
        if not self.known_score_bundle_current(sha256_file(known_bundle)):
            raise ScientificAbort("Known-score bundle is not current before strict lock")
        if not self.inference_equivalence_current(sha256_file(equivalence)):
            raise ScientificAbort("Stage 3M to Stage 3.5M inference equivalence is not current before strict lock")
        payload = {
            "lock_version": "1.0", "created_at": utc_now(), "stage": 8,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_HASH_MANIFEST_SHA256,
            "executable_sha256": self.script_sha, "configuration_sha256": self.config.configuration_sha256(),
            "scorer_definitions": SCORER_DEFINITIONS, "canonical_scorer": CANONICAL_SCORER_POLICY,
            "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY,
            "scorer_fit_sha256": sha256_file(fit), "threshold_manifest_sha256": sha256_file(thresholds),
            "policy_freeze_sha256": sha256_file(policy), "strict_violation_counters_at_lock": self.guard.counters(),
            "known_score_bundle_sha256": sha256_file(known_bundle),
            "stage3m_stage35_inference_equivalence_sha256": sha256_file(equivalence),
            "sealed_strict_declaration_sha256": sha256_file(declaration),
            "post_lock_fitting_permitted": False, "post_lock_calibration_permitted": False,
        }
        atomic_json(lock, payload, self.config.output_root)
        atomic_text(sidecar, sha256_file(lock) + "\n", self.config.output_root)
        if not self.lock_current():
            raise ScientificAbort("Strict evaluation lock failed immediate verification")
        return sha256_file(lock)

    def strict_batches(self, partition: str, indices: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self.guard.authorize_strict("signal", 8, self.lock_current())
        benchmark = self.ensure_benchmark()
        path = self.local_h5 or benchmark.h5_path
        with h5py.File(path, "r", swmr=True) as handle:
            signals = handle[benchmark.signal_key]
            for start in range(0, len(indices), self.config.eval_batch_size):
                global_indices = np.asarray(indices[start:start + self.config.eval_batch_size], dtype=np.int64)
                values = self.stage26.read_h5_rows(signals, global_indices)
                if benchmark.signal_orientation == "channels_last":
                    values = values.transpose(0, 2, 1)
                values = np.ascontiguousarray(values, dtype=np.float32)
                if values.shape[1:] != (2, 256) or not np.isfinite(values).all():
                    raise ScientificAbort(f"Invalid strict signal batch: {partition}")
                yield values, global_indices

    def extract_strict_scores(self, partition: str, indices: np.ndarray) -> Path:
        self.guard.authorize_strict("signal", 8, self.lock_current())
        model, fit, fit_sha = self.ensure_teacher(), self.load_fit(), sha256_file(self.fit_path())
        rows, store = len(indices), self.score_store(partition)
        expected_lock_sha = sha256_file(self.lock_paths()[0])
        if self.score_store_current(partition, rows, fit_sha, indices, strict=True, expected_lock_sha=expected_lock_sha):
            return store
        store.mkdir(parents=True, exist_ok=True)
        atomic_text(store / "INCOMPLETE", utc_now() + "\n", self.config.output_root)
        partial_scores = store / "scores.partial.npy"; partial_predictions = store / "predictions.partial.npy"
        scores = np.lib.format.open_memmap(partial_scores, mode="w+", dtype=np.float32, shape=(rows, len(SCORER_ORDER)))
        predictions = np.lib.format.open_memmap(partial_predictions, mode="w+", dtype=np.int16, shape=(rows,))
        cursor = 0; use_amp = self.config.amp_enabled and self.device.type == "cuda"
        with torch.inference_mode():
            for values, _ in self.strict_batches(partition, indices):
                x = torch.from_numpy(values).to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(x)
                logits = outputs["logits"].float().cpu().numpy(); embeddings = outputs["embedding_normalized"].float().cpu().numpy()
                count = len(values); scores[cursor:cursor + count] = score_outputs(logits, embeddings, fit, self.config.energy_temperature)
                predictions[cursor:cursor + count] = logits.argmax(axis=1).astype(np.int16); cursor += count
        scores.flush(); predictions.flush(); del scores, predictions
        if cursor != rows:
            raise ScientificAbort(f"Strict score extraction row mismatch: {partition}")
        os.replace(partial_scores, store / "scores.npy"); os.replace(partial_predictions, store / "predictions.npy")
        np.save(store / "global_indices.npy", indices.astype(np.int64), allow_pickle=False)
        store_files = ("scores.npy", "predictions.npy", "global_indices.npy")
        atomic_json(store / "store_manifest.json", {
            "complete": True, "pipeline_version": PIPELINE_VERSION, "executable_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(), "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "partition": partition, "rows": rows, "strict": True, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "fit_sha256": fit_sha, "scorer_order": list(SCORER_ORDER),
            "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS), "energy_temperature": self.config.energy_temperature,
            "global_indices_sha256": sha256_int64_array(indices),
            "semantic_target": "UNKNOWN_BY_FROZEN_STRICT_PARTITION_MEMBERSHIP", "strict_labels_loaded": False,
            "evaluation_lock_sha256": expected_lock_sha, "generated_at": utc_now(),
            "files": {name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in store_files},
        }, self.config.output_root)
        (store / "INCOMPLETE").unlink()
        return store

    def stage_01(self) -> None:
        self.ensure_predecessor()
        output = self.config.output_root / "manifests" / "STAGE01_FROZEN_PREDECESSOR.json"
        atomic_json(output, {
            "status": "PASS", "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_HASH_MANIFEST_SHA256,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "teacher_state_file_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "teacher_seed": EXPECTED_TEACHER_SEED, "teacher_version": EXPECTED_TEACHER_VERSION,
            "teacher_retrained": False, "strict_violation_counters": self.guard.counters(), "generated_at": utc_now(),
        }, self.config.output_root)
        inputs = [
            self.config.benchmark_path_resolved,
            self.config.stage3m_path / "MANYTX_STAGE3M_READY.txt",
            self.config.stage3m_path / "manifests" / "STAGE3M_HASH_MANIFEST.json",
            self.config.stage3m_path / "manifests" / "STAGE3M_FINAL_STATUS.json",
            self.config.stage3m_path / "manifests" / "TEACHER_FREEZE.json",
            self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt",
            self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt",
            self.config.stage26_path / "manifests" / "HASH_MANIFEST.json",
        ]
        self.complete_stage(1, "Frozen Stage 3M Predecessor Verification", [output], inputs)

    def stage_02(self) -> None:
        self.ensure_predecessor()
        model = self.ensure_teacher()
        state_before = state_tensor_sha256(model.state_dict())
        strict_evidence = self.declared_strict_evidence()
        if state_before != self.teacher_state_sha:
            raise ScientificAbort("Loaded teacher state differs from frozen export")
        output = self.config.output_root / "manifests" / "STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json"
        atomic_json(output, {
            "status": "PASS", "teacher_parameter_count": sum(p.numel() for p in model.parameters()),
            "teacher_state_element_sha256": state_before, "all_parameters_require_grad_false": all(not p.requires_grad for p in model.parameters()),
            "known_only_partitions": list(NON_STRICT_PARTITIONS), "sealed_strict_partitions": strict_evidence,
            "strict_index_file_contents_loaded": False, "signal_inference_performed": False,
            "unknown_training_class_exists": False, "strict_violation_counters": self.guard.counters(), "generated_at": utc_now(),
        }, self.config.output_root)
        self.complete_stage(2, "Teacher and Partition Exposure Audit", [output], [self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"])

    def stage_03(self) -> None:
        self.assert_prelock_mutation_allowed(); self.guard.assert_fitting_allowed()
        self.ensure_benchmark(); model = self.ensure_teacher()
        counts = np.zeros(EXPECTED_CLASSES, dtype=np.int64)
        sums = np.zeros((EXPECTED_CLASSES, EXPECTED_EMBEDDING_DIM), dtype=np.float64)
        diagonal_sums = np.zeros_like(sums)
        second = np.zeros((EXPECTED_EMBEDDING_DIM, EXPECTED_EMBEDDING_DIM), dtype=np.float64)
        teacher_before = state_tensor_sha256(model.state_dict())
        dataset, loader = self.loader("train_known")
        use_amp = self.config.amp_enabled and self.device.type == "cuda"
        rows = 0
        with torch.inference_mode():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(x)
                z = outputs["embedding_normalized"].float().cpu().numpy().astype(np.float64)
                labels = batch["y"].numpy().astype(np.int64)
                np.add.at(counts, labels, 1); np.add.at(sums, labels, z); np.add.at(diagonal_sums, labels, z * z)
                second += z.T @ z; rows += len(z)
        dataset.close()
        fit = fit_statistics_from_sufficient(counts, sums, second, diagonal_sums, self.config.covariance_regularization, self.config.diagonal_variance_floor)
        fit_path = self.fit_path(); atomic_npz(fit_path, self.config.output_root, **fit)
        if state_tensor_sha256(model.state_dict()) != teacher_before or teacher_before != self.teacher_state_sha:
            raise ScientificAbort("Teacher weights changed during scorer fitting")
        manifest = self.config.output_root / "manifests" / "KNOWN_ONLY_SCORER_FIT.json"
        atomic_json(manifest, {
            "status": "PASS", "source_partition": "TRAIN_KNOWN_ONLY", "rows": rows,
            "classes": EXPECTED_CLASSES, "class_counts": fit["counts"].tolist(), "scorer_fit_sha256": sha256_file(fit_path),
            "covariance_regularization": self.config.covariance_regularization,
            "diagonal_variance_floor": self.config.diagonal_variance_floor,
            "teacher_state_before_sha256": teacher_before, "teacher_state_after_sha256": teacher_before,
            "teacher_weights_immutable": True, "optimizer_created": False, "backward_called": False,
            "strict_data_used": False, "calibration_unknown_used": False, "unknown_training_class_exists": False,
            "generated_at": utc_now(),
        }, self.config.output_root)
        self.complete_stage(3, "Known-Only Scorer Fitting", [fit_path, manifest], [self.config.stage3m_path / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"])

    def stage_04(self) -> None:
        self.assert_prelock_mutation_allowed()
        rows: List[Dict[str, Any]] = []; closed_rows: List[Dict[str, Any]] = []; outputs: List[Path] = []
        teacher_before = state_tensor_sha256(self.ensure_teacher().state_dict())
        for partition in KNOWN_VALIDATION:
            store = self.extract_non_strict_scores(partition); outputs.extend(store / name for name in ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy", "store_manifest.json"))
            scores = np.load(store / "scores.npy", mmap_mode="r"); correct = np.load(store / "known_correct.npy", mmap_mode="r")
            labels = np.asarray(np.load(store / "labels.npy", mmap_mode="r"), dtype=np.int64)
            predictions = np.asarray(np.load(store / "predictions.npy", mmap_mode="r"), dtype=np.int64)
            observed_labels = np.unique(labels)
            closed_rows.append({
                "partition": partition, "rows": len(labels), "accuracy": float(np.mean(predictions == labels)),
                "observed_class_count": len(observed_labels),
                "observed_macro_f1": float(f1_score(labels, predictions, labels=observed_labels, average="macro", zero_division=0)),
                "fixed98_macro_f1": float(f1_score(labels, predictions, labels=np.arange(EXPECTED_CLASSES), average="macro", zero_division=0)),
                "teacher_sha256": EXPECTED_TEACHER_SHA256, "open_set_rejection_applied": False,
            })
            for index, scorer in enumerate(SCORER_ORDER):
                values = np.asarray(scores[:, index], dtype=np.float64)
                rows.append({
                    "protocol": "ZD_STRICT_PRE_EVALUATION_KNOWN_ONLY", "partition": partition, "scorer": scorer,
                    "rows": len(values), "mean": float(values.mean()), "std": float(values.std()),
                    "q05": float(np.quantile(values, 0.05)), "q50": float(np.quantile(values, 0.5)),
                    "q95": float(np.quantile(values, 0.95)), "q99": float(np.quantile(values, 0.99)),
                    "closed_set_accuracy": float(np.mean(correct)), "strict_data_used": False,
                })
        table = self.config.output_root / "tables" / "known_validation_score_characterization.csv"
        closed_table = self.config.output_root / "tables" / "closed_set_teacher_metrics.csv"
        atomic_csv(table, pd.DataFrame(rows), self.config.output_root); atomic_csv(closed_table, pd.DataFrame(closed_rows), self.config.output_root); outputs.extend((table, closed_table))
        if state_tensor_sha256(self.ensure_teacher().state_dict()) != teacher_before:
            raise ScientificAbort("Teacher weights changed during known-validation scoring")
        equivalence, equivalence_inputs = self.write_inference_equivalence(); outputs.append(equivalence)
        self.complete_stage(4, "Known-Validation Score Characterization and Stage-3M Equivalence", outputs, [self.fit_path(), *equivalence_inputs])

    def stage_05(self) -> None:
        self.assert_prelock_mutation_allowed(); self.guard.assert_fitting_allowed()
        known: Dict[str, List[np.ndarray]] = {scorer: [] for scorer in SCORER_ORDER}
        inputs: List[Path] = []
        for partition in KNOWN_VALIDATION:
            path = self.score_store(partition) / "scores.npy"; inputs.append(path); values = np.load(path, mmap_mode="r")
            for index, scorer in enumerate(SCORER_ORDER):
                known[scorer].append(np.asarray(values[:, index], dtype=np.float32))
        concatenated = {scorer: np.concatenate(parts) for scorer, parts in known.items()}
        thresholds = freeze_known_thresholds(concatenated, self.config.target_known_acceptance, self.config.alternate_known_quantile)
        output = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
        atomic_json(output, {
            "protocol": "ZD_STRICT", "status": "FROZEN_BEFORE_STRICT_EVALUATION",
            "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY, "thresholds": thresholds,
            "selection_inputs": list(KNOWN_VALIDATION), "calibration_unknown_used": False, "strict_zero_day_used": False,
            "threshold_policy_sha256": sha256_object({"policy": CANONICAL_THRESHOLD_POLICY, "thresholds": thresholds}),
            "generated_at": utc_now(),
        }, self.config.output_root)
        table = self.config.output_root / "tables" / "known_only_thresholds.csv"
        atomic_csv(table, pd.DataFrame([{"scorer": scorer, **values} for scorer, values in thresholds.items()]), self.config.output_root)
        self.complete_stage(5, "ZD-STRICT Known-Only Threshold Freeze", [output, table], inputs)

    def stage_06(self) -> None:
        self.assert_prelock_mutation_allowed()
        output = self.config.output_root / "manifests" / "ZD_CALIBRATED_ANALYSIS.json"
        table = self.config.output_root / "tables" / "zd_calibrated_thresholds.csv"
        outputs: List[Path] = [output, table]
        if not self.config.calibrated_analysis:
            atomic_json(output, {"status": "DISABLED", "protocol": "ZD_CALIBRATED", "strict_policy_impact": "NONE", "generated_at": utc_now()}, self.config.output_root)
            atomic_csv(table, pd.DataFrame([{"protocol": "ZD_CALIBRATED", "status": "DISABLED", "used_for_zd_strict": False}]), self.config.output_root)
        else:
            self.guard.assert_fitting_allowed()
            store = self.extract_non_strict_scores("calibration_unknown"); scores = np.load(store / "scores.npy", mmap_mode="r")
            known = np.concatenate([np.load(self.score_store(partition) / "scores.npy", mmap_mode="r") for partition in KNOWN_VALIDATION], axis=0)
            strict_manifest = json.loads((self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json").read_text(encoding="utf-8"))
            strict_sha_before = sha256_file(self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json")
            rows = []
            for index, scorer in enumerate(SCORER_ORDER):
                candidates = np.unique(np.quantile(np.concatenate((known[:, index], scores[:, index])), np.linspace(0.01, 0.99, 199)))
                targets = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(scores), dtype=np.int8)))
                values = np.concatenate((known[:, index], scores[:, index]))
                ranked = sorted(((float(f1_score(targets, values > threshold, average="macro", zero_division=0)), float(threshold)) for threshold in candidates), key=lambda pair: (-pair[0], pair[1]))
                rows.append({"protocol": "ZD_CALIBRATED", "scorer": scorer, "threshold": ranked[0][1], "calibration_macro_f1": ranked[0][0], "used_for_zd_strict": False})
            atomic_csv(table, pd.DataFrame(rows), self.config.output_root)
            atomic_json(output, {"status": "COMPLETE", "protocol": "ZD_CALIBRATED", "source": "KNOWN_VALIDATION_PLUS_CALIBRATION_UNKNOWN", "strict_policy_impact": "NONE", "strict_threshold_manifest_sha256": strict_sha_before, "generated_at": utc_now()}, self.config.output_root)
            if sha256_file(self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json") != strict_sha_before or strict_manifest.get("protocol") != "ZD_STRICT":
                raise ScientificAbort("ZD-CALIBRATED analysis altered the ZD-STRICT threshold freeze")
            outputs.extend(store / name for name in ("scores.npy", "store_manifest.json"))
        self.complete_stage(6, "Optional Separately Labelled ZD-CALIBRATED Analysis", outputs, [self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"])

    def stage_07(self) -> None:
        self.assert_prelock_mutation_allowed(); self.guard.assert_fitting_allowed(); self.guard.assert_zero()
        thresholds = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
        fit = self.fit_path()
        output = self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json"
        payload = {
            "status": "FROZEN_BEFORE_STRICT_EVALUATION", "protocol": "ZD_STRICT",
            "canonical_scorer": CANONICAL_SCORER_POLICY,
            "canonical_scorer_rationale": "Unknown-free evidence cannot identify a scientifically defensible detector winner; all predeclared scorers remain co-primary and are reported without post-hoc selection.",
            "scorer_definitions": SCORER_DEFINITIONS, "scorer_order": list(SCORER_ORDER),
            "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY,
            "scorer_fit_sha256": sha256_file(fit), "threshold_manifest_sha256": sha256_file(thresholds),
            "known_validation_only": True, "calibration_unknown_influences_strict_policy": False,
            "strict_zero_day_influences_selection_or_threshold": False, "strict_violation_counters": self.guard.counters(),
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "configuration_sha256": self.config.configuration_sha256(),
            "generated_at": utc_now(),
        }
        payload["policy_sha256"] = sha256_object(payload)
        atomic_json(output, payload, self.config.output_root)
        if not self.inference_equivalence_current():
            raise ScientificAbort("Stage 3M to Stage 3.5M inference equivalence gate is not PASS")
        known_bundle = self.write_known_score_bundle()
        report = self.config.output_root / "reports" / "PRE_STRICT_POLICY_FREEZE.md"
        atomic_text(report, "# Pre-strict policy freeze\n\nAll five deterministic scorers are frozen and will be reported. Each scorer uses its own P0-P3-known-only 95% acceptance threshold. Calibration Unknown and strict zero-day data did not select scorers or thresholds. The complete known-score bundle and Stage 3M inference-equivalence gate are cryptographically frozen. No teacher fitting or unknown training class exists.\n", self.config.output_root)
        self.complete_stage(7, "Scorer, Policy, and Known-Score Bundle Freeze", [output, known_bundle, report], [fit, thresholds, self.inference_equivalence_path()])

    def stage_08(self) -> None:
        lock_sha = self.write_or_verify_lock()
        self.guard.activate_final_lock(8, self.lock_current())
        self.guard.assert_zero()
        evidence_path = self.config.output_root / "manifests" / "STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))["sealed_strict_partitions"]
        benchmark = self.ensure_benchmark()
        authorized_non_strict = np.concatenate([metadata.indices for metadata in benchmark.partitions.values()])
        strict_indices: Dict[str, np.ndarray] = {}
        sealed_paths: Dict[str, Path] = {}
        strict_inputs: List[Path] = []
        for partition, details in evidence.items():
            self.guard.authorize_strict("signal", 8, self.lock_current())
            path = Path(details["path"]); strict_inputs.append(path)
            actual_sha = sha256_file(path)
            if actual_sha not in details["declared_sha256_candidates"]:
                raise ScientificAbort(f"Sealed strict index SHA mismatch: {partition}")
            values = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64).reshape(-1)
            if len(values) != int(details["expected_count"]) or len(np.unique(values)) != len(values):
                raise ScientificAbort(f"Invalid sealed strict index set: {partition}")
            if np.intersect1d(values, authorized_non_strict).size:
                raise ScientificAbort(f"Strict partition overlaps a non-strict partition: {partition}")
            strict_indices[partition] = values
            sealed_paths[partition] = path
        relationship = validate_strict_subset_relationship(
            strict_indices["strict_zero_day_test"], strict_indices["strict_zero_day_shift_test"], authorized_non_strict,
        )
        for partition, values in strict_indices.items():
            self.extract_strict_scores(partition, values)
        threshold_path = self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json"
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
        known_scores = np.concatenate([np.load(self.score_store(partition) / "scores.npy", mmap_mode="r") for partition in KNOWN_VALIDATION], axis=0)
        known_correct = np.concatenate([np.load(self.score_store(partition) / "known_correct.npy", mmap_mode="r") for partition in KNOWN_VALIDATION]).astype(bool)
        rows: List[Dict[str, Any]] = []
        unknown_by_partition = {partition: np.load(self.score_store(partition) / "scores.npy", mmap_mode="r") for partition in STRICT_PARTITIONS}
        for partition, unknown in unknown_by_partition.items():
            for index, scorer in enumerate(SCORER_ORDER):
                metrics = open_set_metrics(known_scores[:, index], known_correct, unknown[:, index], threshold_payload[scorer]["canonical_threshold"])
                rows.append({"protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer, **metrics, "post_hoc_selection": False, "evaluation_lock_sha256": lock_sha})
        table = self.config.output_root / "tables" / "strict_open_set_metrics.csv"
        atomic_csv(table, pd.DataFrame(rows), self.config.output_root)
        audit = self.config.output_root / "manifests" / "STRICT_EVALUATION_ACCESS_AUDIT.json"
        atomic_json(audit, {
            "status": "PASS", "evaluation_lock_sha256": lock_sha, "strict_signal_access_first_stage": 8,
            "strict_partitions": {key: {"rows": len(value), "index_sha256": sha256_file(Path(evidence[key]["path"]))} for key, value in strict_indices.items()},
            "strict_partition_relationship": relationship, "shift_is_nested_subset": True,
            "strict_labels_loaded": False, "semantic_target": "UNKNOWN_BY_FROZEN_PARTITION_MEMBERSHIP",
            "scorer_fitting_after_lock": False, "threshold_fitting_after_lock": False,
            "teacher_state_immutable": state_tensor_sha256(self.ensure_teacher().state_dict()) == self.teacher_state_sha,
            "strict_violation_counters": self.guard.counters(), "generated_at": utc_now(),
        }, self.config.output_root)
        strict_bundle = self.write_strict_score_bundle(strict_indices, sealed_paths)
        outputs = [table, audit, strict_bundle, *self.lock_paths()]
        for partition in STRICT_PARTITIONS:
            outputs.extend(self.score_store(partition) / name for name in ("scores.npy", "predictions.npy", "global_indices.npy", "store_manifest.json"))
        self.guard.assert_zero()
        self.complete_stage(8, "Final Locked Strict Zero-Day Evaluation", outputs, [
            self.config.output_root / "manifests" / "SCORER_POLICY_FREEZE.json", threshold_path,
            self.known_bundle_path(), self.inference_equivalence_path(), self.lock_paths()[0], *strict_inputs,
        ])

    def stage_09(self) -> None:
        if not self.lock_current():
            raise ScientificAbort("Stage 09 requires a current final evaluation lock")
        if not self.known_score_bundle_current() or not self.strict_score_bundle_current():
            raise ScientificAbort("Stage 09 score provenance bundle is stale")
        self.guard.activate_final_lock(9, True); self.guard.authorize_strict("metric", 9, True)
        known_scores = np.concatenate([np.load(self.score_store(partition) / "scores.npy", mmap_mode="r") for partition in KNOWN_VALIDATION], axis=0)
        known_correct = np.concatenate([np.load(self.score_store(partition) / "known_correct.npy", mmap_mode="r") for partition in KNOWN_VALIDATION]).astype(bool)
        thresholds = json.loads((self.config.output_root / "thresholds" / "ZD_STRICT_THRESHOLDS.json").read_text(encoding="utf-8"))["thresholds"]
        rng = np.random.default_rng(self.config.random_seed); rows: List[Dict[str, Any]] = []
        known_pool = np.arange(len(known_scores)); known_size = min(len(known_pool), self.config.bootstrap_max_per_group)
        for partition in STRICT_PARTITIONS:
            unknown_scores = np.load(self.score_store(partition) / "scores.npy", mmap_mode="r")
            unknown_pool = np.arange(len(unknown_scores)); unknown_size = min(len(unknown_pool), self.config.bootstrap_max_per_group)
            for scorer_index, scorer in enumerate(SCORER_ORDER):
                samples: Dict[str, List[float]] = {key: [] for key in (
                    "auroc", "auprc", "unknown_f1", "known_f1", "macro_f1", "fpr_at_95_tpr",
                    "detection_error", "oscr", "known_acceptance_rate", "unknown_rejection_rate",
                )}
                for _ in range(self.config.bootstrap_replicates):
                    known_idx = rng.choice(known_pool, size=known_size, replace=True); unknown_idx = rng.choice(unknown_pool, size=unknown_size, replace=True)
                    measured = open_set_metrics(known_scores[known_idx, scorer_index], known_correct[known_idx], unknown_scores[unknown_idx, scorer_index], thresholds[scorer]["canonical_threshold"])
                    for key in samples:
                        samples[key].append(measured[key])
                for metric, values in samples.items():
                    rows.append({"protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer, "metric": metric, "bootstrap_replicates": self.config.bootstrap_replicates, "bootstrap_known_rows": known_size, "bootstrap_unknown_rows": unknown_size, "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "bootstrap_mean": float(np.mean(values))})
        output = self.config.output_root / "statistics" / "strict_bootstrap_confidence_intervals.csv"
        atomic_csv(output, pd.DataFrame(rows), self.config.output_root)
        self.guard.assert_zero()
        strict_manifests = [self.score_store(partition) / "store_manifest.json" for partition in STRICT_PARTITIONS]
        self.complete_stage(9, "Statistical and OSCR Analysis", [output], [
            self.config.output_root / "tables" / "strict_open_set_metrics.csv", self.lock_paths()[0],
            self.known_bundle_path(), self.strict_bundle_path(), *strict_manifests,
        ])

    def stage_10(self) -> None:
        if not self.lock_current():
            raise ScientificAbort("Publication requires a current strict evaluation lock")
        if not self.known_score_bundle_current() or not self.strict_score_bundle_current():
            raise ScientificAbort("Publication score provenance bundle is stale")
        metrics = pd.read_csv(self.config.output_root / "tables" / "strict_open_set_metrics.csv")
        confidence = pd.read_csv(self.config.output_root / "statistics" / "strict_bootstrap_confidence_intervals.csv")
        known = pd.read_csv(self.config.output_root / "tables" / "known_validation_score_characterization.csv")
        closed = pd.read_csv(self.config.output_root / "tables" / "closed_set_teacher_metrics.csv")
        known_scores = np.concatenate([np.load(self.score_store(partition) / "scores.npy", mmap_mode="r") for partition in KNOWN_VALIDATION], axis=0)
        separation_rows = []
        for partition in STRICT_PARTITIONS:
            unknown_scores = np.load(self.score_store(partition) / "scores.npy", mmap_mode="r")
            for index, scorer in enumerate(SCORER_ORDER):
                known_values = np.asarray(known_scores[:, index], dtype=np.float64); unknown_values = np.asarray(unknown_scores[:, index], dtype=np.float64)
                pooled = math.sqrt(max((known_values.var() + unknown_values.var()) / 2.0, 1e-18))
                separation_rows.append({
                    "protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer, "known_mean": float(known_values.mean()),
                    "known_std": float(known_values.std()), "unknown_mean": float(unknown_values.mean()),
                    "unknown_std": float(unknown_values.std()), "mean_separation": float(unknown_values.mean() - known_values.mean()),
                    "standardized_separation": float((unknown_values.mean() - known_values.mean()) / pooled),
                })
        separation = pd.DataFrame(separation_rows)
        separation_path = self.config.output_root / "tables" / "strict_score_distribution_summary.csv"
        atomic_csv(separation_path, separation, self.config.output_root)
        figures: List[Path] = []
        main_metrics = metrics[metrics.strict_partition == "strict_zero_day_test"]
        for metric in ("auroc", "auprc", "macro_f1", "oscr"):
            fig, ax = plt.subplots(figsize=(9, 5)); frame = main_metrics.sort_values("scorer")
            ax.bar(frame.scorer, frame[metric]); ax.set_ylim(0, 1); ax.set_ylabel(metric.upper()); ax.set_title(f"ZD-STRICT overall {metric.upper()} — frozen teacher")
            ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = self.config.output_root / "figures" / f"strict_{metric}.{suffix}"; fig.savefig(path, dpi=240 if suffix == "png" else None); figures.append(path)
            plt.close(fig)
        rng = np.random.default_rng(self.config.random_seed)
        unknown_scores = np.load(self.score_store("strict_zero_day_test") / "scores.npy", mmap_mode="r")
        fig, axes = plt.subplots(len(SCORER_ORDER), 1, figsize=(9, 15))
        for index, scorer in enumerate(SCORER_ORDER):
            known_index = rng.choice(len(known_scores), size=min(20_000, len(known_scores)), replace=False)
            unknown_index = rng.choice(len(unknown_scores), size=min(20_000, len(unknown_scores)), replace=False)
            axes[index].hist(known_scores[known_index, index], bins=80, density=True, alpha=0.55, label="Known Validation")
            axes[index].hist(unknown_scores[unknown_index, index], bins=80, density=True, alpha=0.55, label="Strict Unknown")
            axes[index].set_title(f"{scorer} score distribution (higher = more unknown)"); axes[index].legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = self.config.output_root / "figures" / f"strict_score_distributions.{suffix}"; fig.savefig(path, dpi=240 if suffix == "png" else None); figures.append(path)
        plt.close(fig)
        report = self.config.output_root / "reports" / "STAGE3_5M_SCIENTIFIC_REPORT.md"
        atomic_text(report, "# Stage 3.5M zero-day/open-set report\n\n## Measured facts\n\nAll rows below use the frozen Stage 3M A3 seed-123 teacher. The ZD-STRICT scorer definitions and known-only thresholds were locked before strict access. All five scorers are reported without post-hoc winner selection. The 3,000-row shifted partition is a nested sensitivity subset of the 216,000-row overall strict test; no combined population is constructed.\n\n" + metrics.to_markdown(index=False) + "\n\n### Closed-set teacher metrics (independent of rejection)\n\n" + closed.to_markdown(index=False) + "\n\n### Known-vs-unknown score separation\n\n" + separation.to_markdown(index=False) + "\n\n## Interpretation\n\nInterpretation must follow the measured tables and confidence intervals. Calibration Unknown, when enabled, is a separate ZD-CALIBRATED analysis and never changes ZD-STRICT policy. Closed-set teacher behavior is reported separately and Stage 3M outputs were not overwritten.\n", self.config.output_root)
        workbook = self.config.output_root / "publication" / "Stage3_5M_tables.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="strict_metrics", index=False); confidence.to_excel(writer, sheet_name="bootstrap_ci", index=False); known.to_excel(writer, sheet_name="known_scores", index=False); closed.to_excel(writer, sheet_name="closed_set", index=False); separation.to_excel(writer, sheet_name="score_separation", index=False)
        pdf = self.config.output_root / "publication" / "Stage3_5M_report.pdf"
        with PdfPages(pdf) as document:
            fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.08, 0.95, "Stage 3.5M Zero-Day/Open-Set Detection", fontsize=17, weight="bold"); fig.text(0.08, 0.89, "Frozen A3 seed-123 teacher\nZD-STRICT policies frozen before strict access\nAll predeclared scorers reported\nTeacher retraining: NO", va="top", fontsize=11); plt.axis("off"); document.savefig(fig); plt.close(fig)
            for path in figures:
                if path.suffix == ".png":
                    image = plt.imread(path); fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.imshow(image); ax.axis("off"); document.savefig(fig, bbox_inches="tight"); plt.close(fig)
        figure_manifest = self.config.output_root / "publication" / "FIGURE_MANIFEST.json"
        atomic_json(figure_manifest, {"figures": [{"path": str(path), "sha256": sha256_file(path)} for path in figures], "measured_inputs_only": True}, self.config.output_root)
        self.complete_stage(10, "Publication Outputs", [report, workbook, pdf, figure_manifest, separation_path, *figures], [
            self.config.output_root / "tables" / "strict_open_set_metrics.csv",
            self.config.output_root / "statistics" / "strict_bootstrap_confidence_intervals.csv",
            self.config.output_root / "tables" / "closed_set_teacher_metrics.csv",
            self.known_bundle_path(), self.strict_bundle_path(), self.lock_paths()[0],
        ])

    def stage_11(self) -> None:
        ready = self.config.output_root / "MANYTX_STAGE3_5M_READY.txt"; not_ready = self.config.output_root / "MANYTX_STAGE3_5M_NOT_READY.txt"
        if ready.exists(): ready.unlink()
        for stage in range(1, 11):
            if not self.stage_current(stage):
                raise ScientificAbort(f"Stage {stage:02d} is stale or incomplete")
        if not self.lock_current():
            raise ScientificAbort("Final strict evaluation lock is not current")
        self.guard.activate_final_lock(11, True); self.guard.assert_zero()
        if state_tensor_sha256(self.ensure_teacher().state_dict()) != self.teacher_state_sha:
            raise ScientificAbort("Frozen teacher changed before READY gate")
        metrics = pd.read_csv(self.config.output_root / "tables" / "strict_open_set_metrics.csv")
        gates = {
            "stage3m_predecessor_verified": True, "teacher_seed_123": True, "teacher_sha_verified": True,
            "teacher_immutable": True, "no_teacher_training": True, "no_unknown_training_class": True,
            "all_five_scorers_reported": set(metrics.scorer) == set(SCORER_ORDER),
            "strict_thresholds_known_only": True, "scorer_policy_frozen_before_strict": True,
            "known_score_bundle_current": self.known_score_bundle_current(),
            "strict_score_bundle_current": self.strict_score_bundle_current(),
            "stage3m_stage35_inference_equivalence": self.inference_equivalence_current(),
            "final_evaluation_lock_current": True, "strict_evaluation_complete": set(STRICT_PARTITIONS) <= set(metrics.strict_partition),
            "strict_violation_counters_zero": not any(self.guard.counters().values()),
            "surrogate_training_not_performed": True, "xai_not_performed": True,
            "publication_complete": (self.config.output_root / "publication" / "Stage3_5M_report.pdf").is_file(),
        }
        if not all(gates.values()):
            raise ScientificAbort(f"Stage 3.5M READY gates failed: {gates}")
        final_status = self.config.output_root / "manifests" / "STAGE3_5M_FINAL_STATUS.json"
        atomic_json(final_status, {
            "status": "MANYTX_STAGE3_5M_READY", "pipeline_version": PIPELINE_VERSION, "gates": gates,
            "canonical_scorer": CANONICAL_SCORER_POLICY, "canonical_threshold_policy": CANONICAL_THRESHOLD_POLICY,
            "strict_evaluation_lock_sha256": sha256_file(self.lock_paths()[0]), "strict_violation_counters": self.guard.counters(),
            "teacher_retrained": False, "surrogate_training_performed": False, "xai_performed": False, "generated_at": utc_now(),
        }, self.config.output_root)
        manifest = self.config.output_root / "manifests" / "STAGE3_5M_HASH_MANIFEST.json"
        excluded = set(FINAL_HASH_EXCLUSIONS)
        files = [path for path in sorted(self.config.output_root.rglob("*")) if path.is_file() and path.relative_to(self.config.output_root).as_posix() not in excluded]
        atomic_json(manifest, {
            "algorithm": "SHA-256", "files": [{"relative_path": path.relative_to(self.config.output_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
            "count": len(files), "exclusions": [{"relative_path": name, "reason": reason} for name, reason in FINAL_HASH_EXCLUSIONS.items()],
        }, self.config.output_root)
        if not final_hash_manifest_current(self.config.output_root):
            raise ScientificAbort("Stage 3.5M final hash manifest is inconsistent")
        if not_ready.exists(): not_ready.unlink()
        ready_text = (
            "MANYTX_STAGE3_5M_READY\n"
            f"teacher_version={EXPECTED_TEACHER_VERSION}\nteacher_seed={EXPECTED_TEACHER_SEED}\nteacher_sha256={EXPECTED_TEACHER_SHA256}\n"
            f"benchmark_sha256={EXPECTED_BENCHMARK_SHA256}\nstage2_6m_artifact_sha256={EXPECTED_STAGE26_ARTIFACT_SHA256}\n"
            f"stage3m_hash_manifest_sha256={EXPECTED_STAGE3M_HASH_MANIFEST_SHA256}\nstrict_protocol=ZD_STRICT\n"
            f"strict_evaluation_lock_sha256={sha256_file(self.lock_paths()[0])}\ncanonical_scorer={CANONICAL_SCORER_POLICY}\n"
            f"canonical_threshold_policy={CANONICAL_THRESHOLD_POLICY}\n"
            + "".join(f"{key}=0\n" for key in STRICT_COUNTER_KEYS)
            + "teacher_retrained=NO\nsurrogate_training_performed=NO\nxai_performed=NO\nnext_stage=STAGE_4M\n"
        )
        atomic_text(ready, ready_text, self.config.output_root)
        if not ready.is_file() or not final_hash_manifest_current(self.config.output_root):
            raise ScientificAbort("READY or final manifest failed post-write verification")
        outputs = [self.config.output_root / relative for relative in FINAL_REQUIRED]
        self.complete_stage(11, "Final Scientific Audit and READY Gate", outputs, [self.stage_manifest_path(stage) for stage in range(1, 11)])
        if not self.stage_current(11):
            raise ScientificAbort("Stage-11 atomic completion verification failed")
        print("\nMANYTX_STAGE3_5M_READY")

    def run(self) -> None:
        print(startup_banner())
        stages = {index: getattr(self, f"stage_{index:02d}") for index in range(1, 12)}
        for stage in range(self.config.stage_start, self.config.stage_end + 1):
            if self.stage_current(stage):
                print(f"[REUSE] Stage {stage:02d} — hash-current")
                continue
            stages[stage]()
        if self.config.preflight:
            print("\nSTAGE3_5M_PREFLIGHT_PASS")


def startup_banner() -> str:
    return "\n".join([
        "=" * 100, "STAGE 3.5M — WISIG MANYTX ZERO-DAY / OPEN-SET DETECTION", "=" * 100,
        f"CANONICAL BENCHMARK: {CANONICAL_BENCHMARK}", "STAGE 3M TEACHER: FROZEN A3 SEED 123",
        "TEACHER TRAINING: DISABLED", "UNKNOWN TRAINING CLASS: FORBIDDEN",
        "ZD-STRICT THRESHOLDS: KNOWN VALIDATION ONLY", "CALIBRATION UNKNOWN: SEPARATE OPTIONAL ZD-CALIBRATED PROTOCOL",
        "STRICT ZERO-DAY ACCESS: STAGE 08 ONLY AFTER EVALUATION LOCK", "SURROGATE TRAINING: DISABLED", "XAI: DISABLED", "=" * 100,
    ])


def synthetic_validation() -> None:
    rng = np.random.default_rng(35)
    embeddings = [] ; labels = []
    for class_index in range(EXPECTED_CLASSES):
        center = np.zeros(EXPECTED_EMBEDDING_DIM); center[class_index] = 1.0
        values = center + rng.normal(0, 0.01, size=(3, EXPECTED_EMBEDDING_DIM)); values /= np.linalg.norm(values, axis=1, keepdims=True)
        embeddings.append(values); labels.extend([class_index] * 3)
    z = np.concatenate(embeddings).astype(np.float32); y = np.asarray(labels)
    fit = fit_statistics(z, y, 0.001, 0.0001)
    logits = np.full((len(z), EXPECTED_CLASSES), -4.0, dtype=np.float32); logits[np.arange(len(z)), y] = 4.0
    scores = score_outputs(logits, z, fit)
    thresholds = freeze_known_thresholds({scorer: scores[:, index] for index, scorer in enumerate(SCORER_ORDER)}, 0.95, 0.99)
    unknown = scores + np.array([0.2, 2.0, 0.2, 5.0, 5.0], dtype=np.float32)
    for index, scorer in enumerate(SCORER_ORDER):
        metrics = open_set_metrics(scores[:, index], np.ones(len(scores), dtype=bool), unknown[:, index], thresholds[scorer]["canonical_threshold"])
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ScientificAbort("Synthetic open-set metric is non-finite")
    print("STAGE3_5M_SYNTHETIC_VALIDATION_PASS")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--branch-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--stage-start", type=int)
    parser.add_argument("--stage-end", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--calibrated-analysis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--synthetic-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.synthetic_validation:
        synthetic_validation(); return 0
    payload: Dict[str, Any] = {}
    if args.config:
        payload.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.branch_root: payload["branch_root"] = str(args.branch_root)
    if args.repository_root: payload["repository_root"] = str(args.repository_root)
    if args.stage_start is not None: payload["stage_start"] = args.stage_start
    if args.stage_end is not None: payload["stage_end"] = args.stage_end
    if args.resume is not None: payload["resume"] = args.resume
    if args.calibrated_analysis is not None: payload["calibrated_analysis"] = args.calibrated_analysis
    if args.preflight:
        payload.update({"preflight": True, "stage_start": 1, "stage_end": 2})
    if "branch_root" not in payload:
        raise SystemExit("--branch-root or config branch_root is required")
    config = Stage35Config(**payload)
    try:
        Stage35Pipeline(config).run()
        return 0
    except Exception as exc:
        output = config.output_root; output.mkdir(parents=True, exist_ok=True)
        ready = output / "MANYTX_STAGE3_5M_READY.txt"
        if ready.exists(): ready.unlink()
        atomic_text(output / "MANYTX_STAGE3_5M_NOT_READY.txt", f"MANYTX_STAGE3_5M_NOT_READY\n{type(exc).__name__}: {exc}\n", output)
        print("MANYTX_STAGE3_5M_NOT_READY", file=sys.stderr); traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
