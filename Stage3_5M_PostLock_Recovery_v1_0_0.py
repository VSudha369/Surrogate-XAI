#!/usr/bin/env python3
"""Audited Stage 3.5M post-lock recovery from immutable persisted score stores.

This executable has no strict-signal data path.  It verifies the original
evaluation lock and persisted score products, audits the frozen shifted-subset
relationship, and can finalize Stage 08-11 exclusively from those products.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


PIPELINE_VERSION = "1.0.0"
RECOVERY_VERSION = "1.0.0"
CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
ORIGINAL_EXECUTION_COMMIT = "311f99517269c389f9a79ccbae6a293ddc6fb628"
ORIGINAL_EXECUTABLE_SHA256 = "a6c767536f21cbd5ab23ad8a555dfe5cfcde29075ce7821510a0754915254351"
ORIGINAL_CONFIGURATION_SHA256 = "e109d1afaf8f084a3032c870449e32746dcd8f0bd4e325f46edd4bcc8f66b75a"
ORIGINAL_LOCK_SHA256 = "257c42de3f486407612c604f7d726aac81232359eff75bd9f63731a0796af8d8"
EXPECTED_FIT_SHA256 = "36aa55f9b680ecabb3bd399acbc212485fdeaf7b9caded323a6605457021cd09"
EXPECTED_KNOWN_BUNDLE_SHA256 = "d708b5d02e7dcc77a9df86bd67f1725f861d101fa864723e95299be9fd3339e5"
EXPECTED_THRESHOLD_SHA256 = "46e5345edf3b4204a681d2d6378ae22d30d8252f4d57b39c55eef1f4c874ebf0"
EXPECTED_POLICY_SHA256 = "c0c2892fb96a651a1966154cf8e6f312ac0f54698c2387efc946774dcb27ac9a"
EXPECTED_EQUIVALENCE_SHA256 = "72d3e00f940ac42ebe42b4e5ae60226ecffeef49269a3bac27a0aa487a30e35a"
EXPECTED_DECLARATION_SHA256 = "df08161d3b8bf771c71c55d62ca54e6850c8ddae412529aa3089fda517f6de59"
EXPECTED_TEACHER_SHA256 = "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE3M_HASH_MANIFEST_SHA256 = "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0"
EXPECTED_STAGE2M_HASH_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
RECOVERY_REASON = "STRICT_SHIFT_IS_FROZEN_SUBSET_NOT_DISJOINT"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_MAX_PER_GROUP = 20_000
BOOTSTRAP_BASE_SEED = 3_500_001
OVERLAP_TOLERANCE = 1e-5
EXPECTED_NOT_READY_LINES = (
    "MANYTX_STAGE3_5M_NOT_READY",
    "ScientificAbort: Strict main and strict shift partitions overlap",
)
KNOWN_VALIDATION = ("p0", "p1", "p2", "p3")
STRICT_PARTITIONS = {"strict_zero_day_test": 216_000, "strict_zero_day_shift_test": 3_000}
NON_STRICT_INDEX_FILES = (
    "train_known_indices.npy", "validation_known_indices.npy", "cross_day_validation_indices.npy",
    "cross_receiver_validation_indices.npy", "cross_day_receiver_validation_indices.npy",
    "calibration_unknown_indices.npy",
)
SCORER_ORDER = ("S0_MSP", "S1_ENERGY", "S2_PROTOTYPE_COSINE", "S3_MAHALANOBIS", "S4_DIAG_GAUSSIAN_NLL")
SCORER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "S0_MSP": {"formula": "1 - max softmax probability", "direction": "higher_is_more_unknown", "fit": "none"},
    "S1_ENERGY": {"formula": "-T * logsumexp(logits / T)", "direction": "higher_is_more_unknown", "fit": "none"},
    "S2_PROTOTYPE_COSINE": {"formula": "1 - max normalized class-prototype cosine", "direction": "higher_is_more_unknown", "fit": "Train Known class means"},
    "S3_MAHALANOBIS": {"formula": "minimum regularized tied-covariance squared distance", "direction": "higher_is_more_unknown", "fit": "Train Known means and pooled covariance"},
    "S4_DIAG_GAUSSIAN_NLL": {"formula": "minimum class-conditional diagonal Gaussian negative log likelihood", "direction": "higher_is_more_unknown", "fit": "Train Known class means and diagonal variances"},
}
STRICT_COUNTER_KEYS = (
    "strict_zero_day_signal_read_violations", "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations", "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations", "strict_zero_day_fit_violations",
)
FINAL_HASH_EXCLUSIONS = {
    "manifests/STAGE3_5M_HASH_MANIFEST.json", "manifests/STAGE_11_CHECKPOINT.json",
    "MANYTX_STAGE3_5M_READY.txt", "MANYTX_STAGE3_5M_NOT_READY.txt",
}
REQUIRED_BRANCH_DIRECTORIES = (
    "01_benchmark_engineering", "02_benchmark_diagnostics",
    "03_representation_ablation", "04_canonical_teacher",
)
IMMUTABLE_STRICT_RELATIVE_PATHS = (
    "manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json",
    "manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.sha256",
    "scores/strict_zero_day_test/scores.npy",
    "scores/strict_zero_day_test/predictions.npy",
    "scores/strict_zero_day_test/global_indices.npy",
    "scores/strict_zero_day_test/store_manifest.json",
    "scores/strict_zero_day_shift_test/scores.npy",
    "scores/strict_zero_day_shift_test/predictions.npy",
    "scores/strict_zero_day_shift_test/global_indices.npy",
    "scores/strict_zero_day_shift_test/store_manifest.json",
)


class RecoveryAbort(RuntimeError):
    """Raised when immutable recovery evidence is absent or stale."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256_int64_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1)).tobytes()).hexdigest()


def validate_branch_root(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    missing = [name for name in REQUIRED_BRANCH_DIRECTORIES if not (candidate / name).is_dir()]
    ready = candidate / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt"
    if candidate.name != CANONICAL_BRANCH or missing or not ready.is_file():
        raise RecoveryAbort(
            f"Invalid canonical branch root {candidate}; expected {CANONICAL_BRANCH}, "
            f"required directories {list(REQUIRED_BRANCH_DIRECTORIES)}, and Stage 3M READY"
        )
    return candidate


def discover_branch_root(
    explicit: Optional[Path] = None,
    search_root: Path = Path("/content/drive/MyDrive"),
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve explicit, environment, then unique Drive root without guessing."""
    if explicit is not None:
        return validate_branch_root(explicit)
    environ = os.environ if environment is None else environment
    configured = environ.get("WISIG_BRANCH_ROOT")
    if configured:
        return validate_branch_root(Path(configured))
    base = search_root.expanduser().resolve()
    if not base.is_dir():
        raise RecoveryAbort(
            f"Google Drive search root is unavailable: {base}. Mount the Google account containing the canonical artifacts."
        )
    candidates: List[Path] = []
    for path in base.rglob(CANONICAL_BRANCH):
        if not path.is_dir():
            continue
        try:
            candidates.append(validate_branch_root(path))
        except RecoveryAbort:
            continue
    candidates = sorted(set(candidates))
    if not candidates:
        raise RecoveryAbort(
            f"No valid {CANONICAL_BRANCH} root found under {base}; the wrong Google account may be mounted."
        )
    if len(candidates) != 1:
        raise RecoveryAbort("Multiple valid canonical roots found; refusing to guess:\n" + "\n".join(map(str, candidates)))
    return candidates[0]


def git_commit(repository_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def bootstrap_seed(partition: str, scorer: str, base_seed: int = BOOTSTRAP_BASE_SEED) -> int:
    material = f"{base_seed}|{partition}|{scorer}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def assert_within(path: Path, root: Path) -> Path:
    resolved, boundary = path.resolve(), root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise RecoveryAbort(f"Recovery output escapes Stage 3.5M root: {resolved}")
    return resolved


def atomic_text(path: Path, value: str, root: Path) -> None:
    assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any], root: Path) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", root)


def atomic_csv(path: Path, value: pd.DataFrame, root: Path) -> None:
    assert_within(path, root); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        value.to_csv(handle, index=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def generated_temporary(path: Path, root: Path) -> Path:
    assert_within(path, root); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    assert_within(temporary, root)
    return temporary


def commit_generated(temporary: Path, path: Path, root: Path) -> None:
    assert_within(temporary, root); assert_within(path, root)
    if not temporary.is_file():
        raise RecoveryAbort(f"Generated temporary output is missing: {temporary}")
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_hash_manifest(output_root: Path, manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "SHA-256":
            return False
        rows = payload.get("files", [])
        if int(payload.get("count", -1)) != len(rows):
            return False
        for row in rows:
            path = assert_within(output_root / row["relative_path"], output_root)
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RecoveryAbort):
        return False
    return True


def strict_subset_relationship(
    main_indices: np.ndarray,
    shift_indices: np.ndarray,
    non_strict_indices: Optional[np.ndarray] = None,
    expected_main_rows: int = 216_000,
    expected_shift_rows: int = 3_000,
) -> Dict[str, Any]:
    main = np.asarray(main_indices, dtype=np.int64).reshape(-1); shift = np.asarray(shift_indices, dtype=np.int64).reshape(-1)
    main_unique = len(np.unique(main)) == len(main); shift_unique = len(np.unique(shift)) == len(shift)
    intersection = np.intersect1d(main, shift); shift_outside = np.setdiff1d(shift, main); main_outside = np.setdiff1d(main, shift)
    non_strict_overlap = 0
    if non_strict_indices is not None:
        known = np.asarray(non_strict_indices, dtype=np.int64).reshape(-1)
        non_strict_overlap = int(np.intersect1d(main, known).size + np.intersect1d(shift, known).size)
    audit = {
        "main_rows": int(len(main)), "shift_rows": int(len(shift)), "main_unique": main_unique,
        "shift_unique": shift_unique, "intersection_rows": int(len(intersection)),
        "shift_subset_of_main": bool(len(shift_outside) == 0), "shift_rows_outside_main": int(len(shift_outside)),
        "main_rows_outside_shift": int(len(main_outside)), "non_strict_overlap_rows": non_strict_overlap,
    }
    if not (
        len(main) == expected_main_rows and len(shift) == expected_shift_rows and main_unique and shift_unique
        and len(intersection) == len(shift) and len(shift_outside) == 0 and non_strict_overlap == 0
    ):
        raise RecoveryAbort(f"Frozen strict shifted-subset relationship failed: {audit}")
    return audit


def overlap_score_consistency(
    main_indices: np.ndarray,
    shift_indices: np.ndarray,
    main_predictions: np.ndarray,
    shift_predictions: np.ndarray,
    main_scores: np.ndarray,
    shift_scores: np.ndarray,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    main = np.asarray(main_indices, dtype=np.int64); shift = np.asarray(shift_indices, dtype=np.int64)
    order = np.argsort(main); sorted_main = main[order]; locations = np.searchsorted(sorted_main, shift)
    if np.any(locations >= len(sorted_main)) or not np.array_equal(sorted_main[locations], shift):
        raise RecoveryAbort("Shift indices are not fully matchable inside strict main")
    main_positions = order[locations]
    prediction_match = np.array_equal(np.asarray(main_predictions)[main_positions], np.asarray(shift_predictions))
    difference = np.abs(np.asarray(main_scores, dtype=np.float64)[main_positions] - np.asarray(shift_scores, dtype=np.float64))
    maxima = difference.max(axis=0) if len(difference) else np.zeros(len(SCORER_ORDER))
    result = {
        "overlap_rows": int(len(shift)), "predictions_exactly_equal": bool(prediction_match),
        "score_tolerance": tolerance, "maximum_absolute_difference_by_scorer": dict(zip(SCORER_ORDER, map(float, maxima))),
        "all_score_vectors_within_tolerance": bool(np.all(maxima <= tolerance)),
    }
    if not prediction_match or not result["all_score_vectors_within_tolerance"]:
        raise RecoveryAbort(f"Persisted overlap score consistency failed: {result}")
    return result


def oscr_score(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray) -> float:
    known = np.asarray(known_scores, dtype=np.float64); correct = np.asarray(known_correct, dtype=bool); unknown = np.asarray(unknown_scores, dtype=np.float64)
    values = np.concatenate((known, unknown)); unknown_marker = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8)))
    correct_marker = np.concatenate((correct.astype(np.int8), np.zeros(len(unknown), dtype=np.int8))); order = np.argsort(values, kind="stable")
    sorted_values = values[order]; cumulative_unknown = np.cumsum(unknown_marker[order]); cumulative_correct = np.cumsum(correct_marker[order])
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_values) != 0), len(sorted_values) - 1]
    fpr = np.r_[0.0, cumulative_unknown[group_ends] / len(unknown)]; ccr = np.r_[0.0, cumulative_correct[group_ends] / len(known)]
    return float(np.trapezoid(ccr, fpr))


def open_set_metrics(known_scores: np.ndarray, known_correct: np.ndarray, unknown_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    known = np.asarray(known_scores, dtype=np.float64); unknown = np.asarray(unknown_scores, dtype=np.float64); correct = np.asarray(known_correct, dtype=bool)
    targets = np.concatenate((np.zeros(len(known), dtype=np.int8), np.ones(len(unknown), dtype=np.int8))); values = np.concatenate((known, unknown)); predicted = (values > threshold).astype(np.int8)
    fpr, tpr, _ = roc_curve(targets, values); candidates = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(targets, values)), "auprc": float(average_precision_score(targets, values)),
        "unknown_f1": float(f1_score(targets, predicted, pos_label=1, zero_division=0)),
        "known_f1": float(f1_score(targets, predicted, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "fpr_at_95_tpr": float(fpr[candidates[0]]) if len(candidates) else 1.0,
        "detection_error": float(np.min(0.5 * (fpr + 1.0 - tpr))), "oscr": oscr_score(known, correct, unknown),
        "threshold": float(threshold), "known_acceptance_rate": float(np.mean(known <= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown > threshold)), "known_rows": float(len(known)), "unknown_rows": float(len(unknown)),
    }


@dataclass
class RecoveryEvidence:
    output_root: Path
    not_ready_bytes: bytes
    lock_payload: Dict[str, Any]
    strict_indices: Dict[str, np.ndarray]
    sealed_paths: Dict[str, Path]
    stores: Dict[str, Dict[str, Any]]
    relationship: Dict[str, Any]
    overlap: Dict[str, Any]
    immutable_hashes: Dict[str, str]


class PostLockRecovery:
    def __init__(self, branch_root: Optional[Path], repository_root: Path, search_root: Path = Path("/content/drive/MyDrive")):
        self.branch_root = discover_branch_root(branch_root, search_root=search_root)
        self.repository_root = repository_root.expanduser().resolve()
        self.output_root = self.branch_root / "05_zero_day_open_set"
        self.recovery_sha = sha256_file(Path(__file__).resolve())
        self.recovery_commit = git_commit(self.repository_root)

    def path(self, relative: str) -> Path:
        return self.output_root / relative

    def ready_content(self) -> str:
        lines = [
            "MANYTX_STAGE3_5M_READY",
            "teacher_version=1.0",
            "teacher_seed=123",
            f"teacher_sha256={EXPECTED_TEACHER_SHA256}",
            f"benchmark_sha256={EXPECTED_BENCHMARK_SHA256}",
            f"stage2_6m_artifact_sha256={EXPECTED_STAGE26_ARTIFACT_SHA256}",
            f"stage3m_hash_manifest_sha256={EXPECTED_STAGE3M_HASH_MANIFEST_SHA256}",
            "strict_protocol=ZD_STRICT",
            "canonical_scorer=ALL_PREDECLARED",
            "post_lock_recovery=YES",
            f"post_lock_recovery_reason={RECOVERY_REASON}",
            f"original_strict_scoring_executable_sha256={ORIGINAL_EXECUTABLE_SHA256}",
            f"original_evaluation_lock_sha256={ORIGINAL_LOCK_SHA256}",
            f"recovery_executable_sha256={self.recovery_sha}",
            "strict_signal_reinference=NO",
            "strict_scores_recomputed=NO",
            "scorer_refit=NO",
            "threshold_refit=NO",
            "policy_changed=NO",
            "strict_labels_loaded=NO",
            "strict_shift_subset_of_main=YES",
            *[f"{key}=0" for key in STRICT_COUNTER_KEYS],
            "teacher_retrained=NO",
            "surrogate_training_performed=NO",
            "xai_performed=NO",
            "next_stage=STAGE_4M",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _verify_file(path: Path, expected_sha: str, label: str) -> None:
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RecoveryAbort(f"{label} is missing or has changed: {path}")

    def verify_not_ready(self) -> bytes:
        path = self.path("MANYTX_STAGE3_5M_NOT_READY.txt")
        if not path.is_file():
            raise RecoveryAbort("Original Stage 3.5M NOT_READY marker is missing")
        value = path.read_bytes()
        if tuple(value.decode("utf-8").splitlines()) != EXPECTED_NOT_READY_LINES:
            raise RecoveryAbort("Original NOT_READY failure does not exactly match the reviewed subset abort")
        return value

    def immutable_snapshot(self) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for relative in IMMUTABLE_STRICT_RELATIVE_PATHS:
            path = self.path(relative)
            if not path.is_file():
                raise RecoveryAbort(f"Immutable original strict evidence is missing: {path}")
            snapshot[relative] = sha256_file(path)
        return snapshot

    def verify_immutable_snapshot(self, before: Mapping[str, str]) -> Dict[str, bool]:
        changed = {
            relative: (not self.path(relative).is_file() or sha256_file(self.path(relative)) != expected)
            for relative, expected in before.items()
        }
        if any(changed.values()):
            raise RecoveryAbort(f"Original lock/strict store changed during recovery: {changed}")
        return {
            "original_lock_modified": changed["manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json"],
            "strict_main_store_modified": any(changed[key] for key in changed if "/strict_zero_day_test/" in key),
            "strict_shift_store_modified": any(changed[key] for key in changed if "/strict_zero_day_shift_test/" in key),
        }

    def checkpoint_current(self, stage: int) -> bool:
        checkpoint = self.path(f"manifests/STAGE_{stage:02d}_CHECKPOINT.json")
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if stage >= 8 and payload.get("recovery_executable_sha256") != self.recovery_sha:
                return False
            for source, expected in payload.get("required_input_hashes", {}).items():
                path = Path(source)
                if not path.is_file() or sha256_file(path) != expected:
                    return False
            for row in payload.get("required_outputs", []):
                path = Path(row["path"])
                if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                    return False
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False
        return True

    def stage11_current(self) -> bool:
        ready = self.path("MANYTX_STAGE3_5M_READY.txt")
        manifest = self.path("manifests/STAGE3_5M_HASH_MANIFEST.json")
        status = self.path("manifests/STAGE3_5M_FINAL_STATUS.json")
        if not (ready.is_file() and status.is_file() and self.checkpoint_current(11) and verify_hash_manifest(self.output_root, manifest)):
            return False
        try:
            return ready.read_text(encoding="utf-8") == self.ready_content()
        except OSError:
            return False

    def recovery_state(self) -> str:
        ready = self.path("MANYTX_STAGE3_5M_READY.txt")
        not_ready = self.path("MANYTX_STAGE3_5M_NOT_READY.txt")
        stage11 = self.stage11_current()
        if stage11 and ready.is_file():
            if not_ready.is_file():
                self.verify_not_ready()
                return "CLEANUP_ONLY"
            return "COMPLETE"
        if not not_ready.is_file():
            raise RecoveryAbort("Unsafe recovery state: neither reviewed NOT_READY nor a valid completed Stage-11 transaction exists")
        self.verify_not_ready()
        return "RESUMABLE_PARTIAL" if ready.is_file() or any(self.checkpoint_current(stage) for stage in range(8, 11)) else "PENDING"

    @staticmethod
    def _ready_values(path: Path) -> Tuple[str, Dict[str, str]]:
        if not path.is_file():
            raise RecoveryAbort(f"Required frozen READY marker is missing: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        return (lines[0] if lines else "", dict(line.split("=", 1) for line in lines[1:] if "=" in line))

    def verify_predecessors(self) -> Dict[str, Any]:
        engineering = self.branch_root / "01_benchmark_engineering"
        diagnostics = self.branch_root / "02_benchmark_diagnostics"
        stage26 = self.branch_root / "03_representation_ablation"
        stage3m = self.branch_root / "04_canonical_teacher"
        benchmark = engineering / "benchmark" / "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3.h5"
        stage1_hash = engineering / "manifests" / "HASH_MANIFEST.json"
        if not stage1_hash.is_file() or not benchmark.is_file() or sha256_file(benchmark) != EXPECTED_BENCHMARK_SHA256:
            raise RecoveryAbort("Stage 1B frozen benchmark/hash-manifest verification failed")
        stage2_status = diagnostics / "manifests" / "STAGE2M_FINAL_STATUS.json"
        stage2_hash = diagnostics / "manifests" / "HASH_MANIFEST.json"
        if not stage2_status.is_file() or not stage2_hash.is_file() or sha256_file(stage2_hash) != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256:
            raise RecoveryAbort("Stage 2M frozen status/hash-manifest verification failed")
        if json.loads(stage2_status.read_text(encoding="utf-8")).get("status") != "MANYTX_STAGE2M_READY":
            raise RecoveryAbort("Stage 2M final status is not READY")
        stage26_marker, stage26_values = self._ready_values(stage26 / "MANYTX_STAGE2_6M_READY.txt")
        if stage26_marker != "MANYTX_STAGE2_6M_READY" or stage26_values.get("artifact_sha256") != EXPECTED_STAGE26_ARTIFACT_SHA256 or stage26_values.get("decision") != "SELECT_CE_SUPCON_PROTOTYPE":
            raise RecoveryAbort("Stage 2.6M frozen READY contract mismatch")
        stage3_marker, stage3_values = self._ready_values(stage3m / "MANYTX_STAGE3M_READY.txt")
        stage3_manifest = stage3m / "manifests" / "STAGE3M_HASH_MANIFEST.json"
        teacher = stage3m / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"
        if (
            stage3_marker != "MANYTX_STAGE3M_READY" or stage3_values.get("selected_seed") != "123"
            or stage3_values.get("canonical_teacher_sha256") != EXPECTED_TEACHER_SHA256
            or not stage3_manifest.is_file() or sha256_file(stage3_manifest) != EXPECTED_STAGE3M_HASH_MANIFEST_SHA256
            or not teacher.is_file() or sha256_file(teacher) != EXPECTED_TEACHER_SHA256
        ):
            raise RecoveryAbort("Stage 3M frozen teacher/READY contract mismatch")
        stage3_status = json.loads((stage3m / "manifests" / "STAGE3M_FINAL_STATUS.json").read_text(encoding="utf-8"))
        counters = stage3_status.get("strict_zero_day_counters", {})
        if stage3_status.get("status") != "MANYTX_STAGE3M_READY" or int(stage3_status.get("selected_seed", -1)) != 123 or any(counters.values()):
            raise RecoveryAbort("Stage 3M final status/strict-counter verification failed")
        return {
            "canonical_branch_root": str(self.branch_root), "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2m_status": "MANYTX_STAGE2M_READY", "stage2_6m_status": stage26_marker,
            "stage3m_status": stage3_marker, "teacher_seed": 123, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "stage3m_strict_counters_zero": True,
        }

    def verify_stage_checkpoints(self) -> None:
        for stage in range(1, 8):
            checkpoint = self.path(f"manifests/STAGE_{stage:02d}_CHECKPOINT.json")
            if not checkpoint.is_file():
                raise RecoveryAbort(f"Original Stage {stage:02d} checkpoint is missing")
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("pipeline_version") != PIPELINE_VERSION or payload.get("executable_sha256") != ORIGINAL_EXECUTABLE_SHA256 or payload.get("configuration_sha256") != ORIGINAL_CONFIGURATION_SHA256:
                raise RecoveryAbort(f"Original Stage {stage:02d} checkpoint provenance mismatch")
            for source, expected in payload.get("required_input_hashes", {}).items():
                path = Path(source)
                if not path.is_file() or sha256_file(path) != expected:
                    raise RecoveryAbort(f"Original Stage {stage:02d} input changed: {path}")
            for row in payload.get("required_outputs", []):
                path = Path(row["path"])
                if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                    raise RecoveryAbort(f"Original Stage {stage:02d} output changed: {path}")

    def verify_known_bundle(self, bundle_path: Path) -> None:
        self._verify_file(bundle_path, EXPECTED_KNOWN_BUNDLE_SHA256, "Pre-strict known-score bundle")
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        if payload.get("status") != "FROZEN_BEFORE_STRICT_EVALUATION" or set(payload.get("partitions", {})) != set(KNOWN_VALIDATION):
            raise RecoveryAbort("Known-score bundle schema/status mismatch")
        for partition, record in payload["partitions"].items():
            files = record.get("files", {})
            required = {"scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy", "store_manifest.json"}
            if set(files) != required:
                raise RecoveryAbort(f"Known-score bundle file set mismatch: {partition}")
            for name, evidence in files.items():
                path = Path(evidence["path"])
                if not path.is_file() or path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != evidence["sha256"]:
                    raise RecoveryAbort(f"Known-score bundle member changed: {partition}/{name}")

    def verify_lock(self) -> Dict[str, Any]:
        lock = self.path("manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json"); sidecar = self.path("manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.sha256")
        self._verify_file(lock, ORIGINAL_LOCK_SHA256, "Original strict evaluation lock")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != ORIGINAL_LOCK_SHA256:
            raise RecoveryAbort("Original strict evaluation lock sidecar mismatch")
        fit = self.path("scores/fitted/known_only_scorer_state.npz"); thresholds = self.path("thresholds/ZD_STRICT_THRESHOLDS.json")
        policy = self.path("manifests/SCORER_POLICY_FREEZE.json"); bundle = self.path("manifests/PRE_STRICT_KNOWN_SCORE_BUNDLE.json")
        equivalence = self.path("manifests/STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json"); declaration = self.path("manifests/STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json")
        for path, expected, label in (
            (fit, EXPECTED_FIT_SHA256, "Fitted scorer"), (thresholds, EXPECTED_THRESHOLD_SHA256, "Strict thresholds"),
            (policy, EXPECTED_POLICY_SHA256, "Scorer policy"), (equivalence, EXPECTED_EQUIVALENCE_SHA256, "Inference equivalence"),
            (declaration, EXPECTED_DECLARATION_SHA256, "Stage-02 sealed declaration"),
        ):
            self._verify_file(path, expected, label)
        self.verify_known_bundle(bundle)
        payload = json.loads(lock.read_text(encoding="utf-8")); expected = {
            "executable_sha256": ORIGINAL_EXECUTABLE_SHA256, "configuration_sha256": ORIGINAL_CONFIGURATION_SHA256,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_HASH_MANIFEST_SHA256,
            "scorer_fit_sha256": EXPECTED_FIT_SHA256, "threshold_manifest_sha256": EXPECTED_THRESHOLD_SHA256,
            "policy_freeze_sha256": EXPECTED_POLICY_SHA256, "known_score_bundle_sha256": EXPECTED_KNOWN_BUNDLE_SHA256,
            "stage3m_stage35_inference_equivalence_sha256": EXPECTED_EQUIVALENCE_SHA256,
            "sealed_strict_declaration_sha256": EXPECTED_DECLARATION_SHA256,
            "post_lock_fitting_permitted": False, "post_lock_calibration_permitted": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RecoveryAbort("Original strict evaluation lock bound-field mismatch")
        counters = payload.get("strict_violation_counters_at_lock")
        if not isinstance(counters, dict) or set(counters) != set(STRICT_COUNTER_KEYS) or any(counters.values()):
            raise RecoveryAbort("Original strict evaluation lock counters are not the exact zero schema")
        return payload

    def sealed_indices(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Path]]:
        declaration = json.loads(self.path("manifests/STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json").read_text(encoding="utf-8"))
        evidence = declaration.get("sealed_strict_partitions", {}); arrays: Dict[str, np.ndarray] = {}; paths: Dict[str, Path] = {}
        if set(evidence) != set(STRICT_PARTITIONS):
            raise RecoveryAbort("Stage-02 sealed partition declaration mismatch")
        for partition, expected_rows in STRICT_PARTITIONS.items():
            details = evidence[partition]; path = Path(details["path"])
            if not path.is_file() or sha256_file(path) not in details.get("declared_sha256_candidates", []):
                raise RecoveryAbort(f"Sealed strict index SHA mismatch: {partition}")
            values = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64).reshape(-1)
            if len(values) != expected_rows:
                raise RecoveryAbort(f"Sealed strict index count mismatch: {partition}")
            arrays[partition] = values; paths[partition] = path
        return arrays, paths

    def non_strict_indices(self) -> np.ndarray:
        engineering = self.branch_root / "01_benchmark_engineering"; arrays: List[np.ndarray] = []
        for filename in NON_STRICT_INDEX_FILES:
            matches = [path for path in engineering.rglob(filename) if path.is_file()]
            if len(matches) != 1:
                raise RecoveryAbort(f"Expected exactly one non-strict {filename}; found {matches}")
            arrays.append(np.asarray(np.load(matches[0], allow_pickle=False), dtype=np.int64).reshape(-1))
        return np.concatenate(arrays)

    def verify_strict_store(self, partition: str, expected_indices: np.ndarray) -> Dict[str, Any]:
        store = self.path(f"scores/{partition}"); expected_names = {"scores.npy", "predictions.npy", "global_indices.npy", "store_manifest.json"}
        if not store.is_dir() or {path.name for path in store.iterdir() if path.is_file()} != expected_names:
            raise RecoveryAbort(f"Strict store contains missing or prohibited files: {partition}")
        manifest_path = store / "store_manifest.json"; payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "complete": True, "pipeline_version": PIPELINE_VERSION, "executable_sha256": ORIGINAL_EXECUTABLE_SHA256,
            "configuration_sha256": ORIGINAL_CONFIGURATION_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "fit_sha256": EXPECTED_FIT_SHA256,
            "scorer_order": list(SCORER_ORDER), "scorer_definitions_sha256": sha256_object(SCORER_DEFINITIONS),
            "energy_temperature": 1.0, "partition": partition, "rows": len(expected_indices), "strict": True,
            "global_indices_sha256": sha256_int64_array(expected_indices), "evaluation_lock_sha256": ORIGINAL_LOCK_SHA256,
            "strict_labels_loaded": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RecoveryAbort(f"Strict store manifest provenance mismatch: {partition}")
        files = payload.get("files", {})
        if set(files) != {"scores.npy", "predictions.npy", "global_indices.npy"}:
            raise RecoveryAbort(f"Strict store manifest file set mismatch: {partition}")
        for name, evidence in files.items():
            path = store / name
            if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != evidence["sha256"]:
                raise RecoveryAbort(f"Strict store member changed: {partition}/{name}")
        scores = np.load(store / "scores.npy", mmap_mode="r"); predictions = np.load(store / "predictions.npy", mmap_mode="r")
        indices = np.asarray(np.load(store / "global_indices.npy", mmap_mode="r"), dtype=np.int64)
        if scores.shape != (len(expected_indices), len(SCORER_ORDER)) or predictions.shape != (len(expected_indices),) or not np.array_equal(indices, expected_indices):
            raise RecoveryAbort(f"Strict store shape/index identity mismatch: {partition}")
        if not np.isfinite(scores).all() or np.any((predictions < 0) | (predictions >= 98)):
            raise RecoveryAbort(f"Strict store contains invalid values: {partition}")
        return {"manifest": payload, "manifest_path": manifest_path, "scores": scores, "predictions": predictions, "indices": indices}

    def preflight(self) -> RecoveryEvidence:
        state = self.recovery_state()
        if state in {"COMPLETE", "CLEANUP_ONLY"}:
            raise RecoveryAbort(f"Recovery is already finalized ({state}); preflight is not a finalization path")
        not_ready = self.verify_not_ready(); self.verify_stage_checkpoints(); lock = self.verify_lock()
        immutable_hashes = self.immutable_snapshot()
        strict_indices, sealed_paths = self.sealed_indices(); relationship = strict_subset_relationship(
            strict_indices["strict_zero_day_test"], strict_indices["strict_zero_day_shift_test"], self.non_strict_indices(),
            STRICT_PARTITIONS["strict_zero_day_test"], STRICT_PARTITIONS["strict_zero_day_shift_test"],
        )
        stores = {partition: self.verify_strict_store(partition, strict_indices[partition]) for partition in STRICT_PARTITIONS}
        overlap = overlap_score_consistency(
            strict_indices["strict_zero_day_test"], strict_indices["strict_zero_day_shift_test"],
            stores["strict_zero_day_test"]["predictions"], stores["strict_zero_day_shift_test"]["predictions"],
            stores["strict_zero_day_test"]["scores"], stores["strict_zero_day_shift_test"]["scores"],
            OVERLAP_TOLERANCE,
        )
        return RecoveryEvidence(self.output_root, not_ready, lock, strict_indices, sealed_paths, stores, relationship, overlap, immutable_hashes)

    def drive_audit(self) -> None:
        self.verify_predecessors()
        state = self.recovery_state()
        if state in {"COMPLETE", "CLEANUP_ONLY"}:
            if not self.stage11_current():
                raise RecoveryAbort("Completed recovery state is not hash-current")
            self.immutable_snapshot()
        else:
            self.preflight()

    def _known_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.concatenate([np.load(self.path(f"scores/{partition}/scores.npy"), mmap_mode="r") for partition in KNOWN_VALIDATION])
        correct = np.concatenate([np.load(self.path(f"scores/{partition}/known_correct.npy"), mmap_mode="r") for partition in KNOWN_VALIDATION]).astype(bool)
        return scores, correct

    def _complete_recovery_stage(self, stage: int, outputs: Iterable[Path], inputs: Iterable[Path]) -> None:
        output_list = [assert_within(path, self.output_root) for path in outputs]
        payload = {
            "stage": stage, "status": "PASS_POST_LOCK_RECOVERY", "pipeline_version": PIPELINE_VERSION,
            "recovery_executable_sha256": self.recovery_sha, "original_executable_sha256": ORIGINAL_EXECUTABLE_SHA256,
            "configuration_sha256": ORIGINAL_CONFIGURATION_SHA256,
            "required_input_hashes": {str(path.resolve()): sha256_file(path) for path in inputs},
            "required_outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in output_list],
            "strict_signal_reinference": False, "completed_at": utc_now(),
        }
        atomic_json(self.path(f"manifests/STAGE_{stage:02d}_CHECKPOINT.json"), payload, self.output_root)
        if not self.checkpoint_current(stage):
            raise RecoveryAbort(f"Recovery Stage {stage:02d} checkpoint failed post-write verification")

    def finalize(self) -> None:
        state = self.recovery_state()
        not_ready = self.path("MANYTX_STAGE3_5M_NOT_READY.txt")
        ready = self.path("MANYTX_STAGE3_5M_READY.txt")
        if state == "COMPLETE":
            print("STAGE3_5M_POSTLOCK_RECOVERY_ALREADY_COMPLETE")
            return
        if state == "CLEANUP_ONLY":
            self.verify_not_ready()
            not_ready.unlink()
            if not self.stage11_current() or not_ready.exists() or not ready.is_file():
                raise RecoveryAbort("Stage-11 cleanup-only transaction failed verification")
            print("STAGE3_5M_POSTLOCK_RECOVERY_CLEANUP_COMPLETE")
            return
        evidence = self.preflight()
        if self.checkpoint_current(8):
            print("[REUSE] Recovery Stage 08 — hash-current")
        else:
            self._finalize_stage08(evidence)
        recovery_path = self.path("manifests/POST_LOCK_RECOVERY_MANIFEST.json")
        strict_bundle_path = self.path("manifests/FINAL_STRICT_SCORE_BUNDLE.json")
        known_scores, known_correct = self._known_arrays()
        thresholds = json.loads(self.path("thresholds/ZD_STRICT_THRESHOLDS.json").read_text(encoding="utf-8"))["thresholds"]
        if self.checkpoint_current(9):
            print("[REUSE] Recovery Stage 09 — hash-current")
        else:
            self._finalize_statistics(evidence, known_scores, known_correct, thresholds, recovery_path, strict_bundle_path)
        if self.checkpoint_current(10):
            print("[REUSE] Recovery Stage 10 — hash-current")
        else:
            self._finalize_publication(evidence, known_scores, recovery_path, strict_bundle_path)
        self.verify_immutable_snapshot(evidence.immutable_hashes)
        self._finalize_ready(evidence, recovery_path, strict_bundle_path)

    def _finalize_stage08(self, evidence: RecoveryEvidence) -> None:
        relationship_path = self.path("manifests/STRICT_PARTITION_RELATIONSHIP_AUDIT.json")
        overlap_path = self.path("manifests/STRICT_OVERLAP_SCORE_CONSISTENCY.json")
        atomic_json(relationship_path, {"status": "PASS", **evidence.relationship, "strict_labels_loaded": False, "generated_at": utc_now()}, self.output_root)
        atomic_json(overlap_path, {"status": "PASS", **evidence.overlap, "strict_labels_loaded": False, "generated_at": utc_now()}, self.output_root)
        recovery_path = self.path("manifests/POST_LOCK_RECOVERY_MANIFEST.json")
        recovery_payload = {
            "status": "POST_LOCK_RECOVERY_AUTHORIZED_ARTIFACT", "recovery_reason": RECOVERY_REASON,
            "original_execution_commit": ORIGINAL_EXECUTION_COMMIT, "original_executable_sha256": ORIGINAL_EXECUTABLE_SHA256,
            "original_configuration_sha256": ORIGINAL_CONFIGURATION_SHA256,
            "original_evaluation_lock_sha256": ORIGINAL_LOCK_SHA256,
            "recovery_commit": self.recovery_commit, "recovery_executable_sha256": self.recovery_sha,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "scorer_fit_sha256": EXPECTED_FIT_SHA256,
            "threshold_manifest_sha256": EXPECTED_THRESHOLD_SHA256, "policy_freeze_sha256": EXPECTED_POLICY_SHA256,
            "known_score_bundle_sha256": EXPECTED_KNOWN_BUNDLE_SHA256, "inference_equivalence_sha256": EXPECTED_EQUIVALENCE_SHA256,
            "strict_partition_relationship_audit_sha256": sha256_file(relationship_path),
            "strict_overlap_score_consistency_audit_sha256": sha256_file(overlap_path),
            "strict_store_manifest_sha256": {partition: sha256_file(store["manifest_path"]) for partition, store in evidence.stores.items()},
            "original_not_ready_sha256": hashlib.sha256(evidence.not_ready_bytes).hexdigest(),
            "original_not_ready_text": evidence.not_ready_bytes.decode("utf-8"),
            "strict_signals_re_read": False, "strict_scores_recomputed": False, "teacher_inference_performed": False,
            "scorer_refit": False, "threshold_refit": False, "policy_changed": False, "teacher_changed": False,
            "strict_labels_loaded": False, "original_lock_modified": False,
            "strict_main_store_modified": False, "strict_shift_store_modified": False, "generated_at": utc_now(),
        }
        atomic_json(recovery_path, recovery_payload, self.output_root)
        known_scores, known_correct = self._known_arrays()
        thresholds = json.loads(self.path("thresholds/ZD_STRICT_THRESHOLDS.json").read_text(encoding="utf-8"))["thresholds"]
        metric_rows: List[Dict[str, Any]] = []
        for partition, store in evidence.stores.items():
            for index, scorer in enumerate(SCORER_ORDER):
                measured = open_set_metrics(known_scores[:, index], known_correct, store["scores"][:, index], thresholds[scorer]["canonical_threshold"])
                metric_rows.append({"protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer, **measured,
                                    "post_hoc_selection": False, "evaluation_lock_sha256": ORIGINAL_LOCK_SHA256,
                                    "recovered_from_existing_locked_scores": True, "strict_signal_reinference": False})
        metrics_path = self.path("tables/strict_open_set_metrics.csv"); atomic_csv(metrics_path, pd.DataFrame(metric_rows), self.output_root)
        access_path = self.path("manifests/STRICT_EVALUATION_ACCESS_AUDIT.json")
        atomic_json(access_path, {
            "status": "PASS_POST_LOCK_RECOVERY", "evaluation_lock_sha256": ORIGINAL_LOCK_SHA256,
            "recovered_from_existing_locked_scores": True, "strict_signal_reinference": False,
            "shift_is_nested_subset": True, "strict_partition_relationship": evidence.relationship,
            "strict_labels_loaded": False, "scorer_fitting_after_lock": False, "threshold_fitting_after_lock": False,
            "strict_violation_counters": {key: 0 for key in STRICT_COUNTER_KEYS}, "generated_at": utc_now(),
        }, self.output_root)
        strict_bundle_path = self.path("manifests/FINAL_STRICT_SCORE_BUNDLE.json")
        partition_records: Dict[str, Any] = {}
        for partition, store in evidence.stores.items():
            directory = self.path(f"scores/{partition}")
            partition_records[partition] = {
                "rows": len(store["indices"]), "store_manifest_sha256": sha256_file(store["manifest_path"]),
                "scores_sha256": sha256_file(directory / "scores.npy"), "predictions_sha256": sha256_file(directory / "predictions.npy"),
                "global_indices_sha256": sha256_file(directory / "global_indices.npy"),
                "global_index_values_sha256": sha256_int64_array(store["indices"]),
                "sealed_index_path": str(evidence.sealed_paths[partition]), "sealed_index_sha256": sha256_file(evidence.sealed_paths[partition]),
            }
        strict_bundle = {
            "status": "FROZEN_FROM_EXISTING_LOCKED_SCORES", "pipeline_version": PIPELINE_VERSION,
            "original_executable_sha256": ORIGINAL_EXECUTABLE_SHA256, "recovery_executable_sha256": self.recovery_sha,
            "configuration_sha256": ORIGINAL_CONFIGURATION_SHA256, "evaluation_lock_sha256": ORIGINAL_LOCK_SHA256,
            "post_lock_recovery_manifest_sha256": sha256_file(recovery_path),
            "partition_relationship_audit_sha256": sha256_file(relationship_path), "partition_relationship": evidence.relationship,
            "shift_is_nested_subset": True, "strict_labels_included": False,
            "strict_signal_reinference": False, "strict_scores_recomputed": False,
            "partitions": partition_records, "generated_at": utc_now(),
        }
        canonical = dict(strict_bundle); canonical.pop("generated_at"); strict_bundle["canonical_content_sha256"] = sha256_object(canonical)
        atomic_json(strict_bundle_path, strict_bundle, self.output_root)
        self._complete_recovery_stage(8, [metrics_path, access_path, relationship_path, overlap_path, recovery_path, strict_bundle_path], [
            self.path("manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json"), self.path("manifests/PRE_STRICT_KNOWN_SCORE_BUNDLE.json"),
            *[store["manifest_path"] for store in evidence.stores.values()],
        ])
        self.verify_immutable_snapshot(evidence.immutable_hashes)

    def _finalize_statistics(
        self,
        evidence: RecoveryEvidence,
        known_scores: np.ndarray,
        known_correct: np.ndarray,
        thresholds: Mapping[str, Any],
        recovery_path: Path,
        strict_bundle_path: Path,
    ) -> None:
        rows: List[Dict[str, Any]] = []; seeds: List[Dict[str, Any]] = []
        known_pool = np.arange(len(known_scores)); known_size = min(len(known_pool), BOOTSTRAP_MAX_PER_GROUP)
        for partition, store in evidence.stores.items():
            unknown = store["scores"]; unknown_pool = np.arange(len(unknown)); unknown_size = min(len(unknown_pool), BOOTSTRAP_MAX_PER_GROUP)
            for scorer_index, scorer in enumerate(SCORER_ORDER):
                effective_seed = bootstrap_seed(partition, scorer)
                rng = np.random.default_rng(effective_seed)
                seeds.append({"base_seed": BOOTSTRAP_BASE_SEED, "effective_seed": effective_seed, "strict_partition": partition, "scorer": scorer})
                samples: Dict[str, List[float]] = {name: [] for name in ("auroc", "auprc", "unknown_f1", "known_f1", "macro_f1", "fpr_at_95_tpr", "detection_error", "oscr", "known_acceptance_rate", "unknown_rejection_rate")}
                for _ in range(BOOTSTRAP_REPLICATES):
                    known_index = rng.choice(known_pool, known_size, replace=True); unknown_index = rng.choice(unknown_pool, unknown_size, replace=True)
                    measured = open_set_metrics(known_scores[known_index, scorer_index], known_correct[known_index], unknown[unknown_index, scorer_index], thresholds[scorer]["canonical_threshold"])
                    for name in samples: samples[name].append(measured[name])
                for name, values in samples.items():
                    rows.append({"protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer, "metric": name,
                                 "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_known_rows": known_size, "bootstrap_unknown_rows": unknown_size,
                                 "bootstrap_base_seed": BOOTSTRAP_BASE_SEED, "bootstrap_effective_seed": effective_seed,
                                 "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "bootstrap_mean": float(np.mean(values)),
                                 "recovered_from_existing_locked_scores": True})
        output = self.path("statistics/strict_bootstrap_confidence_intervals.csv"); atomic_csv(output, pd.DataFrame(rows), self.output_root)
        provenance = self.path("statistics/STRICT_BOOTSTRAP_PROVENANCE.json")
        atomic_json(provenance, {"base_seed": BOOTSTRAP_BASE_SEED, "derivation": "SHA256(base_seed|strict_partition|scorer)[0:8] mod (2^63-1)",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "streams": seeds, "strict_signal_reinference": False}, self.output_root)
        self._complete_recovery_stage(9, [output, provenance], [self.path("tables/strict_open_set_metrics.csv"), recovery_path, strict_bundle_path])
        self.verify_immutable_snapshot(evidence.immutable_hashes)

    def _finalize_publication(self, evidence: RecoveryEvidence, known_scores: np.ndarray, recovery_path: Path, strict_bundle_path: Path) -> None:
        metrics = pd.read_csv(self.path("tables/strict_open_set_metrics.csv")); confidence = pd.read_csv(self.path("statistics/strict_bootstrap_confidence_intervals.csv"))
        known = pd.read_csv(self.path("tables/known_validation_score_characterization.csv")); closed = pd.read_csv(self.path("tables/closed_set_teacher_metrics.csv"))
        separation_rows: List[Dict[str, Any]] = []
        for partition, store in evidence.stores.items():
            for index, scorer in enumerate(SCORER_ORDER):
                known_values = np.asarray(known_scores[:, index], dtype=np.float64); unknown_values = np.asarray(store["scores"][:, index], dtype=np.float64)
                pooled = math.sqrt(max((known_values.var() + unknown_values.var()) / 2.0, 1e-18))
                separation_rows.append({"protocol": "ZD_STRICT", "strict_partition": partition, "scorer": scorer,
                    "known_mean": float(known_values.mean()), "known_std": float(known_values.std()), "unknown_mean": float(unknown_values.mean()),
                    "unknown_std": float(unknown_values.std()), "mean_separation": float(unknown_values.mean() - known_values.mean()),
                    "standardized_separation": float((unknown_values.mean() - known_values.mean()) / pooled)})
        separation = pd.DataFrame(separation_rows); separation_path = self.path("tables/strict_score_distribution_summary.csv"); atomic_csv(separation_path, separation, self.output_root)
        figures: List[Path] = []; main = metrics[metrics.strict_partition == "strict_zero_day_test"]
        for metric in ("auroc", "auprc", "macro_f1", "oscr"):
            frame = main.sort_values("scorer"); fig, axis = plt.subplots(figsize=(9, 5)); axis.bar(frame.scorer, frame[metric]); axis.set_ylim(0, 1)
            axis.set_ylabel(metric.upper()); axis.set_title(f"ZD-STRICT overall {metric.upper()} — recovered locked scores"); axis.tick_params(axis="x", rotation=25); fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = self.path(f"figures/strict_{metric}.{suffix}"); temporary = generated_temporary(path, self.output_root)
                fig.savefig(temporary, dpi=240 if suffix == "png" else None); commit_generated(temporary, path, self.output_root); figures.append(path)
            plt.close(fig)
        report = self.path("reports/STAGE3_5M_SCIENTIFIC_REPORT.md")
        atomic_text(report, "# Stage 3.5M zero-day/open-set report\n\n## Recovery disclosure\n\nThe strict signals were scored once under the original frozen lock. Finalization reused those immutable stores without signal re-read or score recomputation. The 3,000-row shifted partition is a nested sensitivity subset of the 216,000-row overall test; no combined population exists.\n\n" + metrics.to_markdown(index=False) + "\n", self.output_root)
        workbook = self.path("publication/Stage3_5M_tables.xlsx")
        workbook_temporary = generated_temporary(workbook, self.output_root)
        with pd.ExcelWriter(workbook_temporary, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="strict_metrics", index=False); confidence.to_excel(writer, sheet_name="bootstrap_ci", index=False); known.to_excel(writer, sheet_name="known_scores", index=False); closed.to_excel(writer, sheet_name="closed_set", index=False); separation.to_excel(writer, sheet_name="score_separation", index=False)
        commit_generated(workbook_temporary, workbook, self.output_root)
        pdf = self.path("publication/Stage3_5M_report.pdf")
        pdf_temporary = generated_temporary(pdf, self.output_root)
        with PdfPages(pdf_temporary) as document:
            fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.08, 0.95, "Stage 3.5M Post-Lock Recovery", fontsize=17, weight="bold")
            fig.text(0.08, 0.89, "Original locked scores reused\nStrict signal re-inference: NO\nShift is a nested sensitivity subset\nAll predeclared scorers reported", va="top"); plt.axis("off"); document.savefig(fig); plt.close(fig)
            for path in figures:
                if path.suffix == ".png":
                    image = plt.imread(path); fig, axis = plt.subplots(figsize=(8.27, 11.69)); axis.imshow(image); axis.axis("off"); document.savefig(fig); plt.close(fig)
        commit_generated(pdf_temporary, pdf, self.output_root)
        figure_manifest = self.path("publication/FIGURE_MANIFEST.json")
        atomic_json(figure_manifest, {"figures": [{"path": str(path), "sha256": sha256_file(path)} for path in figures], "recovered_from_existing_locked_scores": True}, self.output_root)
        self._complete_recovery_stage(10, [report, workbook, pdf, figure_manifest, separation_path, *figures], [self.path("tables/strict_open_set_metrics.csv"), self.path("statistics/strict_bootstrap_confidence_intervals.csv"), recovery_path, strict_bundle_path])
        self.verify_immutable_snapshot(evidence.immutable_hashes)

    def _finalize_ready(self, evidence: RecoveryEvidence, recovery_path: Path, strict_bundle_path: Path) -> None:
        final_status = self.path("manifests/STAGE3_5M_FINAL_STATUS.json")
        immutable = self.verify_immutable_snapshot(evidence.immutable_hashes)
        gates = {"original_lock_immutable": sha256_file(self.path("manifests/STRICT_ZERO_DAY_EVALUATION_LOCK.json")) == ORIGINAL_LOCK_SHA256,
                 "strict_scores_reused": True, "shift_subset_of_main": evidence.relationship["shift_subset_of_main"],
                 "overlap_scores_consistent": evidence.overlap["all_score_vectors_within_tolerance"], "strict_labels_not_loaded": True,
                 "no_scorer_or_threshold_refit": True, "no_post_hoc_selection": True, "publication_complete": self.path("publication/Stage3_5M_report.pdf").is_file()}
        if not all(gates.values()): raise RecoveryAbort(f"Post-lock READY gates failed: {gates}")
        final_payload = {"status": "MANYTX_STAGE3_5M_READY", "pipeline_version": PIPELINE_VERSION, "gates": gates,
            "post_lock_recovery": True, "post_lock_recovery_reason": RECOVERY_REASON,
            "original_strict_scoring_executable_sha256": ORIGINAL_EXECUTABLE_SHA256, "recovery_executable_sha256": self.recovery_sha,
            "strict_evaluation_lock_sha256": ORIGINAL_LOCK_SHA256, "strict_signal_reinference": False, "strict_scores_recomputed": False,
            "scorer_refit": False, "threshold_refit": False, "policy_changed": False, "strict_labels_loaded": False,
            "strict_shift_subset_of_main": True, **immutable,
            "strict_violation_counters": {key: 0 for key in STRICT_COUNTER_KEYS}, "generated_at": utc_now()}
        atomic_json(final_status, final_payload, self.output_root)
        if json.loads(final_status.read_text(encoding="utf-8")) != final_payload:
            raise RecoveryAbort("Final status failed exact post-write verification")
        manifest = self.path("manifests/STAGE3_5M_HASH_MANIFEST.json")
        files = [path for path in sorted(self.output_root.rglob("*")) if path.is_file() and path.relative_to(self.output_root).as_posix() not in FINAL_HASH_EXCLUSIONS]
        atomic_json(manifest, {"algorithm": "SHA-256", "files": [{"relative_path": path.relative_to(self.output_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files], "count": len(files), "exclusions": sorted(FINAL_HASH_EXCLUSIONS), "post_lock_recovery": True}, self.output_root)
        if not verify_hash_manifest(self.output_root, manifest):
            raise RecoveryAbort("Final hash manifest failed post-write verification")
        not_ready = self.path("MANYTX_STAGE3_5M_NOT_READY.txt"); ready = self.path("MANYTX_STAGE3_5M_READY.txt")
        if sha256_file(not_ready) != hashlib.sha256(evidence.not_ready_bytes).hexdigest(): raise RecoveryAbort("Original NOT_READY changed before READY transaction")
        ready_text = self.ready_content()
        atomic_text(ready, ready_text, self.output_root)
        if ready.read_bytes() != ready_text.encode("utf-8"):
            raise RecoveryAbort("READY marker failed exact byte verification")
        # Stage-11 checkpoint is the durable commit record and is written before NOT_READY removal.
        self._complete_recovery_stage(11, [final_status, manifest, ready], [self.path(f"manifests/STAGE_{stage:02d}_CHECKPOINT.json") for stage in range(1, 11)] + [recovery_path, strict_bundle_path])
        if not self.stage11_current():
            raise RecoveryAbort("Stage-11 durable transaction verification failed")
        self.verify_immutable_snapshot(evidence.immutable_hashes)
        not_ready.unlink()
        if not ready.is_file() or not_ready.exists() or not self.stage11_current():
            raise RecoveryAbort("Final READY-only status transaction failed")
        print("MANYTX_STAGE3_5M_READY")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--branch-root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--search-root", type=Path, default=Path("/content/drive/MyDrive"), help=argparse.SUPPRESS)
    action = parser.add_mutually_exclusive_group(required=True); action.add_argument("--preflight", action="store_true"); action.add_argument("--drive-audit", action="store_true"); action.add_argument("--finalize", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv); recovery = PostLockRecovery(args.branch_root, args.repository_root, args.search_root)
    if args.drive_audit:
        recovery.drive_audit(); print("STAGE3_5M_DRIVE_AUDIT_PASS"); return 0
    if args.preflight:
        recovery.preflight(); print("STAGE3_5M_POSTLOCK_RECOVERY_PREFLIGHT_PASS"); return 0
    recovery.finalize(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
