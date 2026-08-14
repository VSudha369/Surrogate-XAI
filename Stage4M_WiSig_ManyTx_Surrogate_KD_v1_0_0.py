#!/usr/bin/env python3
"""Stage 4M v1.0.0 — WiSig ManyTx surrogate knowledge distillation.

The executable trains one predeclared half-width student architecture under
four frozen KD objectives.  Training is restricted to Train Known, selection
is restricted to P0, and all Stage 3.5M strict-result paths are structurally
denied.  P1-P3 and Calibration Unknown are post-selection diagnostics only.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


PIPELINE_VERSION = "1.0.0"
CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
CANONICAL_BENCHMARK = "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE3M_MANIFEST_SHA256 = "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0"
EXPECTED_TEACHER_SHA256 = "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4"
EXPECTED_TEACHER_STATE_SHA256 = "7d6c6ff609fb86618ae7b92bcd55b0c8a440ed2769561d4de9b4485802e639d7"
EXPECTED_TEACHER_ARCHITECTURE_SHA256 = "d5ed7528ab93246c784fe12ed1bee90d4234753293e89bdf3d226db8f2fb5f9c"
EXPECTED_TEACHER_PARAMETERS = 849_634
EXPECTED_KNOWN_CLASSES = 98
TEACHER_EMBEDDING_DIM = 128
STUDENT_EMBEDDING_DIM = 64
SEEDS = (42, 123, 2026)
ARMS = ("K0", "K1", "K2", "K3")
KNOWN_PROTOCOLS = ("p0", "p1", "p2", "p3")
NON_STRICT_CACHE_PARTITIONS = ("train_known", *KNOWN_PROTOCOLS)
EXPECTED_COUNTS = {
    "train_known": 388_139,
    "p0": 68_495,
    "p1": 153_529,
    "p2": 27_088,
    "p3": 8_992,
    "calibration_unknown": 158_400,
}
REFERENCE_TEACHER_METRICS = {
    "p0": {"accuracy": 0.869377326812176, "fixed98_macro_f1": 0.8668258758471707},
    "p1": {"accuracy": 0.6580320330361039, "fixed98_macro_f1": 0.6300111516084784},
    "p2": {"accuracy": 0.4197430596574129, "fixed98_macro_f1": 0.3931644695389894},
    "p3": {"accuracy": 0.382673487544484, "fixed98_macro_f1": 0.3400004270779978},
}
KD_TEMPERATURE = 4.0
MAX_CONSECUTIVE_AMP_OVERFLOWS = 32
PRE_AMP_HOTFIX_EXECUTABLE_SHA256 = "a770fe52a83a408eedd7e8affaff120dd1d56eb5f243da52f05501252e4b4de3"
ARM_OBJECTIVES: Dict[str, Dict[str, float]] = {
    "K0": {"ce": 1.00, "kd": 0.00, "repr": 0.00, "proto": 0.00},
    "K1": {"ce": 0.50, "kd": 0.50, "repr": 0.00, "proto": 0.00},
    "K2": {"ce": 0.40, "kd": 0.40, "repr": 0.20, "proto": 0.00},
    "K3": {"ce": 0.35, "kd": 0.35, "repr": 0.15, "proto": 0.15},
}
STAGE35_COUNTER_KEYS = (
    "stage35_strict_signal_read_violations",
    "stage35_strict_label_read_violations",
    "stage35_strict_index_read_violations",
    "stage35_strict_score_read_violations",
    "stage35_strict_metric_read_violations",
    "stage35_strict_selection_violations",
)
STAGE35_METADATA_ALLOWLIST = frozenset({
    "MANYTX_STAGE3_5M_READY.txt",
    "STAGE3_5M_FINAL_STATUS.json",
    "STAGE3_5M_HASH_MANIFEST.json",
    "POST_LOCK_RECOVERY_MANIFEST.json",
    "STRICT_ZERO_DAY_EVALUATION_LOCK.json",
})
STRICT_DENY_TOKENS = (
    "strict_zero_day_test",
    "strict_zero_day_shift_test",
    "strict_open_set_metrics",
    "final_strict_score_bundle",
    "strict_bootstrap",
    "strict_prediction",
    "strict_score",
    "strict_global",
    "strict_auroc",
    "strict_auprc",
    "strict_oscr",
)
FINAL_HASH_EXCLUSIONS = {
    "manifests/STAGE4M_HASH_MANIFEST.json": "self-referential",
    "manifests/STAGE_12_CHECKPOINT.json": "written after READY transaction",
    "MANYTX_STAGE4M_READY.txt": "transaction marker",
    "MANYTX_STAGE4M_NOT_READY.txt": "transaction marker",
}


class ScientificAbort(RuntimeError):
    """A scientific/provenance invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array_values(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_within(path: Path, root: Path) -> Path:
    resolved, boundary = path.resolve(), root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ScientificAbort(f"Path escapes approved boundary: {resolved}")
    return resolved


def atomic_text(path: Path, text: str, root: Path) -> None:
    path = assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any, root: Path) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", root)


def atomic_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    atomic_text(path, frame.to_csv(index=False), root)


def atomic_torch_save(path: Path, value: Any, root: Path) -> None:
    path = assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def parse_ready(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
        elif line.strip():
            rows.setdefault("marker", line.strip())
    return rows


def bound_file_rows_current(rows: Sequence[Mapping[str, Any]], require_size: bool) -> bool:
    """Verify every provenance-bound file, including recorded output sizes."""
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            return False
        if require_size and path.stat().st_size != int(row.get("bytes", -1)):
            return False
    return True


def final_hash_manifest_current(output: Path) -> bool:
    path = output / "manifests" / "STAGE4M_HASH_MANIFEST.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("algorithm") == "SHA-256"
            and int(payload.get("count", -1)) == len(payload["files"])
            and all(
                (output / row["relative_path"]).is_file()
                and sha256_file(output / row["relative_path"]) == row["sha256"]
                and (output / row["relative_path"]).stat().st_size == int(row["bytes"])
                for row in payload["files"]
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def completed_state_current(output: Path) -> bool:
    """Validate a completed Stage-12 transaction without mutating its directory."""
    ready = output / "MANYTX_STAGE4M_READY.txt"
    final = output / "manifests" / "STAGE4M_FINAL_STATUS.json"
    checkpoint = output / "manifests" / "STAGE_12_CHECKPOINT.json"
    lock_path = output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"
    if not all(path.is_file() for path in (ready, final, checkpoint, lock_path)):
        return False
    try:
        ready_values = parse_ready(ready)
        final_values = json.loads(final.read_text(encoding="utf-8"))
        stage_values = json.loads(checkpoint.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if ready_values.get("marker") != "MANYTX_STAGE4M_READY":
            return False
        if final_values.get("status") != "MANYTX_STAGE4M_READY":
            return False
        if stage_values.get("stage") != 12 or stage_values.get("status") != "PASS":
            return False
        provenance_paths = {
            "predecessor_lock_sha256": output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json",
            "architecture_freeze_sha256": output / "manifests" / "STUDENT_ARCHITECTURE_FREEZE.json",
            "objective_policy_sha256": output / "manifests" / "KD_OBJECTIVE_POLICY.json",
            "training_target_policy_sha256": output / "manifests" / "TRAINING_TARGET_POLICY.json",
            "amp_runtime_safety_policy_sha256": output / "manifests" / "AMP_RUNTIME_SAFETY_POLICY.json",
            "selection_lock_sha256": lock_path,
        }
        if stage_values.get("pipeline_version") != PIPELINE_VERSION:
            return False
        if stage_values.get("teacher_sha256") != EXPECTED_TEACHER_SHA256 or stage_values.get("benchmark_sha256") != EXPECTED_BENCHMARK_SHA256:
            return False
        if any(not path.is_file() or stage_values.get(field) != sha256_file(path) for field, path in provenance_paths.items()):
            return False
        required_inputs = {str(output / "manifests" / f"STAGE_{stage:02d}_CHECKPOINT.json") for stage in range(1, 12)}
        required_inputs.add(str(lock_path))
        required_outputs = {str(final), str(output / "manifests" / "STAGE4M_HASH_MANIFEST.json"), str(ready)}
        if not required_inputs.issubset({str(row.get("path")) for row in stage_values.get("inputs", [])}):
            return False
        if not required_outputs.issubset({str(row.get("path")) for row in stage_values.get("outputs", [])}):
            return False
        if not bound_file_rows_current(stage_values.get("inputs", []), require_size=False):
            return False
        if not bound_file_rows_current(stage_values.get("outputs", []), require_size=True):
            return False
        for stage in range(1, 12):
            prior = json.loads((output / "manifests" / f"STAGE_{stage:02d}_CHECKPOINT.json").read_text(encoding="utf-8"))
            if prior.get("stage") != stage or prior.get("status") != "PASS":
                return False
            if not prior.get("inputs") or not prior.get("outputs"):
                return False
            if not bound_file_rows_current(prior.get("inputs", []), require_size=False):
                return False
            if not bound_file_rows_current(prior.get("outputs", []), require_size=True):
                return False
        if not final_hash_manifest_current(output):
            return False
        required = {
            "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "selected_kd_arm": str(final_values.get("selected_arm")),
            "selected_seed": str(final_values.get("selected_seed")),
            "canonical_surrogate_sha256": str(lock.get("canonical_surrogate_sha256")),
            "canonical_surrogate_state_sha256": str(lock.get("canonical_surrogate_state_sha256")),
            "amp_runtime_safety_policy_sha256": str(final_values.get("amp_runtime_safety_policy_sha256")),
        }
        return lock.get("status") == "LOCKED" and all(ready_values.get(key) == value for key, value in required.items())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def discover_branch_root(explicit: Optional[str] = None, search_root: Path = Path("/content/drive/MyDrive")) -> Path:
    candidates: List[Path] = []
    requested = explicit or os.environ.get("WISIG_BRANCH_ROOT")
    if requested:
        candidates = [Path(requested).expanduser().resolve()]
    elif search_root.is_dir():
        candidates = [path.resolve() for path in search_root.rglob(CANONICAL_BRANCH) if path.is_dir()]
    valid = []
    required = (
        "01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation",
        "04_canonical_teacher", "05_zero_day_open_set",
    )
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in required) and (
            candidate / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt"
        ).is_file() and (candidate / "05_zero_day_open_set" / "MANYTX_STAGE3_5M_READY.txt").is_file():
            valid.append(candidate)
    if len(valid) == 0:
        raise ScientificAbort("No valid canonical branch root found; the wrong Google account may be mounted")
    if len(valid) > 1:
        raise ScientificAbort("Multiple valid canonical branch roots found: " + ", ".join(map(str, valid)))
    return valid[0]


def load_stage26_module(repository_root: Path) -> Any:
    path = repository_root / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"
    if not path.is_file():
        raise ScientificAbort(f"Frozen Stage 2.6M implementation missing: {path}")
    name = "stage26_frozen_for_stage4m"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScientificAbort("Cannot import frozen Stage 2.6M implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Stage4Config:
    branch_root: str = ""
    repository_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parent))
    output_dir: str = ""
    profile: str = "full"
    seeds: Tuple[int, ...] = SEEDS
    max_epochs: int = 40
    minimum_epochs: int = 5
    patience: int = 8
    batch_size: int = 256
    eval_batch_size: int = 1024
    samples_per_tx: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    amp_enabled: bool = True
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    local_cache_root: str = "/content/wisig_stage4m_cache"
    resume: bool = True
    device: str = "auto"
    stage_start: int = 1
    stage_end: int = 12

    @property
    def branch_root_path(self) -> Path:
        return Path(self.branch_root).expanduser().resolve()

    @property
    def repository_root_path(self) -> Path:
        return Path(self.repository_root).expanduser().resolve()

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else self.branch_root_path / "06_surrogate_kd"

    def validate(self) -> None:
        if tuple(self.seeds) != SEEDS:
            raise ScientificAbort("Stage 4M seeds must be exactly 42, 123, 2026")
        if self.profile not in {"full", "pilot"}:
            raise ValueError("profile must be full or pilot")
        if self.output_root.parent != self.branch_root_path or self.output_root.name != "06_surrogate_kd":
            raise ScientificAbort("Stage 4M output must be <canonical-root>/06_surrogate_kd")
        if not 1 <= self.stage_start <= self.stage_end <= 12:
            raise ValueError("stage range must be within 1..12")
        if self.batch_size != 256 or self.samples_per_tx != 4:
            raise ScientificAbort("Frozen Stage 4M batch/samples-per-Tx policy changed")
        if self.profile == "pilot":
            self.max_epochs = min(self.max_epochs, 2)
            self.minimum_epochs = 1
            self.patience = 1

    def scientific_payload(self) -> Dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "seeds": list(self.seeds), "arms": list(ARMS), "objectives": ARM_OBJECTIVES,
            "temperature": KD_TEMPERATURE, "student_width_multiplier": 0.5,
            "student_embedding_dim": STUDENT_EMBEDDING_DIM, "max_epochs": self.max_epochs,
            "minimum_epochs": self.minimum_epochs, "patience": self.patience,
            "batch_size": self.batch_size, "samples_per_tx": self.samples_per_tx,
            "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "selection_partition": "p0", "training_partition": "train_known",
        }

    def configuration_sha256(self) -> str:
        return sha256_object(self.scientific_payload())


class Stage35StrictGuard:
    """Deny Stage 3.5M strict results and all strict benchmark reads."""

    def __init__(self, branch_root: Path, output_root: Path):
        self.branch_root, self.output_root = branch_root.resolve(), output_root.resolve()
        self._counters = {key: 0 for key in STAGE35_COUNTER_KEYS}
        self._allowed_indices: Dict[str, np.ndarray] = {}

    @staticmethod
    def is_strict_path(path: Path) -> bool:
        token = path.as_posix().lower()
        return any(item in token for item in STRICT_DENY_TOKENS)

    def reject(self, kind: str, detail: str) -> None:
        key = f"stage35_strict_{kind}_read_violations" if kind != "selection" else "stage35_strict_selection_violations"
        if key not in self._counters:
            key = "stage35_strict_metric_read_violations"
        self._counters[key] += 1
        raise ScientificAbort(f"STAGE35_STRICT_ACCESS_VIOLATION[{kind}]: {detail}")

    def authorize_metadata(self, path: Path) -> None:
        if path.name not in STAGE35_METADATA_ALLOWLIST:
            self.reject("metric", str(path))

    def forbid_data_path(self, path: Path, operation: str) -> None:
        if self.is_strict_path(path):
            kind = "index" if "indice" in path.name.lower() else "signal"
            self.reject(kind, f"{operation}: {path}")

    def register_allowed_indices(self, partition: str, indices: np.ndarray) -> None:
        if partition not in EXPECTED_COUNTS:
            self.reject("index", f"unapproved partition {partition}")
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ScientificAbort(f"Invalid authorized indices for {partition}")
        self._allowed_indices[partition] = values

    def authorize_rows(self, partition: str, indices: np.ndarray, operation: str) -> None:
        if partition not in self._allowed_indices:
            self.reject("signal", f"unregistered partition {partition} in {operation}")
        if not np.isin(np.asarray(indices, dtype=np.int64), self._allowed_indices[partition]).all():
            self.reject("signal", f"unauthorized rows in {operation}")

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def assert_zero(self) -> None:
        if any(self._counters.values()):
            raise ScientificAbort(f"Nonzero Stage 3.5M leakage counters: {self._counters}")

    def scan_output(self) -> None:
        if not self.output_root.exists():
            return
        for path in self.output_root.rglob("*"):
            if path.is_file() and self.is_strict_path(path):
                self.reject("metric", f"forbidden output {path}")


def even_width(value: int, multiplier: float = 0.5, minimum: int = 16) -> int:
    scaled = max(minimum, int(round(value * multiplier)))
    return scaled if scaled % 2 == 0 else scaled + 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 5, stride, dilation)
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class WiSigSurrogateNet(nn.Module):
    """Deterministic half-width topology derived from the frozen teacher."""

    def __init__(self, num_classes: int = 98, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        c16, c32, c64, c128 = (even_width(x) for x in (32, 64, 128, 256))
        self.iq_mixer = nn.Sequential(
            nn.Conv1d(2, c16, 1, bias=False), nn.GroupNorm(8, c16), nn.SiLU(inplace=True),
            ConvNormAct(c16, c32, 7, stride=2),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(c32, c32, dilation=1, dropout=dropout),
            ResidualTemporalBlock(c32, c64, stride=2, dilation=1, dropout=dropout),
            ResidualTemporalBlock(c64, c64, dilation=2, dropout=dropout),
            ResidualTemporalBlock(c64, c128, stride=2, dilation=1, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * c128, c128), nn.LayerNorm(c128), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(c128, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim, self.num_classes = embedding_dim, num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.temporal(self.iq_mixer(x))
        pooled = torch.cat((features.mean(-1), features.float().var(-1, unbiased=False).add(1e-8).sqrt().to(features.dtype)), 1)
        raw = self.projection(pooled)
        return {"logits": self.classifier(raw), "embedding_raw": raw, "embedding_normalized": F.normalize(raw.float(), dim=1, eps=1e-8)}


def model_signature(model: nn.Module) -> str:
    return sha256_object([(name, list(value.shape), str(value.dtype)) for name, value in model.state_dict().items()])


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def student_layer_table(model: WiSigSurrogateNet) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.GroupNorm, nn.LayerNorm, nn.SiLU, nn.Dropout)):
            row: Dict[str, Any] = {"name": name, "type": type(module).__name__}
            if isinstance(module, nn.Conv1d):
                row.update({"in": module.in_channels, "out": module.out_channels, "kernel": list(module.kernel_size), "stride": list(module.stride), "dilation": list(module.dilation)})
            elif isinstance(module, nn.Linear):
                row.update({"in": module.in_features, "out": module.out_features})
            rows.append(row)
    return rows


def architecture_freeze_payload() -> Dict[str, Any]:
    model, auxiliary = WiSigSurrogateNet(), nn.Linear(64, 128)
    deployed, aux = model_parameter_count(model), model_parameter_count(auxiliary)
    return {
        "status": "PASS" if deployed <= 0.40 * EXPECTED_TEACHER_PARAMETERS else "FAIL",
        "model": "WiSigSurrogateNet", "derivation": "frozen teacher topology at width_multiplier=0.50",
        "input_shape": ["B", 2, 256], "width_multiplier": 0.5, "even_rounding": True,
        "minimum_hidden_width": 16, "layer_table": student_layer_table(model),
        "native_embedding_dimension": 64, "classifier_dimension": 98,
        "auxiliary_projection": {"shape": [64, 128], "training_only": True, "parameters": aux},
        "deployed_parameter_count": deployed, "training_only_auxiliary_parameter_count": aux,
        "teacher_parameter_count": EXPECTED_TEACHER_PARAMETERS,
        "deployed_to_teacher_ratio": deployed / EXPECTED_TEACHER_PARAMETERS,
        "compression_ratio_teacher_over_student": EXPECTED_TEACHER_PARAMETERS / deployed,
        "architecture_signature_sha256": model_signature(model),
    }


def kd_objective_policy() -> Dict[str, Any]:
    return {
        "version": PIPELINE_VERSION, "temperature": KD_TEMPERATURE, "arms": ARM_OBJECTIVES,
        "ce": "cross_entropy(student_logits,true_known_label)",
        "kd": "T^2*kl_div(log_softmax(student_logits/T),softmax(teacher_logits/T),batchmean)",
        "representation": "mean(1-cosine(normalize(student_projection_128),normalize(detached_teacher_embedding_128)))",
        "prototype": "mean(1-cosine(normalize(student_projection_128),normalized_train_known_teacher_prototype[true_class]))",
        "teacher_targets_detached": True, "training_partition": "train_known",
        "weight_search": False, "temperature_search": False,
    }


def training_target_policy() -> Dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "training_partition": "train_known",
        "augmentation_application_count_per_batch": 1,
        "training_sample_kd_target_source": "ONLINE_TEACHER_FORWARD_ON_SHARED_AUGMENTED_INPUT",
        "teacher_student_input_identity": "EXACT_SAME_AUGMENTED_TENSOR",
        "k0_teacher_forward": "DISABLED",
        "k1_k3_teacher_forward": "FROZEN_INFERENCE_MODE_ON_SHARED_AUGMENTED_INPUT",
        "train_known_clean_cache_used_for_sample_kd": False,
        "teacher_prototype_source": "TRAIN_KNOWN_CLEAN_TEACHER_EMBEDDINGS",
        "teacher_prototypes_used_by": ["K3"],
        "p0_p3_clean_cache_role": "SEMANTIC_EVALUATION_ONLY",
        "teacher_targets_detached": True,
    }


def amp_runtime_safety_policy() -> Dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "runtime_safety_only": True,
        "scientific_independent_variable": False,
        "amp_enabled_on_cuda": True,
        "overflow_policy": "SKIP_STEP_BACKOFF_AND_RETRY_NEXT_BATCH",
        "max_consecutive_amp_overflows": MAX_CONSECUTIVE_AMP_OVERFLOWS,
        "non_amp_nonfinite_gradient": "ABORT",
        "optimizer_step_on_overflow": False,
        "scheduler_scope": "EPOCH",
        "gradient_clip_norm": 5.0,
        "legacy_checkpoint_policy": {
            "previous_executable_sha256": PRE_AMP_HOTFIX_EXECUTABLE_SHA256,
            "action": "REJECT_AND_RESTART_FROM_SCRATCH",
            "scope": {"arm": "K0", "seed": 42, "maximum_completed_epoch": 1},
            "reason": "new executable and predecessor/preflight provenance invalidate the pre-hotfix checkpoint",
        },
    }


def new_amp_runtime_state() -> Dict[str, int]:
    return {
        "consecutive_amp_overflows": 0,
        "total_amp_overflows": 0,
        "consecutive_amp_overflow_peak": 0,
        "total_batches_seen": 0,
        "total_optimizer_steps_completed": 0,
        "total_amp_overflow_skipped_steps": 0,
    }


def new_epoch_amp_accounting() -> Dict[str, int]:
    return {
        "batches_seen": 0,
        "optimizer_steps_completed": 0,
        "amp_overflow_skipped_steps": 0,
        "consecutive_amp_overflow_peak": 0,
        "total_amp_overflows": 0,
    }


def gradients_finite(parameters: Sequence[torch.nn.Parameter], gradient_norm: torch.Tensor) -> bool:
    if not bool(torch.isfinite(gradient_norm).item()):
        return False
    for parameter in parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return False
    return True


def apply_optimizer_step_with_amp_policy(
    optimizer: torch.optim.Optimizer, scaler: Any, use_amp: bool, finite_gradients: bool,
    runtime_state: Dict[str, int], epoch_state: Dict[str, int], context: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Apply one optimizer decision under the frozen AMP runtime-safety policy."""
    runtime_state["total_batches_seen"] += 1
    epoch_state["batches_seen"] += 1
    if not finite_gradients:
        if not use_amp:
            raise ScientificAbort(f"Non-finite gradients with AMP disabled for {context}")
        old_scale = float(scaler.get_scale())
        backoff = float(scaler.get_backoff_factor()) if hasattr(scaler, "get_backoff_factor") else 0.5
        new_scale = old_scale * backoff
        # Use GradScaler's public manual-scale update path so an overflowed batch can
        # never reach optimizer.step, even if the non-finite condition was detected
        # from the clipped total norm rather than GradScaler's found_inf bookkeeping.
        scaler.update(new_scale=new_scale)
        optimizer.zero_grad(set_to_none=True)
        runtime_state["consecutive_amp_overflows"] += 1
        runtime_state["total_amp_overflows"] += 1
        runtime_state["total_amp_overflow_skipped_steps"] += 1
        runtime_state["consecutive_amp_overflow_peak"] = max(
            runtime_state["consecutive_amp_overflow_peak"], runtime_state["consecutive_amp_overflows"]
        )
        epoch_state["amp_overflow_skipped_steps"] += 1
        epoch_state["consecutive_amp_overflow_peak"] = max(
            epoch_state["consecutive_amp_overflow_peak"], runtime_state["consecutive_amp_overflows"]
        )
        epoch_state["total_amp_overflows"] = runtime_state["total_amp_overflows"]
        if logger is not None:
            logger.warning(
                "AMP overflow skipped | %s | old_scale=%.1f new_scale=%.1f consecutive=%d total=%d",
                context, old_scale, new_scale, runtime_state["consecutive_amp_overflows"], runtime_state["total_amp_overflows"],
            )
        if runtime_state["consecutive_amp_overflows"] > MAX_CONSECUTIVE_AMP_OVERFLOWS:
            raise ScientificAbort(
                f"Exceeded {MAX_CONSECUTIVE_AMP_OVERFLOWS} consecutive AMP overflows for {context}"
            )
        return False
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runtime_state["consecutive_amp_overflows"] = 0
    runtime_state["total_optimizer_steps_completed"] += 1
    epoch_state["optimizer_steps_completed"] += 1
    epoch_state["total_amp_overflows"] = runtime_state["total_amp_overflows"]
    return True


def assert_epoch_has_optimizer_step(epoch_state: Mapping[str, int], context: str) -> None:
    if int(epoch_state.get("optimizer_steps_completed", 0)) <= 0:
        raise ScientificAbort(f"Epoch completed with zero optimizer steps for {context}")


def shared_augmented_training_forward(
    arm: str, raw_input: torch.Tensor, augmentation: Any, augmentation_generator: torch.Generator,
    student: nn.Module, teacher: Optional[nn.Module], use_amp: bool,
) -> Tuple[Mapping[str, torch.Tensor], Optional[Mapping[str, torch.Tensor]], torch.Tensor]:
    """Augment once and present the exact same tensor to required model forwards."""
    if arm not in ARM_OBJECTIVES:
        raise ScientificAbort(f"Unknown KD arm: {arm}")
    augmented = augmentation(raw_input, augmentation_generator)
    teacher_output: Optional[Mapping[str, torch.Tensor]] = None
    if arm != "K0":
        if teacher is None:
            raise ScientificAbort(f"{arm} requires the frozen teacher forward")
        with torch.inference_mode(), torch.autocast(
            device_type=raw_input.device.type, dtype=torch.float16, enabled=use_amp
        ):
            teacher_output = teacher(augmented)
        teacher_output = {name: value.detach().clone() for name, value in teacher_output.items()}
    with torch.autocast(device_type=raw_input.device.type, dtype=torch.float16, enabled=use_amp):
        student_output = student(augmented)
    return student_output, teacher_output, augmented


def compute_kd_losses(
    arm: str, student: Mapping[str, torch.Tensor], teacher_logits: Optional[torch.Tensor],
    teacher_embedding: Optional[torch.Tensor], labels: torch.Tensor, auxiliary: Optional[nn.Module],
    teacher_prototypes: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if arm not in ARM_OBJECTIVES:
        raise ScientificAbort(f"Unknown KD arm: {arm}")
    logits, weights = student["logits"], ARM_OBJECTIVES[arm]
    labels = labels.to(logits.device, dtype=torch.long)
    ce = F.cross_entropy(logits.float(), labels)
    kd = logits.new_zeros(())
    repr_loss = logits.new_zeros(())
    proto_loss = logits.new_zeros(())
    if weights["kd"] or weights["repr"]:
        if teacher_logits is None or teacher_embedding is None:
            raise ScientificAbort(f"{arm} requires online teacher targets")
        teacher_logits = teacher_logits.detach().to(logits.device, dtype=torch.float32)
        teacher_embedding = teacher_embedding.detach().to(logits.device, dtype=torch.float32)
    if weights["kd"]:
        teacher_probs = F.softmax(teacher_logits / KD_TEMPERATURE, dim=1)
        student_log_probs = F.log_softmax(logits.float() / KD_TEMPERATURE, dim=1)
        kd = KD_TEMPERATURE ** 2 * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    if weights["repr"] or weights["proto"]:
        if auxiliary is None:
            raise ScientificAbort(f"{arm} requires the training-only 64->128 projection")
        projected = F.normalize(auxiliary(student["embedding_raw"].float()), dim=1, eps=1e-8)
        if weights["repr"]:
            repr_loss = (1.0 - F.cosine_similarity(projected, F.normalize(teacher_embedding, dim=1), dim=1)).mean()
        if weights["proto"]:
            if teacher_prototypes is None:
                raise ScientificAbort("K3 requires Train Known teacher prototypes")
            targets = teacher_prototypes.detach().to(projected.device)[labels]
            proto_loss = (1.0 - F.cosine_similarity(projected, targets, dim=1)).mean()
    losses = {"ce": ce, "kd": kd, "repr": repr_loss, "proto": proto_loss}
    total = sum(weights[name] * losses[name] for name in losses)
    return total, losses


def p0_epoch_better(candidate: Mapping[str, float], incumbent: Optional[Mapping[str, float]], tolerance: float = 0.002) -> bool:
    if incumbent is None:
        return True
    ca, ia = candidate["teacher_student_top1_agreement"], incumbent["teacher_student_top1_agreement"]
    if ca > ia + tolerance:
        return True
    if ia > ca + tolerance:
        return False
    if candidate["teacher_student_kl"] != incumbent["teacher_student_kl"]:
        return candidate["teacher_student_kl"] < incumbent["teacher_student_kl"]
    if candidate["student_fixed98_macro_f1"] != incumbent["student_fixed98_macro_f1"]:
        return candidate["student_fixed98_macro_f1"] > incumbent["student_fixed98_macro_f1"]
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def choose_arm(summary: Mapping[str, Mapping[str, float]], tolerance: float = 0.002) -> str:
    remaining = list(ARMS)
    best_agreement = max(summary[arm]["median_agreement"] for arm in remaining)
    remaining = [arm for arm in remaining if best_agreement - summary[arm]["median_agreement"] <= tolerance]
    best_kl = min(summary[arm]["median_kl"] for arm in remaining)
    remaining = [arm for arm in remaining if math.isclose(summary[arm]["median_kl"], best_kl, rel_tol=0.0, abs_tol=1e-12)]
    best_std = min(summary[arm]["std_agreement"] for arm in remaining)
    remaining = [arm for arm in remaining if math.isclose(summary[arm]["std_agreement"], best_std, rel_tol=0.0, abs_tol=1e-12)]
    best_f1 = max(summary[arm]["median_f1"] for arm in remaining)
    remaining = [arm for arm in remaining if math.isclose(summary[arm]["median_f1"], best_f1, rel_tol=0.0, abs_tol=1e-12)]
    return min(remaining, key=ARMS.index)


def choose_seed(rows: Sequence[Mapping[str, float]], tolerance: float = 0.002) -> int:
    remaining = list(rows)
    best = max(row["teacher_student_top1_agreement"] for row in remaining)
    remaining = [row for row in remaining if best - row["teacher_student_top1_agreement"] <= tolerance]
    best_kl = min(row["teacher_student_kl"] for row in remaining)
    remaining = [row for row in remaining if math.isclose(row["teacher_student_kl"], best_kl, rel_tol=0.0, abs_tol=1e-12)]
    best_f1 = max(row["student_fixed98_macro_f1"] for row in remaining)
    remaining = [row for row in remaining if math.isclose(row["student_fixed98_macro_f1"], best_f1, rel_tol=0.0, abs_tol=1e-12)]
    return min(int(row["seed"]) for row in remaining)


def classification_from_arrays(logits: np.ndarray, labels: np.ndarray, teacher_logits: Optional[np.ndarray] = None) -> Dict[str, float]:
    logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    probabilities = torch.softmax(logits_t, dim=1)
    prediction = logits_t.argmax(1)
    confusion = np.zeros((98, 98), dtype=np.int64)
    np.add.at(confusion, (labels_t.numpy(), prediction.numpy()), 1)
    support, predicted = confusion.sum(1).astype(float), confusion.sum(0).astype(float)
    tp = np.diag(confusion).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    confidence = probabilities.max(1).values.numpy()
    correct = prediction.numpy() == labels_t.numpy()
    ece = 0.0
    for low in np.linspace(0, 1, 16)[:-1]:
        high = low + 1 / 15
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    metrics = {
        "samples": float(len(labels)), "student_accuracy": float(correct.mean()),
        "student_fixed98_macro_f1": float(f1.mean()),
        "student_nll": float(F.cross_entropy(logits_t, labels_t).item()), "student_ece": float(ece),
    }
    if teacher_logits is not None:
        teacher_t = torch.from_numpy(np.asarray(teacher_logits, dtype=np.float32))
        teacher_prob, student_prob = torch.softmax(teacher_t, 1), probabilities
        teacher_prediction = teacher_t.argmax(1)
        middle = 0.5 * (teacher_prob + student_prob)
        kl = F.kl_div(torch.log(student_prob.clamp_min(1e-12)), teacher_prob, reduction="batchmean")
        js = 0.5 * (
            F.kl_div(torch.log(middle.clamp_min(1e-12)), teacher_prob, reduction="batchmean")
            + F.kl_div(torch.log(middle.clamp_min(1e-12)), student_prob, reduction="batchmean")
        )
        metrics.update({
            "teacher_student_top1_agreement": float((teacher_prediction == prediction).float().mean().item()),
            "teacher_student_kl": float(kl.item()), "teacher_student_js": float(js.item()),
        })
    return metrics


def teacher_state_value_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


class Stage4Pipeline:
    def __init__(self, config: Stage4Config):
        self.config = config
        self.config.validate()
        self.root, self.output = config.branch_root_path, config.output_root
        self.repository = config.repository_root_path
        self.script_sha = sha256_file(Path(__file__).resolve())
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.guard = Stage35StrictGuard(self.root, self.output)
        self.stage26 = load_stage26_module(self.repository)
        self.benchmark: Any = None
        self.teacher: Optional[nn.Module] = None
        self.teacher_state_value_hash: Optional[str] = None
        self.predecessor_lock_sha: Optional[str] = None
        self.architecture_freeze_sha: Optional[str] = None
        self.objective_policy_sha: Optional[str] = None
        self.training_target_policy_sha: Optional[str] = None
        self.amp_runtime_safety_policy_sha: Optional[str] = None
        self.selection_lock_hash: Optional[str] = None
        self._setup()

    def _setup(self) -> None:
        for name in (
            "configs", "logs", "manifests", "checkpoints", "teacher_cache_manifests", "metrics",
            "tables", "statistics", "figures", "reports", "performance", "publication",
        ):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            (self.output / "checkpoints" / arm).mkdir(parents=True, exist_ok=True)
        (self.output / "checkpoints" / "canonical").mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("stage4m")
        self.logger.handlers.clear(); self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter); self.logger.addHandler(stream)
        file_handler = logging.FileHandler(self.output / "logs" / "stage4m.log", encoding="utf-8"); file_handler.setFormatter(formatter); self.logger.addHandler(file_handler)

    @property
    def stage3_root(self) -> Path:
        return self.root / "04_canonical_teacher"

    @property
    def stage35_root(self) -> Path:
        return self.root / "05_zero_day_open_set"

    @property
    def teacher_path(self) -> Path:
        return self.stage3_root / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"

    def stage_checkpoint(self, stage: int) -> Path:
        return self.output / "manifests" / f"STAGE_{stage:02d}_CHECKPOINT.json"

    def core_provenance_inputs(self) -> List[Path]:
        return [
            self.output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json",
            self.output / "manifests" / "STUDENT_ARCHITECTURE_FREEZE.json",
            self.output / "manifests" / "KD_OBJECTIVE_POLICY.json",
            self.output / "manifests" / "TRAINING_TARGET_POLICY.json",
            self.output / "manifests" / "AMP_RUNTIME_SAFETY_POLICY.json",
            self.teacher_path,
            self.root / "01_benchmark_engineering" / "benchmark" / f"{CANONICAL_BENCHMARK}.h5",
        ]

    def complete_stage(self, stage: int, name: str, outputs: Sequence[Path], inputs: Sequence[Path] = ()) -> None:
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise ScientificAbort(f"Stage {stage:02d} missing outputs: {missing}")
        missing_inputs = [str(path) for path in inputs if not path.is_file()]
        if missing_inputs:
            raise ScientificAbort(f"Stage {stage:02d} missing bound inputs: {missing_inputs}")
        self.hydrate_provenance()
        payload = {
            "stage": stage, "stage_name": name, "pipeline_version": PIPELINE_VERSION,
            "executable_sha256": self.script_sha, "configuration_sha256": self.config.configuration_sha256(),
            "predecessor_lock_sha256": self.predecessor_lock_sha,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "architecture_freeze_sha256": self.architecture_freeze_sha if stage >= 2 else None,
            "objective_policy_sha256": self.objective_policy_sha if stage >= 2 else None,
            "training_target_policy_sha256": self.training_target_policy_sha if stage >= 2 else None,
            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha if stage >= 2 else None,
            "selection_lock_sha256": self.selection_lock_hash if stage >= 9 else None,
            "inputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in inputs],
            "outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
            "completed_at": utc_now(), "status": "PASS",
        }
        atomic_json(self.stage_checkpoint(stage), payload, self.output)

    def stage_current(self, stage: int) -> bool:
        checkpoint = self.stage_checkpoint(stage)
        if not checkpoint.is_file():
            return False
        try:
            self.hydrate_provenance()
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            expected = {
                "stage": stage,
                "status": "PASS",
                "pipeline_version": PIPELINE_VERSION,
                "configuration_sha256": self.config.configuration_sha256(),
                "executable_sha256": self.script_sha,
                "teacher_sha256": EXPECTED_TEACHER_SHA256,
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
                "predecessor_lock_sha256": self.predecessor_lock_sha,
                "architecture_freeze_sha256": self.architecture_freeze_sha if stage >= 2 else None,
                "objective_policy_sha256": self.objective_policy_sha if stage >= 2 else None,
                "training_target_policy_sha256": self.training_target_policy_sha if stage >= 2 else None,
            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha if stage >= 2 else None,
                "selection_lock_sha256": self.selection_lock_hash if stage >= 9 else None,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                return False
            current = bound_file_rows_current(payload["inputs"], require_size=False)
            current = current and bound_file_rows_current(payload["outputs"], require_size=True)
            dependency_stages = {
                1: (), 2: (1,), 3: (1, 2),
                4: (1, 2, 3), 5: (1, 2, 3), 6: (1, 2, 3), 7: (1, 2, 3),
                8: (4, 5, 6, 7), 9: (8,), 10: (8, 9), 11: (8, 9, 10),
                12: tuple(range(1, 12)),
            }
            current = current and all(self.stage_current(required) for required in dependency_stages[stage])
            if stage == 12:
                current = current and completed_state_current(self.output)
            return current
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def selection_lock_sha(self) -> Optional[str]:
        path = self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"
        return sha256_file(path) if path.is_file() else None

    def verify_small_hash(self, path: Path, expected: str) -> None:
        if not path.is_file() or sha256_file(path) != expected:
            raise ScientificAbort(f"Frozen hash mismatch: {path}")

    def verify_stage35_metadata(self) -> Dict[str, Any]:
        ready_path = self.stage35_root / "MANYTX_STAGE3_5M_READY.txt"
        final_path = self.stage35_root / "manifests" / "STAGE3_5M_FINAL_STATUS.json"
        hash_path = self.stage35_root / "manifests" / "STAGE3_5M_HASH_MANIFEST.json"
        recovery_path = self.stage35_root / "manifests" / "POST_LOCK_RECOVERY_MANIFEST.json"
        lock_path = self.stage35_root / "manifests" / "STRICT_ZERO_DAY_EVALUATION_LOCK.json"
        for path in (ready_path, final_path, hash_path, recovery_path, lock_path):
            self.guard.authorize_metadata(path)
            if not path.is_file(): raise ScientificAbort(f"Missing Stage 3.5M frozen metadata: {path}")
        ready, final = parse_ready(ready_path), json.loads(final_path.read_text(encoding="utf-8"))
        recovery, lock = json.loads(recovery_path.read_text(encoding="utf-8")), json.loads(lock_path.read_text(encoding="utf-8"))
        required_ready = {
            "marker": "MANYTX_STAGE3_5M_READY", "teacher_version": "1.0", "teacher_seed": "123",
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "strict_protocol": "ZD_STRICT", "canonical_scorer": "ALL_PREDECLARED", "post_lock_recovery": "YES",
            "strict_signal_reinference": "NO", "strict_scores_recomputed": "NO", "scorer_refit": "NO",
            "threshold_refit": "NO", "policy_changed": "NO", "strict_labels_loaded": "NO",
            "strict_shift_subset_of_main": "YES", "teacher_retrained": "NO", "surrogate_training_performed": "NO",
            "xai_performed": "NO", "next_stage": "STAGE_4M",
        }
        if any(ready.get(key) != value for key, value in required_ready.items()):
            raise ScientificAbort("Stage 3.5M READY metadata mismatch")
        for key in (
            "strict_zero_day_signal_read_violations", "strict_zero_day_label_read_violations",
            "strict_zero_day_embedding_read_violations", "strict_zero_day_metric_read_violations",
            "strict_zero_day_threshold_read_violations", "strict_zero_day_fit_violations",
        ):
            if int(ready.get(key, -1)) != 0: raise ScientificAbort(f"Stage 3.5M counter nonzero: {key}")
        invariants = {
            "original_lock_modified": False, "strict_main_store_modified": False,
            "strict_shift_store_modified": False, "strict_scores_recomputed": False,
            "strict_signals_re_read": False, "teacher_inference_performed": False,
            "scorer_refit": False, "threshold_refit": False, "policy_changed": False, "strict_labels_loaded": False,
        }
        if any(recovery.get(key) is not value for key, value in invariants.items()):
            raise ScientificAbort("Stage 3.5M recovery provenance mismatch")
        if final.get("status") != "MANYTX_STAGE3_5M_READY" or lock.get("teacher_sha256") != EXPECTED_TEACHER_SHA256:
            raise ScientificAbort("Stage 3.5M final/lock metadata mismatch")
        return {"ready": ready, "final": final, "recovery": recovery, "lock": lock, "hash_manifest_sha256": sha256_file(hash_path)}

    def stage_01(self) -> None:
        preflight = self.output / "manifests" / "STAGE4M_PREFLIGHT.json"
        if not preflight.is_file():
            raise ScientificAbort("Fresh Stage 4M --preflight is required before canonical execution")
        preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
        if preflight_payload.get("status") != "STAGE4M_PREFLIGHT_PASS" or preflight_payload.get("executable_sha256") != self.script_sha or preflight_payload.get("configuration_sha256") != self.config.configuration_sha256():
            raise ScientificAbort("Stage 4M preflight is stale for the current executable/configuration")
        benchmark = self.root / "01_benchmark_engineering" / "benchmark" / f"{CANONICAL_BENCHMARK}.h5"
        stage2m = self.root / "02_benchmark_diagnostics" / "manifests" / "HASH_MANIFEST.json"
        stage26 = self.root / "03_representation_ablation" / "manifests" / "HASH_MANIFEST.json"
        stage3_ready = self.stage3_root / "MANYTX_STAGE3M_READY.txt"
        stage3_hash = self.stage3_root / "manifests" / "STAGE3M_HASH_MANIFEST.json"
        if not benchmark.is_file() or sha256_file(benchmark) != EXPECTED_BENCHMARK_SHA256:
            raise ScientificAbort("Canonical benchmark hash mismatch")
        self.verify_small_hash(stage2m, EXPECTED_STAGE2M_MANIFEST_SHA256)
        self.verify_small_hash(stage26, EXPECTED_STAGE26_ARTIFACT_SHA256)
        self.verify_small_hash(stage3_hash, EXPECTED_STAGE3M_MANIFEST_SHA256)
        stage3 = parse_ready(stage3_ready)
        if stage3.get("marker") != "MANYTX_STAGE3M_READY" or stage3.get("selected_seed") != "123" or stage3.get("canonical_teacher_sha256") != EXPECTED_TEACHER_SHA256:
            raise ScientificAbort("Stage 3M READY identity mismatch")
        if sha256_file(self.teacher_path) != EXPECTED_TEACHER_SHA256:
            raise ScientificAbort("Canonical teacher checkpoint hash mismatch")
        state_path = self.stage3_root / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt"
        if sha256_file(state_path) != EXPECTED_TEACHER_STATE_SHA256:
            raise ScientificAbort("Canonical teacher state-dict file hash mismatch")
        stage35 = self.verify_stage35_metadata()
        bound = (preflight, stage3_ready, stage3_hash, self.teacher_path, benchmark,
                 self.stage35_root / "MANYTX_STAGE3_5M_READY.txt",
                 self.stage35_root / "manifests" / "STAGE3_5M_FINAL_STATUS.json",
                 self.stage35_root / "manifests" / "STAGE3_5M_HASH_MANIFEST.json",
                 self.stage35_root / "manifests" / "POST_LOCK_RECOVERY_MANIFEST.json")
        payload = {
            "status": "PASS", "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2m_hash_manifest_sha256": EXPECTED_STAGE2M_MANIFEST_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage3m_hash_manifest_sha256": EXPECTED_STAGE3M_MANIFEST_SHA256,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "teacher_state_dict_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "stage35_metadata_hash_manifest_sha256": stage35["hash_manifest_sha256"],
            "bound_files": [{"path": str(path), "sha256": sha256_file(path)} for path in bound],
            "stage35_strict_scientific_results_opened": False, "strict_index_contents_opened": False,
        }
        path = self.output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json"
        atomic_json(path, payload, self.output); self.predecessor_lock_sha = sha256_file(path)
        self.complete_stage(1, "Frozen predecessor verification", [path], list(bound))

    def load_teacher(self) -> nn.Module:
        if self.teacher is not None: return self.teacher
        payload = safe_torch_load(self.teacher_path, "cpu")
        required = {
            "selected_seed": 123, "source_arm": "A3", "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "architecture_signature": EXPECTED_TEACHER_ARCHITECTURE_SHA256, "parameter_count": EXPECTED_TEACHER_PARAMETERS,
            "embedding_dimension": 128, "known_classes": 98,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise ScientificAbort("Canonical teacher checkpoint metadata mismatch")
        model = self.stage26.WiSigRepresentationNet(num_classes=98, embedding_dim=128, dropout=0.1)
        model.load_state_dict(payload["model_state"], strict=True); model.eval(); model.requires_grad_(False)
        if sum(p.numel() for p in model.parameters()) != EXPECTED_TEACHER_PARAMETERS:
            raise ScientificAbort("Teacher parameter-count mismatch")
        self.teacher_state_value_hash = teacher_state_value_sha(model)
        self.teacher = model.to(self.device)
        return self.teacher

    def stage_02(self) -> None:
        teacher = self.load_teacher()
        state_before = teacher_state_value_sha(teacher)
        with torch.inference_mode():
            sample = torch.zeros(2, 2, 256, device=self.device)
            first, second = teacher(sample), teacher(sample)
        teacher_checks = {
            "parameter_count": model_parameter_count(teacher) == EXPECTED_TEACHER_PARAMETERS,
            "logits_shape": tuple(first["logits"].shape) == (2, 98),
            "embedding_shape": tuple(first["embedding_normalized"].shape) == (2, 128),
            "normalized_embedding": torch.allclose(first["embedding_normalized"].norm(dim=1), torch.ones(2, device=self.device), atol=1e-5),
            "deterministic_forward": torch.equal(first["logits"], second["logits"]),
            "gradients_disabled": not any(parameter.requires_grad for parameter in teacher.parameters()),
            "state_unchanged": teacher_state_value_sha(teacher) == state_before,
        }
        if not all(teacher_checks.values()): raise ScientificAbort(f"Teacher equivalence failed: {teacher_checks}")
        freeze, objective, targets = architecture_freeze_payload(), kd_objective_policy(), training_target_policy()
        amp_policy = amp_runtime_safety_policy()
        if freeze["status"] != "PASS": raise ScientificAbort("Deterministic student violates the 40% compression gate")
        equivalence_path = self.output / "manifests" / "STAGE3M_STAGE4M_TEACHER_EQUIVALENCE.json"
        architecture_path = self.output / "manifests" / "STUDENT_ARCHITECTURE_FREEZE.json"
        objective_path = self.output / "manifests" / "KD_OBJECTIVE_POLICY.json"
        target_policy_path = self.output / "manifests" / "TRAINING_TARGET_POLICY.json"
        amp_policy_path = self.output / "manifests" / "AMP_RUNTIME_SAFETY_POLICY.json"
        atomic_json(equivalence_path, {"status": "PASS", "checks": teacher_checks, "teacher_state_value_sha256": state_before,
                                      "frozen_stage3m_primitives": "verified during Stage 03 for P0-P3"}, self.output)
        atomic_json(architecture_path, freeze, self.output); atomic_json(objective_path, objective, self.output)
        atomic_json(target_policy_path, targets, self.output); atomic_json(amp_policy_path, amp_policy, self.output)
        self.architecture_freeze_sha, self.objective_policy_sha = sha256_file(architecture_path), sha256_file(objective_path)
        self.training_target_policy_sha = sha256_file(target_policy_path)
        self.amp_runtime_safety_policy_sha = sha256_file(amp_policy_path)
        self.complete_stage(
            2, "Teacher equivalence, architecture freeze, leakage audit",
            [equivalence_path, architecture_path, objective_path, target_policy_path, amp_policy_path],
            [self.output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json", self.teacher_path],
        )

    def ensure_context(self) -> Any:
        if self.benchmark is not None: return self.benchmark
        compatibility = self.stage26.Stage26Config(branch_root=str(self.root), device=str(self.device), output_dir=str(self.root / "03_representation_ablation"))
        self.benchmark = self.stage26.resolve_benchmark(compatibility, self.guard)
        return self.benchmark

    def cache_dir(self, partition: str) -> Path:
        return Path(self.config.local_cache_root).resolve() / "teacher_targets" / partition

    def cache_current(self, partition: str) -> bool:
        store, rows = self.cache_dir(partition), EXPECTED_COUNTS[partition]
        manifest = store / "store_manifest.json"
        required = {"logits.npy": (rows, 98), "embedding.npy": (rows, 128), "labels.npy": (rows,), "global_indices.npy": (rows,)}
        if (store / "INCOMPLETE").exists() or not manifest.is_file(): return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("complete") is not True or payload.get("teacher_sha256") != EXPECTED_TEACHER_SHA256 or payload.get("benchmark_sha256") != EXPECTED_BENCHMARK_SHA256:
                return False
            if payload.get("training_sample_kd_target_source") != "ONLINE_TEACHER_FORWARD_ON_SHARED_AUGMENTED_INPUT" or payload.get("train_known_clean_cache_used_for_sample_kd") is not False:
                return False
            return all(
                (store / name).is_file()
                and tuple(np.load(store / name, mmap_mode="r").shape) == shape
                and sha256_file(store / name) == payload["files"][name]["sha256"]
                and (store / name).stat().st_size == int(payload["files"][name]["bytes"])
                for name, shape in required.items()
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError): return False

    def build_loader(self, partition: str, batches: Optional[Sequence[Sequence[int]]] = None, seed: int = 0) -> Tuple[Any, DataLoader]:
        benchmark = self.ensure_context(); dataset = self.stage26.WiSigH5Dataset(benchmark, partition, self.guard)
        compatibility = self.stage26.Stage26Config(
            branch_root=str(self.root), output_dir=str(self.root / "03_representation_ablation"),
            batch_size=self.config.batch_size, eval_batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers, prefetch_factor=self.config.prefetch_factor,
        )
        return dataset, self.stage26.build_loader(dataset, compatibility, batches=batches, seed=seed)

    def prepare_teacher_cache(self, partition: str) -> Path:
        if partition not in NON_STRICT_CACHE_PARTITIONS: self.guard.reject("signal", f"teacher cache {partition}")
        store = self.cache_dir(partition)
        if self.cache_current(partition): return store
        store.mkdir(parents=True, exist_ok=True); atomic_text(store / "INCOMPLETE", utc_now() + "\n", store)
        teacher, benchmark = self.load_teacher(), self.ensure_context(); metadata = benchmark.partitions[partition]
        logits = np.lib.format.open_memmap(store / "logits.partial.npy", mode="w+", dtype=np.float16, shape=(len(metadata.indices), 98))
        embedding = np.lib.format.open_memmap(store / "embedding.partial.npy", mode="w+", dtype=np.float16, shape=(len(metadata.indices), 128))
        dataset, loader = self.build_loader(partition, seed=4_000_000 + list(NON_STRICT_CACHE_PARTITIONS).index(partition))
        cursor, use_amp = 0, self.config.amp_enabled and self.device.type == "cuda"
        with torch.inference_mode():
            for batch in loader:
                values = batch["x"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp): outputs = teacher(values)
                count = len(values); logits[cursor:cursor+count] = outputs["logits"].float().cpu().numpy().astype(np.float16)
                embedding[cursor:cursor+count] = outputs["embedding_normalized"].float().cpu().numpy().astype(np.float16); cursor += count
        dataset.close(); logits.flush(); embedding.flush(); del logits, embedding
        if cursor != len(metadata.indices): raise ScientificAbort(f"Teacher cache row mismatch for {partition}")
        os.replace(store / "logits.partial.npy", store / "logits.npy"); os.replace(store / "embedding.partial.npy", store / "embedding.npy")
        np.save(store / "labels.npy", metadata.labels.astype(np.int16), allow_pickle=False)
        np.save(store / "global_indices.npy", metadata.indices.astype(np.int64), allow_pickle=False)
        cache_files = {
            name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size}
            for name in ("logits.npy", "embedding.npy", "labels.npy", "global_indices.npy")
        }
        manifest = {
            "complete": True, "partition": partition, "rows": cursor, "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_state_sha256": EXPECTED_TEACHER_STATE_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "global_index_value_sha256": sha256_array_values(metadata.indices.astype(np.int64)),
            "logit_shape": [cursor, 98], "embedding_shape": [cursor, 128], "dtype": "float16",
            "cache_role": "TRAIN_KNOWN_CLEAN_PROTOTYPE_SOURCE" if partition == "train_known" else "SEMANTIC_EVALUATION_CACHE",
            "training_sample_kd_target_source": "ONLINE_TEACHER_FORWARD_ON_SHARED_AUGMENTED_INPUT",
            "train_known_clean_cache_used_for_sample_kd": False,
            "teacher_prototype_source": "TRAIN_KNOWN_CLEAN_TEACHER_EMBEDDINGS",
            "files": cache_files,
            "strict_rows": False, "generated_at": utc_now(),
        }
        atomic_json(store / "store_manifest.json", manifest, store); (store / "INCOMPLETE").unlink()
        return store

    def stage_03(self) -> None:
        outputs: List[Path] = []
        equivalence_rows = []
        for partition in NON_STRICT_CACHE_PARTITIONS:
            store = self.prepare_teacher_cache(partition); manifest_copy = self.output / "teacher_cache_manifests" / f"{partition}.json"
            payload = json.loads((store / "store_manifest.json").read_text(encoding="utf-8")); atomic_json(manifest_copy, payload, self.output); outputs.append(manifest_copy)
            if partition in KNOWN_PROTOCOLS:
                stage3_store = self.stage3_root / "embeddings" / "seed_123" / partition
                available = all((stage3_store / name).is_file() for name in ("logits.npy", "labels.npy", "global_indices.npy"))
                row: Dict[str, Any] = {"partition": partition, "frozen_primitives_available": available}
                if available:
                    current_logits = np.load(store / "logits.npy", mmap_mode="r")
                    frozen_logits = np.load(stage3_store / "logits.npy", mmap_mode="r")
                    current_labels = np.load(store / "labels.npy", mmap_mode="r"); frozen_labels = np.load(stage3_store / "labels.npy", mmap_mode="r")
                    current_indices = np.load(store / "global_indices.npy", mmap_mode="r"); frozen_indices = np.load(stage3_store / "global_indices.npy", mmap_mode="r")
                    row.update({
                        "row_count_equal": len(current_labels) == len(frozen_labels),
                        "global_indices_exact": np.array_equal(current_indices, frozen_indices),
                        "labels_exact": np.array_equal(current_labels, frozen_labels),
                        "predictions_exact": np.array_equal(np.asarray(current_logits).argmax(1), np.asarray(frozen_logits).argmax(1)),
                    })
                    metrics = classification_from_arrays(np.asarray(current_logits), np.asarray(current_labels))
                    row["accuracy"] = metrics["student_accuracy"]; row["fixed98_macro_f1"] = metrics["student_fixed98_macro_f1"]
                    row["accuracy_reference_delta"] = abs(row["accuracy"] - REFERENCE_TEACHER_METRICS[partition]["accuracy"])
                    row["fixed98_macro_f1_reference_delta"] = abs(row["fixed98_macro_f1"] - REFERENCE_TEACHER_METRICS[partition]["fixed98_macro_f1"])
                    if not all(row[key] for key in ("row_count_equal", "global_indices_exact", "labels_exact", "predictions_exact")) or row["accuracy_reference_delta"] > 5e-4 or row["fixed98_macro_f1_reference_delta"] > 5e-4:
                        raise ScientificAbort(f"Stage3M/Stage4M teacher equivalence failed for {partition}: {row}")
                else:
                    row["exact_equivalence"] = "NOT_AVAILABLE_IN_FROZEN_STAGE3M"
                equivalence_rows.append(row)
        equivalence_path = self.output / "manifests" / "STAGE3M_STAGE4M_TEACHER_EQUIVALENCE.json"
        atomic_json(equivalence_path, {"status": "PASS", "partitions": equivalence_rows, "teacher_sha256": EXPECTED_TEACHER_SHA256}, self.output); outputs.append(equivalence_path)
        self.complete_stage(
            3, "Non-strict teacher target preparation", outputs,
            [*self.core_provenance_inputs(), self.output / "manifests" / "STAGE_02_CHECKPOINT.json"],
        )

    def compatibility_config(self) -> Any:
        return self.stage26.Stage26Config(
            branch_root=str(self.root), output_dir=str(self.root / "03_representation_ablation"),
            batch_size=self.config.batch_size, samples_per_tx=self.config.samples_per_tx,
            eval_batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory, persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor, learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay, max_epochs=self.config.max_epochs,
            minimum_epochs=self.config.minimum_epochs, early_stopping_patience=self.config.patience,
            amp_enabled=self.config.amp_enabled,
        )

    def teacher_prototypes(self) -> torch.Tensor:
        store = self.prepare_teacher_cache("train_known")
        embedding = np.load(store / "embedding.npy", mmap_mode="r")
        labels = np.load(store / "labels.npy", mmap_mode="r")
        prototypes = np.zeros((98, 128), dtype=np.float64); counts = np.zeros(98, dtype=np.int64)
        for start in range(0, len(labels), 8192):
            block_labels = np.asarray(labels[start:start+8192], dtype=np.int64)
            block_embedding = np.asarray(embedding[start:start+8192], dtype=np.float32)
            np.add.at(prototypes, block_labels, block_embedding); np.add.at(counts, block_labels, 1)
        if (counts == 0).any(): raise ScientificAbort("Train Known teacher prototype is missing a class")
        prototypes /= counts[:, None]
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
        return torch.from_numpy(prototypes.astype(np.float32))

    def evaluate_student(self, model: nn.Module, partition: str, auxiliary: Optional[nn.Module] = None) -> Dict[str, float]:
        if partition not in KNOWN_PROTOCOLS: raise ScientificAbort(f"Selection/evaluation partition forbidden: {partition}")
        store = self.prepare_teacher_cache(partition)
        teacher_logits = np.load(store / "logits.npy", mmap_mode="r")
        teacher_embedding = np.load(store / "embedding.npy", mmap_mode="r")
        labels = np.load(store / "labels.npy", mmap_mode="r")
        student_logits = np.empty((len(labels), 98), dtype=np.float32)
        projected_cosine_sum, cursor = 0.0, 0
        dataset, loader = self.build_loader(partition, seed=4_100_000 + KNOWN_PROTOCOLS.index(partition))
        model.eval()
        if auxiliary is not None: auxiliary.eval()
        with torch.inference_mode():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True); count = len(x)
                output = model(x); student_logits[cursor:cursor+count] = output["logits"].float().cpu().numpy()
                if auxiliary is not None:
                    projected = F.normalize(auxiliary(output["embedding_raw"].float()), dim=1)
                    target = torch.from_numpy(np.asarray(teacher_embedding[cursor:cursor+count], dtype=np.float32)).to(self.device)
                    projected_cosine_sum += float(F.cosine_similarity(projected, F.normalize(target, dim=1), dim=1).sum().cpu())
                cursor += count
        dataset.close()
        if cursor != len(labels): raise ScientificAbort(f"Student evaluation row mismatch: {partition}")
        metrics = classification_from_arrays(student_logits, np.asarray(labels), np.asarray(teacher_logits))
        metrics["partition"] = partition
        metrics["projected_teacher_cosine"] = projected_cosine_sum / cursor if auxiliary is not None else math.nan
        return metrics

    def checkpoint_validation_fields(self, arm: str, seed: int) -> Dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION, "executable_sha256": self.script_sha,
            "configuration_sha256": self.config.configuration_sha256(),
            "objective_policy_sha256": self.objective_policy_sha,
            "training_target_policy_sha256": self.training_target_policy_sha,
            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
            "architecture_freeze_sha256": self.architecture_freeze_sha,
            "predecessor_lock_sha256": self.predecessor_lock_sha,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "teacher_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "arm": arm, "seed": seed,
            "objective": ARM_OBJECTIVES[arm], "student_architecture_signature": model_signature(WiSigSurrogateNet()),
        }

    def training_checkpoint_dir(self, arm: str, seed: int) -> Path:
        return self.output / "checkpoints" / arm / f"seed_{seed}"

    def validate_training_checkpoint(self, payload: Mapping[str, Any], arm: str, seed: int) -> None:
        expected = self.checkpoint_validation_fields(arm, seed)
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches: raise ScientificAbort(f"STALE_STAGE4M_CHECKPOINT {arm}/seed={seed}: {mismatches}")

    @staticmethod
    def known_pre_amp_checkpoint_requires_restart(payload: Mapping[str, Any], arm: str, seed: int) -> bool:
        return (
            payload.get("executable_sha256") == PRE_AMP_HOTFIX_EXECUTABLE_SHA256
            and payload.get("arm") == arm == "K0"
            and int(payload.get("seed", -1)) == seed == 42
            and int(payload.get("epoch", -1)) <= 1
            and payload.get("amp_runtime_safety_policy_sha256") in (None, "")
        )

    def discard_known_pre_amp_checkpoint(self, base: Path, payload: Mapping[str, Any], arm: str, seed: int) -> None:
        if not self.known_pre_amp_checkpoint_requires_restart(payload, arm, seed):
            raise ScientificAbort(f"STALE_STAGE4M_CHECKPOINT {arm}/seed={seed}: unknown incompatible checkpoint")
        latest, best_path = base / "latest.pt", base / "best.pt"
        audit = {
            "status": "REJECTED_AND_RESTARTED",
            "reason": "AMP overflow hotfix changed executable/runtime provenance; clean restart required",
            "previous_executable_sha256": payload.get("executable_sha256"),
            "previous_epoch": int(payload.get("epoch", -1)),
            "arm": arm, "seed": seed,
            "previous_latest_sha256": sha256_file(latest) if latest.is_file() else None,
            "previous_best_sha256": sha256_file(best_path) if best_path.is_file() else None,
            "new_amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
            "recorded_at": utc_now(),
        }
        audit_path = self.output / "manifests" / f"PRE_AMP_HOTFIX_RESTART_{arm}_SEED_{seed}.json"
        atomic_json(audit_path, audit, self.output)
        for path in (latest, best_path):
            if path.is_file():
                path.unlink()
        self.logger.warning("Rejected pre-AMP-hotfix checkpoint for %s seed %d; restarting from epoch 1", arm, seed)

    def save_training_checkpoint(
        self, path: Path, arm: str, seed: int, epoch: int, model: nn.Module,
        auxiliary: Optional[nn.Module], optimizer: torch.optim.Optimizer, scheduler: Any,
        scaler: Any, best_metrics: Optional[Mapping[str, float]], stale_epochs: int,
        loader_generator_state: Optional[torch.Tensor], exposure_sha: str,
        amp_runtime_state: Mapping[str, int], epoch_amp_accounting: Mapping[str, int],
    ) -> None:
        payload = {
            **self.checkpoint_validation_fields(arm, seed), "epoch": epoch,
            "student_state": model.state_dict(), "auxiliary_state": auxiliary.state_dict() if auxiliary is not None else None,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "best_p0_metrics": dict(best_metrics or {}),
            "early_stop_state": {"stale_epochs": stale_epochs}, "rng_state": capture_rng(),
            "dataloader_generator_state": loader_generator_state, "sampler_exposure_sha256": exposure_sha,
            "amp_runtime_state": dict(amp_runtime_state), "epoch_amp_accounting": dict(epoch_amp_accounting),
            "batches_seen": int(epoch_amp_accounting["batches_seen"]),
            "optimizer_steps_completed": int(epoch_amp_accounting["optimizer_steps_completed"]),
            "amp_overflow_skipped_steps": int(epoch_amp_accounting["amp_overflow_skipped_steps"]),
            "consecutive_amp_overflow_peak": int(epoch_amp_accounting["consecutive_amp_overflow_peak"]),
            "total_amp_overflows": int(amp_runtime_state["total_amp_overflows"]),
            "teacher_optimizer_owned": False, "teacher_targets_detached": True, "saved_at": utc_now(),
        }
        atomic_torch_save(path, payload, self.output)

    def create_training_objects(self, arm: str, seed: int) -> Tuple[nn.Module, Optional[nn.Module], Any, Any, Any]:
        seed_everything(seed)
        model = WiSigSurrogateNet().to(self.device)
        auxiliary = nn.Linear(64, 128).to(self.device) if arm in {"K2", "K3"} else None
        parameters = list(model.parameters()) + (list(auxiliary.parameters()) if auxiliary is not None else [])
        teacher_ids = {id(parameter) for parameter in self.load_teacher().parameters()}
        if any(id(parameter) in teacher_ids for parameter in parameters): raise ScientificAbort("Teacher parameter entered student optimizer")
        optimizer = torch.optim.AdamW(parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs, eta_min=self.config.learning_rate * 0.01)
        try: scaler = torch.amp.GradScaler("cuda", enabled=self.config.amp_enabled and self.device.type == "cuda")
        except (AttributeError, TypeError): scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp_enabled and self.device.type == "cuda")
        return model, auxiliary, optimizer, scheduler, scaler

    def train_arm_seed(self, arm: str, seed: int) -> Dict[str, Any]:
        base = self.training_checkpoint_dir(arm, seed); base.mkdir(parents=True, exist_ok=True)
        latest, best_path = base / "latest.pt", base / "best.pt"
        model, auxiliary, optimizer, scheduler, scaler = self.create_training_objects(arm, seed)
        start_epoch, best_metrics, stale_epochs = 0, None, 0
        amp_runtime_state = new_amp_runtime_state()
        if self.config.resume and latest.is_file():
            payload = safe_torch_load(latest, self.device)
            try:
                self.validate_training_checkpoint(payload, arm, seed)
            except ScientificAbort:
                if self.known_pre_amp_checkpoint_requires_restart(payload, arm, seed):
                    self.discard_known_pre_amp_checkpoint(base, payload, arm, seed)
                else:
                    raise
            else:
                model.load_state_dict(payload["student_state"], strict=True)
                if auxiliary is not None: auxiliary.load_state_dict(payload["auxiliary_state"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"])
                scaler.load_state_dict(payload["scaler_state"]); restore_rng(payload["rng_state"])
                start_epoch, best_metrics = int(payload["epoch"]), dict(payload.get("best_p0_metrics") or {}) or None
                stale_epochs = int(payload.get("early_stop_state", {}).get("stale_epochs", 0))
                loaded_runtime = payload.get("amp_runtime_state") or {}
                amp_runtime_state.update({key: int(loaded_runtime.get(key, value)) for key, value in amp_runtime_state.items()})
                self.logger.info("Resuming %s seed %d after epoch %d", arm, seed, start_epoch)
        train_meta = self.ensure_context().partitions["train_known"]
        teacher = None if arm == "K0" else self.load_teacher()
        prototypes = self.teacher_prototypes().to(self.device) if arm == "K3" else None
        augmentation = self.stage26.RFAugmentation(self.compatibility_config())
        history: List[Dict[str, Any]] = []
        teacher_value_before = teacher_state_value_sha(self.load_teacher())
        for epoch in range(start_epoch + 1, self.config.max_epochs + 1):
            sampler = self.stage26.DomainBalancedTxSampler(
                train_meta.labels, train_meta.receiver, train_meta.day, train_meta.equalized,
                self.config.batch_size, self.config.samples_per_tx, seed, epoch,
            )
            batches = list(iter(sampler)); exposure_sha = self.stage26.batch_exposure_sha256(batches)
            dataset, loader = self.build_loader("train_known", batches=batches, seed=seed * 1000 + epoch)
            model.train()
            if auxiliary is not None: auxiliary.train()
            augmentation_generator = torch.Generator(device=self.device.type); augmentation_generator.manual_seed(seed * 1_000_003 + epoch)
            sums = {name: 0.0 for name in ("total", "ce", "kd", "repr", "proto")}; samples = 0
            epoch_amp_accounting = new_epoch_amp_accounting()
            trainable_parameters = list(model.parameters()) + (list(auxiliary.parameters()) if auxiliary is not None else [])
            for batch_index, batch in enumerate(loader, start=1):
                raw_input = batch["x"].to(self.device, non_blocking=True); labels = batch["y"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                use_amp = self.config.amp_enabled and self.device.type == "cuda"
                output, teacher_output, augmented = shared_augmented_training_forward(
                    arm, raw_input, augmentation, augmentation_generator, model, teacher, use_amp
                )
                target_logits = teacher_output["logits"] if teacher_output is not None else None
                target_embedding = teacher_output["embedding_normalized"] if teacher_output is not None else None
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    loss, parts = compute_kd_losses(arm, output, target_logits, target_embedding, labels, auxiliary, prototypes)
                if not torch.isfinite(loss): raise ScientificAbort(f"Non-finite loss for {arm}/seed={seed}/epoch={epoch}")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, self.config.gradient_clip_norm)
                finite = gradients_finite(trainable_parameters, gradient_norm)
                updated = apply_optimizer_step_with_amp_policy(
                    optimizer, scaler, use_amp, finite, amp_runtime_state, epoch_amp_accounting,
                    f"arm={arm} seed={seed} epoch={epoch} batch={batch_index}", self.logger,
                )
                if not updated:
                    continue
                count = len(augmented); samples += count; sums["total"] += float(loss.detach().cpu()) * count
                for name, value in parts.items(): sums[name] += float(value.detach().cpu()) * count
            dataset.close()
            assert_epoch_has_optimizer_step(epoch_amp_accounting, f"{arm}/seed={seed}/epoch={epoch}")
            scheduler.step()
            p0 = self.evaluate_student(model, "p0", auxiliary); p0.update({"arm": arm, "seed": seed, "epoch": epoch})
            improved = p0_epoch_better(p0, best_metrics)
            stale_epochs = 0 if improved else stale_epochs + 1
            if improved: best_metrics = dict(p0)
            row = {
                **p0, **{f"train_{name}": value / samples for name, value in sums.items()},
                "exposure_sha256": exposure_sha,
                **epoch_amp_accounting,
            }
            history.append(row)
            loader_state = loader.generator.get_state() if getattr(loader, "generator", None) is not None else None
            self.save_training_checkpoint(
                latest, arm, seed, epoch, model, auxiliary, optimizer, scheduler, scaler, best_metrics,
                stale_epochs, loader_state, exposure_sha, amp_runtime_state, epoch_amp_accounting,
            )
            if improved: shutil.copy2(latest, best_path)
            self.logger.info("%s seed %d epoch %d | agreement %.5f | KL %.5f | F1 %.5f", arm, seed, epoch,
                             p0["teacher_student_top1_agreement"], p0["teacher_student_kl"], p0["student_fixed98_macro_f1"])
            if epoch >= self.config.minimum_epochs and stale_epochs >= self.config.patience: break
        if not best_path.is_file(): raise ScientificAbort(f"Best checkpoint missing for {arm}/seed={seed}")
        if teacher_state_value_sha(self.load_teacher()) != teacher_value_before or sha256_file(self.teacher_path) != EXPECTED_TEACHER_SHA256:
            raise ScientificAbort("Teacher mutated during student training")
        best_payload = safe_torch_load(best_path, "cpu"); self.validate_training_checkpoint(best_payload, arm, seed)
        history_path = base / "history.csv"; atomic_csv(history_path, pd.DataFrame(history), self.output)
        result = dict(best_payload["best_p0_metrics"]); result.update({
            "arm": arm, "seed": seed, "best_checkpoint_sha256": sha256_file(best_path), "teacher_immutable": True,
            "total_amp_overflows": int(best_payload.get("amp_runtime_state", {}).get("total_amp_overflows", 0)),
        })
        return result

    def train_arm_stage(self, stage: int, arm: str) -> None:
        results = [self.train_arm_seed(arm, seed) for seed in SEEDS]
        result_path = self.output / "tables" / f"{arm}_P0_RESULTS.csv"
        atomic_csv(result_path, pd.DataFrame(results), self.output)
        outputs = [result_path]
        for seed in SEEDS:
            base = self.training_checkpoint_dir(arm, seed); outputs.extend((base / "latest.pt", base / "best.pt", base / "history.csv"))
        dependencies = [
            *self.core_provenance_inputs(), self.output / "manifests" / "STAGE_03_CHECKPOINT.json",
        ]
        if arm == "K3":
            dependencies.append(self.output / "teacher_cache_manifests" / "train_known.json")
        self.complete_stage(stage, f"Train {arm}", outputs, dependencies)

    def stage_04(self) -> None: self.train_arm_stage(4, "K0")
    def stage_05(self) -> None: self.train_arm_stage(5, "K1")
    def stage_06(self) -> None: self.train_arm_stage(6, "K2")
    def stage_07(self) -> None: self.train_arm_stage(7, "K3")

    def load_best(self, arm: str, seed: int) -> Tuple[nn.Module, Optional[nn.Module], Mapping[str, Any]]:
        path = self.training_checkpoint_dir(arm, seed) / "best.pt"
        if not path.is_file(): raise ScientificAbort(f"Missing best checkpoint: {arm}/seed={seed}")
        payload = safe_torch_load(path, self.device); self.validate_training_checkpoint(payload, arm, seed)
        model = WiSigSurrogateNet().to(self.device); model.load_state_dict(payload["student_state"], strict=True); model.eval()
        auxiliary = nn.Linear(64, 128).to(self.device) if arm in {"K2", "K3"} else None
        if auxiliary is not None: auxiliary.load_state_dict(payload["auxiliary_state"], strict=True); auxiliary.eval()
        return model, auxiliary, payload

    def stage_08(self) -> None:
        seed_rows, summary = [], {}
        for arm in ARMS:
            frame = pd.read_csv(self.output / "tables" / f"{arm}_P0_RESULTS.csv")
            if set(frame["seed"].astype(int)) != set(SEEDS) or set(frame["arm"]) != {arm}: raise ScientificAbort(f"Incomplete P0-only evidence for {arm}")
            seed_rows.extend(frame.to_dict("records"))
            agreements = frame["teacher_student_top1_agreement"].to_numpy(float)
            summary[arm] = {
                "median_agreement": float(np.median(agreements)),
                "median_kl": float(np.median(frame["teacher_student_kl"])),
                "std_agreement": float(np.std(agreements, ddof=1)),
                "median_f1": float(np.median(frame["student_fixed98_macro_f1"])),
                "median_accuracy": float(np.median(frame["student_accuracy"])),
            }
        passing = {
            arm: values for arm, values in summary.items()
            if values["median_agreement"] >= 0.90 and values["median_accuracy"] >= 0.8193773268
            and values["median_f1"] >= 0.8168258758
        }
        freeze = architecture_freeze_payload()
        if freeze["deployed_parameter_count"] > 0.40 * EXPECTED_TEACHER_PARAMETERS:
            passing = {}
        if not passing:
            decision = "SURROGATE_HYPOTHESIS_NOT_SUPPORTED_UNDER_FROZEN_STAGE4M_V1_0_0"
            atomic_text(self.output / "MANYTX_STAGE4M_NOT_READY.txt", f"MANYTX_STAGE4M_NOT_READY\ndecision={decision}\n", self.output)
            raise ScientificAbort(decision)
        selected_arm = choose_arm(passing)
        arm_rows = [row for row in seed_rows if row["arm"] == selected_arm]
        selected_seed = choose_seed(arm_rows)
        source_path = self.training_checkpoint_dir(selected_arm, selected_seed) / "best.pt"
        source_payload = safe_torch_load(source_path, "cpu"); self.validate_training_checkpoint(source_payload, selected_arm, selected_seed)
        canonical_dir = self.output / "checkpoints" / "canonical"
        training_path = canonical_dir / "canonical_surrogate_training_checkpoint.pt"
        deploy_path = canonical_dir / "canonical_surrogate_deploy.pt"
        state_path = canonical_dir / "canonical_surrogate_state_dict.pt"
        atomic_torch_save(training_path, source_payload, self.output)
        atomic_torch_save(state_path, source_payload["student_state"], self.output)
        deploy = {
            "pipeline_version": PIPELINE_VERSION, "model": "WiSigSurrogateNet", "student_state": source_payload["student_state"],
            "selected_arm": selected_arm, "selected_seed": selected_seed, "parameter_count": freeze["deployed_parameter_count"],
            "native_embedding_dimension": 64, "known_classes": 98,
            "architecture_signature": freeze["architecture_signature_sha256"], "teacher_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_state_sha256": EXPECTED_TEACHER_STATE_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "configuration_sha256": self.config.configuration_sha256(), "objective_policy_sha256": self.objective_policy_sha,
            "training_target_policy_sha256": self.training_target_policy_sha,
            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
            "architecture_freeze_sha256": self.architecture_freeze_sha, "predecessor_lock_sha256": self.predecessor_lock_sha,
            "training_only_auxiliary_excluded": True,
        }
        atomic_torch_save(deploy_path, deploy, self.output)
        selection_path = self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION.json"
        selection_payload = {
            "status": "PASS", "selection_partition": "p0", "p1_p3_used_for_selection": False,
            "calibration_unknown_used_for_selection": False, "stage35_strict_used_for_selection": False,
            "arm_summary": summary, "passing_arms": list(passing), "selected_arm": selected_arm, "selected_seed": selected_seed,
            "arm_policy": ["highest median P0 agreement within 0.002", "lower median P0 KL", "lower agreement std", "higher median fixed98 F1", "simpler K0<K1<K2<K3"],
            "seed_policy": ["highest P0 agreement within 0.002", "lower P0 KL", "higher fixed98 F1", "lower numeric seed"],
            "freeze_gates": {"agreement": 0.90, "accuracy": 0.8193773268, "fixed98_macro_f1": 0.8168258758, "compression_fraction": 0.40},
            "seed_results": seed_rows,
        }
        atomic_json(selection_path, selection_payload, self.output)
        lock_path = self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"
        lock = {
            "status": "LOCKED", "canonical_surrogate_sha256": sha256_file(deploy_path),
            "canonical_surrogate_state_sha256": sha256_file(state_path), "selected_arm": selected_arm, "selected_seed": selected_seed,
            "selection_manifest_sha256": sha256_file(selection_path), "architecture_sha256": self.architecture_freeze_sha,
            "objective_policy_sha256": self.objective_policy_sha, "configuration_sha256": self.config.configuration_sha256(),
            "training_target_policy_sha256": self.training_target_policy_sha,
            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
            "teacher_sha256": EXPECTED_TEACHER_SHA256, "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "predecessor_lock_sha256": self.predecessor_lock_sha, "post_selection_model_changes_permitted": False,
            "calibration_unknown_used_for_selection": False, "stage35_strict_used_for_selection": False,
        }
        atomic_json(lock_path, lock, self.output)
        outputs = [selection_path, lock_path, training_path, deploy_path, state_path]
        inputs = [*self.core_provenance_inputs()]
        for completed_stage, completed_arm in zip(range(4, 8), ARMS):
            inputs.extend((self.stage_checkpoint(completed_stage), self.output / "tables" / f"{completed_arm}_P0_RESULTS.csv"))
            inputs.extend(self.training_checkpoint_dir(completed_arm, seed) / "best.pt" for seed in SEEDS)
        self.complete_stage(8, "P0-only canonical surrogate selection and freeze", outputs, inputs)

    def canonical_model(self) -> Tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
        deploy_path = self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt"
        payload = safe_torch_load(deploy_path, self.device)
        model = WiSigSurrogateNet().to(self.device); model.load_state_dict(payload["student_state"], strict=True); model.eval(); model.requires_grad_(False)
        selection = json.loads((self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION.json").read_text(encoding="utf-8"))
        _, auxiliary, _ = self.load_best(selection["selected_arm"], int(selection["selected_seed"]))
        if auxiliary is not None: auxiliary.eval(); auxiliary.requires_grad_(False)
        return model, auxiliary, selection

    def student_outputs(self, model: nn.Module, partition: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if partition not in (*NON_STRICT_CACHE_PARTITIONS, "calibration_unknown"):
            self.guard.reject("signal", partition)
        metadata = self.ensure_context().partitions[partition]
        logits = np.empty((len(metadata.indices), 98), dtype=np.float32)
        embedding = np.empty((len(metadata.indices), 64), dtype=np.float32)
        dataset, loader = self.build_loader(partition, seed=4_200_000 + list(EXPECTED_COUNTS).index(partition))
        cursor = 0
        with torch.inference_mode():
            for batch in loader:
                output = model(batch["x"].to(self.device, non_blocking=True)); count = len(batch["x"])
                logits[cursor:cursor+count] = output["logits"].float().cpu().numpy()
                embedding[cursor:cursor+count] = output["embedding_normalized"].float().cpu().numpy(); cursor += count
        dataset.close()
        if cursor != len(metadata.indices): raise ScientificAbort(f"Student output row mismatch: {partition}")
        return logits, embedding, metadata.labels.astype(np.int64)

    @staticmethod
    def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
        x = x - x.mean(0, keepdims=True); y = y - y.mean(0, keepdims=True)
        numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
        denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
        return float(numerator / max(denominator, 1e-12))

    def stage_09(self) -> None:
        model, auxiliary, selection = self.canonical_model(); rows, representation = [], []
        for partition in ("p1", "p2", "p3"):
            metrics = self.evaluate_student(model, partition, auxiliary)
            metrics.update({"selected_arm": selection["selected_arm"], "selected_seed": selection["selected_seed"], "selection_role": "REPORTING_ONLY"}); rows.append(metrics)
            logits, embedding, labels = self.student_outputs(model, partition)
            teacher_embedding = np.load(self.prepare_teacher_cache(partition) / "embedding.npy", mmap_mode="r")
            rng = np.random.default_rng(4_300_000 + KNOWN_PROTOCOLS.index(partition)); positions = np.sort(rng.choice(len(labels), min(10_000, len(labels)), replace=False))
            projected_cosine, cka = math.nan, math.nan
            if auxiliary is not None:
                with torch.inference_mode():
                    projected = F.normalize(auxiliary(torch.from_numpy(embedding[positions]).to(self.device)), dim=1).cpu().numpy()
                target = np.asarray(teacher_embedding[positions], dtype=np.float32)
                projected_cosine = float(np.mean(np.sum(projected * target, axis=1)))
                cka = self.linear_cka(projected, target)
            representation.append({"partition": partition, "projected_teacher_cosine": projected_cosine, "linear_cka": cka, "rows_sampled": len(positions), "selection_role": "REPORTING_ONLY"})
        fidelity_path = self.output / "tables" / "KNOWN_DOMAIN_FIDELITY.csv"
        representation_path = self.output / "tables" / "REPRESENTATION_FIDELITY.csv"
        atomic_csv(fidelity_path, pd.DataFrame(rows), self.output); atomic_csv(representation_path, pd.DataFrame(representation), self.output)
        self.complete_stage(
            9, "P1-P3 external known-domain diagnostics", [fidelity_path, representation_path],
            [*self.core_provenance_inputs(), self.stage_checkpoint(8),
             self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json",
             self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt"],
        )

    @staticmethod
    def fitted_scores(logits: np.ndarray, embedding: np.ndarray, prototypes: np.ndarray, precision: np.ndarray, variances: np.ndarray) -> np.ndarray:
        probabilities = np.exp(logits - logits.max(1, keepdims=True)); probabilities /= probabilities.sum(1, keepdims=True)
        msp = 1.0 - probabilities.max(1); energy = -np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) - logits.max(1)
        cosine = 1.0 - (embedding @ prototypes.T).max(1)
        diff = embedding[:, None, :] - prototypes[None, :, :]
        mahalanobis = np.einsum("ncd,df,ncf->nc", diff, precision, diff, optimize=True).min(1)
        nll = (0.5 * ((diff ** 2) / variances[None, :, :] + np.log(variances[None, :, :])).sum(2)).min(1)
        return np.column_stack((msp, energy, cosine, mahalanobis, nll))

    def stage_10(self) -> None:
        lock = self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"
        if not lock.is_file() or json.loads(lock.read_text(encoding="utf-8")).get("post_selection_model_changes_permitted") is not False:
            raise ScientificAbort("Calibration diagnostic requires an immutable canonical selection lock")
        model, _, _ = self.canonical_model(); before = sha256_file(self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt")
        train_logits, train_embedding, train_labels = self.student_outputs(model, "train_known")
        p0_logits, p0_embedding, _ = self.student_outputs(model, "p0")
        unknown_logits, unknown_embedding, _ = self.student_outputs(model, "calibration_unknown")
        prototypes, variances = np.zeros((98, 64)), np.zeros((98, 64))
        for label in range(98):
            values = train_embedding[train_labels == label]
            if len(values) == 0: raise ScientificAbort(f"Missing Train Known student class {label}")
            prototypes[label] = values.mean(0); variances[label] = values.var(0) + 1e-4
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
        fit_positions = np.linspace(0, len(train_embedding)-1, min(100_000, len(train_embedding)), dtype=np.int64)
        precision = LedoitWolf().fit(train_embedding[fit_positions]).precision_.astype(np.float64)
        known_scores = self.fitted_scores(p0_logits, p0_embedding, prototypes, precision, variances)
        unknown_scores = self.fitted_scores(unknown_logits, unknown_embedding, prototypes, precision, variances)
        rows = []
        for index, scorer in enumerate(("S0_MSP", "S1_ENERGY", "S2_PROTOTYPE_COSINE", "S3_MAHALANOBIS", "S4_DIAG_GAUSSIAN_NLL")):
            threshold = float(np.quantile(known_scores[:, index], 0.95))
            y = np.concatenate((np.zeros(len(known_scores)), np.ones(len(unknown_scores))))
            score = np.concatenate((known_scores[:, index], unknown_scores[:, index]))
            rows.append({
                "protocol": "ZD_CALIBRATED_DIAGNOSTIC", "scorer": scorer, "known_threshold_source": "P0_TARGET_ACCEPTANCE_0.95",
                "threshold": threshold, "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
                "known_false_reject_rate": float((known_scores[:, index] > threshold).mean()),
                "unknown_rejection_rate": float((unknown_scores[:, index] > threshold).mean()),
                "fpr_at_95_tpr": float((known_scores[:, index] >= np.quantile(unknown_scores[:, index], 0.05)).mean()),
                "calibration_unknown_used_for_training": False, "calibration_unknown_used_for_selection": False,
                "teacher_surrogate_score_correlation": "NOT_COMPUTED_NO_TEACHER_CALIBRATION_CACHE",
            })
        diagnostic = self.output / "tables" / "ZD_CALIBRATED_DIAGNOSTIC.csv"; atomic_csv(diagnostic, pd.DataFrame(rows), self.output)
        after = sha256_file(self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt")
        if before != after: raise ScientificAbort("Calibration Unknown mutated the canonical surrogate")
        self.complete_stage(
            10, "Optional ZD_CALIBRATED_DIAGNOSTIC", [diagnostic],
            [*self.core_provenance_inputs(), self.stage_checkpoint(8), self.stage_checkpoint(9), lock,
             self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt"],
        )

    def latency_rows(self, name: str, model: nn.Module) -> List[Dict[str, Any]]:
        rows = []
        targets = [("cpu", 1)]
        if torch.cuda.is_available(): targets = [("cuda", 1), ("cuda", 64), ("cpu", 1)]
        for device_name, batch in targets:
            device = torch.device(device_name); measured_model = copy.deepcopy(model).to(device).eval()
            sample = torch.randn(batch, 2, 256, device=device)
            with torch.inference_mode():
                for _ in range(50): measured_model(sample)
                if device.type == "cuda": torch.cuda.synchronize()
                timings = []
                for _ in range(200):
                    if device.type == "cuda": torch.cuda.synchronize()
                    started = time.perf_counter(); measured_model(sample)
                    if device.type == "cuda": torch.cuda.synchronize()
                    timings.append((time.perf_counter() - started) * 1000)
            rows.append({"model": name, "device": device_name, "batch_size": batch, "warmup_iterations": 50,
                         "measured_iterations": 200, "median_ms": float(np.median(timings)),
                         "mean_ms": float(np.mean(timings)), "p95_ms": float(np.quantile(timings, 0.95)),
                         "selection_role": "REPORTING_ONLY"})
            del measured_model
        return rows

    def stage_11(self) -> None:
        model, auxiliary, selection = self.canonical_model(); teacher = self.load_teacher()
        freeze = architecture_freeze_payload()
        deploy_path = self.output / "checkpoints" / "canonical" / "canonical_surrogate_deploy.pt"
        state_path = self.output / "checkpoints" / "canonical" / "canonical_surrogate_state_dict.pt"
        compression = {
            "teacher_parameter_count": EXPECTED_TEACHER_PARAMETERS,
            "student_deployed_parameter_count": freeze["deployed_parameter_count"],
            "training_only_auxiliary_parameter_count": freeze["training_only_auxiliary_parameter_count"],
            "deployed_parameter_fraction": freeze["deployed_to_teacher_ratio"],
            "parameter_compression_ratio": freeze["compression_ratio_teacher_over_student"],
            "teacher_checkpoint_bytes": self.teacher_path.stat().st_size, "student_deploy_bytes": deploy_path.stat().st_size,
            "student_state_dict_bytes": state_path.stat().st_size, "compression_gate_pass": freeze["status"] == "PASS",
            "mac_flop_estimate": "MAC_FLOP_ESTIMATE_NOT_AVAILABLE", "selection_role": "REPORTING_ONLY",
        }
        compression_path = self.output / "performance" / "SURROGATE_COMPRESSION_SUMMARY.json"; atomic_json(compression_path, compression, self.output)
        latency = self.latency_rows("teacher", teacher) + self.latency_rows("surrogate", model)
        latency_path = self.output / "performance" / "PRELIMINARY_LATENCY.csv"; atomic_csv(latency_path, pd.DataFrame(latency), self.output)
        seed_frames = [pd.read_csv(self.output / "tables" / f"{arm}_P0_RESULTS.csv") for arm in ARMS]
        seeds = pd.concat(seed_frames, ignore_index=True)
        seed_path = self.output / "tables" / "KD_SEED_RESULTS.csv"; atomic_csv(seed_path, seeds, self.output)
        summary_rows = []
        for arm, frame in seeds.groupby("arm"):
            row: Dict[str, Any] = {"arm": arm}
            for metric in ("teacher_student_top1_agreement", "teacher_student_kl", "student_accuracy", "student_fixed98_macro_f1"):
                values = frame[metric].to_numpy(float)
                for statistic_name, value in (("mean", np.mean(values)), ("median", np.median(values)), ("std", np.std(values, ddof=1)), ("min", np.min(values)), ("max", np.max(values))):
                    row[f"{metric}_{statistic_name}"] = float(value)
            summary_rows.append(row)
        summary_path = self.output / "tables" / "KD_ARM_SUMMARY.csv"; atomic_csv(summary_path, pd.DataFrame(summary_rows), self.output)
        figures = []
        specs = [
            ("p0_agreement_by_arm", "teacher_student_top1_agreement", "P0 teacher/student top-1 agreement"),
            ("p0_f1_by_arm", "student_fixed98_macro_f1", "P0 student fixed-98 macro-F1"),
            ("p0_kl_by_arm", "teacher_student_kl", "P0 teacher-to-student KL"),
        ]
        for filename, metric, title in specs:
            fig, ax = plt.subplots(figsize=(7, 4.5)); seeds.boxplot(column=metric, by="arm", ax=ax, grid=False)
            ax.set_title(title); ax.set_xlabel("Frozen KD arm"); fig.suptitle(""); fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = self.output / "figures" / f"{filename}.{suffix}"; fig.savefig(path, dpi=220 if suffix == "png" else None); figures.append(path)
            plt.close(fig)
        report = self.output / "reports" / "Stage4M_Surrogate_KD_Report.md"
        atomic_text(report, f"# Stage 4M surrogate KD report\n\nCanonical selection used P0 only and selected **{selection['selected_arm']} seed {selection['selected_seed']}**. P1-P3 and ZD_CALIBRATED_DIAGNOSTIC are descriptive only. Stage 3.5M strict results were neither read nor used.\n\n## Compression\n\n```json\n{json.dumps(compression, indent=2)}\n```\n", self.output)
        workbook = self.output / "publication" / "Stage4M_tables.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            seeds.to_excel(writer, sheet_name="seed_results", index=False); pd.DataFrame(summary_rows).to_excel(writer, sheet_name="arm_summary", index=False)
            pd.read_csv(self.output / "tables" / "KNOWN_DOMAIN_FIDELITY.csv").to_excel(writer, sheet_name="known_fidelity", index=False)
            pd.read_csv(self.output / "tables" / "REPRESENTATION_FIDELITY.csv").to_excel(writer, sheet_name="repr_fidelity", index=False)
            pd.read_csv(self.output / "tables" / "ZD_CALIBRATED_DIAGNOSTIC.csv").to_excel(writer, sheet_name="zd_calibrated", index=False)
        pdf = self.output / "publication" / "Stage4M_report.pdf"
        with PdfPages(pdf) as pages:
            cover = plt.figure(figsize=(8.27, 11.69)); cover.text(0.08, 0.94, "Stage 4M Surrogate KD", fontsize=20, weight="bold")
            cover.text(0.08, 0.86, f"Selected arm: {selection['selected_arm']}\nSelected seed: {selection['selected_seed']}\nStrict zero-day evaluation: NOT PERFORMED\nCalibration Unknown: post-selection diagnostic only", va="top"); plt.axis("off"); pages.savefig(cover); plt.close(cover)
            for path in figures:
                if path.suffix == ".png":
                    image = plt.imread(path); fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.imshow(image); ax.axis("off"); pages.savefig(fig, bbox_inches="tight"); plt.close(fig)
        outputs = [compression_path, latency_path, seed_path, summary_path, report, workbook, pdf, *figures]
        self.complete_stage(
            11, "Compression, preliminary latency, publication", outputs,
            [*self.core_provenance_inputs(), self.stage_checkpoint(8), self.stage_checkpoint(9), self.stage_checkpoint(10),
             self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json", deploy_path,
             self.output / "tables" / "KNOWN_DOMAIN_FIDELITY.csv",
             self.output / "tables" / "REPRESENTATION_FIDELITY.csv",
             self.output / "tables" / "ZD_CALIBRATED_DIAGNOSTIC.csv"],
        )

    def create_final_hash_manifest(self) -> Path:
        manifest = self.output / "manifests" / "STAGE4M_HASH_MANIFEST.json"
        excluded = {(self.output / relative).resolve() for relative in FINAL_HASH_EXCLUSIONS}
        rows = []
        for path in sorted(self.output.rglob("*")):
            if path.is_file() and path.resolve() not in excluded:
                rows.append({"relative_path": path.relative_to(self.output).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        atomic_json(manifest, {"algorithm": "SHA-256", "count": len(rows), "files": rows,
                               "exclusions": [{"relative_path": key, "reason": value} for key, value in FINAL_HASH_EXCLUSIONS.items()]}, self.output)
        if not self.final_hash_manifest_current(): raise ScientificAbort("Stage 4M final hash manifest is inconsistent")
        return manifest

    def final_hash_manifest_current(self) -> bool:
        return final_hash_manifest_current(self.output)

    def stage_12(self) -> None:
        ready, not_ready = self.output / "MANYTX_STAGE4M_READY.txt", self.output / "MANYTX_STAGE4M_NOT_READY.txt"
        if ready.exists(): ready.unlink()
        for stage in range(1, 12):
            if not self.stage_current(stage): raise ScientificAbort(f"Stage {stage:02d} is stale or incomplete")
        self.guard.scan_output(); self.guard.assert_zero()
        selection = json.loads((self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION.json").read_text(encoding="utf-8"))
        lock = json.loads((self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json").read_text(encoding="utf-8"))
        compression = json.loads((self.output / "performance" / "SURROGATE_COMPRESSION_SUMMARY.json").read_text(encoding="utf-8"))
        selected = next(row for row in selection["seed_results"] if row["arm"] == selection["selected_arm"] and int(row["seed"]) == int(selection["selected_seed"]))
        gates = {
            "predecessor_frozen": True, "teacher_equivalence_pass": True, "teacher_immutable": sha256_file(self.teacher_path) == EXPECTED_TEACHER_SHA256,
            "student_architecture_predeclared": True, "single_student_architecture": True,
            "all_four_kd_arms_completed": all(self.stage_current(stage) for stage in range(4, 8)),
            "all_three_seeds_per_arm_completed": all(len(pd.read_csv(self.output / "tables" / f"{arm}_P0_RESULTS.csv")) == 3 for arm in ARMS),
            "selection_used_p0_only": selection["selection_partition"] == "p0", "p1_p3_used_for_selection": False,
            "calibration_unknown_used_for_training": False, "calibration_unknown_used_for_selection": False,
            "stage35_strict_data_used": False, "strict_zero_day_evaluation_performed": False,
            "all_stage35_strict_violation_counters_zero": not any(self.guard.counters().values()),
            "canonical_surrogate_frozen": lock["status"] == "LOCKED", "compression_gate_pass": compression["compression_gate_pass"],
            "p0_fidelity_gate_pass": selected["teacher_student_top1_agreement"] >= 0.90 and selected["student_accuracy"] >= 0.8193773268 and selected["student_fixed98_macro_f1"] >= 0.8168258758,
            "publication_complete": (self.output / "publication" / "Stage4M_tables.xlsx").is_file() and (self.output / "publication" / "Stage4M_report.pdf").is_file(),
        }
        if not all(gates.values()): raise ScientificAbort(f"Stage 4M final gates failed: {gates}")
        final = self.output / "manifests" / "STAGE4M_FINAL_STATUS.json"
        atomic_json(final, {"status": "MANYTX_STAGE4M_READY", "pipeline_version": PIPELINE_VERSION, "selected_arm": selection["selected_arm"],
                            "selected_seed": selection["selected_seed"], "gates": gates, "stage35_strict_violation_counters": self.guard.counters(),
                            "amp_runtime_safety_policy_sha256": self.amp_runtime_safety_policy_sha,
                            "generated_at": utc_now()}, self.output)
        manifest = self.create_final_hash_manifest()
        ready_text = "\n".join([
            "MANYTX_STAGE4M_READY", f"pipeline_version={PIPELINE_VERSION}", "teacher_version=1.0", "teacher_seed=123",
            f"teacher_sha256={EXPECTED_TEACHER_SHA256}", f"benchmark_sha256={EXPECTED_BENCHMARK_SHA256}",
            f"selected_kd_arm={selection['selected_arm']}", f"selected_seed={selection['selected_seed']}",
            f"canonical_surrogate_sha256={lock['canonical_surrogate_sha256']}", f"canonical_surrogate_state_sha256={lock['canonical_surrogate_state_sha256']}",
            f"amp_runtime_safety_policy_sha256={self.amp_runtime_safety_policy_sha}",
            f"student_deployed_parameter_count={compression['student_deployed_parameter_count']}", f"teacher_parameter_count={EXPECTED_TEACHER_PARAMETERS}",
            f"parameter_compression_ratio={compression['parameter_compression_ratio']}", "student_native_embedding_dim=64", "teacher_embedding_dim=128",
            "calibration_unknown_used_for_training=NO", "calibration_unknown_used_for_selection=NO", "strict_zero_day_evaluation_performed=NO",
            "strict_zero_day_signal_read=NO", "strict_zero_day_label_read=NO", "strict_zero_day_index_read=NO", "strict_zero_day_score_read=NO", "strict_zero_day_metric_read=NO",
            *[f"{key}=0" for key in STAGE35_COUNTER_KEYS], "teacher_modified=NO", "teacher_retrained=NO", "surrogate_training_performed=YES", "xai_performed=NO", "next_stage=STAGE_5M", "",
        ])
        atomic_text(ready, ready_text, self.output)
        if parse_ready(ready).get("marker") != "MANYTX_STAGE4M_READY" or not self.final_hash_manifest_current(): raise ScientificAbort("READY/hash verification failed")
        stage_dependencies = [self.stage_checkpoint(stage) for stage in range(1, 12)]
        self.complete_stage(
            12, "Final audit, hash manifest, READY transaction", [final, manifest, ready],
            [*stage_dependencies, self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"],
        )
        if not self.stage_current(12): raise ScientificAbort("Stage 12 durable checkpoint verification failed")
        if not_ready.exists(): not_ready.unlink()
        if not ready.is_file() or not_ready.exists(): raise ScientificAbort("READY-only final state verification failed")
        print("MANYTX_STAGE4M_READY")

    def hydrate_provenance(self) -> None:
        predecessor = self.output / "manifests" / "STAGE4M_PREDECESSOR_LOCK.json"
        architecture = self.output / "manifests" / "STUDENT_ARCHITECTURE_FREEZE.json"
        objective = self.output / "manifests" / "KD_OBJECTIVE_POLICY.json"
        targets = self.output / "manifests" / "TRAINING_TARGET_POLICY.json"
        amp_policy = self.output / "manifests" / "AMP_RUNTIME_SAFETY_POLICY.json"
        selection = self.output / "manifests" / "CANONICAL_SURROGATE_SELECTION_LOCK.json"
        self.predecessor_lock_sha = sha256_file(predecessor) if predecessor.is_file() else None
        self.architecture_freeze_sha = sha256_file(architecture) if architecture.is_file() else None
        self.objective_policy_sha = sha256_file(objective) if objective.is_file() else None
        self.training_target_policy_sha = sha256_file(targets) if targets.is_file() else None
        self.amp_runtime_safety_policy_sha = sha256_file(amp_policy) if amp_policy.is_file() else None
        self.selection_lock_hash = sha256_file(selection) if selection.is_file() else None

    def preflight(self) -> None:
        benchmark = self.root / "01_benchmark_engineering" / "benchmark" / f"{CANONICAL_BENCHMARK}.h5"
        checks: Dict[str, Any] = {
            "canonical_root": str(self.root), "root_name": self.root.name == CANONICAL_BRANCH,
            "output_boundary": self.output.parent == self.root and self.output.name == "06_surrogate_kd",
            "gpu_available": torch.cuda.is_available(), "teacher_checkpoint_sha": self.teacher_path.is_file() and sha256_file(self.teacher_path) == EXPECTED_TEACHER_SHA256,
            "benchmark_sha": benchmark.is_file() and sha256_file(benchmark) == EXPECTED_BENCHMARK_SHA256,
            "ready_absent": not (self.output / "MANYTX_STAGE4M_READY.txt").exists(),
        }
        stage35 = self.verify_stage35_metadata(); checks["stage35_ready"] = stage35["ready"].get("marker") == "MANYTX_STAGE3_5M_READY"
        stage3 = parse_ready(self.stage3_root / "MANYTX_STAGE3M_READY.txt"); checks["stage3_ready"] = stage3.get("marker") == "MANYTX_STAGE3M_READY"
        with h5py.File(benchmark, "r", swmr=True) as handle:
            checks["signal_schema"] = "signals/X" in handle and tuple(handle["signals/X"].shape) == (1_020_643, 2, 256) and str(handle["signals/X"].dtype) == "float32"
        split_root = self.root / "01_benchmark_engineering" / "splits"
        for name in ("train_known_indices.npy", "validation_known_indices.npy", "cross_day_validation_indices.npy", "cross_receiver_validation_indices.npy", "cross_day_receiver_validation_indices.npy"):
            checks[f"split_exists_{name}"] = (split_root / name).is_file()
        self.load_teacher(); checks["teacher_cpu_metadata_and_forward"] = True
        probe = self.output / ".preflight_write_probe"; atomic_text(probe, "probe\n", self.output); probe.unlink(); checks["output_writable"] = True
        self.guard.assert_zero(); checks["strict_counters_zero"] = True
        if not all(value for key, value in checks.items() if isinstance(value, bool)):
            raise ScientificAbort(f"Stage 4M preflight failed: {checks}")
        report = self.output / "manifests" / "STAGE4M_PREFLIGHT.json"
        atomic_json(report, {"status": "STAGE4M_PREFLIGHT_PASS", "pipeline_version": PIPELINE_VERSION,
                             "executable_sha256": self.script_sha, "configuration_sha256": self.config.configuration_sha256(),
                             "checks": checks, "runtime": runtime_manifest(), "student_architecture": architecture_freeze_payload(),
                             "training_performed": False, "teacher_target_cache_created": False,
                             "calibration_unknown_accessed": False, "strict_index_contents_accessed": False,
                             "stage04_plus_executed": False, "ready_created": False, "generated_at": utc_now()}, self.output)
        print("STAGE4M_PREFLIGHT_PASS")

    def run(self) -> None:
        self.hydrate_provenance()
        stages = {1: self.stage_01, 2: self.stage_02, 3: self.stage_03, 4: self.stage_04, 5: self.stage_05, 6: self.stage_06,
                  7: self.stage_07, 8: self.stage_08, 9: self.stage_09, 10: self.stage_10, 11: self.stage_11, 12: self.stage_12}
        for stage in range(self.config.stage_start, self.config.stage_end + 1):
            self.hydrate_provenance()
            if self.config.resume and self.stage_current(stage):
                print(f"[REUSE] Stage {stage:02d} — hash-current"); continue
            stages[stage](); print(f"[PASS] Stage {stage:02d}")


def runtime_manifest() -> Dict[str, Any]:
    return {
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "numpy": np.__version__,
        "seed_panel": list(SEEDS), "worker_seed_policy": "torch.initial_seed modulo 2^32 -> Python and NumPy",
        "amp_policy": "enabled on CUDA", "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
    }


def synthetic_validation() -> None:
    freeze = architecture_freeze_payload()
    if freeze["status"] != "PASS" or freeze["native_embedding_dimension"] != 64: raise ScientificAbort("Synthetic architecture validation failed")
    torch.manual_seed(42)
    student = WiSigSurrogateNet(); auxiliary = nn.Linear(64, 128)
    output = student(torch.randn(8, 2, 256)); teacher_logits = torch.randn(8, 98); teacher_embedding = F.normalize(torch.randn(8, 128), dim=1)
    labels = torch.arange(8) % 4; prototypes = F.normalize(torch.randn(98, 128), dim=1)
    for arm in ARMS:
        targets = (None, None) if arm == "K0" else (teacher_logits, teacher_embedding)
        loss, parts = compute_kd_losses(arm, output, targets[0], targets[1], labels, auxiliary if arm in {"K2", "K3"} else None, prototypes)
        if not torch.isfinite(loss) or set(parts) != {"ce", "kd", "repr", "proto"}: raise ScientificAbort(f"Synthetic objective failed: {arm}")
    print("STAGE4M_SYNTHETIC_VALIDATION_PASS")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root"); parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-dir"); parser.add_argument("--config"); parser.add_argument("--profile", choices=("full", "pilot"), default="full")
    parser.add_argument("--device", default="auto"); parser.add_argument("--stage-start", type=int, default=1); parser.add_argument("--stage-end", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true"); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--synthetic-validation", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Stage4Config:
    values = load_config(args.config)
    root = discover_branch_root(args.branch_root)
    values.update({"branch_root": str(root), "repository_root": args.repository_root, "profile": args.profile, "device": args.device,
                   "stage_start": args.stage_start, "stage_end": args.stage_end, "resume": not args.no_resume})
    if args.output_dir: values["output_dir"] = args.output_dir
    if "seeds" in values: values["seeds"] = tuple(values["seeds"])
    allowed = {field.name for field in dataclasses.fields(Stage4Config)}
    unknown = set(values) - allowed
    if unknown: raise ValueError(f"Unknown Stage 4M configuration keys: {sorted(unknown)}")
    return Stage4Config(**values)


def write_not_ready(config: Stage4Config, exc: BaseException) -> bool:
    output = config.output_root
    if completed_state_current(output):
        print("STAGE4M_COMPLETED_STATE_PROTECTED")
        return False
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "MANYTX_STAGE4M_READY.txt"
    if ready.exists(): ready.unlink()
    atomic_text(output / "MANYTX_STAGE4M_NOT_READY.txt", f"MANYTX_STAGE4M_NOT_READY\n{type(exc).__name__}: {exc}\n", output)
    manifests = output / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    atomic_json(manifests / "STAGE4M_FAILURE.json", {"status": "MANYTX_STAGE4M_NOT_READY", "error_type": type(exc).__name__,
                                                       "error": str(exc), "traceback": traceback.format_exc(), "generated_at": utc_now()}, output)
    return True


def invocation_output(args: argparse.Namespace, config: Optional[Stage4Config]) -> Optional[Path]:
    if config is not None:
        return config.output_root
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.branch_root:
        return Path(args.branch_root).expanduser().resolve() / "06_surrogate_kd"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.synthetic_validation: synthetic_validation(); return 0
    config: Optional[Stage4Config] = None
    try:
        config = config_from_args(args)
        if completed_state_current(config.output_root):
            print("MANYTX_STAGE4M_ALREADY_READY" if not args.preflight else "STAGE4M_COMPLETED_STATE_PROTECTED")
            return 0
        print("=" * 100 + "\nSTAGE 4M — WISIG MANYTX SURROGATE KNOWLEDGE DISTILLATION\n" + "=" * 100)
        pipeline = Stage4Pipeline(config)
        if args.preflight: pipeline.preflight()
        else: pipeline.run()
        return 0
    except BaseException as exc:
        output = invocation_output(args, config)
        if output is not None and completed_state_current(output):
            print("STAGE4M_COMPLETED_STATE_PROTECTED")
            print(f"{type(exc).__name__}: {exc}")
            return 1
        if config is not None:
            write_not_ready(config, exc)
        print(f"MANYTX_STAGE4M_NOT_READY\n{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
