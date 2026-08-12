#!/usr/bin/env python3
"""Stage 3M v1.0.0 — promote and freeze one canonical WiSig ManyTx teacher.

Canonical execution is verification and promotion only. It never trains a model,
never reads strict-zero-day rows, and never mutates Stage 1B/2M/2.6M artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

from stage2_6m_performance_v1_0_2 import BatchedSignalDataset, LocalCacheManager, identity_collate


PIPELINE_VERSION = "1.0.0"
TEACHER_VERSION = "1.0"
CANONICAL_BRANCH = "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
CANONICAL_BENCHMARK = "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_VERSION = "1.0.5"
EXPECTED_STAGE2M_SCRIPT_SHA256 = "46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba"
EXPECTED_STAGE2M_HASH_MANIFEST_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
EXPECTED_STAGE26_ARTIFACT_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE26_DECISION = "SELECT_CE_SUPCON_PROTOTYPE"
EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_ARM = "A3"
EXPECTED_LOSS = {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1}
EXPECTED_PARAMETER_COUNT = 849_634
STRICT_COUNTER_KEYS = (
    "strict_zero_day_signal_read_violations",
    "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations",
    "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations",
)
FORBIDDEN_TOKENS = ("strict_zero_day", "zero_day_shift_test", "strict_test")
KNOWN_PROTOCOLS = ("p0", "p1", "p2", "p3")
REQUIRED_OUTPUT_DIRS = (
    "configs", "checkpoints", "logs", "manifests", "metrics", "tables", "embeddings",
    "statistics", "figures", "reports", "publication", "performance",
)

SELECTION_POLICY = {
    "policy_version": "stage3m_teacher_selection_v1",
    "candidate_scope": {"arm": "A3", "seeds": [42, 123, 2026]},
    "data_scope": ["P0", "P1", "P2", "P3"],
    "hierarchy": [
        {"rank": 1, "metric": "mean_fixed98_macro_f1_p0_p3", "direction": "higher"},
        {"rank": 2, "metric": "mean_fixed98_balanced_accuracy_p0_p3", "direction": "higher"},
        {"rank": 3, "metric": "mean_absolute_macro_f1_degradation_from_p0", "direction": "lower"},
        {"rank": 4, "metric": "p0_fisher_ratio", "direction": "higher"},
        {"rank": 5, "metric": "numeric_seed", "direction": "lower"},
    ],
    "forbidden_selection_inputs": [
        "calibration_unknown", "strict_zero_day", "strict_zero_day_shift",
        "surrogate_fidelity", "xai", "future_deployment",
    ],
}


class ScientificAbort(RuntimeError):
    pass


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


def assert_within(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(), root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ScientificAbort(f"Path escapes Stage 3M output root: {resolved}")
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


def atomic_json(path: Path, payload: Mapping[str, Any], root: Path) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", root)


def atomic_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    path = assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ScientificAbort(f"Checkpoint load failed: {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ScientificAbort(f"Checkpoint payload is not a mapping: {path}")
    return payload


def atomic_torch_save(path: Path, payload: Any, root: Path) -> None:
    path = assert_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_ready_marker(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise ScientificAbort(f"Required READY marker missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "MANYTX_STAGE2_6M_READY":
        raise ScientificAbort("Stage 2.6M READY marker header is invalid")
    result: Dict[str, str] = {}
    for line in lines[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def load_stage26_module(repo_root: Path) -> Any:
    path = repo_root / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"
    if not path.is_file():
        raise ScientificAbort(f"Frozen Stage 2.6M executable missing from repository: {path}")
    name = "stage2_6m_frozen_for_stage3m"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScientificAbort("Cannot load frozen Stage 2.6M executable for equivalence audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StrictZeroDayGuard:
    """Structural guard; strict partitions have no authorized row-registration route."""
    def __init__(self) -> None:
        self.allowed: Dict[str, np.ndarray] = {}
        self.signal_reads = 0
        self.label_reads = 0
        self.embedding_reads = 0
        self.metric_reads = 0
        self.threshold_reads = 0

    @staticmethod
    def is_strict(value: Any) -> bool:
        token = str(value).lower().replace("-", "_")
        return any(forbidden in token for forbidden in FORBIDDEN_TOKENS)

    def reject(self, kind: str, value: Any) -> None:
        if self.is_strict(value):
            if kind == "signal": self.signal_reads += 1
            elif kind == "label": self.label_reads += 1
            elif kind == "embedding": self.embedding_reads += 1
            elif kind == "metric": self.metric_reads += 1
            elif kind == "threshold": self.threshold_reads += 1
            raise ScientificAbort(f"STRICT_ZERO_DAY_ACCESS_VIOLATION: {kind}: {value}")

    def forbid_data_path(self, path: Path, operation: str) -> None:
        self.reject("signal", f"{operation}:{path}")

    def register_allowed_indices(self, partition: str, indices: np.ndarray) -> None:
        if partition not in {"train_known", "p0", "p1", "p2", "p3", "calibration_unknown"}:
            self.reject("signal", partition)
            raise ScientificAbort(f"Unauthorized partition: {partition}")
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ScientificAbort(f"Invalid authorized indices for {partition}")
        self.allowed[partition] = values

    def authorize_rows(self, partition: str, indices: np.ndarray, operation: str) -> None:
        self.reject("signal", partition)
        if partition not in self.allowed:
            raise ScientificAbort(f"Partition not authorized: {partition}")
        if not np.isin(np.asarray(indices, dtype=np.int64), self.allowed[partition]).all():
            raise ScientificAbort(f"Unauthorized rows requested for {partition}: {operation}")

    def counters(self) -> Dict[str, int]:
        return {
            STRICT_COUNTER_KEYS[0]: self.signal_reads,
            STRICT_COUNTER_KEYS[1]: self.label_reads,
            STRICT_COUNTER_KEYS[2]: self.embedding_reads,
            STRICT_COUNTER_KEYS[3]: self.metric_reads,
            STRICT_COUNTER_KEYS[4]: self.threshold_reads,
        }

    def assert_zero(self) -> None:
        if any(self.counters().values()):
            raise ScientificAbort(f"Strict-zero-day counters are non-zero: {self.counters()}")

    def scan_output(self, root: Path) -> None:
        for path in root.rglob("*"):
            if path.is_file() and self.is_strict(path.name) and path.name not in {"STRICT_ZERO_DAY_GUARD.json"}:
                raise ScientificAbort(f"Forbidden strict-zero-day artifact: {path}")


@dataclass
class Stage3MConfig:
    branch_root: str
    repository_root: str = ""
    benchmark_path: str = ""
    stage2m_dir: str = ""
    stage26_dir: str = ""
    output_dir: str = ""
    teacher_source: str = "stage2_6m_promote"
    seeds: Tuple[int, ...] = EXPECTED_SEEDS
    eval_batch_size: int = 1024
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    local_cache_root: str = "/content/wisig_stage3m_cache"
    representation_samples_per_class: int = 100
    representation_sampling_seed: int = 3_000_001
    calibration_unknown_diagnostic: bool = False
    amp_enabled: bool = True
    device: str = "auto"
    stage_start: int = 1
    stage_end: int = 10
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
    def stage2m_path(self) -> Path:
        return Path(self.stage2m_dir).expanduser().resolve() if self.stage2m_dir else self.branch_root_path / "02_benchmark_diagnostics"

    @property
    def stage26_path(self) -> Path:
        return Path(self.stage26_dir).expanduser().resolve() if self.stage26_dir else self.branch_root_path / "03_representation_ablation"

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else self.branch_root_path / "04_canonical_teacher"

    def validate(self) -> None:
        if self.branch_root_path.name != CANONICAL_BRANCH:
            raise ScientificAbort(f"Branch root must be {CANONICAL_BRANCH}")
        if self.output_root.parent != self.branch_root_path or self.output_root.name != "04_canonical_teacher":
            raise ScientificAbort("Stage 3M output must be branch-root/04_canonical_teacher")
        if self.teacher_source != "stage2_6m_promote":
            raise ScientificAbort("Canonical Stage 3M supports PROMOTE_AND_VERIFY only; automatic retraining is forbidden")
        if tuple(self.seeds) != EXPECTED_SEEDS:
            raise ScientificAbort("Teacher candidate seeds must be exactly 42, 123, 2026")
        if not 1 <= self.stage_start <= self.stage_end <= 10:
            raise ValueError("Stage range must be within 1..10")
        if self.eval_batch_size <= 0 or self.num_workers < 0 or self.representation_samples_per_class <= 1:
            raise ValueError("Invalid execution configuration")

    def scientific_payload(self) -> Dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "teacher_source": self.teacher_source,
            "candidate_arm": EXPECTED_ARM,
            "seeds": list(self.seeds),
            "known_protocols": list(KNOWN_PROTOCOLS),
            "known_classes": 98,
            "embedding_dimension": 128,
            "loss": EXPECTED_LOSS,
            "temperature": 0.07,
            "prototype_momentum": 0.95,
            "representation_samples_per_class": self.representation_samples_per_class,
            "representation_sampling_seed": self.representation_sampling_seed,
            "calibration_unknown_diagnostic": self.calibration_unknown_diagnostic,
            "selection_policy": SELECTION_POLICY,
        }

    def configuration_sha256(self) -> str:
        return sha256_object(self.scientific_payload())


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
        self.conv1 = ConvNormAct(in_channels, out_channels, 5, stride=stride, dilation=dilation)
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


class WiSigRepresentationNet(nn.Module):
    """Independent exact copy of the frozen Stage 2.6M common A3 architecture."""
    def __init__(self, num_classes: int = 98, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.iq_mixer = nn.Sequential(
            nn.Conv1d(2, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.SiLU(inplace=True),
            ConvNormAct(32, 64, 7, stride=2),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(64, 64, dilation=1, dropout=dropout),
            ResidualTemporalBlock(64, 128, stride=2, dilation=1, dropout=dropout),
            ResidualTemporalBlock(128, 128, dilation=2, dropout=dropout),
            ResidualTemporalBlock(128, 256, stride=2, dilation=1, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(inplace=True), nn.Dropout(dropout),
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
        return {"logits": self.classifier(embedding_raw), "embedding_raw": embedding_raw, "embedding_normalized": embedding_normalized}


def architecture_signature(model: nn.Module) -> str:
    return sha256_object([(name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()])


def model_schema(model: nn.Module) -> List[Dict[str, Any]]:
    return [{"key": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in model.state_dict().items()]


class StreamingMetrics:
    def __init__(self, classes: int = 98, bins: int = 15):
        self.classes, self.bins = classes, bins
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.count = self.correct = self.top5 = 0
        self.ce_sum = 0.0
        self.bin_count = np.zeros(bins, dtype=np.int64)
        self.bin_conf = np.zeros(bins, dtype=np.float64)
        self.bin_correct = np.zeros(bins, dtype=np.float64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.detach().float()
        labels = labels.detach().to(logits.device, dtype=torch.long, non_blocking=True)
        if not torch.isfinite(logits).all():
            raise ScientificAbort("NaN/Inf logits during known evaluation")
        probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(1)
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        self.ce_sum += float(F.cross_entropy(logits, labels, reduction="sum").cpu())
        self.count += len(labels)
        self.correct += int((prediction == labels).sum().cpu())
        self.top5 += int(top5.eq(labels[:, None]).any(1).sum().cpu())
        y, pred = labels.cpu().numpy(), prediction.cpu().numpy()
        np.add.at(self.confusion, (y, pred), 1)
        confidence = probabilities.max(1).values.cpu().numpy()
        correct = (pred == y).astype(np.float64)
        assignments = np.minimum((confidence * self.bins).astype(np.int64), self.bins - 1)
        for index in range(self.bins):
            mask = assignments == index
            if mask.any():
                self.bin_count[index] += int(mask.sum())
                self.bin_conf[index] += float(confidence[mask].sum())
                self.bin_correct[index] += float(correct[mask].sum())

    def finish(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        if not self.count:
            raise ScientificAbort("Cannot finalize empty metrics")
        support = self.confusion.sum(1).astype(float)
        predicted = self.confusion.sum(0).astype(float)
        tp = np.diag(self.confusion).astype(float)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
        observed = support > 0
        ece = 0.0
        for index in range(self.bins):
            if self.bin_count[index]:
                ece += self.bin_count[index] / self.count * abs(self.bin_correct[index] / self.bin_count[index] - self.bin_conf[index] / self.bin_count[index])
        metrics = {
            "samples": float(self.count), "accuracy": self.correct / self.count, "top5_accuracy": self.top5 / self.count,
            "cross_entropy": self.ce_sum / self.count, "ece": ece,
            "observed_macro_f1": float(f1[observed].mean()), "fixed98_macro_f1": float(f1.mean()),
            "observed_balanced_accuracy": float(recall[observed].mean()), "fixed98_balanced_accuracy": float(recall.mean()),
        }
        per_class = pd.DataFrame({"class_index": np.arange(self.classes), "support": support.astype(int), "precision": precision, "recall": recall, "f1": f1})
        return metrics, per_class


def deterministic_selection(known: pd.DataFrame, representation: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    required = {(seed, protocol) for seed in EXPECTED_SEEDS for protocol in KNOWN_PROTOCOLS}
    observed = {(int(row.seed), str(row.protocol)) for row in known.itertuples()}
    if required != observed:
        raise ScientificAbort(f"Candidate known evaluation is incomplete: missing={sorted(required - observed)}")
    rows = []
    for seed in EXPECTED_SEEDS:
        frame = known[known.seed == seed].set_index("protocol")
        p0 = frame.loc["p0"]
        p0_rep = representation[(representation.seed == seed) & (representation.protocol == "p0")]
        if len(p0_rep) != 1:
            raise ScientificAbort(f"P0 Fisher ratio missing for seed {seed}")
        degradations = [abs(float(p0.fixed98_macro_f1) - float(frame.loc[p].fixed98_macro_f1)) for p in ("p1", "p2", "p3")]
        rows.append({
            "seed": seed,
            "primary_mean_fixed98_macro_f1": float(frame.fixed98_macro_f1.mean()),
            "secondary_mean_fixed98_balanced_accuracy": float(frame.fixed98_balanced_accuracy.mean()),
            "tertiary_mean_absolute_degradation": float(np.mean(degradations)),
            "quaternary_p0_fisher_ratio": float(p0_rep.iloc[0].fisher_ratio),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["primary_mean_fixed98_macro_f1", "secondary_mean_fixed98_balanced_accuracy", "tertiary_mean_absolute_degradation", "quaternary_p0_fisher_ratio", "seed"],
        ascending=[False, False, True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return int(ranking.iloc[0].seed), ranking


def state_dict_equal(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    return list(first) == list(second) and all(torch.equal(first[key].detach().cpu(), second[key].detach().cpu()) for key in first)


def validate_candidate_checkpoint_payload(
    payload: Mapping[str, Any],
    expected_seed: int,
    expected_configuration_sha: str,
    expected_architecture_signature: str,
) -> WiSigRepresentationNet:
    required = {"model_state", "arm", "seed", "benchmark_sha", "stage2m_sha", "configuration_sha", "architecture_signature", "loss_coefficients"}
    missing = required - set(payload)
    if missing:
        raise ScientificAbort(f"A3 seed {expected_seed} checkpoint missing fields: {sorted(missing)}")
    checks = {
        "arm": payload.get("arm") == "A3",
        "seed": int(payload.get("seed", -1)) == expected_seed,
        "benchmark SHA": payload.get("benchmark_sha") == EXPECTED_BENCHMARK_SHA256,
        "Stage 2M SHA": payload.get("stage2m_sha") == EXPECTED_STAGE2M_SCRIPT_SHA256,
        "configuration SHA": payload.get("configuration_sha") == expected_configuration_sha,
        "architecture signature": payload.get("architecture_signature") == expected_architecture_signature,
        "loss coefficients": payload.get("loss_coefficients") == EXPECTED_LOSS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ScientificAbort(f"A3 seed {expected_seed} checkpoint provenance mismatch: {failed}")
    model = WiSigRepresentationNet()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise ScientificAbort(f"A3 seed {expected_seed} model-state integrity failure: {exc}") from exc
    if architecture_signature(model) != expected_architecture_signature or sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ScientificAbort(f"A3 seed {expected_seed} architecture mismatch")
    return model


def ready_gate(gates: Mapping[str, bool]) -> bool:
    return bool(gates) and all(gates.values())


class Stage3MPipeline:
    def __init__(self, config: Stage3MConfig):
        self.config = config
        config.validate()
        for name in REQUIRED_OUTPUT_DIRS:
            assert_within(config.output_root / name, config.output_root).mkdir(parents=True, exist_ok=True)
        self.script_sha = sha256_file(Path(__file__).resolve())
        self.guard = StrictZeroDayGuard()
        self.device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
        self.logger = self._logger()
        self.stage26_module: Any = None
        self.benchmark: Any = None
        self.local_h5: Optional[Path] = None
        self.stage26_status: Dict[str, Any] = {}
        self.stage26_hash_manifest: Dict[str, Any] = {}
        self.candidate_payloads: Dict[int, Mapping[str, Any]] = {}
        self.candidate_paths: Dict[int, Path] = {}

    def _logger(self) -> logging.Logger:
        logger = logging.getLogger(f"stage3m-{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter); logger.addHandler(stream)
        file_handler = logging.FileHandler(self.config.output_root / "logs" / "stage3m.log", encoding="utf-8"); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
        return logger

    def stage_manifest_path(self, stage: int) -> Path:
        return self.config.output_root / "manifests" / f"STAGE_{stage:02d}_CHECKPOINT.json"

    def _input_hashes(self, paths: Iterable[Path]) -> Dict[str, str]:
        return {str(path.resolve()): sha256_file(path) for path in paths}

    def stage_current(self, stage: int) -> bool:
        path = self.stage_manifest_path(stage)
        if not self.config.resume or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("pipeline_version") != PIPELINE_VERSION or payload.get("executable_sha256") != self.script_sha or payload.get("configuration_sha256") != self.config.configuration_sha256():
                return False
            for source, expected in payload.get("required_input_hashes", {}).items():
                candidate = Path(source)
                if not candidate.is_file() or sha256_file(candidate) != expected:
                    return False
            for output in payload.get("required_outputs", []):
                candidate = Path(output["path"])
                if not candidate.is_file() or sha256_file(candidate) != output["sha256"]:
                    return False
            return True
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    def complete_stage(self, stage: int, name: str, outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
        payload = {
            "stage": stage, "name": name, "status": "PASS", "pipeline_version": PIPELINE_VERSION,
            "executable_sha256": self.script_sha, "configuration_sha256": self.config.configuration_sha256(),
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "stage2m_script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "required_input_hashes": self._input_hashes(inputs),
            "required_outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
            "completed_at": utc_now(),
        }
        atomic_json(self.stage_manifest_path(stage), payload, self.config.output_root)
        print(f"[PASS] Stage {stage:02d} — {name}")

    def ensure_predecessor(self) -> None:
        if self.stage26_status:
            return
        if sha256_file(self.config.benchmark_path_resolved) != EXPECTED_BENCHMARK_SHA256:
            raise ScientificAbort("Canonical benchmark SHA mismatch")
        stage2m_status = self.config.stage2m_path / "manifests" / "STAGE2M_FINAL_STATUS.json"
        stage2m_hash = self.config.stage2m_path / "manifests" / "HASH_MANIFEST.json"
        if not stage2m_status.is_file() or not stage2m_hash.is_file() or sha256_file(stage2m_hash) != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256:
            raise ScientificAbort("Frozen Stage 2M provenance mismatch")
        stage2m = json.loads(stage2m_status.read_text(encoding="utf-8"))
        stage2m_required = {
            "status": "MANYTX_STAGE2M_READY",
            "stage_version": EXPECTED_STAGE2M_VERSION,
            "script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "recommendation": "PROCEED_STAGE_2_6M_WITH_CAUTION",
            "failed_gates": [],
            "final_test_model_evaluation_performed": False,
            "final_test_threshold_selection_performed": False,
        }
        if any(stage2m.get(key) != expected for key, expected in stage2m_required.items()):
            raise ScientificAbort("Frozen Stage 2M structured final status mismatch")
        stage2m_guard = stage2m.get("strict_test_guard", {})
        if any(stage2m_guard.get(key) != expected for key, expected in {
            "strict_index_arrays_loaded": False,
            "strict_test_signal_reads": 0,
            "strict_test_label_reads": 0,
            "strict_test_feature_reads": 0,
        }.items()):
            raise ScientificAbort("Frozen Stage 2M strict-test guard mismatch")
        ready_path = self.config.stage26_path / "MANYTX_STAGE2_6M_READY.txt"
        ready = parse_ready_marker(ready_path)
        if ready.get("decision") != EXPECTED_STAGE26_DECISION or ready.get("artifact_sha256") != EXPECTED_STAGE26_ARTIFACT_SHA256:
            raise ScientificAbort("Stage 2.6M READY decision/artifact mismatch")
        for key in STRICT_COUNTER_KEYS:
            if ready.get(key) != "0":
                raise ScientificAbort(f"Stage 2.6M strict counter is not zero: {key}")
        status_path = self.config.stage26_path / "manifests" / "STAGE2_6M_FINAL_STATUS.json"
        objective_path = self.config.stage26_path / "manifests" / "CANONICAL_STAGE3M_OBJECTIVE.json"
        hash_path = self.config.stage26_path / "manifests" / "HASH_MANIFEST.json"
        if not all(path.is_file() for path in (status_path, objective_path, hash_path)) or sha256_file(hash_path) != EXPECTED_STAGE26_ARTIFACT_SHA256:
            raise ScientificAbort("Frozen Stage 2.6M artifact hash mismatch")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        if (
            status.get("status") != "MANYTX_STAGE2_6M_READY"
            or status.get("decision") != EXPECTED_STAGE26_DECISION
            or status.get("selected_arm") != "A3"
            or status.get("benchmark_sha256") != EXPECTED_BENCHMARK_SHA256
            or status.get("stage2m_script_sha256") != EXPECTED_STAGE2M_SCRIPT_SHA256
            or status.get("stage2m_artifact_manifest_sha256") != EXPECTED_STAGE2M_HASH_MANIFEST_SHA256
            or tuple(status.get("seed_panel", ())) != EXPECTED_SEEDS
        ):
            raise ScientificAbort("Stage 2.6M final status is not the frozen A3 decision")
        if objective.get("decision") != EXPECTED_STAGE26_DECISION or objective.get("selected_arm") != "A3" or objective.get("loss_coefficients") != EXPECTED_LOSS:
            raise ScientificAbort("Stage 2.6M canonical objective mismatch")
        if any(status.get("strict_zero_day_violation_counters", {}).values()):
            raise ScientificAbort("Stage 2.6M final strict-zero-day counters are non-zero")
        self.stage26_status = status
        self.stage26_hash_manifest = json.loads(hash_path.read_text(encoding="utf-8"))

    def ensure_benchmark(self) -> Any:
        self.ensure_predecessor()
        if self.benchmark is not None:
            return self.benchmark
        performance_root = self.config.output_root / "performance"
        self.local_h5, report = LocalCacheManager(Path(self.config.local_cache_root), performance_root).ensure(self.config.benchmark_path_resolved, EXPECTED_BENCHMARK_SHA256)
        atomic_json(performance_root / "LOCAL_CACHE_REPORT.json", report, self.config.output_root)
        self.stage26_module = load_stage26_module(self.config.repository_root_path)
        self.benchmark = self.stage26_module.resolve_benchmark(self.config, self.guard, self.local_h5)
        return self.benchmark

    def checkpoint_hash_from_manifest(self, path: Path) -> str:
        relative = str(path.relative_to(self.config.stage26_path)).replace("\\", "/")
        rows = self.stage26_hash_manifest.get("files", [])
        matches = [row for row in rows if str(row.get("relative_path", "")).replace("\\", "/") == relative]
        if len(matches) != 1:
            raise ScientificAbort(f"Stage 2.6M hash manifest lacks unique checkpoint record: {relative}")
        return str(matches[0]["sha256"])

    def ensure_candidates(self) -> None:
        self.ensure_predecessor()
        if self.candidate_payloads:
            return
        for seed in EXPECTED_SEEDS:
            path = self.config.stage26_path / "checkpoints" / "A3" / f"seed_{seed}" / "best_selection.pt"
            if not path.is_file():
                raise ScientificAbort(f"A3 best-selection checkpoint missing: seed {seed}")
            actual = sha256_file(path)
            if actual != self.checkpoint_hash_from_manifest(path):
                raise ScientificAbort(f"A3 checkpoint hash mismatch: seed {seed}")
            payload = safe_torch_load(path)
            validate_candidate_checkpoint_payload(payload, seed, self.stage26_status["configuration_sha256"], self.stage26_status["architecture_signature"])
            self.candidate_paths[seed] = path
            self.candidate_payloads[seed] = payload

    def dataset(self, partition: str) -> BatchedSignalDataset:
        benchmark = self.ensure_benchmark()
        self.guard.reject("signal", partition)
        if partition not in (*KNOWN_PROTOCOLS, "calibration_unknown"):
            raise ScientificAbort(f"Stage 3M signal loader forbids partition: {partition}")
        metadata = benchmark.partitions[partition]
        self.guard.authorize_rows(partition, metadata.indices, "Stage 3M evaluation")
        return BatchedSignalDataset(benchmark.h5_path, benchmark.signal_key, benchmark.signal_orientation, partition, metadata, "single_local")

    def loader(self, dataset: BatchedSignalDataset) -> DataLoader:
        kwargs: Dict[str, Any] = {
            "dataset": dataset, "batch_size": self.config.eval_batch_size, "shuffle": False,
            "num_workers": self.config.num_workers, "pin_memory": self.config.pin_memory and self.device.type == "cuda",
            "collate_fn": identity_collate,
        }
        if self.config.num_workers:
            kwargs.update({"prefetch_factor": self.config.prefetch_factor, "persistent_workers": self.config.persistent_workers})
        return DataLoader(**kwargs)

    def model_for(self, seed: int) -> WiSigRepresentationNet:
        self.ensure_candidates()
        model = WiSigRepresentationNet().to(self.device)
        model.load_state_dict(self.candidate_payloads[seed]["model_state"], strict=True)
        return model.eval()

    def stage_01(self) -> None:
        self.ensure_predecessor()
        output = self.config.output_root / "manifests" / "STAGE01_INPUT_FREEZE.json"
        selection_policy = self.config.output_root / "configs" / "CANONICAL_TEACHER_SELECTION_POLICY.json"
        policy = {**SELECTION_POLICY, "selection_policy_sha256": sha256_object(SELECTION_POLICY), "written_before_selection": True}
        atomic_json(selection_policy, policy, self.config.output_root)
        payload = {
            "status": "PASS", "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "stage2m_script_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "stage2m_artifact_manifest_sha256": EXPECTED_STAGE2M_HASH_MANIFEST_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256,
            "stage2_6m_decision": EXPECTED_STAGE26_DECISION, "selected_arm": "A3",
            "stage2_6m_read_only": True, "teacher_source": "stage2_6m_promote",
            "strict_zero_day_counters": self.guard.counters(), "generated_at": utc_now(),
        }
        atomic_json(output, payload, self.config.output_root)
        predecessor_inputs = [
            self.config.benchmark_path_resolved,
            self.config.stage2m_path / "manifests" / "STAGE2M_FINAL_STATUS.json",
            self.config.stage2m_path / "manifests" / "HASH_MANIFEST.json",
            self.config.stage26_path / "MANYTX_STAGE2_6M_READY.txt",
            self.config.stage26_path / "manifests" / "STAGE2_6M_FINAL_STATUS.json",
            self.config.stage26_path / "manifests" / "CANONICAL_STAGE3M_OBJECTIVE.json",
            self.config.stage26_path / "manifests" / "HASH_MANIFEST.json",
        ]
        self.complete_stage(1, "Frozen Predecessor Verification", [output, selection_policy], predecessor_inputs)

    def stage_02(self) -> None:
        self.ensure_candidates()
        rows = []
        copied = []
        for seed, source in self.candidate_paths.items():
            destination = self.config.output_root / "checkpoints" / "candidates" / f"seed_{seed}" / "best_selection.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".copying")
            shutil.copyfile(source, temporary); os.replace(temporary, destination)
            if sha256_file(destination) != sha256_file(source):
                raise ScientificAbort(f"Candidate promotion copy mismatch: seed {seed}")
            copied.append(destination)
            payload = self.candidate_payloads[seed]
            rows.append({
                "seed": seed, "source_arm": "A3", "source_path": str(source), "source_checkpoint_sha256": sha256_file(source),
                "promoted_candidate_path": str(destination), "promoted_candidate_sha256": sha256_file(destination),
                "benchmark_sha256": payload["benchmark_sha"], "configuration_sha256": payload["configuration_sha"],
                "architecture_signature": payload["architecture_signature"], "loss_coefficients": payload["loss_coefficients"],
                "strict_model_state_load": True,
            })
        output = self.config.output_root / "manifests" / "A3_CANDIDATE_CHECKPOINT_MANIFEST.json"
        atomic_json(output, {"candidates": rows, "candidate_count": 3, "a0_a1_a2_excluded": True, "generated_at": utc_now()}, self.config.output_root)
        self.complete_stage(2, "A3 Candidate Checkpoint Provenance Audit", [output, *copied], list(self.candidate_paths.values()))

    def stage_03(self) -> None:
        self.ensure_candidates()
        frozen = load_stage26_module(self.config.repository_root_path)
        reference = frozen.WiSigRepresentationNet(98, 128, 0.1).eval()
        teacher = WiSigRepresentationNet(98, 128, 0.1).eval()
        x = torch.linspace(-1, 1, 4 * 2 * 256, dtype=torch.float32).reshape(4, 2, 256)
        with torch.inference_mode():
            first, second = teacher(x), teacher(x)
        norms = first["embedding_normalized"].norm(2, dim=1)
        checks = {
            "parameter_count_equivalent": sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in teacher.parameters()) == EXPECTED_PARAMETER_COUNT,
            "architecture_signature_equivalent": architecture_signature(reference) == architecture_signature(teacher) == self.stage26_status["architecture_signature"],
            "ordered_state_dict_keys_equivalent": list(reference.state_dict()) == list(teacher.state_dict()),
            "state_dict_tensor_shapes_equivalent": model_schema(reference) == model_schema(teacher),
            "logits_shape": list(first["logits"].shape) == [4, 98], "embedding_shape": list(first["embedding_normalized"].shape) == [4, 128],
            "normalized_embedding": bool(torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5)),
            "finite_outputs": all(torch.isfinite(value).all().item() for value in first.values()),
            "deterministic_eval_forward": all(torch.equal(first[key], second[key]) for key in first),
            "gradient_required": False,
        }
        if not all(value is True for value in checks.values()):
            raise ScientificAbort(f"Teacher architecture/forward equivalence failed: {checks}")
        output = self.config.output_root / "manifests" / "TEACHER_ARCHITECTURE_EQUIVALENCE.json"
        atomic_json(output, {"status": "PASS", "checks": checks, "parameter_count": EXPECTED_PARAMETER_COUNT, "architecture_signature": architecture_signature(teacher), "state_dict_schema": model_schema(teacher)}, self.config.output_root)
        self.complete_stage(3, "Architecture and Forward-Contract Equivalence", [output], [self.config.repository_root_path / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"])

    def embedding_store(self, seed: int, protocol: str) -> Path:
        return self.config.output_root / "embeddings" / f"seed_{seed}" / protocol

    def store_current(self, store: Path, checkpoint_sha: str, rows: int) -> bool:
        manifest = store / "store_manifest.json"
        required = ("embedding_normalized.npy", "logits.npy", "labels.npy", "global_indices.npy")
        if (store / "INCOMPLETE").exists() or not manifest.is_file() or not all((store / name).is_file() for name in required):
            return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if payload.get("complete") is not True or payload.get("checkpoint_sha256") != checkpoint_sha or int(payload.get("rows", -1)) != rows:
            return False
        try:
            shapes = {
                "embedding_normalized.npy": (rows, 128),
                "logits.npy": (rows, 98),
                "labels.npy": (rows,),
                "global_indices.npy": (rows,),
            }
            return all(tuple(np.load(store / name, mmap_mode="r").shape) == shape for name, shape in shapes.items())
        except (OSError, ValueError):
            return False

    def evaluate_candidate_protocol(self, seed: int, protocol: str) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
        benchmark = self.ensure_benchmark(); metadata = benchmark.partitions[protocol]
        checkpoint_sha = sha256_file(self.candidate_paths[seed]); store = self.embedding_store(seed, protocol)
        if self.store_current(store, checkpoint_sha, len(metadata.indices)):
            logits = np.load(store / "logits.npy", mmap_mode="r"); labels = np.load(store / "labels.npy", mmap_mode="r")
            accumulator = StreamingMetrics()
            for start in range(0, len(labels), self.config.eval_batch_size):
                end = min(start + self.config.eval_batch_size, len(labels))
                accumulator.update(torch.from_numpy(np.asarray(logits[start:end], dtype=np.float32)), torch.from_numpy(np.asarray(labels[start:end], dtype=np.int64)))
            metrics, per_class = accumulator.finish(); return metrics, per_class, accumulator.confusion
        store.mkdir(parents=True, exist_ok=True)
        atomic_text(store / "INCOMPLETE", utc_now() + "\n", self.config.output_root)
        partial_embedding = store / "embedding_normalized.partial.npy"; partial_logits = store / "logits.partial.npy"
        embeddings = np.lib.format.open_memmap(partial_embedding, mode="w+", dtype=np.float16, shape=(len(metadata.indices), 128))
        logits_store = np.lib.format.open_memmap(partial_logits, mode="w+", dtype=np.float16, shape=(len(metadata.indices), 98))
        model, dataset, accumulator, cursor = self.model_for(seed), self.dataset(protocol), StreamingMetrics(), 0
        use_amp = self.config.amp_enabled and self.device.type == "cuda"
        with torch.inference_mode():
            for batch in self.loader(dataset):
                x = batch["x"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(x)
                count = len(x)
                embeddings[cursor:cursor+count] = outputs["embedding_normalized"].float().cpu().numpy().astype(np.float16)
                logits_store[cursor:cursor+count] = outputs["logits"].float().cpu().numpy().astype(np.float16)
                accumulator.update(outputs["logits"], batch["y"]); cursor += count
        dataset.close(); embeddings.flush(); logits_store.flush(); del embeddings, logits_store, model
        if cursor != len(metadata.indices):
            raise ScientificAbort(f"Embedding extraction row mismatch: seed={seed}, protocol={protocol}")
        os.replace(partial_embedding, store / "embedding_normalized.npy"); os.replace(partial_logits, store / "logits.npy")
        np.save(store / "labels.npy", metadata.labels.astype(np.int16), allow_pickle=False); np.save(store / "global_indices.npy", metadata.indices.astype(np.int64), allow_pickle=False)
        atomic_json(store / "store_manifest.json", {"complete": True, "seed": seed, "protocol": protocol, "rows": cursor, "checkpoint_sha256": checkpoint_sha, "strict_zero_day_access": False, "generated_at": utc_now()}, self.config.output_root)
        (store / "INCOMPLETE").unlink()
        metrics, per_class = accumulator.finish(); return metrics, per_class, accumulator.confusion

    def stage_04(self) -> None:
        self.ensure_candidates(); known_rows, per_class_rows, outputs = [], [], []
        confusion_root = self.config.output_root / "metrics" / "confusion_matrices"; confusion_root.mkdir(parents=True, exist_ok=True)
        for seed in EXPECTED_SEEDS:
            for protocol in KNOWN_PROTOCOLS:
                metrics, per_class, confusion = self.evaluate_candidate_protocol(seed, protocol)
                known_rows.append({"seed": seed, "arm": "A3", "protocol": protocol, **metrics})
                per_class.insert(0, "protocol", protocol); per_class.insert(0, "seed", seed); per_class_rows.append(per_class)
                confusion_path = confusion_root / f"seed_{seed}_{protocol}.npy"; np.save(confusion_path, confusion, allow_pickle=False); outputs.append(confusion_path)
                self.logger.info("Evaluated seed %s protocol %s with %s rows", seed, protocol.upper(), f"{int(metrics['samples']):,}")
        known_path = self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv"
        per_class_path = self.config.output_root / "tables" / "teacher_candidate_per_class_metrics.csv"
        atomic_csv(known_path, pd.DataFrame(known_rows), self.config.output_root); atomic_csv(per_class_path, pd.concat(per_class_rows, ignore_index=True), self.config.output_root)
        store_outputs = []
        for seed in EXPECTED_SEEDS:
            for protocol in KNOWN_PROTOCOLS:
                store = self.embedding_store(seed, protocol)
                store_outputs.extend(store / name for name in ("embedding_normalized.npy", "logits.npy", "labels.npy", "global_indices.npy", "store_manifest.json"))
        self.complete_stage(4, "Known-Domain Canonical Evaluation", [known_path, per_class_path, *outputs, *store_outputs], list(self.candidate_paths.values()))

    def representation_metrics(self, embedding: np.ndarray, labels: np.ndarray, seed: int, protocol: str) -> Tuple[Dict[str, float], np.ndarray]:
        rng = np.random.default_rng(self.config.representation_sampling_seed + seed * 101 + KNOWN_PROTOCOLS.index(protocol))
        positions = []
        for label in range(98):
            members = np.flatnonzero(labels == label)
            if len(members) == 0:
                raise ScientificAbort(f"Representation diagnostics missing fixed-98 class {label}: seed={seed}, protocol={protocol}")
            take = min(len(members), self.config.representation_samples_per_class)
            positions.extend(rng.choice(members, size=take, replace=False).tolist())
        positions_array = np.asarray(sorted(positions), dtype=np.int64)
        x = np.asarray(embedding[positions_array], dtype=np.float32); y = np.asarray(labels[positions_array], dtype=np.int64)
        centroids = np.stack([x[y == label].mean(0) for label in range(98)])
        global_centroid = x.mean(0)
        intra_sq = np.concatenate([np.sum((x[y == label] - centroids[label]) ** 2, axis=1) for label in range(98)])
        between_sq = np.sum((centroids - global_centroid) ** 2, axis=1)
        centroid_distances = np.sqrt(np.maximum(((centroids[:, None] - centroids[None, :]) ** 2).sum(2), 0))
        upper = centroid_distances[np.triu_indices(98, 1)]
        compactness = float(np.sqrt(intra_sq).mean()); separation = float(upper.mean())
        result = {
            "silhouette_score": float(silhouette_score(x, y)), "davies_bouldin_index": float(davies_bouldin_score(x, y)),
            "calinski_harabasz_index": float(calinski_harabasz_score(x, y)), "mean_intra_class_distance": compactness,
            "mean_inter_class_distance": separation, "inter_intra_ratio": separation / max(compactness, 1e-12),
            "fisher_ratio": float(between_sq.mean() / max(intra_sq.mean(), 1e-12)),
            "prototype_separation": separation, "prototype_compactness": compactness, "sample_count": len(x),
        }
        return result, positions_array

    def stage_05(self) -> None:
        rows, samples = [], []
        for seed in EXPECTED_SEEDS:
            for protocol in KNOWN_PROTOCOLS:
                store = self.embedding_store(seed, protocol)
                if not self.store_current(store, sha256_file(self.candidate_paths[seed]), len(self.ensure_benchmark().partitions[protocol].indices)):
                    raise ScientificAbort(f"Stage 04 embedding store is incomplete: seed={seed}, protocol={protocol}")
                embedding = np.load(store / "embedding_normalized.npy", mmap_mode="r"); labels = np.load(store / "labels.npy", mmap_mode="r")
                metrics, positions = self.representation_metrics(embedding, labels, seed, protocol)
                rows.append({"seed": seed, "protocol": protocol, **metrics})
                samples.append({"seed": seed, "protocol": protocol, "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(), "sample_count": len(positions), "sampling_seed": self.config.representation_sampling_seed})
        table = self.config.output_root / "tables" / "teacher_representation_metrics.csv"
        sampling = self.config.output_root / "manifests" / "REPRESENTATION_METRIC_SAMPLING.json"
        atomic_csv(table, pd.DataFrame(rows), self.config.output_root); atomic_json(sampling, {"policy": "per-class deterministic without replacement", "records": samples}, self.config.output_root)
        store_inputs = []
        for seed in EXPECTED_SEEDS:
            for protocol in KNOWN_PROTOCOLS:
                store = self.embedding_store(seed, protocol)
                store_inputs.extend((store / "embedding_normalized.npy", store / "labels.npy", store / "store_manifest.json"))
        self.complete_stage(5, "Representation-Quality Diagnostics", [table, sampling], store_inputs)

    def stage_06(self) -> None:
        known_path = self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv"; frame = pd.read_csv(known_path)
        rows = []
        for seed in EXPECTED_SEEDS:
            data = frame[frame.seed == seed].set_index("protocol"); p0 = data.loc["p0"]
            for protocol in ("p1", "p2", "p3"):
                rows.append({"seed": seed, "source_protocol": "p0", "target_protocol": protocol, "fixed98_macro_f1_degradation": float(p0.fixed98_macro_f1 - data.loc[protocol].fixed98_macro_f1), "fixed98_balanced_accuracy_degradation": float(p0.fixed98_balanced_accuracy - data.loc[protocol].fixed98_balanced_accuracy)})
        table = self.config.output_root / "tables" / "teacher_domain_degradation.csv"; report = self.config.output_root / "reports" / "teacher_domain_robustness_report.md"
        degradation = pd.DataFrame(rows); atomic_csv(table, degradation, self.config.output_root)
        atomic_text(report, "# Teacher domain robustness\n\nMeasured known-validation P0-to-P1/P2/P3 degradation only. No strict-zero-day or Calibration Unknown input contributed to these values.\n\n" + degradation.to_markdown(index=False) + "\n", self.config.output_root)
        self.complete_stage(6, "Domain Robustness Analysis", [table, report], [known_path])

    def stage_07(self) -> None:
        output = self.config.output_root / "tables" / "calibration_unknown_teacher_diagnostics.csv"
        if not self.config.calibration_unknown_diagnostic:
            atomic_csv(output, pd.DataFrame([{"status": "DISABLED", "label": "CALIBRATION_UNKNOWN_DIAGNOSTIC_ONLY", "used_for_selection": False, "threshold_fitting": False}]), self.config.output_root)
        else:
            rows = []
            for seed in EXPECTED_SEEDS:
                dataset, model = self.dataset("calibration_unknown"), self.model_for(seed)
                norms, cursor = [], 0
                with torch.inference_mode():
                    for batch in self.loader(dataset):
                        x = batch["x"].to(self.device, non_blocking=True); values = model(x)["embedding_normalized"].norm(2, dim=1).cpu().numpy(); norms.extend(values.tolist()); cursor += len(x)
                dataset.close(); del model
                rows.append({"seed": seed, "status": "CALIBRATION_UNKNOWN_DIAGNOSTIC_ONLY", "samples": cursor, "mean_embedding_norm": float(np.mean(norms)), "std_embedding_norm": float(np.std(norms)), "used_for_selection": False, "threshold_fitting": False})
            atomic_csv(output, pd.DataFrame(rows), self.config.output_root)
        self.complete_stage(7, "Optional Calibration Unknown Diagnostic", [output], [])

    def stage_08(self) -> None:
        policy_path = self.config.output_root / "configs" / "CANONICAL_TEACHER_SELECTION_POLICY.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("selection_policy_sha256") != sha256_object(SELECTION_POLICY):
            raise ScientificAbort("Canonical teacher selection policy hash mismatch")
        known_path = self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv"; rep_path = self.config.output_root / "tables" / "teacher_representation_metrics.csv"
        selected_seed, ranking = deterministic_selection(pd.read_csv(known_path), pd.read_csv(rep_path))
        table = self.config.output_root / "tables" / "canonical_teacher_selection_table.csv"
        selection = self.config.output_root / "manifests" / "CANONICAL_TEACHER_SELECTION.json"
        report = self.config.output_root / "reports" / "CANONICAL_TEACHER_SELECTION_REPORT.md"
        atomic_csv(table, ranking, self.config.output_root)
        payload = {"status": "PASS", "selected_seed": selected_seed, "source_arm": "A3", "objective": "CE_SUPCON_PROTOTYPE", "selection_policy_sha256": sha256_object(SELECTION_POLICY), "ranking": ranking.to_dict("records"), "known_validation_only": True, "generated_at": utc_now()}
        atomic_json(selection, payload, self.config.output_root)
        atomic_text(report, f"# Canonical teacher selection\n\nSelected A3 seed **{selected_seed}** by the predeclared known-validation-only deterministic hierarchy. Calibration Unknown and strict-zero-day data were not selection inputs.\n\n" + ranking.to_markdown(index=False) + "\n", self.config.output_root)
        self.complete_stage(8, "Canonical Teacher Deterministic Selection", [selection, report, table], [policy_path, known_path, rep_path])

    def stage_09(self) -> None:
        self.ensure_candidates(); selection_path = self.config.output_root / "manifests" / "CANONICAL_TEACHER_SELECTION.json"; selection = json.loads(selection_path.read_text(encoding="utf-8")); seed = int(selection["selected_seed"])
        source_path, source_payload = self.candidate_paths[seed], self.candidate_payloads[seed]
        source_state = source_payload["model_state"]
        canonical_dir = self.config.output_root / "checkpoints" / "canonical"; canonical_dir.mkdir(parents=True, exist_ok=True)
        state_path = canonical_dir / "canonical_teacher_state_dict.pt"; checkpoint_path = canonical_dir / "canonical_teacher_v1_0.pt"
        atomic_torch_save(state_path, source_state, self.config.output_root)
        metadata = {
            "teacher_version": TEACHER_VERSION, "model_state": source_state, "selected_seed": seed, "source_arm": "A3",
            "objective": "CE + SupCon + Prototype", "loss_coefficients": EXPECTED_LOSS,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "stage2m_sha256": EXPECTED_STAGE2M_SCRIPT_SHA256,
            "stage2_6m_artifact_sha256": EXPECTED_STAGE26_ARTIFACT_SHA256, "source_stage2_6m_checkpoint_sha256": sha256_file(source_path),
            "stage3m_executable_sha256": self.script_sha, "stage3m_configuration_sha256": self.config.configuration_sha256(),
            "architecture_signature": source_payload["architecture_signature"], "parameter_count": EXPECTED_PARAMETER_COUNT,
            "embedding_dimension": 128, "known_classes": 98, "selection_score": selection["ranking"][0],
            "p0_p3_metrics": pd.read_csv(self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv").query("seed == @seed").to_dict("records"),
            "selection_policy_sha256": sha256_object(SELECTION_POLICY), "strict_zero_day_counters": self.guard.counters(),
            "provenance_chain": ["Stage1B v1.0.3", "Stage2M v1.0.5", "Stage2.6M A3 best-selection", "Stage3M promotion"],
            "freeze_timestamp": utc_now(), "training_performed": False,
        }
        atomic_torch_save(checkpoint_path, metadata, self.config.output_root)
        exported_state = torch.load(state_path, map_location="cpu", weights_only=False); exported_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not state_dict_equal(source_state, exported_state) or not state_dict_equal(source_state, exported_checkpoint["model_state"]):
            raise ScientificAbort("Canonical teacher export differs from selected source weights")
        freeze = self.config.output_root / "manifests" / "TEACHER_FREEZE.json"; model_card = self.config.output_root / "reports" / "TEACHER_MODEL_CARD.md"; hash_manifest = self.config.output_root / "manifests" / "TEACHER_HASH_MANIFEST.json"
        atomic_json(freeze, {**{key: value for key, value in metadata.items() if key != "model_state"}, "source_export_state_dict_element_equivalent": True}, self.config.output_root)
        atomic_text(model_card, f"# Stage 3M canonical teacher model card\n\n- Source: frozen Stage 2.6M A3 seed {seed}\n- Objective: CE + 0.1 SupCon + 0.1 Prototype\n- Input: 2×256 I/Q\n- Output: 98 logits and normalized 128-D embedding\n- Intended downstream use: Stage 3.5M, 4M, 5M, 6M, 7M, and locked Stage 9M workflows\n- Prohibited claim: this artifact contains no final zero-day evaluation.\n", self.config.output_root)
        atomic_json(hash_manifest, {"algorithm": "SHA-256", "files": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in (checkpoint_path, state_path, freeze, model_card)], "source_weight_element_equivalence": True}, self.config.output_root)
        self.complete_stage(9, "Canonical Teacher Export and Freeze", [checkpoint_path, state_path, freeze, model_card, hash_manifest], [source_path, selection_path, self.config.output_root / "configs" / "CANONICAL_TEACHER_SELECTION_POLICY.json", self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv"])

    def generate_publication(self) -> List[Path]:
        known = pd.read_csv(self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv"); degradation = pd.read_csv(self.config.output_root / "tables" / "teacher_domain_degradation.csv"); representation = pd.read_csv(self.config.output_root / "tables" / "teacher_representation_metrics.csv")
        figures: List[Path] = []
        specs = [
            ("known_protocol_macro_f1", known, "protocol", "fixed98_macro_f1", "Known protocol fixed-98 macro-F1"),
            ("domain_degradation", degradation, "target_protocol", "fixed98_macro_f1_degradation", "Known-domain macro-F1 degradation"),
            ("representation_quality", representation, "protocol", "fisher_ratio", "Representation Fisher ratio"),
        ]
        for name, frame, x_col, y_col, title in specs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for seed, group in frame.groupby("seed"):
                group.groupby(x_col)[y_col].mean().plot(marker="o", ax=ax, label=f"seed {seed}")
            ax.set_title(title); ax.set_ylabel(y_col.replace("_", " ")); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = self.config.output_root / "figures" / f"{name}.{suffix}"; fig.savefig(path, dpi=220 if suffix == "png" else None); figures.append(path)
            plt.close(fig)
        report = self.config.output_root / "reports" / "Stage3M_Canonical_Teacher_Report.md"
        selection = json.loads((self.config.output_root / "manifests" / "CANONICAL_TEACHER_SELECTION.json").read_text(encoding="utf-8"))
        numeric_metrics = [column for column in known.select_dtypes(include=[np.number]).columns if column != "seed"]
        summary = known.groupby("protocol")[numeric_metrics].agg(["mean", "std", "median", "min", "max"])
        summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
        summary = summary.reset_index()
        summary_path = self.config.output_root / "statistics" / "teacher_candidate_metric_summary.csv"
        atomic_csv(summary_path, summary, self.config.output_root)
        atomic_text(report, f"# Stage 3M canonical teacher report\n\nMeasured known-validation results select A3 seed **{selection['selected_seed']}** by the frozen hierarchy. No retraining, strict-zero-day evaluation, surrogate training, or XAI occurred. With only three candidate seeds, no new significance claim is made.\n\n## Candidate summary\n\n{summary.to_markdown(index=False)}\n", self.config.output_root)
        workbook = self.config.output_root / "publication" / "Stage3M_tables.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            known.to_excel(writer, sheet_name="known_metrics", index=False); degradation.to_excel(writer, sheet_name="degradation", index=False); representation.to_excel(writer, sheet_name="representation", index=False)
        latex = self.config.output_root / "publication" / "Stage3M_latex_tables.tex"; atomic_text(latex, known.to_latex(index=False) + "\n\n" + degradation.to_latex(index=False) + "\n\n" + representation.to_latex(index=False), self.config.output_root)
        report_pdf = self.config.output_root / "publication" / "Stage3M_report.pdf"
        with PdfPages(report_pdf) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.08, 0.95, "Stage 3M Canonical Teacher Report", fontsize=18, weight="bold"); fig.text(0.08, 0.88, f"Selected A3 seed: {selection['selected_seed']}\nKnown-validation-only deterministic selection\nStrict zero-day access: 0\nTraining performed: NO", fontsize=12, va="top"); plt.axis("off"); pdf.savefig(fig); plt.close(fig)
            for path in figures:
                if path.suffix == ".png":
                    image = plt.imread(path); fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.imshow(image); ax.axis("off"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        figure_manifest = self.config.output_root / "publication" / "FIGURE_MANIFEST.json"; atomic_json(figure_manifest, {"figures": [{"path": str(path), "sha256": sha256_file(path)} for path in figures], "generated_from_measured_outputs": True}, self.config.output_root)
        return [report, summary_path, workbook, latex, report_pdf, figure_manifest, *figures]

    def stage_10(self) -> None:
        self.ensure_candidates()
        self.guard.scan_output(self.config.output_root); self.guard.assert_zero()
        for stage in range(1, 10):
            if not self.stage_current(stage):
                raise ScientificAbort(f"Stage {stage:02d} is stale or incomplete")
        publication = self.generate_publication()
        selection_path = self.config.output_root / "manifests" / "CANONICAL_TEACHER_SELECTION.json"; selection = json.loads(selection_path.read_text(encoding="utf-8")); seed = int(selection["selected_seed"])
        canonical = self.config.output_root / "checkpoints" / "canonical" / "canonical_teacher_v1_0.pt"; state = self.config.output_root / "checkpoints" / "canonical" / "canonical_teacher_state_dict.pt"
        gates = {
            "predecessor_verified": True, "a3_only": True, "three_candidates": set(self.candidate_paths) == set(EXPECTED_SEEDS),
            "known_evaluation_complete": len(pd.read_csv(self.config.output_root / "tables" / "teacher_candidate_known_metrics.csv")) == 12,
            "representation_complete": len(pd.read_csv(self.config.output_root / "tables" / "teacher_representation_metrics.csv")) == 12,
            "selection_unique": seed in EXPECTED_SEEDS, "canonical_checkpoint_exists": canonical.is_file(), "canonical_state_exists": state.is_file(),
            "strict_zero_day_zero": not any(self.guard.counters().values()), "final_zero_day_evaluation_not_performed": True,
            "surrogate_training_not_performed": True, "xai_not_performed": True, "publication_complete": all(path.is_file() for path in publication),
        }
        if not ready_gate(gates):
            raise ScientificAbort(f"Stage 3M READY gates failed: {gates}")
        final_status = self.config.output_root / "manifests" / "STAGE3M_FINAL_STATUS.json"
        atomic_json(final_status, {"status": "MANYTX_STAGE3M_READY", "pipeline_version": PIPELINE_VERSION, "selected_seed": seed, "gates": gates, "strict_zero_day_counters": self.guard.counters(), "generated_at": utc_now()}, self.config.output_root)
        outputs = [final_status, *publication]
        self.complete_stage(10, "Final Audit and READY Gate", outputs, [self.stage_manifest_path(stage) for stage in range(1, 10)])
        hash_manifest = self.config.output_root / "manifests" / "STAGE3M_HASH_MANIFEST.json"
        excluded = {hash_manifest.resolve()}
        files = [path for path in sorted(self.config.output_root.rglob("*")) if path.is_file() and path.resolve() not in excluded and path.name not in {"MANYTX_STAGE3M_READY.txt", "MANYTX_STAGE3M_NOT_READY.txt"}]
        hash_rows = [{"relative_path": str(path.relative_to(self.config.output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files]
        atomic_json(hash_manifest, {"algorithm": "SHA-256", "files": hash_rows, "count": len(hash_rows)}, self.config.output_root)
        verified_manifest = json.loads(hash_manifest.read_text(encoding="utf-8"))
        if verified_manifest.get("count") != len(hash_rows) or any(
            not (self.config.output_root / row["relative_path"]).is_file()
            or sha256_file(self.config.output_root / row["relative_path"]) != row["sha256"]
            or (self.config.output_root / row["relative_path"]).stat().st_size != row["bytes"]
            for row in verified_manifest.get("files", [])
        ):
            raise ScientificAbort("Final Stage 3M hash manifest is inconsistent")
        freeze = json.loads((self.config.output_root / "manifests" / "TEACHER_FREEZE.json").read_text(encoding="utf-8")); policy_sha = sha256_object(SELECTION_POLICY)
        ready = self.config.output_root / "MANYTX_STAGE3M_READY.txt"; not_ready = self.config.output_root / "MANYTX_STAGE3M_NOT_READY.txt"
        if not_ready.exists(): not_ready.unlink()
        ready_text = (
            "MANYTX_STAGE3M_READY\n" f"teacher_version={TEACHER_VERSION}\n" "objective=CE_SUPCON_PROTOTYPE\nsource_arm=A3\n"
            f"selected_seed={seed}\nbenchmark_sha256={EXPECTED_BENCHMARK_SHA256}\nstage2m_sha256={EXPECTED_STAGE2M_SCRIPT_SHA256}\n"
            f"stage2_6m_artifact_sha256={EXPECTED_STAGE26_ARTIFACT_SHA256}\nsource_checkpoint_sha256={freeze['source_stage2_6m_checkpoint_sha256']}\n"
            f"canonical_teacher_sha256={sha256_file(canonical)}\narchitecture_signature={freeze['architecture_signature']}\n"
            f"configuration_sha256={self.config.configuration_sha256()}\nselection_policy_sha256={policy_sha}\n"
            + "".join(f"{key}=0\n" for key in STRICT_COUNTER_KEYS)
            + "final_zero_day_evaluation_performed=NO\nsurrogate_training_performed=NO\nxai_performed=NO\nnext_stage=STAGE_3_5M\n"
        )
        atomic_text(ready, ready_text, self.config.output_root)
        print("\nMANYTX_STAGE3M_READY")

    def run(self) -> None:
        print(startup_banner())
        stages = {1: self.stage_01, 2: self.stage_02, 3: self.stage_03, 4: self.stage_04, 5: self.stage_05, 6: self.stage_06, 7: self.stage_07, 8: self.stage_08, 9: self.stage_09, 10: self.stage_10}
        for stage in range(self.config.stage_start, self.config.stage_end + 1):
            if self.stage_current(stage):
                print(f"[REUSE] Stage {stage:02d} — hash-current")
                continue
            stages[stage]()
        if self.config.preflight:
            print("\nSTAGE3M_PREFLIGHT_PASS")


def startup_banner() -> str:
    return "\n".join([
        "=" * 100, "STAGE 3M — WISIG MANYTX CANONICAL TEACHER FREEZE", "=" * 100,
        f"CANONICAL BENCHMARK: {CANONICAL_BENCHMARK}", "STAGE 2.6M: VERIFIED / FROZEN",
        "SELECTED OBJECTIVE: CE + SUPCON + PROTOTYPE", "TEACHER CANDIDATES: SEEDS 42 / 123 / 2026",
        "TEACHER SOURCE: STAGE 2.6M PROMOTION", "ARCHITECTURE SEARCH: DISABLED",
        "TRAINING: DISABLED IN CANONICAL PROMOTION MODE", "STRICT ZERO-DAY ACCESS: FORBIDDEN",
        "FINAL ZERO-DAY EVALUATION: DISABLED", "SURROGATE TRAINING: DISABLED", "XAI: DISABLED", "=" * 100,
    ])


def synthetic_validation() -> None:
    model = WiSigRepresentationNet().eval(); x = torch.randn(8, 2, 256)
    with torch.inference_mode(): outputs = model(x)
    assert outputs["logits"].shape == (8, 98) and outputs["embedding_normalized"].shape == (8, 128)
    assert torch.allclose(outputs["embedding_normalized"].norm(2, dim=1), torch.ones(8), atol=2e-5)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETER_COUNT
    known = pd.DataFrame([{"seed": seed, "protocol": protocol, "fixed98_macro_f1": 0.5 + seed / 1e6, "fixed98_balanced_accuracy": 0.6 + seed / 1e6} for seed in EXPECTED_SEEDS for protocol in KNOWN_PROTOCOLS])
    representation = pd.DataFrame([{"seed": seed, "protocol": protocol, "fisher_ratio": float(seed)} for seed in EXPECTED_SEEDS for protocol in KNOWN_PROTOCOLS])
    selected, _ = deterministic_selection(known, representation); assert selected == 2026
    assert not ready_gate({"a": True, "b": False}) and ready_gate({"a": True, "b": True})
    print("STAGE3M_SYNTHETIC_VALIDATION_PASS")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3M WiSig ManyTx canonical teacher freeze")
    parser.add_argument("--config", type=Path); parser.add_argument("--branch-root"); parser.add_argument("--repository-root")
    parser.add_argument("--stage-start", type=int); parser.add_argument("--stage-end", type=int); parser.add_argument("--device")
    parser.add_argument("--teacher-source", choices=("stage2_6m_promote",)); parser.add_argument("--resume", action="store_true"); parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preflight", action="store_true"); parser.add_argument("--synthetic-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.synthetic_validation:
        synthetic_validation(); return 0
    values: Dict[str, Any] = {}
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    for name in ("branch_root", "repository_root", "stage_start", "stage_end", "device", "teacher_source"):
        value = getattr(args, name)
        if value is not None: values[name] = value
    if args.resume and args.no_resume:
        raise ValueError("--resume and --no-resume are mutually exclusive")
    if args.resume: values["resume"] = True
    if args.no_resume: values["resume"] = False
    if args.preflight:
        values.update({"preflight": True, "stage_start": 1, "stage_end": 3})
    if "branch_root" not in values:
        values["branch_root"] = os.environ.get("WISIG_BRANCH_ROOT", "/content/drive/MyDrive/colab files /Surrogate-XAI/project_root/MANYTX_ZERO_DAY_BRANCH_v1.0.3")
    config = Stage3MConfig(**values)
    try:
        Stage3MPipeline(config).run(); return 0
    except Exception as exc:
        output = config.output_root; output.mkdir(parents=True, exist_ok=True)
        ready = output / "MANYTX_STAGE3M_READY.txt"
        if ready.exists(): ready.unlink()
        atomic_text(output / "MANYTX_STAGE3M_NOT_READY.txt", f"MANYTX_STAGE3M_NOT_READY\n{type(exc).__name__}: {exc}\n", output)
        print(f"MANYTX_STAGE3M_NOT_READY\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
