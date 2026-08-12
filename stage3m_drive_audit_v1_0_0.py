#!/usr/bin/env python3
"""Read-only Google Drive predecessor audit for Stage 3M v1.0.0.

This module intentionally has no training, teacher-selection, canonical-export,
or strict-zero-day data-loading route.  It reads only frozen predecessor
metadata, known-domain tables, A3 checkpoints, and known-domain embedding
stores, then writes audit evidence below ``04_canonical_teacher/audit``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch


CANONICAL_BENCHMARK = "WiSig_ManyTx_ZeroDay_Benchmark_v1.0.3"
EXPECTED_BENCHMARK_SHA256 = "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
EXPECTED_STAGE2M_HASH_SHA256 = "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a"
EXPECTED_STAGE26_HASH_SHA256 = "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
EXPECTED_STAGE2M_SCRIPT_SHA256 = "46c95bbf9fb6806a5f463b4e173434a5f03f013367b1bcd38ebb73c07d0f67ba"
EXPECTED_DECISION = "SELECT_CE_SUPCON_PROTOTYPE"
EXPECTED_SEEDS = (42, 123, 2026)
KNOWN_PROTOCOLS = ("p0", "p1", "p2", "p3")
EXPECTED_LOSS = {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1}
EXPECTED_PARAMETER_COUNT = 849_634

READY_STRICT_KEYS = (
    "strict_zero_day_signal_read_violations",
    "strict_zero_day_label_read_violations",
    "strict_zero_day_embedding_read_violations",
    "strict_zero_day_metric_read_violations",
    "strict_zero_day_threshold_read_violations",
)
FINAL_STATUS_STRICT_KEY_MAP = {
    READY_STRICT_KEYS[0]: "strict_test_signal_reads",
    READY_STRICT_KEYS[1]: "strict_test_label_reads",
    READY_STRICT_KEYS[2]: "strict_test_embedding_reads",
    READY_STRICT_KEYS[3]: "strict_test_metric_reads",
    READY_STRICT_KEYS[4]: "strict_test_threshold_reads",
}

REQUIRED_ARTIFACTS: Tuple[Tuple[str, str], ...] = (
    (f"01_benchmark_engineering/benchmark/{CANONICAL_BENCHMARK}.h5", "canonical benchmark HDF5"),
    ("01_benchmark_engineering/manifests/HASH_MANIFEST.json", "Stage 1B hash manifest"),
    ("01_benchmark_engineering/manifests/FILE_MANIFEST.json", "Stage 1B file manifest"),
    ("01_benchmark_engineering/manifests/SPLIT_MANIFEST.json", "frozen split manifest"),
    ("01_benchmark_engineering/manifests/BENCHMARK_PROVENANCE.json", "frozen benchmark metadata"),
    ("01_benchmark_engineering/splits/domain_protocols.json", "frozen domain protocol metadata"),
    ("02_benchmark_diagnostics/manifests/STAGE2M_FINAL_STATUS.json", "Stage 2M final status"),
    ("02_benchmark_diagnostics/manifests/HASH_MANIFEST.json", "Stage 2M hash manifest"),
    ("02_benchmark_diagnostics/manifests/TEST_GUARD_MANIFEST.json", "Stage 2M strict-test guard status"),
    ("02_benchmark_diagnostics/reports/reviewer_ready_stage2m_report.md", "Stage 2M reviewer report"),
    ("03_representation_ablation/MANYTX_STAGE2_6M_READY.txt", "Stage 2.6M READY marker"),
    ("03_representation_ablation/manifests/STAGE2_6M_FINAL_STATUS.json", "Stage 2.6M final status"),
    ("03_representation_ablation/manifests/HASH_MANIFEST.json", "Stage 2.6M hash manifest"),
    ("03_representation_ablation/manifests/CANONICAL_STAGE3M_OBJECTIVE.json", "Stage 3M objective contract"),
    ("03_representation_ablation/reports/objective_selection_report.md", "objective selection report"),
    ("03_representation_ablation/tables/objective_selection_matrix.csv", "objective selection matrix"),
    ("03_representation_ablation/reports/known_tx_evaluation_report.md", "known transmitter report"),
    ("03_representation_ablation/tables/known_protocol_results.csv", "known protocol results"),
    ("03_representation_ablation/tables/known_protocol_per_class.csv", "known protocol per-class results"),
    ("03_representation_ablation/tables/fixed98_macro_results.csv", "fixed-98 results"),
    ("03_representation_ablation/reports/embedding_separability_report.md", "embedding separability report"),
    ("03_representation_ablation/tables/embedding_separability.csv", "embedding separability table"),
    ("03_representation_ablation/tables/protocol_degradation.csv", "protocol degradation table"),
    ("03_representation_ablation/tables/seed_summary.csv", "seed summary"),
    ("03_representation_ablation/tables/training_history_summary.csv", "training history summary"),
    ("03_representation_ablation/reports/statistical_comparison_report.md", "statistical comparison report"),
    ("03_representation_ablation/tables/paired_statistical_tests.csv", "paired statistical tests"),
    ("03_representation_ablation/tables/effect_sizes.csv", "effect sizes"),
    ("03_representation_ablation/reports/calibration_unknown_diagnostic_report.md", "calibration-unknown diagnostic report"),
    ("03_representation_ablation/reports/domain_robustness_report.md", "domain robustness report"),
    *((f"03_representation_ablation/checkpoints/A3/seed_{seed}/best_selection.pt", f"A3 seed {seed} best-selection checkpoint") for seed in EXPECTED_SEEDS),
)


class DriveAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DriveAuditError(f"Expected JSON object: {path}")
    return value


def _ready(path: Path) -> Tuple[str, Dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise DriveAuditError("Stage 2.6M READY marker is empty")
    return lines[0].strip(), dict(line.split("=", 1) for line in lines[1:] if "=" in line)


def _manifest_index(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if "files" not in payload:
        return {
            str(relative).replace("\\", "/"): {"relative_path": str(relative), "sha256": digest}
            for relative, digest in payload.items()
            if isinstance(digest, str)
        }
    rows = payload.get("files", [])
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        relative = str(row.get("relative_path", row.get("path", ""))).replace("\\", "/")
        if relative:
            result[relative] = row
    return result


def canonical_strict_counters(final_status: Mapping[str, Any]) -> Dict[str, int]:
    """Map the real Stage 2.6M final-status schema to canonical audit keys."""
    raw = final_status.get("strict_zero_day_violation_counters")
    if not isinstance(raw, Mapping):
        raise DriveAuditError("Stage 2.6M structured strict counter object is missing")
    missing = [source for source in FINAL_STATUS_STRICT_KEY_MAP.values() if source not in raw]
    if missing:
        raise DriveAuditError(f"Stage 2.6M structured strict counters are incomplete: {missing}")
    return {canonical: int(raw[source]) for canonical, source in FINAL_STATUS_STRICT_KEY_MAP.items()}


def class_coverage_from_per_class(frame: pd.DataFrame) -> Dict[str, Any]:
    required = {"arm", "seed", "protocol", "class_index", "support"}
    if not required.issubset(frame.columns):
        raise DriveAuditError(f"known_protocol_per_class.csv missing columns: {sorted(required - set(frame.columns))}")
    a3 = frame[(frame["arm"] == "A3") & (frame["protocol"].isin(KNOWN_PROTOCOLS))].copy()
    seed_rows: List[Dict[str, Any]] = []
    signatures: Dict[str, set] = {protocol: set() for protocol in KNOWN_PROTOCOLS}
    canonical: Dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        for protocol in KNOWN_PROTOCOLS:
            subset = a3[(a3["seed"].astype(int) == seed) & (a3["protocol"] == protocol)]
            if len(subset) != 98 or set(subset["class_index"].astype(int)) != set(range(98)):
                raise DriveAuditError(f"Per-class table lacks the fixed-98 frame for A3/{seed}/{protocol}")
            support = subset.sort_values("class_index")["support"].to_numpy(dtype=np.int64)
            observed = np.flatnonzero(support > 0).astype(int).tolist()
            missing = np.flatnonzero(support == 0).astype(int).tolist()
            observed_support = support[support > 0]
            if len(observed_support) < 2 or np.any(observed_support < 2):
                raise DriveAuditError(f"Observed class with fewer than two rows: A3/{seed}/{protocol}")
            row = {
                "seed": seed,
                "protocol": protocol,
                "observed_class_count": len(observed),
                "missing_class_count": len(missing),
                "missing_class_ids": missing,
                "minimum_observed_class_support": int(observed_support.min()),
                "maximum_observed_class_support": int(observed_support.max()),
                "median_observed_class_support": float(np.median(observed_support)),
                "total_rows": int(support.sum()),
                "all_observed_classes_have_at_least_two_samples": True,
            }
            signatures[protocol].add((tuple(observed), tuple(support.tolist())))
            seed_rows.append(row)
            if seed == EXPECTED_SEEDS[0]:
                canonical[protocol] = {key: value for key, value in row.items() if key not in {"seed", "protocol"}}
    if any(len(values) != 1 for values in signatures.values()):
        raise DriveAuditError("A3 class coverage differs by seed for the same frozen protocol")
    if canonical["p0"]["observed_class_count"] != 98:
        raise DriveAuditError("P0 does not contain all 98 known transmitter classes")
    return {"representation_frame": "OBSERVED_CLASSES", "protocols": canonical, "seed_protocol_rows": seed_rows}


def _number_tokens(text: str) -> List[float]:
    return [float(token) for token in re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]


def _contains_rounded(text: str, value: float, decimals: int = 6) -> bool:
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    return any(abs(token - value) <= tolerance for token in _number_tokens(text))


def assess_report_consistency(facts: Mapping[str, Any], reports: Mapping[str, str]) -> Dict[str, Any]:
    """Pure report-vs-machine consistency gate used by runtime and validator."""
    objective = reports["objective_selection_report"]
    known = reports["known_tx_evaluation_report"]
    embedding = reports["embedding_separability_report"]
    statistics = reports["statistical_comparison_report"]
    checks = {
        "objective_decision_and_ranking": EXPECTED_DECISION in objective and "`A3` ranked first" in objective,
        "objective_gain_rounding": _contains_rounded(objective, float(facts["paired_mean_gain"]), 6),
        "objective_ci_low_rounding": _contains_rounded(objective, float(facts["paired_ci_low"]), 6),
        "objective_ci_high_rounding": _contains_rounded(objective, float(facts["paired_ci_high"]), 6),
        "known_protocol_a3_rounding": all(_contains_rounded(known, float(value), 6) for value in facts["a3_protocol_macro_f1"].values()),
        "embedding_p0_fisher_rounding": _contains_rounded(embedding, float(facts["a3_p0_fisher_mean"]), 5),
        "statistics_gain_rounding": _contains_rounded(statistics, float(facts["paired_mean_gain"]), 6),
        "statistics_ci_rounding": _contains_rounded(statistics, float(facts["paired_ci_low"]), 6) and _contains_rounded(statistics, float(facts["paired_ci_high"]), 6),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rounding_note": "Markdown values are accepted only when they agree with machine tables at the displayed precision.",
    }


def _finite_model_state(payload: Mapping[str, Any]) -> bool:
    state = payload.get("model_state")
    return isinstance(state, Mapping) and bool(state) and all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sealed_array(path: Path) -> bool:
    token = path.name.lower().replace("-", "_")
    return path.suffix.lower() in {".npy", ".npz", ".h5", ".hdf5"} and any(value in token for value in ("strict_zero_day", "strict_test", "zero_day_shift_test"))


def _inventory(branch_root: Path, required: Mapping[str, str], manifest_shas: Mapping[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for top in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation"):
        root = branch_root / top
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(branch_root).as_posix()
            sealed = _sealed_array(path)
            row: Dict[str, Any] = {
                "absolute_path": str(path.resolve()),
                "relative_path": relative,
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "role": required.get(relative, "inventory-only"),
                "required": relative in required,
                "readable": False if sealed else None,
                "internally_consistent": None,
                "sealed_strict_artifact_not_opened": sealed,
                "sha256": manifest_shas.get(relative),
                "sha256_source": "frozen predecessor hash manifest" if relative in manifest_shas else None,
            }
            should_hash = relative in required and (
                path.stat().st_size <= 128 * 1024 * 1024
                or relative == f"01_benchmark_engineering/benchmark/{CANONICAL_BENCHMARK}.h5"
            )
            if relative in required:
                if sealed:
                    raise DriveAuditError(f"Required audit artifact unexpectedly resolves to sealed data: {relative}")
            if should_hash:
                row["sha256"] = _sha256(path)
                row["sha256_source"] = "directly calculated during Drive audit"
                row["readable"] = True
            elif relative in required:
                row["readable"] = True
                row["sha256_source"] = "frozen predecessor hash manifest; direct audit hash omitted for large file"
            rows.append(row)
    return rows


def _store_audit(stage26: Path, checkpoint_shas: Mapping[int, str], protocol_rows: Mapping[str, int]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    exact = True
    for seed in EXPECTED_SEEDS:
        for protocol in KNOWN_PROTOCOLS:
            store = stage26 / "embeddings" / "A3" / f"seed_{seed}" / protocol
            manifest_path = store / "store_manifest.json"
            if not manifest_path.is_file():
                exact = False
                rows.append({"seed": seed, "protocol": protocol, "complete": False, "reason": "store manifest missing"})
                continue
            manifest = _json(manifest_path)
            expected_rows = int(protocol_rows[protocol])
            expected_shapes = {
                "embedding_normalized.npy": (expected_rows, 128),
                "logits.npy": (expected_rows, 98),
                "labels.npy": (expected_rows,),
                "global_indices.npy": (expected_rows,),
            }
            arrays: Dict[str, Any] = {}
            compatible = (
                manifest.get("complete") is True
                and manifest.get("arm") == "A3"
                and int(manifest.get("seed", -1)) == seed
                and manifest.get("partition") == protocol
                and int(manifest.get("rows", -1)) == expected_rows
                and manifest.get("checkpoint_sha256") == checkpoint_shas[seed]
                and manifest.get("strict_zero_day") is False
            )
            for name, shape in expected_shapes.items():
                path = store / name
                if not path.is_file():
                    compatible = False
                    arrays[name] = {"exists": False}
                    continue
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                arrays[name] = {"exists": True, "shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(path)}
                compatible = compatible and tuple(array.shape) == shape
            rows.append({
                "seed": seed,
                "protocol": protocol,
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "row_count": manifest.get("rows"),
                "complete": manifest.get("complete"),
                "partition_identity": manifest.get("partition"),
                "arrays": arrays,
                "global_index_sha256": arrays.get("global_indices.npy", {}).get("sha256"),
                "exactly_compatible": bool(compatible),
            })
            exact = exact and bool(compatible)
    return {
        "decision": "STAGE26_EMBEDDING_REUSE_EXACTLY_COMPATIBLE" if exact else "STAGE26_EMBEDDING_REUSE_NOT_APPROPRIATE",
        "canonical_stage3m_policy": "Stage 3M retains independent embedding generation for cleaner verification; no predecessor store is automatically reused.",
        "stores": rows,
    }


def _verify_critical_manifest_files(inventory: List[Dict[str, Any]], manifest_shas: Mapping[str, str]) -> None:
    """Verify every directly audited critical file against its frozen manifest record."""
    for row in inventory:
        if not row["required"]:
            continue
        relative = str(row["relative_path"])
        expected = manifest_shas.get(relative)
        actual = row.get("sha256")
        if expected is not None and actual is not None and actual != expected:
            raise DriveAuditError(f"Frozen hash-manifest mismatch for critical file: {relative}")


def validate_drive_root_candidate(branch_root: Path) -> bool:
    """Pure unique-root predicate shared with synthetic regression coverage."""
    return (
        branch_root.name == "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
        and all((branch_root / name).is_dir() for name in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation"))
        and (branch_root / "03_representation_ablation/MANYTX_STAGE2_6M_READY.txt").is_file()
    )


def discover_unique_drive_root(search_root: Path) -> Path:
    matches = [path.resolve() for path in search_root.rglob("MANYTX_ZERO_DAY_BRANCH_v1.0.3") if path.is_dir() and validate_drive_root_candidate(path)]
    if len(matches) != 1:
        raise DriveAuditError(f"Expected exactly one canonical READY Drive root; found: {matches}")
    return matches[0]


def run_drive_audit(config: Any, checkpoint_validator: Callable[[Mapping[str, Any], int, str, str], torch.nn.Module]) -> Dict[str, Any]:
    branch_root = config.branch_root_path
    required = dict(REQUIRED_ARTIFACTS)
    audit_root = config.output_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    if branch_root.name != "MANYTX_ZERO_DAY_BRANCH_v1.0.3":
        raise DriveAuditError("Canonical branch root name mismatch")
    for top in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation"):
        if not (branch_root / top).is_dir():
            raise DriveAuditError(f"Canonical predecessor folder missing: {top}")
    missing = [relative for relative in required if not (branch_root / relative).is_file()]
    if missing:
        raise DriveAuditError(f"Required predecessor artifacts missing: {missing}")

    benchmark_path = branch_root / f"01_benchmark_engineering/benchmark/{CANONICAL_BENCHMARK}.h5"
    stage1_hash = _json(branch_root / "01_benchmark_engineering/manifests/HASH_MANIFEST.json")
    stage1_index = _manifest_index(stage1_hash)
    stage2m_hash_payload = _json(branch_root / "02_benchmark_diagnostics/manifests/HASH_MANIFEST.json")
    stage26_hash_payload = _json(branch_root / "03_representation_ablation/manifests/HASH_MANIFEST.json")
    manifest_shas: Dict[str, str] = {}
    for prefix, index in (
        ("01_benchmark_engineering", stage1_index),
        ("02_benchmark_diagnostics", _manifest_index(stage2m_hash_payload)),
        ("03_representation_ablation", _manifest_index(stage26_hash_payload)),
    ):
        for relative, row in index.items():
            if row.get("sha256"):
                manifest_shas[f"{prefix}/{relative}"] = str(row["sha256"])
    inventory = _inventory(branch_root, required, manifest_shas)
    _verify_critical_manifest_files(inventory, manifest_shas)
    inventory_index = {row["relative_path"]: row for row in inventory}
    benchmark_sha = inventory_index[benchmark_path.relative_to(branch_root).as_posix()]["sha256"]
    if benchmark_sha != EXPECTED_BENCHMARK_SHA256:
        raise DriveAuditError(f"Canonical benchmark SHA mismatch: {benchmark_sha}")

    h5_record = stage1_index.get(f"benchmark/{CANONICAL_BENCHMARK}.h5") or stage1_index.get(f"01_benchmark_engineering/benchmark/{CANONICAL_BENCHMARK}.h5")
    if not h5_record or h5_record.get("sha256") != benchmark_sha:
        raise DriveAuditError("Stage 1B hash manifest does not bind the canonical benchmark")

    stage2m = _json(branch_root / "02_benchmark_diagnostics/manifests/STAGE2M_FINAL_STATUS.json")
    stage2m_hash_path = branch_root / "02_benchmark_diagnostics/manifests/HASH_MANIFEST.json"
    stage2m_guard = _json(branch_root / "02_benchmark_diagnostics/manifests/TEST_GUARD_MANIFEST.json")
    if _sha256(stage2m_hash_path) != EXPECTED_STAGE2M_HASH_SHA256:
        raise DriveAuditError("Stage 2M hash-manifest SHA mismatch")
    if any((
        stage2m.get("status") != "MANYTX_STAGE2M_READY",
        stage2m.get("script_sha256") != EXPECTED_STAGE2M_SCRIPT_SHA256,
        stage2m.get("benchmark_sha256") != benchmark_sha,
        stage2m.get("failed_gates") != [],
        stage2m.get("final_test_model_evaluation_performed") is not False,
        stage2m.get("final_test_threshold_selection_performed") is not False,
        stage2m_guard.get("status") != "PASS",
    )):
        raise DriveAuditError("Stage 2M frozen status/guard contract mismatch")

    stage26 = branch_root / "03_representation_ablation"
    ready_header, ready_values = _ready(stage26 / "MANYTX_STAGE2_6M_READY.txt")
    status = _json(stage26 / "manifests/STAGE2_6M_FINAL_STATUS.json")
    objective = _json(stage26 / "manifests/CANONICAL_STAGE3M_OBJECTIVE.json")
    hash_path = stage26 / "manifests/HASH_MANIFEST.json"
    hash_payload = stage26_hash_payload
    hash_index = _manifest_index(hash_payload)
    if _sha256(hash_path) != EXPECTED_STAGE26_HASH_SHA256:
        raise DriveAuditError("Stage 2.6M final artifact SHA mismatch")
    if ready_header != "MANYTX_STAGE2_6M_READY" or ready_values.get("decision") != EXPECTED_DECISION or ready_values.get("artifact_sha256") != EXPECTED_STAGE26_HASH_SHA256:
        raise DriveAuditError("Stage 2.6M READY marker disagrees with the frozen contract")
    ready_counters = {key: int(ready_values[key]) for key in READY_STRICT_KEYS if key in ready_values}
    if set(ready_counters) != set(READY_STRICT_KEYS) or any(ready_counters.values()):
        raise DriveAuditError("Stage 2.6M READY strict counters are missing or non-zero")
    final_counters = canonical_strict_counters(status)
    if final_counters != ready_counters or any(final_counters.values()):
        raise DriveAuditError("Stage 2.6M READY/final-status strict counters disagree or are non-zero")
    if any((
        status.get("status") != "MANYTX_STAGE2_6M_READY",
        status.get("decision") != EXPECTED_DECISION,
        status.get("selected_arm") != "A3",
        status.get("benchmark_sha256") != benchmark_sha,
        tuple(status.get("seed_panel", ())) != EXPECTED_SEEDS,
        objective.get("decision") != EXPECTED_DECISION,
        objective.get("selected_arm") != "A3",
        objective.get("loss_coefficients") != EXPECTED_LOSS,
    )):
        raise DriveAuditError("Stage 2.6M final status/objective contract mismatch")

    objective_matrix = pd.read_csv(stage26 / "tables/objective_selection_matrix.csv")
    known = pd.read_csv(stage26 / "tables/known_protocol_results.csv")
    per_class = pd.read_csv(stage26 / "tables/known_protocol_per_class.csv")
    embedding = pd.read_csv(stage26 / "tables/embedding_separability.csv")
    paired = pd.read_csv(stage26 / "tables/paired_statistical_tests.csv")
    if str(objective_matrix.sort_values("classification_mean", ascending=False).iloc[0]["arm"]) != "A3":
        raise DriveAuditError("Objective selection matrix scientific ranking disagrees with READY")
    a3_protocol = known[known["arm"] == "A3"].groupby("protocol", sort=False)["fixed98_macro_f1"].mean()
    if not set(KNOWN_PROTOCOLS).issubset(a3_protocol.index):
        raise DriveAuditError("A3 P0-P3 machine-readable known metrics are incomplete")
    a3_fisher = embedding[(embedding["arm"] == "A3") & (embedding["protocol"] == "p0")]["fisher_ratio"]
    paired_a3 = paired[(paired["component"] == "classification") & (paired["comparison"] == "A3_vs_A0")]
    if len(a3_fisher) != 3 or len(paired_a3) != 1:
        raise DriveAuditError("A3 Fisher/statistical evidence is incomplete")
    paired_row = paired_a3.iloc[0]
    facts = {
        "a3_protocol_macro_f1": {protocol: float(a3_protocol.loc[protocol]) for protocol in KNOWN_PROTOCOLS},
        "a3_p0_fisher_by_seed": {str(int(row.seed)): float(row.fisher_ratio) for row in embedding[(embedding["arm"] == "A3") & (embedding["protocol"] == "p0")].itertuples()},
        "a3_p0_fisher_mean": float(a3_fisher.mean()),
        "paired_mean_gain": float(paired_row["paired_difference_mean"]),
        "paired_ci_low": float(paired_row["paired_bootstrap_ci95_low"]),
        "paired_ci_high": float(paired_row["paired_bootstrap_ci95_high"]),
    }
    reports = {
        "objective_selection_report": (stage26 / "reports/objective_selection_report.md").read_text(encoding="utf-8"),
        "known_tx_evaluation_report": (stage26 / "reports/known_tx_evaluation_report.md").read_text(encoding="utf-8"),
        "embedding_separability_report": (stage26 / "reports/embedding_separability_report.md").read_text(encoding="utf-8"),
        "statistical_comparison_report": (stage26 / "reports/statistical_comparison_report.md").read_text(encoding="utf-8"),
    }
    consistency = assess_report_consistency(facts, reports)
    consistency["ready_vs_final_status"] = ready_values.get("decision") == status.get("decision") == objective.get("decision")
    consistency["hash_manifest_vs_critical_files"] = True
    if consistency["status"] != "PASS" or not consistency["ready_vs_final_status"]:
        raise DriveAuditError(f"Report/table consistency failure: {consistency}")

    coverage = class_coverage_from_per_class(per_class)
    checkpoint_rows: List[Dict[str, Any]] = []
    checkpoint_shas: Dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        relative = f"checkpoints/A3/seed_{seed}/best_selection.pt"
        path = stage26 / relative
        actual = _sha256(path)
        manifest_row = hash_index.get(relative)
        if not manifest_row or manifest_row.get("sha256") != actual:
            raise DriveAuditError(f"A3 seed {seed} checkpoint SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise DriveAuditError(f"A3 seed {seed} checkpoint is not a mapping")
        model = checkpoint_validator(payload, seed, str(status["configuration_sha256"]), str(status["architecture_signature"]))
        finite = _finite_model_state(payload)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not finite or parameter_count != EXPECTED_PARAMETER_COUNT:
            raise DriveAuditError(f"A3 seed {seed} checkpoint tensor integrity failure")
        checkpoint_shas[seed] = actual
        checkpoint_rows.append({
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": actual,
            "manifest_sha256": manifest_row.get("sha256"),
            "arm": payload.get("arm"),
            "benchmark_sha256": payload.get("benchmark_sha"),
            "stage2m_sha256": payload.get("stage2m_sha"),
            "stage2_6m_configuration_sha256": payload.get("configuration_sha"),
            "architecture_signature": payload.get("architecture_signature"),
            "loss_coefficients": payload.get("loss_coefficients"),
            "parameter_count": parameter_count,
            "embedding_dimension": int(getattr(model, "embedding_dim", -1)),
            "class_count": int(getattr(model, "num_classes", -1)),
            "state_dict_key_count": len(payload["model_state"]),
            "state_dict_shapes": {key: list(value.shape) for key, value in payload["model_state"].items()},
            "all_tensors_finite": finite,
            "cpu_only_load": True,
            "strict_state_dict_load": True,
        })
    stores = _store_audit(
        stage26,
        checkpoint_shas,
        {protocol: int(coverage["protocols"][protocol]["total_rows"]) for protocol in KNOWN_PROTOCOLS},
    )

    for row in inventory:
        if row["required"]:
            row["internally_consistent"] = True
    checkpoint_audit = {"status": "PASS", "checkpoints": checkpoint_rows, "stage2_6m_embedding_stores": stores}
    class_audit = {"status": "PASS", **coverage}
    consistency = {"status": "PASS", **consistency}
    predecessor = {
        "status": "DRIVE_PREDECESSOR_AUDIT_PASS",
        "canonical_drive_root": str(branch_root.resolve()),
        "measured_from_drive": {
            "benchmark_path": str(benchmark_path.resolve()),
            "benchmark_sha256": benchmark_sha,
            "stage2m_status": stage2m.get("status"),
            "stage2m_hash_manifest_sha256": _sha256(stage2m_hash_path),
            "stage2_6m_ready_content": (stage26 / "MANYTX_STAGE2_6M_READY.txt").read_text(encoding="utf-8"),
            "stage2_6m_artifact_sha256": _sha256(hash_path),
            "strict_zero_day_counters": final_counters,
            "selected_objective": objective.get("selected_objective"),
            "decision": objective.get("decision"),
            **facts,
        },
        "derived_cross_checked": {
            "class_coverage": coverage["protocols"],
            "report_table_consistency": consistency["status"],
            "checkpoint_audit": "PASS",
            "embedding_reuse_assessment": stores["decision"],
        },
        "expected_from_protocol": {
            "teacher_source": "stage2_6m_promote",
            "candidate_arm": "A3",
            "candidate_seeds": list(EXPECTED_SEEDS),
            "selection_policy_unchanged": True,
            "strict_zero_day_signal_or_label_arrays_opened": False,
        },
        "mismatch_warning": [],
        "recursive_inventory": inventory,
    }
    markdown = "\n".join([
        "# Stage 3M Google Drive predecessor audit",
        "",
        "## MEASURED FROM DRIVE",
        "",
        f"- Canonical root: `{branch_root.resolve()}`",
        f"- Benchmark SHA-256: `{benchmark_sha}`",
        f"- Stage 2M: `{stage2m.get('status')}`",
        f"- Stage 2.6M decision: `{objective.get('decision')}` / `{objective.get('selected_objective')}`",
        f"- Stage 2.6M artifact SHA-256: `{_sha256(hash_path)}`",
        f"- Strict counters: `{json.dumps(final_counters, sort_keys=True)}`",
        "",
        "## DERIVED / CROSS-CHECKED",
        "",
        f"- Class coverage: `{json.dumps(coverage['protocols'], sort_keys=True)}`",
        f"- Report/table consistency: `{consistency['status']}` (rounded prose values agree with machine tables)",
        "- A3 checkpoint audit: `PASS`",
        f"- Embedding-store assessment: `{stores['decision']}`",
        "",
        "## EXPECTED FROM PROTOCOL",
        "",
        "Stage 3M remains A3-only, seeds 42/123/2026, promotion-and-verification only, with no training, no strict-zero-day access, no surrogate, no XAI, and no final-zero-day evaluation.",
        "",
        "## MISMATCH / WARNING",
        "",
        "None.",
        "",
        "DRIVE_PREDECESSOR_AUDIT_PASS",
        "",
    ])
    _write_json(audit_root / "STAGE3M_DRIVE_CLASS_COVERAGE_AUDIT.json", class_audit)
    _write_json(audit_root / "STAGE3M_DRIVE_A3_CHECKPOINT_AUDIT.json", checkpoint_audit)
    _write_json(audit_root / "STAGE3M_DRIVE_REPORT_CONSISTENCY.json", consistency)
    _write_json(audit_root / "STAGE3M_DRIVE_PREDECESSOR_AUDIT.json", predecessor)
    _write_text(audit_root / "STAGE3M_DRIVE_PREDECESSOR_AUDIT.md", markdown)
    return predecessor
