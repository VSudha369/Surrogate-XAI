#!/usr/bin/env python3
"""Offline validator for Stage 3M v1.0.0 source and scientific contracts."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.py"
LAUNCHER = ROOT / "Stage3M_Colab_Launcher_v1_0_0.py"
STAGE26 = ROOT / "Stage2_6M_WiSig_ManyTx_Controlled_Representation_Ablation_v1_0_2.py"
NOTEBOOK = ROOT / "Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.ipynb"
MANIFEST = ROOT / "STAGE3M_CODE_MANIFEST.json"
PACKAGE = ROOT / "Stage3M_WiSig_ManyTx_Canonical_Teacher_v1_0_0.zip"
DRIVE_AUDIT = ROOT / "stage3m_drive_audit_v1_0_0.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expect_abort(module: Any, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except module.ScientificAbort:
        return
    raise AssertionError("Expected ScientificAbort")


def test_static(module: Any, require_package: bool) -> None:
    for path in (MAIN, LAUNCHER, DRIVE_AUDIT, ROOT / "validate_stage3m_v1_0_0.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    assert "%,d" not in MAIN.read_text(encoding="utf-8")
    assert "architecture search" not in MAIN.read_text(encoding="utf-8").lower().replace("architecture search: disabled", "")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook.get("cells")
    pyflakes = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(MAIN), str(LAUNCHER), str(DRIVE_AUDIT), str(Path(__file__))],
        capture_output=True, text=True, check=False,
    )
    if pyflakes.returncode:
        raise AssertionError(f"pyflakes failed or is unavailable:\n{pyflakes.stdout}{pyflakes.stderr}")
    if require_package:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = ROOT / row["path"]
            assert path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"]
        with zipfile.ZipFile(PACKAGE) as archive:
            assert archive.testzip() is None
            members = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
            for row in manifest["files"]:
                assert members[row["path"]].file_size == row["bytes"]
                assert archive.read(row["path"]) == (ROOT / row["path"]).read_bytes()


def test_contract(module: Any) -> None:
    assert module.PIPELINE_VERSION == "1.0.0"
    assert module.EXPECTED_ARM == "A3" and module.EXPECTED_SEEDS == (42, 123, 2026)
    assert module.EXPECTED_BENCHMARK_SHA256 == "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9"
    assert module.EXPECTED_STAGE26_ARTIFACT_SHA256 == "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660"
    assert module.EXPECTED_STAGE26_DECISION == "SELECT_CE_SUPCON_PROTOTYPE"
    assert module.EXPECTED_LOSS == {"name": "CE + SupCon + Prototype", "supcon_weight": 0.1, "prototype_weight": 0.1}
    config = module.Stage3MConfig(branch_root=str(ROOT / module.CANONICAL_BRANCH))
    payload = config.scientific_payload()
    assert payload["embedding_dimension"] == 128 and payload["known_classes"] == 98
    assert payload["temperature"] == 0.07 and payload["prototype_momentum"] == 0.95
    assert config.teacher_source == "stage2_6m_promote"


def test_architecture(module: Any, stage26: Any) -> None:
    left, right = module.WiSigRepresentationNet().eval(), stage26.WiSigRepresentationNet().eval()
    assert sum(p.numel() for p in left.parameters()) == sum(p.numel() for p in right.parameters()) == module.EXPECTED_PARAMETER_COUNT
    assert list(left.state_dict()) == list(right.state_dict())
    assert {k: tuple(v.shape) for k, v in left.state_dict().items()} == {k: tuple(v.shape) for k, v in right.state_dict().items()}
    sample = torch.randn(4, 2, 256)
    with torch.inference_mode():
        first, second = left(sample), left(sample)
    assert first["logits"].shape == (4, 98) and first["embedding_normalized"].shape == (4, 128)
    assert torch.allclose(first["embedding_normalized"].norm(dim=1), torch.ones(4), atol=2e-5)
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_checkpoint_provenance(module: Any) -> None:
    model = module.WiSigRepresentationNet()
    signature = module.architecture_signature(model)
    config_sha = "c" * 64
    base: Dict[str, Any] = {
        "model_state": model.state_dict(), "arm": "A3", "seed": 42,
        "benchmark_sha": module.EXPECTED_BENCHMARK_SHA256,
        "stage2m_sha": module.EXPECTED_STAGE2M_SCRIPT_SHA256,
        "configuration_sha": config_sha, "architecture_signature": signature,
        "loss_coefficients": dict(module.EXPECTED_LOSS),
    }
    module.validate_candidate_checkpoint_payload(base, 42, config_sha, signature)
    for arm in ("A0", "A1", "A2"):
        candidate = dict(base, arm=arm)
        expect_abort(module, lambda c=candidate: module.validate_candidate_checkpoint_payload(c, 42, config_sha, signature))
    mutations = {
        "wrong seed": dict(base, seed=123),
        "benchmark": dict(base, benchmark_sha="0" * 64),
        "configuration": dict(base, configuration_sha="0" * 64),
        "architecture": dict(base, architecture_signature="0" * 64),
        "loss": dict(base, loss_coefficients={"name": "CE + SupCon + Prototype", "supcon_weight": 0.0, "prototype_weight": 0.1}),
    }
    for candidate in mutations.values():
        expect_abort(module, lambda c=candidate: module.validate_candidate_checkpoint_payload(c, 42, config_sha, signature))
    corrupted = copy.deepcopy(base)
    corrupted["model_state"].pop(next(iter(corrupted["model_state"])))
    expect_abort(module, lambda: module.validate_candidate_checkpoint_payload(corrupted, 42, config_sha, signature))


def selection_tables(values: Mapping[int, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    known, representation = [], []
    for seed in (42, 123, 2026):
        row = values[seed]
        macros = row.get("macros", [0.5] * 4)
        for index, protocol in enumerate(("p0", "p1", "p2", "p3")):
            known.append({"seed": seed, "protocol": protocol, "fixed98_macro_f1": macros[index], "fixed98_balanced_accuracy": row.get("balanced", 0.5)})
            representation.append({"seed": seed, "protocol": protocol, "fisher_ratio": row.get("fisher", 1.0)})
    return pd.DataFrame(known), pd.DataFrame(representation)


def selected(module: Any, values: Mapping[int, Mapping[str, Any]]) -> int:
    known, representation = selection_tables(values)
    first = module.deterministic_selection(known, representation)[0]
    second = module.deterministic_selection(known.sample(frac=1, random_state=9), representation.sample(frac=1, random_state=8))[0]
    assert first == second
    return first


def test_selection(module: Any) -> None:
    neutral = {42: {}, 123: {}, 2026: {}}
    primary = copy.deepcopy(neutral); primary[123]["macros"] = [0.51] * 4
    assert selected(module, primary) == 123
    secondary = copy.deepcopy(neutral); secondary[123]["balanced"] = 0.51
    assert selected(module, secondary) == 123
    degradation = {42: {"macros": [0.53, 0.49, 0.49, 0.49]}, 123: {"macros": [0.5] * 4}, 2026: {"macros": [0.53, 0.49, 0.49, 0.49]}}
    assert selected(module, degradation) == 123
    fisher = copy.deepcopy(neutral); fisher[123]["fisher"] = 2.0
    assert selected(module, fisher) == 123
    assert selected(module, neutral) == 42


def synthetic_representation(observed_labels: list[int], samples_per_class: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for label in observed_labels:
        center = np.zeros(128, dtype=np.float32); center[label] = 1.0
        for sample in range(samples_per_class):
            value = center.copy(); value[(label + sample + 1) % 128] += 0.01 * (sample + 1)
            value /= np.linalg.norm(value)
            embeddings.append(value); labels.append(label)
    global_indices = np.arange(10_000, 10_000 + len(labels), dtype=np.int64)
    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.int64), global_indices


def test_representation_missing_classes(module: Any) -> None:
    pipeline = module.Stage3MPipeline.__new__(module.Stage3MPipeline)
    pipeline.config = SimpleNamespace(representation_sampling_seed=3_000_001, representation_samples_per_class=3)
    geometry = (
        "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index",
        "mean_intra_class_distance", "mean_inter_class_distance", "inter_intra_ratio",
        "fisher_ratio", "prototype_separation", "prototype_compactness",
    )
    cases = {
        "p0": list(range(98)),
        "p2": [0, 3, 7, 12, 19, 31, 48, 63, 77, 97],
        "p3": [2, 17, 44, 89],
    }
    for protocol, observed in cases.items():
        embedding, labels, global_indices = synthetic_representation(observed)
        first_metrics, first_positions = pipeline.representation_metrics(embedding, labels, 42, protocol)
        second_metrics, second_positions = pipeline.representation_metrics(embedding, labels, 42, protocol)
        first_hashes = module.representation_sampling_hashes(first_positions, global_indices)
        second_hashes = module.representation_sampling_hashes(second_positions, global_indices)
        assert np.array_equal(first_positions, second_positions) and first_hashes == second_hashes
        assert first_metrics == second_metrics
        assert first_metrics["observed_class_count"] == len(observed)
        assert first_metrics["missing_class_count"] == 98 - len(observed)
        assert first_metrics["representation_frame"] == "OBSERVED_CLASSES"
        assert all(np.isfinite(float(first_metrics[key])) for key in geometry)
    embedding, labels, _ = synthetic_representation(list(range(97)))
    expect_abort(module, lambda: pipeline.representation_metrics(embedding, labels, 42, "p0"))
    embedding, labels, _ = synthetic_representation([0, 98])
    expect_abort(module, lambda: pipeline.representation_metrics(embedding, labels, 42, "p2"))
    embedding, labels, _ = synthetic_representation([7])
    expect_abort(module, lambda: pipeline.representation_metrics(embedding, labels, 42, "p3"))


def test_zero_day(module: Any) -> None:
    for kind in ("signal", "label", "embedding", "metric", "threshold"):
        guard = module.StrictZeroDayGuard()
        expect_abort(module, lambda g=guard, k=kind: g.reject(k, "strict_zero_day_test"))
        assert sum(guard.counters().values()) == 1
    pristine = module.StrictZeroDayGuard()
    pristine.assert_zero()
    assert all(value == 0 for value in pristine.counters().values())


def test_storage(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = Path(temporary)
        np.save(store / "embedding_normalized.npy", np.zeros((2, 128)), allow_pickle=False)
        np.save(store / "logits.npy", np.zeros((2, 98)), allow_pickle=False)
        np.save(store / "labels.npy", np.zeros(2), allow_pickle=False)
        np.save(store / "global_indices.npy", np.zeros(2), allow_pickle=False)
        (store / "store_manifest.json").write_text(json.dumps({"complete": True, "checkpoint_sha256": "good", "rows": 2}), encoding="utf-8")
        assert module.Stage3MPipeline.store_current(object(), store, "good", 2)
        assert not module.Stage3MPipeline.store_current(object(), store, "bad", 2)
        assert not module.Stage3MPipeline.store_current(object(), store, "good", 3)
        (store / "INCOMPLETE").write_text("partial", encoding="utf-8")
        assert not module.Stage3MPipeline.store_current(object(), store, "good", 2)


def test_logging_and_ready(module: Any) -> None:
    stream = io.StringIO(); logger = logging.getLogger("stage3m-validator"); logger.handlers.clear()
    handler = logging.StreamHandler(stream); handler.setFormatter(logging.Formatter("%(levelname)s %(message)s")); logger.addHandler(handler); logger.setLevel(logging.INFO)
    logger.info("Evaluated seed %s protocol %s with %s rows", 42, "P0", f"{1000:,}")
    assert "1,000 rows" in stream.getvalue()
    gates = {"predecessor": True, "candidates": True, "architecture": True, "known": True, "representation": True, "selection": True, "weights": True, "strict_zero": True}
    assert module.ready_gate(gates)
    for key in gates:
        failed = dict(gates); failed[key] = False
        assert not module.ready_gate(failed)


def write_stage10_checkpoint(module: Any, pipeline: Any, outputs: list[Path]) -> None:
    payload = {
        "pipeline_version": module.PIPELINE_VERSION,
        "executable_sha256": pipeline.script_sha,
        "configuration_sha256": pipeline.config.configuration_sha256(),
        "required_input_hashes": {},
        "required_outputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
    }
    path = pipeline.stage_manifest_path(10); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_interrupted_stage10_resume(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        class FixtureConfig:
            resume = True
            output_root = root
            @staticmethod
            def configuration_sha256() -> str:
                return "config-sha"
        pipeline = module.Stage3MPipeline.__new__(module.Stage3MPipeline)
        pipeline.config = FixtureConfig(); pipeline.script_sha = "script-sha"
        checkpoint = root / "manifests" / "STAGE_10_CHECKPOINT.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"pipeline_version": module.PIPELINE_VERSION}), encoding="utf-8")
        assert not pipeline.stage_current(10)  # Case A: checkpoint-like partial state, no READY.
        ready = root / "MANYTX_STAGE3M_READY.txt"; ready.write_text("MANYTX_STAGE3M_READY\n", encoding="utf-8")
        corrupt = root / "manifests" / "STAGE3M_HASH_MANIFEST.json"; corrupt.write_text("{corrupt", encoding="utf-8")
        assert not pipeline.stage_current(10)  # Case B: READY with corrupt final manifest.
        for relative in module.FINAL_HASH_REQUIRED_RELATIVE_PATHS:
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"fixture:{relative}".encode())
        rows = []
        for relative in module.FINAL_HASH_REQUIRED_RELATIVE_PATHS:
            path = root / relative
            rows.append({"relative_path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        exclusions = [{"relative_path": path, "reason": reason} for path, reason in module.FINAL_HASH_EXCLUSIONS.items()]
        corrupt.write_text(json.dumps({"algorithm": "SHA-256", "files": rows, "count": len(rows), "exclusions": exclusions}), encoding="utf-8")
        outputs = [root / relative for relative in module.FINAL_STAGE_REQUIRED_RELATIVE_PATHS]
        write_stage10_checkpoint(module, pipeline, outputs)
        assert pipeline.stage_current(10)  # Case C: complete and hash-current final transaction.
        (root / "publication" / "Stage3M_report.pdf").write_bytes(b"corrupt-after-checkpoint")
        assert not pipeline.stage_current(10)


def test_preflight_scope(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        branch = Path(temporary) / module.CANONICAL_BRANCH
        captured: Dict[str, Any] = {}
        original_pipeline = module.Stage3MPipeline
        original_audit = module.run_drive_audit
        class CapturePipeline:
            def __init__(self, config: Any):
                captured["config"] = config
            def run(self) -> None:
                captured["run"] = True
        module.Stage3MPipeline = CapturePipeline
        module.run_drive_audit = lambda config, validator: captured.setdefault("audit", True)
        try:
            assert module.main(["--preflight", "--branch-root", str(branch)]) == 0
        finally:
            module.Stage3MPipeline = original_pipeline
            module.run_drive_audit = original_audit
        config = captured["config"]
        assert captured["audit"] and captured["run"] and config.preflight is True and config.stage_start == 1 and config.stage_end == 3
        calls: list[int] = []
        harness = original_pipeline.__new__(original_pipeline)
        harness.config = SimpleNamespace(stage_start=1, stage_end=3, preflight=True)
        harness.stage_current = lambda stage: False
        for stage in range(1, 11):
            setattr(harness, f"stage_{stage:02d}", lambda stage=stage: calls.append(stage))
        original_pipeline.run(harness)
        assert calls == [1, 2, 3]
        assert not (branch / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt").exists()


def test_drive_audit_mode(module: Any) -> None:
    audit_module = load_module(DRIVE_AUDIT, "stage3m_drive_audit_validator")
    good = {source: 0 for source in audit_module.FINAL_STATUS_STRICT_KEY_MAP.values()}
    canonical = audit_module.canonical_strict_counters({"strict_zero_day_violation_counters": good})
    assert set(canonical) == set(audit_module.READY_STRICT_KEYS) and not any(canonical.values())
    for source in good:
        malformed = dict(good); malformed.pop(source)
        try:
            audit_module.canonical_strict_counters({"strict_zero_day_violation_counters": malformed})
        except audit_module.DriveAuditError:
            pass
        else:
            raise AssertionError("Missing structured strict counter must fail")

    rows = []
    missing = {"p0": set(), "p1": set(), "p2": {72}, "p3": {50, 52, 58, 71, 72}}
    for seed in (42, 123, 2026):
        for protocol in ("p0", "p1", "p2", "p3"):
            for label in range(98):
                rows.append({"arm": "A3", "seed": seed, "protocol": protocol, "class_index": label, "support": 0 if label in missing[protocol] else 3})
    coverage = audit_module.class_coverage_from_per_class(pd.DataFrame(rows))
    assert coverage["protocols"]["p0"]["observed_class_count"] == 98
    assert coverage["protocols"]["p2"]["missing_class_ids"] == [72]
    assert coverage["protocols"]["p3"]["missing_class_ids"] == [50, 52, 58, 71, 72]
    bad = pd.DataFrame(rows); bad.loc[(bad.protocol == "p0") & (bad.class_index == 97), "support"] = 0
    try:
        audit_module.class_coverage_from_per_class(bad)
    except audit_module.DriveAuditError:
        pass
    else:
        raise AssertionError("P0 with 97 observed classes must fail")

    facts = {
        "paired_mean_gain": 0.014854654467257533,
        "paired_ci_low": 0.009930833520440108,
        "paired_ci_high": 0.020772814494475544,
        "a3_protocol_macro_f1": {"p0": 0.8638864, "p1": 0.62955027, "p2": 0.3919334, "p3": 0.3323978},
        "a3_p0_fisher_mean": 1.9547617,
    }
    reports = {
        "objective_selection_report": "`A3` ranked first. 0.014855 [0.009931, 0.020773] SELECT_CE_SUPCON_PROTOTYPE",
        "known_tx_evaluation_report": "A3 0.863886 0.629550 0.391933 0.332398",
        "embedding_separability_report": "A3 1.95476",
        "statistical_comparison_report": "A3_vs_A0 0.014855 0.009931 0.020773",
    }
    assert audit_module.assess_report_consistency(facts, reports)["status"] == "PASS"
    disagree = dict(reports); disagree["objective_selection_report"] = "`A0` ranked first. SELECT_CE"
    assert audit_module.assess_report_consistency(facts, disagree)["status"] == "FAIL"

    source = DRIVE_AUDIT.read_text(encoding="utf-8")
    assert "checkpoint SHA mismatch" in source and "checkpoint_validator" in source
    assert "Stage 2.6M READY marker disagrees" in source and "Required predecessor artifacts missing" in source
    assert "MANYTX_STAGE3M_READY" not in source and "optimizer.step" not in source and "backward(" not in source
    with tempfile.TemporaryDirectory() as temporary:
        branch = Path(temporary) / module.CANONICAL_BRANCH
        captured: Dict[str, Any] = {}
        original_pipeline, original_audit = module.Stage3MPipeline, module.run_drive_audit
        class ForbiddenPipeline:
            def __init__(self, config: Any):
                raise AssertionError("--drive-audit must not instantiate Stage3MPipeline")
        module.Stage3MPipeline = ForbiddenPipeline
        module.run_drive_audit = lambda config, validator: captured.setdefault("audit", config)
        try:
            assert module.main(["--drive-audit", "--branch-root", str(branch)]) == 0
        finally:
            module.Stage3MPipeline, module.run_drive_audit = original_pipeline, original_audit
        assert captured["audit"].drive_audit is True
        assert not (branch / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt").exists()


def test_drive_audit_failure_matrix() -> None:
    audit_module = load_module(DRIVE_AUDIT, "stage3m_drive_audit_failure_validator")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            audit_module.discover_unique_drive_root(root)
        except audit_module.DriveAuditError:
            pass
        else:
            raise AssertionError("Missing READY root must fail")
        candidate = root / "MANYTX_ZERO_DAY_BRANCH_v1.0.3"
        for name in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation"):
            (candidate / name).mkdir(parents=True)
        ready = candidate / "03_representation_ablation/MANYTX_STAGE2_6M_READY.txt"
        ready.write_text("MANYTX_STAGE2_6M_READY\n", encoding="utf-8")
        assert audit_module.discover_unique_drive_root(root) == candidate.resolve()
        second = root / "nested/MANYTX_ZERO_DAY_BRANCH_v1.0.3"
        for name in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation"):
            (second / name).mkdir(parents=True)
        (second / "03_representation_ablation/MANYTX_STAGE2_6M_READY.txt").write_text("MANYTX_STAGE2_6M_READY\n", encoding="utf-8")
        try:
            audit_module.discover_unique_drive_root(root)
        except audit_module.DriveAuditError:
            pass
        else:
            raise AssertionError("Multiple READY roots must fail")
    assert "Required predecessor artifacts missing" in DRIVE_AUDIT.read_text(encoding="utf-8")
    assert "checkpoint SHA mismatch" in DRIVE_AUDIT.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true", help="Skip manifest/ZIP checks while assembling the package")
    args = parser.parse_args()
    module = load_module(MAIN, "stage3m_validator_target")
    stage26 = load_module(STAGE26, "stage26_validator_reference")
    tests = {
        "static": lambda: test_static(module, not args.source_only),
        "scientific_contract": lambda: test_contract(module),
        "architecture_equivalence": lambda: test_architecture(module, stage26),
        "checkpoint_provenance": lambda: test_checkpoint_provenance(module),
        "deterministic_selection": lambda: test_selection(module),
        "missing_protocol_classes": lambda: test_representation_missing_classes(module),
        "strict_zero_day_guard": lambda: test_zero_day(module),
        "embedding_store_resume": lambda: test_storage(module),
        "logging_and_ready_gate": lambda: test_logging_and_ready(module),
        "interrupted_stage10_resume": lambda: test_interrupted_stage10_resume(module),
        "preflight_scope": lambda: test_preflight_scope(module),
        "drive_audit_mode": lambda: test_drive_audit_mode(module),
        "drive_audit_failure_matrix": test_drive_audit_failure_matrix,
    }
    for name, operation in tests.items():
        operation(); print(f"[PASS] {name}")
    print(f"STAGE3M_VALIDATION_PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
