#!/usr/bin/env python3
"""Local/static regression validator for Stage 3.5M v1.0.0."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.py"
RECOVERY = ROOT / "Stage3_5M_PostLock_Recovery_v1_0_0.py"
LAUNCHER = ROOT / "Stage3_5M_Colab_Launcher_v1_0_0.py"
CONFIG = ROOT / "stage3_5m_config_v1_0_0.example.json"
NOTEBOOK = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.ipynb"
MANIFEST = ROOT / "STAGE3_5M_CODE_MANIFEST.json"
PACKAGE = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.zip"
REQUIRED_PACKAGE = (
    MAIN.name, RECOVERY.name, LAUNCHER.name, CONFIG.name, Path(__file__).name, NOTEBOOK.name,
    "STAGE3_5M_SCIENTIFIC_PROTOCOL.md", "STAGE3_5M_OUTPUT_SCHEMA.md",
    "STAGE3_5M_VALIDATION_CHECKLIST.md", "STAGE3_5M_README.md",
    "STAGE3_5M_CODE_MANIFEST.json", "requirements_stage3_5m.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expect_abort(module: Any, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except module.ScientificAbort:
        return
    raise AssertionError("Expected ScientificAbort")


def synthetic_known(module: Any, samples_per_class: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(350)
    embeddings, labels = [], []
    for class_index in range(module.EXPECTED_CLASSES):
        center = np.zeros(module.EXPECTED_EMBEDDING_DIM, dtype=np.float64); center[class_index] = 1.0
        values = center + rng.normal(0, 0.008, size=(samples_per_class, module.EXPECTED_EMBEDDING_DIM))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        embeddings.append(values); labels.extend([class_index] * samples_per_class)
    z = np.concatenate(embeddings).astype(np.float32); y = np.asarray(labels, dtype=np.int64)
    logits = np.full((len(z), module.EXPECTED_CLASSES), -5.0, dtype=np.float32); logits[np.arange(len(z)), y] = 5.0
    return z, y, logits


def test_static(module: Any, full: bool) -> None:
    for path in [MAIN, RECOVERY, LAUNCHER, CONFIG, NOTEBOOK, *[ROOT / name for name in REQUIRED_PACKAGE[6:10]], ROOT / "requirements_stage3_5m.txt"]:
        assert path.is_file(), path
    ast.parse(MAIN.read_text(encoding="utf-8")); ast.parse(RECOVERY.read_text(encoding="utf-8")); ast.parse(LAUNCHER.read_text(encoding="utf-8")); ast.parse(Path(__file__).read_text(encoding="utf-8"))
    json.loads(CONFIG.read_text(encoding="utf-8")); notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); assert notebook["nbformat"] == 4
    source = MAIN.read_text(encoding="utf-8")
    assert ".backward(" not in source and "optimizer.step(" not in source and "CrossEntropyLoss(" not in source
    assert "UNKNOWN training class" not in source
    assert source.index("write_or_verify_lock()") < source.index("strict_indices: Dict[str, np.ndarray]")
    if full:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8")); rows = {row["path"]: row for row in payload["files"]}
        assert set(rows) == set(REQUIRED_PACKAGE) - {MANIFEST.name}
        for name, row in rows.items():
            path = ROOT / name; assert path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"]
        with zipfile.ZipFile(PACKAGE) as archive:
            assert archive.testzip() is None and set(archive.namelist()) == set(REQUIRED_PACKAGE)
            for name in REQUIRED_PACKAGE:
                assert archive.read(name) == (ROOT / name).read_bytes()


def test_contract(module: Any) -> None:
    assert module.EXPECTED_TEACHER_SEED == 123
    assert module.EXPECTED_TEACHER_SHA256 == "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4"
    assert module.EXPECTED_STAGE3M_HASH_MANIFEST_SHA256 == "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0"
    assert module.CANONICAL_SCORER_POLICY == "ALL_PREDECLARED" and len(module.SCORER_ORDER) == 5
    assert all(value["direction"] == "higher_is_more_unknown" for value in module.SCORER_DEFINITIONS.values())
    assert len(module.STRICT_COUNTER_KEYS) == 6


def test_strict_guard(module: Any) -> None:
    for kind in ("signal", "label", "embedding", "metric", "threshold", "fit"):
        guard = module.StrictZeroDayGuard()
        expect_abort(module, lambda g=guard, k=kind: g.authorize_strict(k, 7, False))
        assert sum(guard.counters().values()) == 1
    guard = module.StrictZeroDayGuard(); guard.activate_final_lock(8, True); guard.authorize_strict("signal", 8, True); guard.assert_zero()


def test_no_unknown_training_class(module: Any) -> None:
    z, y, _ = synthetic_known(module)
    fit = module.fit_statistics(z, y, 0.001, 0.0001)
    assert fit["counts"].shape == (98,) and fit["counts"].sum() == len(y)
    bad = y.copy(); bad[0] = 98
    expect_abort(module, lambda: module.fit_statistics(z, bad, 0.001, 0.0001))


def test_scorers(module: Any) -> None:
    z, y, logits = synthetic_known(module)
    fit = module.fit_statistics(z, y, 0.001, 0.0001)
    known = module.score_outputs(logits, z, fit, 1.0)
    unknown_z = np.full((8, 128), 1 / np.sqrt(128), dtype=np.float32)
    unknown_logits = np.zeros((8, 98), dtype=np.float32)
    unknown = module.score_outputs(unknown_logits, unknown_z, fit, 1.0)
    assert known.shape == (len(z), 5) and np.isfinite(known).all()
    assert unknown[:, 0].mean() > known[:, 0].mean()  # MSP convention.
    assert unknown[:, 1].mean() > known[:, 1].mean()  # Energy convention.
    assert unknown[:, 2].mean() > known[:, 2].mean()  # Prototype distance.
    assert unknown[:, 3].mean() > known[:, 3].mean()  # Mahalanobis distance.
    assert unknown[:, 4].mean() > known[:, 4].mean()  # Density NLL.
    assert np.allclose(np.linalg.norm(fit["prototypes"], axis=1), 1.0, atol=1e-5)


def test_covariance_regularization(module: Any) -> None:
    z, y, _ = synthetic_known(module, 2)
    fit = module.fit_statistics(z, y, 0.01, 0.0005)
    assert np.isfinite(fit["precision"]).all()
    assert np.linalg.eigvalsh(fit["precision"].astype(np.float64)).min() > 0
    assert fit["diagonal_variance"].min() >= 0.0005 - 1e-8


def test_threshold_known_only(module: Any) -> None:
    values = {scorer: np.linspace(index, index + 1, 101) for index, scorer in enumerate(module.SCORER_ORDER)}
    first = module.freeze_known_thresholds(values, 0.95, 0.99); second = module.freeze_known_thresholds(values, 0.95, 0.99)
    assert first == second
    for scorer in module.SCORER_ORDER:
        assert first[scorer]["source"] == "P0_P1_P2_P3_KNOWN_VALIDATION_ONLY"
        assert first[scorer]["direction"] == "unknown_if_score_above_threshold"
    expect_abort(module, lambda: module.freeze_known_thresholds({"S0_MSP": np.ones(2)}, 0.95, 0.99))


def test_open_set_metrics(module: Any) -> None:
    known = np.linspace(0.0, 0.4, 100); unknown = np.linspace(0.6, 1.0, 100); correct = np.ones(100, dtype=bool)
    first = module.open_set_metrics(known, correct, unknown, 0.5); second = module.open_set_metrics(known, correct, unknown, 0.5)
    assert first == second and first["auroc"] == 1.0 and first["macro_f1"] == 1.0 and first["oscr"] > 0.99


def fake_config(root: Path) -> Any:
    class Config:
        output_root = root
        resume = True
        energy_temperature = 1.0
        config_sha = "config-sha"
        def configuration_sha256(self) -> str:
            return self.config_sha
    return Config()


def make_pipeline(module: Any, root: Path) -> Any:
    pipeline = module.Stage35Pipeline.__new__(module.Stage35Pipeline)
    pipeline.config = fake_config(root); pipeline.script_sha = "script-sha"; pipeline.guard = module.StrictZeroDayGuard()
    return pipeline


def write_score_store(
    module: Any,
    pipeline: Any,
    partition: str,
    indices: np.ndarray,
    fit_sha: str,
    strict: bool = False,
    lock_sha: str | None = None,
) -> Path:
    indices = np.asarray(indices, dtype=np.int64)
    rows, store = len(indices), pipeline.score_store(partition); store.mkdir(parents=True, exist_ok=True)
    np.save(store / "scores.npy", np.zeros((rows, len(module.SCORER_ORDER)), dtype=np.float32), allow_pickle=False)
    np.save(store / "predictions.npy", np.arange(rows, dtype=np.int16) % module.EXPECTED_CLASSES, allow_pickle=False)
    np.save(store / "global_indices.npy", indices, allow_pickle=False)
    required = ["scores.npy", "predictions.npy", "global_indices.npy"]
    if not strict:
        np.save(store / "labels.npy", np.arange(rows, dtype=np.int16) % module.EXPECTED_CLASSES, allow_pickle=False)
        np.save(store / "known_correct.npy", np.ones(rows, dtype=bool), allow_pickle=False)
        required.extend(["labels.npy", "known_correct.npy"])
    payload = {
        "complete": True, "pipeline_version": module.PIPELINE_VERSION,
        "executable_sha256": pipeline.script_sha, "configuration_sha256": pipeline.config.configuration_sha256(),
        "benchmark_sha256": module.EXPECTED_BENCHMARK_SHA256, "teacher_sha256": module.EXPECTED_TEACHER_SHA256,
        "fit_sha256": fit_sha, "scorer_order": list(module.SCORER_ORDER),
        "scorer_definitions_sha256": module.sha256_object(module.SCORER_DEFINITIONS),
        "energy_temperature": pipeline.config.energy_temperature, "partition": partition, "rows": rows,
        "strict": strict, "global_indices_sha256": module.sha256_int64_array(indices),
        "files": {name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in required},
    }
    if strict:
        payload["evaluation_lock_sha256"] = lock_sha
    (store / "store_manifest.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return store


def prepare_lock_fixture(module: Any, root: Path) -> Any:
    for directory in (root / "manifests", root / "thresholds", root / "scores/fitted"):
        directory.mkdir(parents=True, exist_ok=True)
    pipeline = make_pipeline(module, root)
    (root / "scores/fitted/known_only_scorer_state.npz").write_bytes(b"fit")
    (root / "thresholds/ZD_STRICT_THRESHOLDS.json").write_text("{}", encoding="utf-8")
    (root / "manifests/SCORER_POLICY_FREEZE.json").write_text("{}", encoding="utf-8")
    (root / "manifests/STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json").write_text("{}", encoding="utf-8")
    (root / "manifests/STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json").write_text(json.dumps({
        "status": "PASS", "teacher_seed": module.EXPECTED_TEACHER_SEED,
        "teacher_sha256": module.EXPECTED_TEACHER_SHA256, "strict_data_accessed": False,
    }), encoding="utf-8")
    fit_sha = sha256_file(pipeline.fit_path())
    for position, partition in enumerate(module.KNOWN_VALIDATION):
        write_score_store(module, pipeline, partition, np.arange(4, dtype=np.int64) + position * 10, fit_sha)
    pipeline.write_known_score_bundle(); pipeline.write_or_verify_lock()
    assert pipeline.lock_current()
    return pipeline


def test_evaluation_lock(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; pipeline = prepare_lock_fixture(module, root)
        pipeline.guard.activate_final_lock(8, True)
        expect_abort(module, pipeline.guard.assert_fitting_allowed)
        lock, sidecar = pipeline.lock_paths(); lock.write_text("{}", encoding="utf-8"); sidecar.write_text(sha256_file(lock), encoding="utf-8")
        assert not pipeline.lock_current()


def test_lock_mutation_matrix(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; pipeline = prepare_lock_fixture(module, root)
        lock, sidecar = pipeline.lock_paths(); original_lock = lock.read_bytes(); original_sidecar = sidecar.read_bytes()
        bound_files = [
            pipeline.fit_path(), root / "thresholds/ZD_STRICT_THRESHOLDS.json",
            root / "manifests/SCORER_POLICY_FREEZE.json", pipeline.known_bundle_path(),
            pipeline.inference_equivalence_path(), root / "manifests/STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json",
        ]
        for path in bound_files:
            original = path.read_bytes(); path.write_bytes(original + b"mutation"); assert not pipeline.lock_current(), path
            path.write_bytes(original); assert pipeline.lock_current(), path
        def mutate_lock(operation: Callable[[Dict[str, Any]], None]) -> None:
            payload = json.loads(original_lock); operation(payload); lock.write_text(json.dumps(payload), encoding="utf-8")
            sidecar.write_text(sha256_file(lock), encoding="utf-8"); assert not pipeline.lock_current()
            lock.write_bytes(original_lock); sidecar.write_bytes(original_sidecar); assert pipeline.lock_current()
        mutate_lock(lambda p: p["strict_violation_counters_at_lock"].pop(module.STRICT_COUNTER_KEYS[0]))
        mutate_lock(lambda p: p["strict_violation_counters_at_lock"].__setitem__(module.STRICT_COUNTER_KEYS[0], 1))
        mutate_lock(lambda p: p.__setitem__("post_lock_fitting_permitted", True))
        mutate_lock(lambda p: p.__setitem__("post_lock_calibration_permitted", True))
        pipeline.script_sha = "mutated-script"; assert not pipeline.lock_current(); pipeline.script_sha = "script-sha"; assert pipeline.lock_current()
        pipeline.config.config_sha = "mutated-config"; assert not pipeline.lock_current(); pipeline.config.config_sha = "config-sha"; assert pipeline.lock_current()


def test_known_bundle_mutation_matrix(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; pipeline = prepare_lock_fixture(module, root)
        for partition in module.KNOWN_VALIDATION:
            store = pipeline.score_store(partition)
            for name in ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy", "store_manifest.json"):
                path = store / name; original = path.read_bytes(); path.write_bytes(original + b"mutation")
                assert not pipeline.known_score_bundle_current() and not pipeline.lock_current(), (partition, name)
                path.write_bytes(original); assert pipeline.known_score_bundle_current() and pipeline.lock_current(), (partition, name)


def test_protocol_separation(module: Any) -> None:
    source = MAIN.read_text(encoding="utf-8")
    assert "strict_policy_impact\": \"NONE" in source
    assert "ZD-CALIBRATED analysis altered the ZD-STRICT threshold freeze" in source
    assert "calibration_unknown_influences_strict_policy\": False" in source
    assert "strict_zero_day_influences_selection_or_threshold\": False" in source


def test_teacher_immutability(module: Any) -> None:
    model = torch.nn.Linear(4, 3); state_before = module.state_tensor_sha256(model.state_dict())
    for parameter in model.parameters(): parameter.requires_grad_(False)
    with torch.inference_mode(): model(torch.ones(2, 4))
    assert module.state_tensor_sha256(model.state_dict()) == state_before and all(not parameter.requires_grad for parameter in model.parameters())


def test_resume_hash_invalidation(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; (root / "manifests").mkdir(parents=True)
        output = root / "tables/result.csv"; output.parent.mkdir(); output.write_text("ok", encoding="utf-8")
        pipeline = module.Stage35Pipeline.__new__(module.Stage35Pipeline); pipeline.config = fake_config(root); pipeline.script_sha = "script-sha"
        payload = {"pipeline_version": module.PIPELINE_VERSION, "executable_sha256": "script-sha", "configuration_sha256": "config-sha", "required_input_hashes": {}, "required_outputs": [{"path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size}]}
        pipeline.stage_manifest_path(4).write_text(json.dumps(payload), encoding="utf-8")
        assert pipeline.stage_current(4)
        output.write_text("changed", encoding="utf-8"); assert not pipeline.stage_current(4)
        output.write_text("ok", encoding="utf-8"); pipeline.script_sha = "changed-script"; assert not pipeline.stage_current(4)


def test_incomplete_score_store(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; pipeline = make_pipeline(module, root); indices = np.arange(3, dtype=np.int64)
        store = write_score_store(module, pipeline, "p0", indices, "fit")
        (store / "INCOMPLETE").write_text("partial", encoding="utf-8")
        assert not pipeline.score_store_current("p0", len(indices), "fit", indices)
        (store / "INCOMPLETE").unlink(); assert pipeline.score_store_current("p0", len(indices), "fit", indices)


def test_score_store_provenance_matrix(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; pipeline = make_pipeline(module, root)
        indices = np.asarray([5, 9, 12], dtype=np.int64); store = write_score_store(module, pipeline, "p0", indices, "fit")
        assert pipeline.score_store_current("p0", 3, "fit", indices)
        manifest = store / "store_manifest.json"; original_manifest = manifest.read_bytes()
        for field, value in (
            ("teacher_sha256", "mutated"), ("fit_sha256", "mutated"),
            ("scorer_definitions_sha256", "mutated"), ("energy_temperature", 2.0),
            ("benchmark_sha256", "mutated"), ("pipeline_version", "mutated"),
        ):
            payload = json.loads(original_manifest); payload[field] = value; manifest.write_text(json.dumps(payload), encoding="utf-8")
            assert not pipeline.score_store_current("p0", 3, "fit", indices), field
            manifest.write_bytes(original_manifest)
        pipeline.script_sha = "mutated"; assert not pipeline.score_store_current("p0", 3, "fit", indices); pipeline.script_sha = "script-sha"
        pipeline.config.config_sha = "mutated"; assert not pipeline.score_store_current("p0", 3, "fit", indices); pipeline.config.config_sha = "config-sha"
        assert not pipeline.score_store_current("p0", 3, "fit", np.asarray([5, 9, 13], dtype=np.int64))
        original_indices = (store / "global_indices.npy").read_bytes(); np.save(store / "global_indices.npy", np.asarray([5, 9, 13], dtype=np.int64), allow_pickle=False)
        assert not pipeline.score_store_current("p0", 3, "fit", indices); (store / "global_indices.npy").write_bytes(original_indices)
        lock_sha = "lock-sha"; strict_indices = np.asarray([100, 101], dtype=np.int64)
        strict_store = write_score_store(module, pipeline, "strict_zero_day", strict_indices, "fit", True, lock_sha)
        assert pipeline.score_store_current("strict_zero_day", 2, "fit", strict_indices, True, lock_sha)
        assert not pipeline.score_store_current("strict_zero_day", 2, "fit", strict_indices, True, "other-lock")
        assert not pipeline.score_store_current("strict_zero_day", 2, "fit", np.asarray([100, 102]), True, lock_sha)
        scores = strict_store / "scores.npy"; original_scores = scores.read_bytes(); scores.write_bytes(original_scores + b"mutation")
        assert not pipeline.score_store_current("strict_zero_day", 2, "fit", strict_indices, True, lock_sha)


def test_inference_equivalence_matrix(module: Any) -> None:
    indices = np.asarray([10, 20, 30], dtype=np.int64); labels = np.asarray([0, 1, 2], dtype=np.int64); predictions = np.asarray([0, 1, 1], dtype=np.int64)
    arguments = ["p0", indices, labels, predictions, indices.copy(), labels.copy(), predictions.copy(), 2 / 3, 0.5, 2 / 3, 0.5]
    assert module.compare_inference_evidence(*arguments)["status"] == "PASS"
    for position, mutation in (
        (4, np.asarray([10, 20, 31])), (5, np.asarray([0, 1, 3])), (6, np.asarray([0, 1, 2])),
    ):
        changed = list(arguments); changed[position] = mutation; assert module.compare_inference_evidence(*changed)["status"] == "FAIL"
    changed = list(arguments); changed[9] += 1e-4; assert module.compare_inference_evidence(*changed)["status"] == "FAIL"
    changed = list(arguments); changed[10] += 1e-4; assert module.compare_inference_evidence(*changed)["status"] == "FAIL"


def test_preflight_scope(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        branch = Path(temporary) / module.CANONICAL_BRANCH; captured: Dict[str, Any] = {}
        original = module.Stage35Pipeline
        class Capture:
            def __init__(self, config: Any): captured["config"] = config
            def run(self) -> None: captured["run"] = True
        module.Stage35Pipeline = Capture
        try: assert module.main(["--branch-root", str(branch), "--preflight"]) == 0
        finally: module.Stage35Pipeline = original
        assert captured["run"] and captured["config"].stage_start == 1 and captured["config"].stage_end == 2
        assert not (branch / "05_zero_day_open_set/MANYTX_STAGE3_5M_READY.txt").exists()


def test_logging_regression() -> None:
    stream = io.StringIO(); logger = logging.getLogger("stage35-validator"); logger.handlers.clear(); logger.addHandler(logging.StreamHandler(stream)); logger.setLevel(logging.INFO)
    logger.info("Evaluated %s rows for %s", f"{219000:,}", "ZD-STRICT")
    assert "219,000 rows" in stream.getvalue()


def test_final_transaction_source(module: Any) -> None:
    source = MAIN.read_text(encoding="utf-8")
    stage11 = source[source.index("    def stage_11"):source.index("    def run", source.index("    def stage_11"))]
    assert stage11.index("atomic_json(final_status") < stage11.index("atomic_json(manifest") < stage11.index("atomic_text(ready") < stage11.index("self.complete_stage(11")
    assert "final_hash_manifest_current" in stage11 and "self.stage_current(11)" in stage11
    for stage_name in ("stage_09", "stage_10", "stage_11"):
        start = source.index(f"    def {stage_name}"); end = source.find("\n    def ", start + 8)
        stage_source = source[start:end if end >= 0 else None]
        assert "known_score_bundle_current" in stage_source and "strict_score_bundle_current" in stage_source


def test_strict_subset_semantics(module: Any, recovery: Any) -> None:
    main = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64); shift = np.asarray([2, 5], dtype=np.int64); known = np.asarray([20, 21], dtype=np.int64)
    result = recovery.strict_subset_relationship(main, shift, known, 6, 2)
    assert result["shift_subset_of_main"] and result["intersection_rows"] == 2 and result["shift_rows_outside_main"] == 0
    assert module.validate_strict_subset_relationship(main, shift, known, 6, 2) == result
    for bad_main, bad_shift, bad_known in (
        (main, np.asarray([2, 9]), known),                    # row outside main / partial overlap
        (np.asarray([1, 2, 2, 4, 5, 6]), shift, known),      # duplicate main
        (main, np.asarray([2, 2]), known),                    # duplicate shift
        (main, shift, np.asarray([5, 20])),                   # non-strict overlap
    ):
        try:
            recovery.strict_subset_relationship(bad_main, bad_shift, bad_known, 6, 2)
        except recovery.RecoveryAbort:
            pass
        else:
            raise AssertionError("Invalid strict subset relationship must fail")


def test_overlap_consistency(recovery: Any) -> None:
    main_indices = np.asarray([10, 20, 30, 40]); shift_indices = np.asarray([20, 40])
    main_predictions = np.asarray([1, 2, 3, 4]); shift_predictions = np.asarray([2, 4])
    main_scores = np.arange(20, dtype=np.float32).reshape(4, 5); shift_scores = main_scores[[1, 3]].copy()
    assert recovery.overlap_score_consistency(main_indices, shift_indices, main_predictions, shift_predictions, main_scores, shift_scores)["predictions_exactly_equal"]
    within = shift_scores.copy(); within[0, 0] += 5e-6
    assert recovery.overlap_score_consistency(main_indices, shift_indices, main_predictions, shift_predictions, main_scores, within, 1e-5)["all_score_vectors_within_tolerance"]
    for predictions, scores in ((np.asarray([9, 4]), shift_scores), (shift_predictions, shift_scores + 1e-3)):
        try:
            recovery.overlap_score_consistency(main_indices, shift_indices, main_predictions, predictions, main_scores, scores, 1e-5)
        except recovery.RecoveryAbort:
            pass
        else:
            raise AssertionError("Persisted overlap inconsistency must fail")


def make_recovery_fixture(recovery: Any, root: Path) -> tuple[Any, Dict[str, Path]]:
    branch = root / recovery.CANONICAL_BRANCH; output = branch / "05_zero_day_open_set"; manifests = output / "manifests"
    for directory in (manifests, output / "thresholds", output / "scores/fitted", output / "tables", output / "statistics",
                      output / "figures", output / "reports", output / "publication", branch / "01_benchmark_engineering/splits"):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "MANYTX_STAGE3_5M_NOT_READY.txt").write_text("\n".join(recovery.EXPECTED_NOT_READY_LINES) + "\n", encoding="utf-8")
    fit = output / "scores/fitted/known_only_scorer_state.npz"; fit.write_bytes(b"fit")
    threshold = output / "thresholds/ZD_STRICT_THRESHOLDS.json"; threshold.write_text(json.dumps({
        "thresholds": {scorer: {"canonical_threshold": 0.5} for scorer in recovery.SCORER_ORDER}
    }), encoding="utf-8")
    policy = manifests / "SCORER_POLICY_FREEZE.json"; policy.write_text("{}", encoding="utf-8")
    equivalence = manifests / "STAGE3M_STAGE35_INFERENCE_EQUIVALENCE.json"; equivalence.write_text("{}", encoding="utf-8")
    main_indices = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64); shift_indices = np.asarray([2, 5], dtype=np.int64)
    sealed_dir = branch / "01_benchmark_engineering/sealed"; sealed_dir.mkdir(parents=True)
    main_path = sealed_dir / "strict_zero_day_test_indices.npy"; shift_path = sealed_dir / "strict_zero_day_shift_test_indices.npy"
    np.save(main_path, main_indices, allow_pickle=False); np.save(shift_path, shift_indices, allow_pickle=False)
    declaration = manifests / "STAGE02_TEACHER_PARTITION_EXPOSURE_AUDIT.json"
    declaration.write_text(json.dumps({"sealed_strict_partitions": {
        "strict_zero_day_test": {"path": str(main_path), "declared_sha256_candidates": [sha256_file(main_path)]},
        "strict_zero_day_shift_test": {"path": str(shift_path), "declared_sha256_candidates": [sha256_file(shift_path)]},
    }}), encoding="utf-8")
    split_dir = branch / "01_benchmark_engineering/splits"
    for position, name in enumerate(recovery.NON_STRICT_INDEX_FILES):
        np.save(split_dir / name, np.asarray([100 + position], dtype=np.int64), allow_pickle=False)
    known_bundle = manifests / "PRE_STRICT_KNOWN_SCORE_BUNDLE.json"; known_partitions: Dict[str, Any] = {}
    for position, partition in enumerate(recovery.KNOWN_VALIDATION):
        store = output / "scores" / partition; store.mkdir(parents=True)
        arrays = {
            "scores.npy": np.zeros((2, 5), dtype=np.float32), "predictions.npy": np.zeros(2, dtype=np.int16),
            "labels.npy": np.zeros(2, dtype=np.int16), "known_correct.npy": np.ones(2, dtype=bool),
            "global_indices.npy": np.asarray([200 + position * 2, 201 + position * 2], dtype=np.int64),
        }
        for name, values in arrays.items(): np.save(store / name, values, allow_pickle=False)
        (store / "store_manifest.json").write_text("{}", encoding="utf-8")
        files = {name: {"path": str(store / name), "sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in (*arrays, "store_manifest.json")}
        known_partitions[partition] = {"files": files}
    pd.DataFrame([{"partition": partition, "scorer": scorer, "mean": 0.0} for partition in recovery.KNOWN_VALIDATION for scorer in recovery.SCORER_ORDER]).to_csv(output / "tables/known_validation_score_characterization.csv", index=False)
    pd.DataFrame([{"partition": partition, "accuracy": 1.0, "fixed98_macro_f1": 1.0} for partition in recovery.KNOWN_VALIDATION]).to_csv(output / "tables/closed_set_teacher_metrics.csv", index=False)
    known_bundle.write_text(json.dumps({"status": "FROZEN_BEFORE_STRICT_EVALUATION", "partitions": known_partitions}), encoding="utf-8")
    recovery.EXPECTED_FIT_SHA256 = sha256_file(fit); recovery.EXPECTED_THRESHOLD_SHA256 = sha256_file(threshold)
    recovery.EXPECTED_POLICY_SHA256 = sha256_file(policy); recovery.EXPECTED_EQUIVALENCE_SHA256 = sha256_file(equivalence)
    recovery.EXPECTED_DECLARATION_SHA256 = sha256_file(declaration); recovery.EXPECTED_KNOWN_BUNDLE_SHA256 = sha256_file(known_bundle)
    lock = manifests / "STRICT_ZERO_DAY_EVALUATION_LOCK.json"
    lock_payload = {
        "executable_sha256": recovery.ORIGINAL_EXECUTABLE_SHA256, "configuration_sha256": recovery.ORIGINAL_CONFIGURATION_SHA256,
        "teacher_sha256": recovery.EXPECTED_TEACHER_SHA256, "benchmark_sha256": recovery.EXPECTED_BENCHMARK_SHA256,
        "stage2_6m_artifact_sha256": recovery.EXPECTED_STAGE26_ARTIFACT_SHA256,
        "stage3m_hash_manifest_sha256": recovery.EXPECTED_STAGE3M_HASH_MANIFEST_SHA256,
        "scorer_fit_sha256": recovery.EXPECTED_FIT_SHA256, "threshold_manifest_sha256": recovery.EXPECTED_THRESHOLD_SHA256,
        "policy_freeze_sha256": recovery.EXPECTED_POLICY_SHA256, "known_score_bundle_sha256": recovery.EXPECTED_KNOWN_BUNDLE_SHA256,
        "stage3m_stage35_inference_equivalence_sha256": recovery.EXPECTED_EQUIVALENCE_SHA256,
        "sealed_strict_declaration_sha256": recovery.EXPECTED_DECLARATION_SHA256,
        "post_lock_fitting_permitted": False, "post_lock_calibration_permitted": False,
        "strict_violation_counters_at_lock": {key: 0 for key in recovery.STRICT_COUNTER_KEYS},
    }
    lock.write_text(json.dumps(lock_payload), encoding="utf-8"); recovery.ORIGINAL_LOCK_SHA256 = sha256_file(lock)
    sidecar = manifests / "STRICT_ZERO_DAY_EVALUATION_LOCK.sha256"; sidecar.write_text(recovery.ORIGINAL_LOCK_SHA256 + "\n", encoding="utf-8")
    scores = np.arange(30, dtype=np.float32).reshape(6, 5); predictions = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int16)
    strict_paths: Dict[str, Path] = {}
    for partition, indices, partition_scores, partition_predictions in (
        ("strict_zero_day_test", main_indices, scores, predictions),
        ("strict_zero_day_shift_test", shift_indices, scores[[1, 4]], predictions[[1, 4]]),
    ):
        store = output / "scores" / partition; store.mkdir(parents=True); strict_paths[partition] = store
        np.save(store / "scores.npy", partition_scores, allow_pickle=False); np.save(store / "predictions.npy", partition_predictions, allow_pickle=False); np.save(store / "global_indices.npy", indices, allow_pickle=False)
        files = {name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in ("scores.npy", "predictions.npy", "global_indices.npy")}
        manifest = {"complete": True, "pipeline_version": recovery.PIPELINE_VERSION, "executable_sha256": recovery.ORIGINAL_EXECUTABLE_SHA256,
            "configuration_sha256": recovery.ORIGINAL_CONFIGURATION_SHA256, "benchmark_sha256": recovery.EXPECTED_BENCHMARK_SHA256,
            "teacher_sha256": recovery.EXPECTED_TEACHER_SHA256, "fit_sha256": recovery.EXPECTED_FIT_SHA256,
            "scorer_order": list(recovery.SCORER_ORDER), "scorer_definitions_sha256": recovery.sha256_object(recovery.SCORER_DEFINITIONS),
            "energy_temperature": 1.0, "partition": partition, "rows": len(indices), "strict": True,
            "global_indices_sha256": recovery.sha256_int64_array(indices), "evaluation_lock_sha256": recovery.ORIGINAL_LOCK_SHA256,
            "strict_labels_loaded": False, "files": files}
        (store / "store_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stage_output = manifests / "fixture_stage_output.txt"; stage_output.write_text("frozen", encoding="utf-8")
    for stage in range(1, 8):
        (manifests / f"STAGE_{stage:02d}_CHECKPOINT.json").write_text(json.dumps({
            "pipeline_version": recovery.PIPELINE_VERSION, "executable_sha256": recovery.ORIGINAL_EXECUTABLE_SHA256,
            "configuration_sha256": recovery.ORIGINAL_CONFIGURATION_SHA256, "required_input_hashes": {},
            "required_outputs": [{"path": str(stage_output), "sha256": sha256_file(stage_output), "bytes": stage_output.stat().st_size}],
        }), encoding="utf-8")
    recovery.STRICT_PARTITIONS = {"strict_zero_day_test": 6, "strict_zero_day_shift_test": 2}
    runner = recovery.PostLockRecovery(branch, ROOT)
    paths = {"lock": lock, "sidecar": sidecar, "stage_output": stage_output, "threshold": threshold, "fit": fit,
             "known_bundle": known_bundle, "main_scores": strict_paths["strict_zero_day_test"] / "scores.npy",
             "main_manifest": strict_paths["strict_zero_day_test"] / "store_manifest.json",
             "main_store": strict_paths["strict_zero_day_test"]}
    return runner, paths


def test_recovery_precondition_matrix(recovery: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        runner, paths = make_recovery_fixture(recovery, Path(temporary)); assert runner.preflight().relationship["shift_subset_of_main"]
        inventory_before = {str(path): sha256_file(path) for path in runner.output_root.rglob("*") if path.is_file()}
        assert recovery.main(["--branch-root", str(runner.branch_root), "--repository-root", str(ROOT), "--preflight"]) == 0
        inventory_after = {str(path): sha256_file(path) for path in runner.output_root.rglob("*") if path.is_file()}
        assert inventory_before == inventory_after and not (runner.output_root / "MANYTX_STAGE3_5M_READY.txt").exists()
        for key in ("lock", "stage_output", "threshold", "fit", "known_bundle", "main_scores"):
            path = paths[key]; original = path.read_bytes(); path.write_bytes(original + b"mutation")
            try: runner.preflight()
            except recovery.RecoveryAbort: pass
            else: raise AssertionError(f"Recovery accepted modified {key}")
            path.write_bytes(original); assert runner.preflight()
        manifest = paths["main_manifest"]; original_manifest = manifest.read_bytes(); payload = json.loads(original_manifest); payload["evaluation_lock_sha256"] = "wrong"; manifest.write_text(json.dumps(payload), encoding="utf-8")
        try: runner.preflight()
        except recovery.RecoveryAbort: pass
        else: raise AssertionError("Recovery accepted mismatched strict-store lock SHA")
        manifest.write_bytes(original_manifest); assert runner.preflight()
        label_path = paths["main_store"] / "labels.npy"; np.save(label_path, np.zeros(6), allow_pickle=False)
        try: runner.preflight()
        except recovery.RecoveryAbort: pass
        else: raise AssertionError("Recovery accepted a strict label file")


def test_recovery_static_isolation(recovery: Any) -> None:
    source = RECOVERY.read_text(encoding="utf-8")
    for forbidden in ("import h5py", "import torch", "DataLoader", "extract_strict_scores", "score_outputs(", "fit_statistics(", ".backward(", "optimizer.step("):
        assert forbidden not in source
    normal_source = MAIN.read_text(encoding="utf-8")
    assert "strict_combined" not in normal_source and "strict_combined" not in source
    assert normal_source.index("validate_strict_subset_relationship(", normal_source.index("    def stage_08")) < normal_source.index("self.extract_strict_scores(", normal_source.index("    def stage_08"))
    assert "post_lock_recovery=YES" in source and "strict_signal_reinference=NO" in source and "original_lock_modified" in source


def test_synthetic_recovery_finalization(recovery: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        runner, paths = make_recovery_fixture(recovery, Path(temporary)); recovery.BOOTSTRAP_REPLICATES = 2; recovery.BOOTSTRAP_MAX_PER_GROUP = 4
        lock_before = paths["lock"].read_bytes()
        store_before = {str(path): sha256_file(path) for partition in recovery.STRICT_PARTITIONS for path in (runner.output_root / "scores" / partition).iterdir() if path.is_file()}
        runner.finalize()
        assert paths["lock"].read_bytes() == lock_before
        store_after = {str(path): sha256_file(path) for partition in recovery.STRICT_PARTITIONS for path in (runner.output_root / "scores" / partition).iterdir() if path.is_file()}
        assert store_before == store_after
        metrics = pd.read_csv(runner.output_root / "tables/strict_open_set_metrics.csv")
        assert set(metrics.strict_partition) == set(recovery.STRICT_PARTITIONS) and "strict_combined" not in set(metrics.strict_partition)
        ready = (runner.output_root / "MANYTX_STAGE3_5M_READY.txt").read_text(encoding="utf-8")
        assert "post_lock_recovery=YES" in ready and "strict_signal_reinference=NO" in ready and "strict_shift_subset_of_main=YES" in ready
        assert not (runner.output_root / "MANYTX_STAGE3_5M_NOT_READY.txt").exists()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-only", action="store_true"); args = parser.parse_args()
    module = load_module(MAIN, "stage35_validator_target")
    recovery = load_module(RECOVERY, "stage35_recovery_validator_target")
    tests: Mapping[str, Callable[[], None]] = {
        "static_and_package": lambda: test_static(module, not args.source_only),
        "scientific_contract": lambda: test_contract(module),
        "strict_partition_guard": lambda: test_strict_guard(module),
        "no_unknown_training_class": lambda: test_no_unknown_training_class(module),
        "scorer_correctness_and_direction": lambda: test_scorers(module),
        "covariance_regularization": lambda: test_covariance_regularization(module),
        "known_only_thresholds": lambda: test_threshold_known_only(module),
        "open_set_metrics_and_oscr": lambda: test_open_set_metrics(module),
        "final_evaluation_lock": lambda: test_evaluation_lock(module),
        "lock_mutation_matrix": lambda: test_lock_mutation_matrix(module),
        "known_bundle_mutation_matrix": lambda: test_known_bundle_mutation_matrix(module),
        "zd_protocol_separation": lambda: test_protocol_separation(module),
        "teacher_immutability": lambda: test_teacher_immutability(module),
        "resume_hash_invalidation": lambda: test_resume_hash_invalidation(module),
        "incomplete_score_store": lambda: test_incomplete_score_store(module),
        "score_store_provenance_matrix": lambda: test_score_store_provenance_matrix(module),
        "stage3m_stage35_inference_equivalence": lambda: test_inference_equivalence_matrix(module),
        "preflight_scope": lambda: test_preflight_scope(module),
        "logging_regression": test_logging_regression,
        "stage11_transaction_order": lambda: test_final_transaction_source(module),
        "strict_shift_subset_semantics": lambda: test_strict_subset_semantics(module, recovery),
        "strict_overlap_score_consistency": lambda: test_overlap_consistency(recovery),
        "postlock_recovery_precondition_matrix": lambda: test_recovery_precondition_matrix(recovery),
        "postlock_recovery_static_isolation": lambda: test_recovery_static_isolation(recovery),
        "synthetic_postlock_recovery_finalization": lambda: test_synthetic_recovery_finalization(recovery),
    }
    for name, operation in tests.items(): operation(); print(f"[PASS] {name}")
    print(f"STAGE3_5M_VALIDATION_PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
