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
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.py"
LAUNCHER = ROOT / "Stage3_5M_Colab_Launcher_v1_0_0.py"
CONFIG = ROOT / "stage3_5m_config_v1_0_0.example.json"
NOTEBOOK = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.ipynb"
MANIFEST = ROOT / "STAGE3_5M_CODE_MANIFEST.json"
PACKAGE = ROOT / "Stage3_5M_WiSig_ManyTx_ZeroDay_OpenSet_v1_0_0.zip"
REQUIRED_PACKAGE = (
    MAIN.name, LAUNCHER.name, CONFIG.name, Path(__file__).name, NOTEBOOK.name,
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
    for path in [MAIN, LAUNCHER, CONFIG, NOTEBOOK, *[ROOT / name for name in REQUIRED_PACKAGE[5:9]], ROOT / "requirements_stage3_5m.txt"]:
        assert path.is_file(), path
    ast.parse(MAIN.read_text(encoding="utf-8")); ast.parse(LAUNCHER.read_text(encoding="utf-8")); ast.parse(Path(__file__).read_text(encoding="utf-8"))
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
        def configuration_sha256(self) -> str:
            return "config-sha"
    return Config()


def test_evaluation_lock(module: Any) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "05_zero_day_open_set"; (root / "manifests").mkdir(parents=True); (root / "thresholds").mkdir(); (root / "scores/fitted").mkdir(parents=True)
        pipeline = module.Stage35Pipeline.__new__(module.Stage35Pipeline); pipeline.config = fake_config(root); pipeline.script_sha = "script-sha"; pipeline.guard = module.StrictZeroDayGuard()
        (root / "scores/fitted/known_only_scorer_state.npz").write_bytes(b"fit")
        (root / "thresholds/ZD_STRICT_THRESHOLDS.json").write_text("{}", encoding="utf-8")
        (root / "manifests/SCORER_POLICY_FREEZE.json").write_text("{}", encoding="utf-8")
        digest = pipeline.write_or_verify_lock(); assert len(digest) == 64 and pipeline.lock_current()
        pipeline.guard.activate_final_lock(8, True)
        expect_abort(module, pipeline.guard.assert_fitting_allowed)
        lock = pipeline.lock_paths()[0]; lock.write_text("{}", encoding="utf-8"); assert not pipeline.lock_current()


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
        root = Path(temporary) / "05_zero_day_open_set"; store = root / "scores/p0"; store.mkdir(parents=True)
        pipeline = module.Stage35Pipeline.__new__(module.Stage35Pipeline); pipeline.config = SimpleNamespace(output_root=root)
        rows, fit_sha = 3, "fit"
        np.save(store / "scores.npy", np.zeros((rows, 5), dtype=np.float32)); np.save(store / "predictions.npy", np.zeros(rows)); np.save(store / "labels.npy", np.zeros(rows)); np.save(store / "known_correct.npy", np.ones(rows)); np.save(store / "global_indices.npy", np.arange(rows))
        names = ("scores.npy", "predictions.npy", "labels.npy", "known_correct.npy", "global_indices.npy")
        files = {name: {"sha256": sha256_file(store / name), "bytes": (store / name).stat().st_size} for name in names}
        (store / "store_manifest.json").write_text(json.dumps({"complete": True, "partition": "p0", "rows": rows, "fit_sha256": fit_sha, "teacher_sha256": module.EXPECTED_TEACHER_SHA256, "strict": False, "scorer_order": list(module.SCORER_ORDER), "files": files}), encoding="utf-8")
        (store / "INCOMPLETE").write_text("partial", encoding="utf-8"); assert not pipeline.score_store_current("p0", rows, fit_sha)
        (store / "INCOMPLETE").unlink(); assert pipeline.score_store_current("p0", rows, fit_sha)


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


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-only", action="store_true"); args = parser.parse_args()
    module = load_module(MAIN, "stage35_validator_target")
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
        "zd_protocol_separation": lambda: test_protocol_separation(module),
        "teacher_immutability": lambda: test_teacher_immutability(module),
        "resume_hash_invalidation": lambda: test_resume_hash_invalidation(module),
        "incomplete_score_store": lambda: test_incomplete_score_store(module),
        "preflight_scope": lambda: test_preflight_scope(module),
        "logging_regression": test_logging_regression,
        "stage11_transaction_order": lambda: test_final_transaction_source(module),
    }
    for name, operation in tests.items(): operation(); print(f"[PASS] {name}")
    print(f"STAGE3_5M_VALIDATION_PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
