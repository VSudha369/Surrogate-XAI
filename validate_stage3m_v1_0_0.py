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
    for path in (MAIN, LAUNCHER, ROOT / "validate_stage3m_v1_0_0.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    assert "%,d" not in MAIN.read_text(encoding="utf-8")
    assert "architecture search" not in MAIN.read_text(encoding="utf-8").lower().replace("architecture search: disabled", "")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook.get("cells")
    pyflakes = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(MAIN), str(LAUNCHER), str(Path(__file__))],
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
        "strict_zero_day_guard": lambda: test_zero_day(module),
        "embedding_store_resume": lambda: test_storage(module),
        "logging_and_ready_gate": lambda: test_logging_and_ready(module),
    }
    for name, operation in tests.items():
        operation(); print(f"[PASS] {name}")
    print(f"STAGE3M_VALIDATION_PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
