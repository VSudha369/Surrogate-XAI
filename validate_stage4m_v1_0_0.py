#!/usr/bin/env python3
"""Static and synthetic validator for Stage 4M v1.0.0."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import py_compile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.py"
LAUNCHER = ROOT / "Stage4M_Colab_Launcher_v1_0_0.py"
NOTEBOOK = ROOT / "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.ipynb"
ZIP = ROOT / "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.zip"
MANIFEST = ROOT / "STAGE4M_CODE_MANIFEST.json"
EXPECTED_MEMBERS = {
    "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.py", "Stage4M_Colab_Launcher_v1_0_0.py",
    "Stage4M_WiSig_ManyTx_Surrogate_KD_v1_0_0.ipynb", "validate_stage4m_v1_0_0.py",
    "stage4m_config_v1_0_0.example.json", "requirements_stage4m.txt", "STAGE4M_CODE_MANIFEST.json",
    "STAGE4M_SCIENTIFIC_PROTOCOL.md", "STAGE4M_OUTPUT_SCHEMA.md", "STAGE4M_VALIDATION_CHECKLIST.md", "STAGE4M_README.md",
    "STAGE4M_PREDECESSOR_DRIVE_AUDIT.json", "STAGE4M_PREDECESSOR_DRIVE_AUDIT.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_main() -> Any:
    spec = importlib.util.spec_from_file_location("stage4m_validation_target", MAIN)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load Stage 4M")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def main() -> int:
    passed = 0

    def check(name: str, condition: Any) -> None:
        nonlocal passed
        value = condition() if callable(condition) else condition
        if not value: raise AssertionError(name)
        passed += 1; print(f"[{passed:02d}] PASS — {name}")

    source = MAIN.read_text(encoding="utf-8"); launcher_source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source); module = load_main()
    check("Python compilation main", lambda: py_compile.compile(str(MAIN), doraise=True))
    check("Python compilation launcher", lambda: py_compile.compile(str(LAUNCHER), doraise=True))
    check("Python compilation validator", lambda: py_compile.compile(str(Path(__file__)), doraise=True))
    check("AST parse", isinstance(tree, ast.Module))
    check("pyflakes main", lambda: __import__("pyflakes.api").api.checkPath(str(MAIN)) == 0)
    check("pyflakes launcher", lambda: __import__("pyflakes.api").api.checkPath(str(LAUNCHER)) == 0)
    check("notebook JSON", lambda: isinstance(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"], list))
    check("predecessor audit PASS", json.loads((ROOT / "STAGE4M_PREDECESSOR_DRIVE_AUDIT.json").read_text())["audit_result"] == "STAGE4M_PREDECESSOR_DRIVE_AUDIT_PASS")
    check("benchmark SHA constant", module.EXPECTED_BENCHMARK_SHA256 == "9cce10dcee47c81dad855da3bd5ff845af2b955cee1a0fe03084609560cbd3b9")
    check("Stage2M SHA constant", module.EXPECTED_STAGE2M_MANIFEST_SHA256 == "0a8853d782006ce8af2d7b798a61c1e141afbeb55066cb70115ae41c8d24f16a")
    check("Stage2.6M SHA constant", module.EXPECTED_STAGE26_ARTIFACT_SHA256 == "83b1eec28b36afd39fffb4d3b719d92ccd3f0caaa270df0d16f4f28eab209660")
    check("Stage3M SHA constant", module.EXPECTED_STAGE3M_MANIFEST_SHA256 == "5aeaa4a2b0ec65642853426dfea56223ea223bbd027769009f705b6fd59d3ea0")
    check("teacher checkpoint SHA", module.EXPECTED_TEACHER_SHA256 == "ed8698ca9ac6ba813e6d74734ac16987129b0e3079b865f9502974119414aaf4")
    check("teacher state SHA", module.EXPECTED_TEACHER_STATE_SHA256 == "7d6c6ff609fb86618ae7b92bcd55b0c8a440ed2769561d4de9b4485802e639d7")
    check("teacher parameter count", module.EXPECTED_TEACHER_PARAMETERS == 849634)
    check("teacher embedding", module.TEACHER_EMBEDDING_DIM == 128)
    check("known classes", module.EXPECTED_KNOWN_CLASSES == 98)
    fixture = "MANYTX_STAGE3_5M_READY\nteacher_seed=123\nstrict_scores_recomputed=NO\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "READY.txt"; path.write_text(fixture)
        ready = module.parse_ready(path)
    check("actual READY schema fixture", ready == {"marker": "MANYTX_STAGE3_5M_READY", "teacher_seed": "123", "strict_scores_recomputed": "NO"})
    student = module.WiSigSurrogateNet(); freeze = module.architecture_freeze_payload()
    check("one fixed student architecture", module.model_signature(student) == freeze["architecture_signature_sha256"])
    check("architecture compression gate", freeze["status"] == "PASS" and freeze["deployed_parameter_count"] <= .4 * 849634)
    check("native embedding dimension 64", student.embedding_dim == 64)
    check("classifier output dimension 98", student.classifier.out_features == 98)
    check("deterministic even width", [module.even_width(x) for x in (32,64,128,256)] == [16,32,64,128])
    check("minimum hidden width", min(module.even_width(x) for x in (32,64,128,256)) >= 16)
    check("auxiliary shape", freeze["auxiliary_projection"]["shape"] == [64,128])
    check("auxiliary training only", freeze["auxiliary_projection"]["training_only"] is True)
    check("deployed excludes auxiliary", freeze["deployed_parameter_count"] == module.model_parameter_count(student))
    output = student(torch.randn(4,2,256))
    check("student logits shape", tuple(output["logits"].shape) == (4,98))
    check("student embedding shape", tuple(output["embedding_normalized"].shape) == (4,64))
    check("student normalized embedding", torch.allclose(output["embedding_normalized"].norm(dim=1), torch.ones(4), atol=1e-5))
    check("K0 exact loss", module.ARM_OBJECTIVES["K0"] == {"ce":1.0,"kd":0.0,"repr":0.0,"proto":0.0})
    check("K1 exact loss", module.ARM_OBJECTIVES["K1"] == {"ce":0.5,"kd":0.5,"repr":0.0,"proto":0.0})
    check("K2 exact loss", module.ARM_OBJECTIVES["K2"] == {"ce":0.4,"kd":0.4,"repr":0.2,"proto":0.0})
    check("K3 exact loss", module.ARM_OBJECTIVES["K3"] == {"ce":0.35,"kd":0.35,"repr":0.15,"proto":0.15})
    check("temperature 4.0", module.KD_TEMPERATURE == 4.0)
    check("no loss weight search", module.kd_objective_policy()["weight_search"] is False)
    check("no temperature search", module.kd_objective_policy()["temperature_search"] is False)
    teacher_logits = torch.randn(4,98,requires_grad=True); teacher_embedding = torch.randn(4,128,requires_grad=True)
    labels = torch.tensor([0,1,2,3]); auxiliary = torch.nn.Linear(64,128); prototypes = torch.nn.functional.normalize(torch.randn(98,128),dim=1)
    loss, _ = module.compute_kd_losses("K3", output, teacher_logits, teacher_embedding, labels, auxiliary, prototypes); loss.backward()
    check("teacher logits detached", teacher_logits.grad is None)
    check("teacher embedding detached", teacher_embedding.grad is None)
    check("student receives gradient", any(parameter.grad is not None for parameter in student.parameters()))
    check("KL orientation source", "F.kl_div(student_log_probs, teacher_probs" in source)
    check("seed set exact", module.SEEDS == (42,123,2026))
    check("four arms exact", module.ARMS == ("K0","K1","K2","K3"))
    check("Train Known only canonical training", 'teacher_store = self.prepare_teacher_cache("train_known")' in source)
    check("P0-only epoch selection", 'self.evaluate_student(model, "p0"' in source)
    check("P1 selection forbidden", 'selection_partition": "p0"' in source)
    check("P2 selection forbidden", 'p1_p3_used_for_selection": False' in source)
    check("P3 selection forbidden", 'selection_role": "REPORTING_ONLY"' in source)
    check("calibration training forbidden", 'calibration_unknown_used_for_training": False' in source)
    check("calibration selection forbidden", 'calibration_unknown_used_for_selection": False' in source)
    guard = module.Stage35StrictGuard(Path.cwd(), Path.cwd())
    for kind, filename in (("signal","strict_zero_day_test.h5"),("index","strict_zero_day_test_indices.npy"),("score","strict_zero_day_test_scores.npy"),("metric","strict_open_set_metrics.csv")):
        rejected = False
        try: guard.reject(kind, filename)
        except module.ScientificAbort: rejected = True
        check(f"strict {kind} access forbidden", rejected)
    check("strict selection access forbidden", "stage35_strict_selection_violations" in module.STAGE35_COUNTER_KEYS)
    check("six leakage counters", len(module.STAGE35_COUNTER_KEYS) == 6)
    check("Stage3.5 strict-result denylist", all(token in module.STRICT_DENY_TOKENS for token in ("strict_zero_day_test","strict_open_set_metrics","final_strict_score_bundle")))
    check("predecessor metadata whitelist", module.STAGE35_METADATA_ALLOWLIST == frozenset({"MANYTX_STAGE3_5M_READY.txt","STAGE3_5M_FINAL_STATUS.json","STAGE3_5M_HASH_MANIFEST.json","POST_LOCK_RECOVERY_MANIFEST.json","STRICT_ZERO_DAY_EVALUATION_LOCK.json"}))
    incumbent = {"epoch":4,"teacher_student_top1_agreement":.91,"teacher_student_kl":.2,"student_fixed98_macro_f1":.82}
    check("best epoch agreement", module.p0_epoch_better({**incumbent,"epoch":5,"teacher_student_top1_agreement":.92},incumbent))
    check("best epoch tolerance KL", module.p0_epoch_better({**incumbent,"epoch":5,"teacher_student_top1_agreement":.911,"teacher_student_kl":.1},incumbent))
    check("best epoch F1", module.p0_epoch_better({**incumbent,"epoch":5,"teacher_student_kl":.2,"student_fixed98_macro_f1":.83},incumbent))
    check("best epoch earlier tie", not module.p0_epoch_better({**incumbent,"epoch":5},incumbent))
    arm_summary = {arm:{"median_agreement":.91,"median_kl":.2,"std_agreement":.01,"median_f1":.82} for arm in module.ARMS}
    check("simpler arm tie-break", module.choose_arm(arm_summary) == "K0")
    arm_summary["K3"]["median_agreement"] = .92
    check("arm agreement policy", module.choose_arm(arm_summary) == "K3")
    seed_rows = [{"seed":42,"teacher_student_top1_agreement":.91,"teacher_student_kl":.2,"student_fixed98_macro_f1":.82},{"seed":123,"teacher_student_top1_agreement":.92,"teacher_student_kl":.2,"student_fixed98_macro_f1":.82},{"seed":2026,"teacher_student_top1_agreement":.90,"teacher_student_kl":.1,"student_fixed98_macro_f1":.83}]
    check("canonical seed P0 policy", module.choose_seed(seed_rows) == 123)
    check("checkpoint stores RNG", '"rng_state": capture_rng()' in source)
    check("checkpoint stores optimizer", '"optimizer_state": optimizer.state_dict()' in source)
    check("checkpoint stores scheduler", '"scheduler_state": scheduler.state_dict()' in source)
    check("checkpoint stores loader generator", '"dataloader_generator_state"' in source)
    check("stale checkpoint rejection", "STALE_STAGE4M_CHECKPOINT" in source)
    check("architecture mutation rejection", '"architecture_freeze_sha256"' in source)
    check("objective mutation rejection", '"objective_policy_sha256"' in source)
    check("predecessor mutation rejection", '"predecessor_lock_sha256"' in source)
    check("selection mutation rejection", '"selection_lock_sha256"' in source)
    check("calibration blocked before lock", "Calibration diagnostic requires an immutable canonical selection lock" in source)
    check("calibration cannot mutate surrogate", "Calibration Unknown mutated the canonical surrogate" in source)
    check("READY fidelity gate", '"p0_fidelity_gate_pass"' in source)
    check("READY compression gate", '"compression_gate_pass"' in source)
    check("READY leakage gate", '"all_stage35_strict_violation_counters_zero"' in source)
    check("READY declares no strict eval", 'strict_zero_day_evaluation_performed=NO' in source)
    order = [source.index(fragment) for fragment in ("atomic_json(final", "manifest = self.create_final_hash_manifest()", "atomic_text(ready", "self.complete_stage(12", "if not_ready.exists(): not_ready.unlink()")]
    check("Stage12 transaction order", order == sorted(order))
    check("Stage12 resume READY gate", "if stage == 12" in source and "final_hash_manifest_current" in source)
    check("idempotent finalization", "if ready.exists(): ready.unlink()" in source)
    with tempfile.TemporaryDirectory() as directory:
        search = Path(directory); root = search / module.CANONICAL_BRANCH
        zero_failed = False
        try: module.discover_branch_root(search_root=search)
        except module.ScientificAbort: zero_failed = True
        check("Drive discovery zero-root failure", zero_failed)
        for name in ("01_benchmark_engineering","02_benchmark_diagnostics","03_representation_ablation","04_canonical_teacher","05_zero_day_open_set"): (root/name).mkdir(parents=True)
        (root/"04_canonical_teacher"/"MANYTX_STAGE3M_READY.txt").write_text("x")
        (root/"05_zero_day_open_set"/"MANYTX_STAGE3_5M_READY.txt").write_text("x")
        check("Drive discovery one-root pass", module.discover_branch_root(search_root=search) == root.resolve())
        duplicate = search / "copy" / module.CANONICAL_BRANCH
        for name in ("01_benchmark_engineering","02_benchmark_diagnostics","03_representation_ablation","04_canonical_teacher","05_zero_day_open_set"): (duplicate/name).mkdir(parents=True)
        (duplicate/"04_canonical_teacher"/"MANYTX_STAGE3M_READY.txt").write_text("x"); (duplicate/"05_zero_day_open_set"/"MANYTX_STAGE3_5M_READY.txt").write_text("x")
        multiple_failed = False
        try: module.discover_branch_root(search_root=search)
        except module.ScientificAbort: multiple_failed = True
        check("Drive discovery multiple-root failure", multiple_failed)
    check("launcher explicit/env/search order", "explicit or os.environ.get(\"WISIG_BRANCH_ROOT\")" in launcher_source)
    preflight_body = source[source.index("    def preflight(self)"):source.index("    def run(self)")]
    check("preflight cannot train", "train_arm_seed" not in preflight_body)
    check("preflight cannot cache teacher targets", "prepare_teacher_cache" not in preflight_body)
    check("preflight cannot access calibration", "student_outputs" not in preflight_body and '"calibration_unknown_accessed": False' in preflight_body)
    check("preflight cannot create READY", "atomic_text(ready" not in preflight_body and '"ready_created": False' in preflight_body)
    check("preflight marker", "STAGE4M_PREFLIGHT_PASS" in preflight_body)
    with zipfile.ZipFile(ZIP) as archive:
        check("ZIP CRC", archive.testzip() is None)
        check("ZIP member set", set(archive.namelist()) == EXPECTED_MEMBERS)
        check("ZIP byte-for-byte members", all(archive.read(name) == (ROOT/name).read_bytes() for name in EXPECTED_MEMBERS))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check("code manifest member set", {row["path"] for row in manifest["files"]} == EXPECTED_MEMBERS - {"STAGE4M_CODE_MANIFEST.json"})
    check("code manifest hashes", all(sha256(ROOT/row["path"]) == row["sha256"] for row in manifest["files"]))
    check("no canonical READY in repository", not (ROOT/"MANYTX_STAGE4M_READY.txt").exists())
    validator_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    check("no canonical training in validator", not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Stage4Pipeline" for node in ast.walk(validator_tree)))
    if passed < 62: raise AssertionError(f"validator coverage too small: {passed}")
    print(f"\nSTAGE4M_VALIDATION_PASS ({passed}/{passed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
