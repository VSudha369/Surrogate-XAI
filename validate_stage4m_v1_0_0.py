#!/usr/bin/env python3
"""Static and synthetic validator for Stage 4M v1.0.0."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
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

    class CountingAugmentation:
        def __init__(self) -> None: self.calls = 0
        def __call__(self, value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
            del generator; self.calls += 1; return value + 0.125

    class RecordingModel(torch.nn.Module):
        def __init__(self, teacher: bool) -> None:
            super().__init__(); self.teacher = teacher; self.scale = torch.nn.Parameter(torch.ones(()))
            if teacher: self.requires_grad_(False)
            self.input_ids: list[int] = []; self.calls = 0
        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            self.calls += 1; self.input_ids.append(id(value)); base = value.mean((1, 2))[:, None] * self.scale
            dimension = 128 if self.teacher else 64
            raw = base.expand(len(value), dimension)
            return {"logits": base.expand(len(value), 98), "embedding_raw": raw,
                    "embedding_normalized": torch.nn.functional.normalize(raw, dim=1)}

    shared_results = {}
    for arm in module.ARMS:
        aug = CountingAugmentation(); candidate = RecordingModel(False); frozen = None if arm == "K0" else RecordingModel(True)
        result, target, augmented = module.shared_augmented_training_forward(
            arm, torch.randn(3, 2, 256), aug, torch.Generator().manual_seed(7), candidate, frozen, False
        )
        shared_results[arm] = (aug, candidate, frozen, result, target, augmented)
    check("A shared augmented input K1", shared_results["K1"][1].input_ids == shared_results["K1"][2].input_ids)
    check("A shared augmented input K2", shared_results["K2"][1].input_ids == shared_results["K2"][2].input_ids)
    check("A shared augmented input K3", shared_results["K3"][1].input_ids == shared_results["K3"][2].input_ids)
    check("A augmentation exactly once", all(row[0].calls == 1 for row in shared_results.values()))
    check("C K0 no teacher forward", shared_results["K0"][2] is None and shared_results["K0"][4] is None)
    check("D K1-K3 online teacher forward", all(shared_results[arm][2].calls == 1 for arm in ("K1", "K2", "K3")))
    check("teacher online logits detached", all(not shared_results[arm][4]["logits"].requires_grad for arm in ("K1", "K2", "K3")))
    check("teacher online embeddings detached", all(not shared_results[arm][4]["embedding_normalized"].requires_grad for arm in ("K1", "K2", "K3")))
    k1 = shared_results["K1"]
    shared_loss, _ = module.compute_kd_losses("K1", k1[3], k1[4]["logits"], k1[4]["embedding_normalized"], torch.tensor([0, 1, 2]), None, None)
    shared_loss.backward()
    check("teacher receives no backward update", all(parameter.grad is None for parameter in k1[2].parameters()))
    train_body = source[source.index("    def train_arm_seed"):source.index("    def train_arm_stage")]
    check("B clean cache rejected for sample KD", "teacher_logits[positions]" not in train_body and "teacher_embedding[positions]" not in train_body)
    policy = module.training_target_policy()
    check("training target policy online shared", policy["training_sample_kd_target_source"] == "ONLINE_TEACHER_FORWARD_ON_SHARED_AUGMENTED_INPUT")
    check("Train Known clean cache not sample KD", policy["train_known_clean_cache_used_for_sample_kd"] is False)
    check("K3 prototypes Train Known only", policy["teacher_prototype_source"] == "TRAIN_KNOWN_CLEAN_TEACHER_EMBEDDINGS" and policy["teacher_prototypes_used_by"] == ["K3"])
    check("P0-P3 clean cache remains evaluation-only", policy["p0_p3_clean_cache_role"] == "SEMANTIC_EVALUATION_ONLY")
    check("seed set exact", module.SEEDS == (42,123,2026))
    check("four arms exact", module.ARMS == ("K0","K1","K2","K3"))
    check("Train Known only canonical training", 'partitions["train_known"]' in source and 'build_loader("train_known"' in source)
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
    check("training checkpoint target-policy binding", '"training_target_policy_sha256"' in source)
    with tempfile.TemporaryDirectory() as directory:
        branch = Path(directory) / "branch"; output_root = branch / "06_surrogate_kd"; manifests = output_root / "manifests"
        manifests.mkdir(parents=True)
        files = {
            "STAGE4M_PREDECESSOR_LOCK.json": "predecessor\n",
            "STUDENT_ARCHITECTURE_FREEZE.json": "architecture\n",
            "KD_OBJECTIVE_POLICY.json": "objective\n",
            "TRAINING_TARGET_POLICY.json": "targets\n",
            "CANONICAL_SURROGATE_SELECTION_LOCK.json": "selection\n",
        }
        for name, value in files.items(): (manifests / name).write_text(value, encoding="utf-8")
        bound_input = output_root / "bound_input.bin"; bound_output = output_root / "bound_output.bin"
        bound_input.write_bytes(b"input"); bound_output.write_bytes(b"output")
        fixture = object.__new__(module.Stage4Pipeline)
        fixture.output = output_root; fixture.script_sha = "current-script"
        fixture.config = module.Stage4Config(branch_root=str(branch), output_dir=str(output_root))
        fixture.predecessor_lock_sha = fixture.architecture_freeze_sha = fixture.objective_policy_sha = None
        fixture.training_target_policy_sha = fixture.selection_lock_hash = None
        fixture.hydrate_provenance()

        def checkpoint_payload(stage: int) -> dict[str, Any]:
            return {
                "stage": stage, "stage_name": "fixture", "status": "PASS", "pipeline_version": module.PIPELINE_VERSION,
                "executable_sha256": fixture.script_sha, "configuration_sha256": fixture.config.configuration_sha256(),
                "teacher_sha256": module.EXPECTED_TEACHER_SHA256, "benchmark_sha256": module.EXPECTED_BENCHMARK_SHA256,
                "predecessor_lock_sha256": fixture.predecessor_lock_sha,
                "architecture_freeze_sha256": fixture.architecture_freeze_sha if stage >= 2 else None,
                "objective_policy_sha256": fixture.objective_policy_sha if stage >= 2 else None,
                "training_target_policy_sha256": fixture.training_target_policy_sha if stage >= 2 else None,
                "selection_lock_sha256": fixture.selection_lock_hash if stage >= 9 else None,
                "inputs": [{"path": str(bound_input), "sha256": sha256(bound_input), "bytes": bound_input.stat().st_size}],
                "outputs": [{"path": str(bound_output), "sha256": sha256(bound_output), "bytes": bound_output.stat().st_size}],
            }

        for prerequisite in range(1, 9):
            (manifests / f"STAGE_{prerequisite:02d}_CHECKPOINT.json").write_text(json.dumps(checkpoint_payload(prerequisite)), encoding="utf-8")
        check("hash-complete stage baseline current", fixture.stage_current(4))
        bound_input.write_bytes(b"mutated-input"); check("E input-hash mutation invalidation", not fixture.stage_current(4))
        check("E dependent-stage graph invalidation", not fixture.stage_current(8)); bound_input.write_bytes(b"input")
        (manifests / "STAGE4M_PREDECESSOR_LOCK.json").write_text("changed\n"); check("F predecessor-lock mutation invalidation", not fixture.stage_current(4)); (manifests / "STAGE4M_PREDECESSOR_LOCK.json").write_text(files["STAGE4M_PREDECESSOR_LOCK.json"])
        (manifests / "STUDENT_ARCHITECTURE_FREEZE.json").write_text("changed\n"); check("G architecture mutation invalidation", not fixture.stage_current(4)); (manifests / "STUDENT_ARCHITECTURE_FREEZE.json").write_text(files["STUDENT_ARCHITECTURE_FREEZE.json"])
        (manifests / "KD_OBJECTIVE_POLICY.json").write_text("changed\n"); check("H objective mutation invalidation", not fixture.stage_current(4)); (manifests / "KD_OBJECTIVE_POLICY.json").write_text(files["KD_OBJECTIVE_POLICY.json"])
        (manifests / "TRAINING_TARGET_POLICY.json").write_text("changed\n"); check("training-target-policy mutation invalidation", not fixture.stage_current(4)); (manifests / "TRAINING_TARGET_POLICY.json").write_text(files["TRAINING_TARGET_POLICY.json"])
        wrong_size = checkpoint_payload(4); wrong_size["outputs"][0]["bytes"] += 1
        (manifests / "STAGE_04_CHECKPOINT.json").write_text(json.dumps(wrong_size), encoding="utf-8")
        check("J recorded output byte-size mutation", not fixture.stage_current(4))
        (manifests / "STAGE_04_CHECKPOINT.json").write_text(json.dumps(checkpoint_payload(4)), encoding="utf-8")
        fixture.predecessor_lock_sha = fixture.architecture_freeze_sha = fixture.objective_policy_sha = None
        fixture.training_target_policy_sha = fixture.selection_lock_hash = None
        check("K fresh-process provenance hydration", fixture.stage_current(4) and all((fixture.predecessor_lock_sha, fixture.architecture_freeze_sha, fixture.objective_policy_sha, fixture.training_target_policy_sha, fixture.selection_lock_hash)))
        stage9 = checkpoint_payload(9); (manifests / "STAGE_09_CHECKPOINT.json").write_text(json.dumps(stage9), encoding="utf-8")
        check("selection-bound stage baseline current", fixture.stage_current(9))
        (manifests / "CANONICAL_SURROGATE_SELECTION_LOCK.json").write_text("changed\n")
        check("I selection-lock mutation invalidation", not fixture.stage_current(9))
    check("calibration blocked before lock", "Calibration diagnostic requires an immutable canonical selection lock" in source)
    check("calibration cannot mutate surrogate", "Calibration Unknown mutated the canonical surrogate" in source)
    check("READY fidelity gate", '"p0_fidelity_gate_pass"' in source)
    check("READY compression gate", '"compression_gate_pass"' in source)
    check("READY leakage gate", '"all_stage35_strict_violation_counters_zero"' in source)
    check("READY declares no strict eval", 'strict_zero_day_evaluation_performed=NO' in source)
    order = [source.index(fragment) for fragment in ("atomic_json(final", "manifest = self.create_final_hash_manifest()", "atomic_text(ready", "12, \"Final audit", "if not_ready.exists(): not_ready.unlink()")]
    check("Stage12 transaction order", order == sorted(order))
    check("Stage12 resume READY gate", "if stage == 12" in source and "final_hash_manifest_current" in source)
    check("idempotent finalization guard", "MANYTX_STAGE4M_ALREADY_READY" in source and "completed_state_current" in source)
    with tempfile.TemporaryDirectory() as directory:
        branch = Path(directory) / module.CANONICAL_BRANCH; completed = branch / "06_surrogate_kd"; manifests = completed / "manifests"
        manifests.mkdir(parents=True)
        for name in ("01_benchmark_engineering", "02_benchmark_diagnostics", "03_representation_ablation", "04_canonical_teacher", "05_zero_day_open_set"):
            (branch / name).mkdir(parents=True, exist_ok=True)
        (branch / "04_canonical_teacher" / "MANYTX_STAGE3M_READY.txt").write_text("x")
        (branch / "05_zero_day_open_set" / "MANYTX_STAGE3_5M_READY.txt").write_text("x")
        lock = {"status": "LOCKED", "canonical_surrogate_sha256": "deploy-sha", "canonical_surrogate_state_sha256": "state-sha"}
        lock_path = manifests / "CANONICAL_SURROGATE_SELECTION_LOCK.json"; lock_path.write_text(json.dumps(lock))
        provenance = {}
        for name in ("STAGE4M_PREDECESSOR_LOCK.json", "STUDENT_ARCHITECTURE_FREEZE.json", "KD_OBJECTIVE_POLICY.json", "TRAINING_TARGET_POLICY.json"):
            path = manifests / name; path.write_text(name); provenance[name] = path
        prior_checkpoints = []
        for stage in range(1, 12):
            row = {"path": str(lock_path), "sha256": sha256(lock_path), "bytes": lock_path.stat().st_size}
            path = manifests / f"STAGE_{stage:02d}_CHECKPOINT.json"
            path.write_text(json.dumps({"stage": stage, "status": "PASS", "inputs": [row], "outputs": [row]})); prior_checkpoints.append(path)
        final_path = manifests / "STAGE4M_FINAL_STATUS.json"
        final_path.write_text(json.dumps({"status": "MANYTX_STAGE4M_READY", "selected_arm": "K2", "selected_seed": 123}))
        ready_path = completed / "MANYTX_STAGE4M_READY.txt"
        ready_path.write_text("\n".join(("MANYTX_STAGE4M_READY", f"teacher_sha256={module.EXPECTED_TEACHER_SHA256}",
                                         f"benchmark_sha256={module.EXPECTED_BENCHMARK_SHA256}", "selected_kd_arm=K2", "selected_seed=123",
                                         "canonical_surrogate_sha256=deploy-sha", "canonical_surrogate_state_sha256=state-sha", "")))
        manifest_path = manifests / "STAGE4M_HASH_MANIFEST.json"
        manifest_rows = [{"relative_path": path.relative_to(completed).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
                         for path in (lock_path, final_path)]
        manifest_path.write_text(json.dumps({"algorithm": "SHA-256", "count": len(manifest_rows), "files": manifest_rows}))
        stage12_path = manifests / "STAGE_12_CHECKPOINT.json"
        stage12_path.write_text(json.dumps({
            "stage": 12, "status": "PASS", "pipeline_version": module.PIPELINE_VERSION,
            "teacher_sha256": module.EXPECTED_TEACHER_SHA256, "benchmark_sha256": module.EXPECTED_BENCHMARK_SHA256,
            "predecessor_lock_sha256": sha256(provenance["STAGE4M_PREDECESSOR_LOCK.json"]),
            "architecture_freeze_sha256": sha256(provenance["STUDENT_ARCHITECTURE_FREEZE.json"]),
            "objective_policy_sha256": sha256(provenance["KD_OBJECTIVE_POLICY.json"]),
            "training_target_policy_sha256": sha256(provenance["TRAINING_TARGET_POLICY.json"]),
            "selection_lock_sha256": sha256(lock_path),
            "inputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in (*prior_checkpoints, lock_path)],
            "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                        for path in (final_path, manifest_path, ready_path)],
        }))
        check("completed-state fixture current", module.completed_state_current(completed))
        before = {path.relative_to(completed).as_posix(): (sha256(path), path.stat().st_size) for path in completed.rglob("*") if path.is_file()}
        completed_config = module.Stage4Config(branch_root=str(branch), output_dir=str(completed))
        check("L valid READY refuses NOT_READY conversion", module.write_not_ready(completed_config, RuntimeError("later failure")) is False and ready_path.is_file() and not (completed / "MANYTX_STAGE4M_NOT_READY.txt").exists())
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture): second_code = module.main(["--branch-root", str(branch)])
        after = {path.relative_to(completed).as_posix(): (sha256(path), path.stat().st_size) for path in completed.rglob("*") if path.is_file()}
        check("M idempotent second invocation marker", second_code == 0 and "MANYTX_STAGE4M_ALREADY_READY" in capture.getvalue())
        check("M idempotent second invocation no writes", before == after)
        bad_config = Path(directory) / "bad.json"; bad_config.write_text('{"unknown_scientific_key": true}')
        with contextlib.redirect_stdout(io.StringIO()): bad_code = module.main(["--branch-root", str(branch), "--config", str(bad_config)])
        check("L unrelated config error preserves READY", bad_code == 1 and ready_path.is_file() and module.completed_state_current(completed))
        original_manifest = manifest_path.read_text(); manifest_path.write_text("{}")
        check("N stale final hash manifest rejected", not module.completed_state_current(completed))
        manifest_path.write_text(original_manifest); original_stage12 = stage12_path.read_text(); stage12_path.write_text("{}")
        check("N stale Stage12 checkpoint rejected", not module.completed_state_current(completed))
        stage12_path.write_text(original_stage12)
    with tempfile.TemporaryDirectory() as directory:
        branch = Path(directory) / "branch"; output_root = branch / "06_surrogate_kd"
        failed_config = module.Stage4Config(branch_root=str(branch), output_dir=str(output_root))
        check("READY matrix no READY creates NOT_READY", module.write_not_ready(failed_config, RuntimeError("failure")) is True and (output_root / "MANYTX_STAGE4M_NOT_READY.txt").is_file())
        partial_ready = output_root / "MANYTX_STAGE4M_READY.txt"; partial_ready.write_text("MANYTX_STAGE4M_READY\n")
        module.write_not_ready(failed_config, RuntimeError("partial"))
        check("READY matrix partial transaction recovered", not partial_ready.exists() and (output_root / "MANYTX_STAGE4M_NOT_READY.txt").is_file())
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
    check("O preflight cannot create training checkpoints", "save_training_checkpoint" not in preflight_body and "train_arm_stage" not in preflight_body)
    check("O preflight cannot freeze architecture/objectives", "STUDENT_ARCHITECTURE_FREEZE" not in preflight_body and "KD_OBJECTIVE_POLICY" not in preflight_body)
    check("O preflight cannot create selection artifacts", "CANONICAL_SURROGATE_SELECTION" not in preflight_body)
    check("preflight cannot access calibration", "student_outputs" not in preflight_body and '"calibration_unknown_accessed": False' in preflight_body)
    check("preflight cannot create READY", "atomic_text(ready" not in preflight_body and '"ready_created": False' in preflight_body)
    check("O preflight write probe deleted", "probe.unlink()" in preflight_body)
    check("O preflight strict counters zero", "self.guard.assert_zero()" in preflight_body)
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
