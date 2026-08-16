#!/usr/bin/env python3
"""Stage 4M v1.0.3 restart recovery: normalize RNG state back to CPU bytes.

This runtime-only layer extends the v1.0.2 scientific-current recovery. The
preserved Stage 4M checkpoints are loaded with ``map_location=self.device``;
on CUDA this also remaps the checkpoint's CPU RNG ByteTensor to CUDA. PyTorch
``torch.set_rng_state`` is CPU-only and therefore rejects that otherwise valid
state. This layer restores the exact saved RNG bytes after moving RNG-state
tensors back to CPU. No model, optimizer, scheduler, scaler, objective, data,
selection rule, metric, or strict-data guard is changed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PARENT_FILENAME = "Stage4M_Resume_Scientific_Current_Hotfix_v1_0_2.py"
EXPECTED_PARENT_GIT_BLOB_SHA1 = "b83b34aa0a7ef3b6989f49d8139727fca890c67c"
EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256 = "78a6437a153f3a764fd3255bd6625bb350e24643611434deca60bbed9a566a80"
RNG_AUDIT_FILENAME = "STAGE4M_RNG_RESTORE_COMPATIBILITY_v1_0_3.json"


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _load_parent() -> Any:
    path = Path(__file__).resolve().with_name(PARENT_FILENAME)
    spec = importlib.util.spec_from_file_location("stage4m_resume_parent_v102", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage 4M v1.0.2 recovery executable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if _git_blob_sha1(path) != EXPECTED_PARENT_GIT_BLOB_SHA1:
        raise RuntimeError("Stage 4M v1.0.2 recovery content mismatch; refuse v1.0.3")
    if module.base.Stage4Config().configuration_sha256() != EXPECTED_SCIENTIFIC_CONFIGURATION_SHA256:
        raise RuntimeError("Stage 4M scientific configuration changed; refuse v1.0.3")
    return module


parent = _load_parent()
base = parent.base


def _rng_byte_tensor(value: Any, label: str) -> Any:
    """Return an exact CPU uint8 tensor representation of saved RNG bytes."""
    if not base.torch.is_tensor(value):
        raise base.ScientificAbort(f"Saved {label} RNG state is not a torch.Tensor")
    if value.dtype != base.torch.uint8:
        raise base.ScientificAbort(f"Saved {label} RNG state dtype is not uint8: {value.dtype}")
    if value.numel() <= 0:
        raise base.ScientificAbort(f"Saved {label} RNG state is empty")
    return value.detach().to(device="cpu", dtype=base.torch.uint8).contiguous()


def restore_rng_cpu_safe(payload: Mapping[str, Any]) -> None:
    """Restore exact checkpoint RNG bytes independent of checkpoint map_location."""
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - set(payload))
    if missing:
        raise base.ScientificAbort(f"Saved RNG payload missing fields: {missing}")

    base.random.setstate(payload["python"])
    base.np.random.set_state(payload["numpy"])
    base.torch.set_rng_state(_rng_byte_tensor(payload["torch_cpu"], "CPU"))

    if base.torch.cuda.is_available():
        states = payload.get("torch_cuda")
        if not isinstance(states, (list, tuple)):
            raise base.ScientificAbort("Saved CUDA RNG state is missing or not a sequence")
        device_count = int(base.torch.cuda.device_count())
        if len(states) != device_count:
            raise base.ScientificAbort(
                f"Saved CUDA RNG device count mismatch: checkpoint={len(states)} runtime={device_count}"
            )
        normalized = [_rng_byte_tensor(state, f"CUDA[{index}]") for index, state in enumerate(states)]
        base.torch.cuda.set_rng_state_all(normalized)


# The validated base training function resolves restore_rng from this module's
# global namespace. Replace only that runtime restore function; checkpoint bytes
# and every scientific/training object remain unchanged.
base.restore_rng = restore_rng_cpu_safe


class Stage4Pipeline(parent.Stage4Pipeline):
    """v1.0.2 recovery plus exact RNG-device normalization on checkpoint resume."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.script_sha = base.sha256_file(Path(__file__).resolve())

    def validate_training_checkpoint(self, payload: Mapping[str, Any], arm: str, seed: int) -> None:
        super().validate_training_checkpoint(payload, arm, seed)
        rng = payload.get("rng_state")
        if not isinstance(rng, Mapping):
            raise base.ScientificAbort(f"Checkpoint RNG payload missing for {arm}/seed={seed}")
        cpu = rng.get("torch_cpu")
        _rng_byte_tensor(cpu, "CPU")
        cuda_states = rng.get("torch_cuda")
        if base.torch.cuda.is_available():
            if not isinstance(cuda_states, (list, tuple)):
                raise base.ScientificAbort(f"Checkpoint CUDA RNG payload missing for {arm}/seed={seed}")
            if len(cuda_states) != int(base.torch.cuda.device_count()):
                raise base.ScientificAbort(
                    f"Checkpoint CUDA RNG count mismatch for {arm}/seed={seed}: "
                    f"checkpoint={len(cuda_states)} runtime={base.torch.cuda.device_count()}"
                )
            for index, state in enumerate(cuda_states):
                _rng_byte_tensor(state, f"CUDA[{index}]")

        audit_path = self.output / "manifests" / RNG_AUDIT_FILENAME
        existing = []
        if audit_path.is_file():
            try:
                existing = list(json.loads(audit_path.read_text(encoding="utf-8")).get("validated_checkpoints", []))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                existing = []
        row = {
            "arm": arm,
            "seed": int(seed),
            "epoch": int(payload.get("epoch", -1)),
            "checkpoint_executable_sha256": payload.get("executable_sha256"),
            "checkpoint_cpu_rng_device_after_map_location": str(getattr(cpu, "device", None)),
            "checkpoint_cpu_rng_dtype": str(getattr(cpu, "dtype", None)),
            "checkpoint_cpu_rng_numel": int(cpu.numel()),
            "cuda_rng_state_count": len(cuda_states) if isinstance(cuda_states, (list, tuple)) else None,
        }
        key = (row["arm"], row["seed"], row["epoch"])
        existing = [item for item in existing if (item.get("arm"), int(item.get("seed", -1)), int(item.get("epoch", -1))) != key]
        existing.append(row)
        base.atomic_json(audit_path, {
            "status": "PASS",
            "compatibility_scope": "RNG_TENSOR_DEVICE_NORMALIZATION_ONLY",
            "scientific_changes": False,
            "checkpoint_mutated": False,
            "rng_bytes_mutated": False,
            "restore_policy": "EXACT_UINT8_BYTES_MOVED_TO_CPU_BEFORE_SET_RNG_STATE",
            "current_executable_sha256": self.script_sha,
            "scientific_configuration_sha256": self.config.configuration_sha256(),
            "strict_counters": self.guard.counters(),
            "validated_checkpoints": existing,
            "updated_at": base.utc_now(),
        }, self.output)


parent.base.Stage4Pipeline = Stage4Pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    return parent.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
