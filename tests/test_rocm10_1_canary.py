from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "rocm10_1_canary.py"
SPEC = importlib.util.spec_from_file_location("rocm10_1_canary", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_pytorch_canary_is_isolated() -> None:
    plan = MOD.build_plan("pytorch")
    assert "rocm10.1.0" in plan.image
    assert "pytorch_release_2.14.0" in plan.image
    assert plan.stock_runtime_mutated is False
    assert plan.gpu_mode == "shared-smoke-only"


def test_vllm_canary_uses_amd_supported_pairing() -> None:
    plan = MOD.build_plan("vllm")
    assert "vllm-0.29.0" in plan.image
    assert "pytorch_2.13.0" in plan.image
    assert plan.gpu_mode == "exclusive-benchmark-window"
    command = MOD.docker_smoke_command(plan)
    assert "--device=/dev/kfd" in command
    assert "--device=/dev/dri" in command
