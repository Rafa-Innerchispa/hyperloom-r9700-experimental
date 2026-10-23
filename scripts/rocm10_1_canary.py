#!/usr/bin/env python3
"""Plan isolated ROCm 10.1 canaries without replacing the stock R9700 runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

PYTORCH_IMAGE = "rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
VLLM_IMAGE = "rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0"


@dataclass(frozen=True)
class CanaryPlan:
    profile: str
    image: str
    container_name: str
    purpose: str
    gpu_mode: str
    stock_runtime_mutated: bool
    notes: list[str]


def build_plan(profile: str) -> CanaryPlan:
    if profile == "pytorch":
        return CanaryPlan(
            profile=profile,
            image=PYTORCH_IMAGE,
            container_name="hyperloom-rocm101-pytorch-canary",
            purpose="ROCm 10.1 / PyTorch 2.14 compatibility and framework smoke",
            gpu_mode="shared-smoke-only",
            stock_runtime_mutated=False,
            notes=[
                "May coexist with the stock service for lightweight stack checks.",
                "Do not load the 30B benchmark model while stock vLLM occupies the GPU.",
            ],
        )
    if profile == "vllm":
        return CanaryPlan(
            profile=profile,
            image=VLLM_IMAGE,
            container_name="hyperloom-rocm101-vllm029-canary",
            purpose="ROCm 10.1 / PyTorch 2.13 / vLLM 0.29 benchmark candidate",
            gpu_mode="exclusive-benchmark-window",
            stock_runtime_mutated=False,
            notes=[
                "AMD documents vLLM 0.29 with PyTorch 2.13 on ROCm 10.1.",
                "Use an exclusive GPU benchmark window for the same 30B model.",
                "Restore/verify the stock service immediately after each benchmark window.",
            ],
        )
    raise ValueError(f"unknown profile: {profile}")


def docker_smoke_command(plan: CanaryPlan) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        plan.container_name,
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add=video",
        "--ipc=host",
        "--security-opt",
        "seccomp=unconfined",
        plan.image,
        "python3",
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({'torch': torch.__version__, "
            "'hip': getattr(torch.version, 'hip', None), "
            "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
            "'available': torch.cuda.is_available()}))"
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["pytorch", "vllm"])
    parser.add_argument("--format", choices=["json", "command"], default="json")
    args = parser.parse_args()

    plan = build_plan(args.profile)
    if args.format == "command":
        print(" ".join(docker_smoke_command(plan)))
    else:
        print(json.dumps({**asdict(plan), "smoke_command": docker_smoke_command(plan)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
