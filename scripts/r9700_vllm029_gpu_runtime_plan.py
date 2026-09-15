#!/usr/bin/env python3
"""Emit the fail-closed contract for the isolated vLLM 0.29 R9700 GPU gate.

This file is intentionally non-mutating. It records exactly what must be true
before the project may claim that the new ROCm 10 + vLLM 0.29.0 stack works on
the physical gfx1201 R9700. Runtime execution is a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

VLLM_TAG = "v0.29.0"
VLLM_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"
PATCH_REL = "docs/evidence/vllm_v0_29_0_r9700_awq_triton_candidate.patch"
PATCH_SHA256 = "372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_PATH = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
GPU_NAME = "AMD Radeon AI PRO R9700"
GPU_ARCH = "gfx1201"
ROCM_GENERATION = "10.0"
BUILD_PYTHON = "3.13"
BUILD_TORCH = "2.13.0"
BUILD_BASE_IMAGE = "rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.13.0"
BUILD_BASE_DIGEST = "sha256:c820e27bba8090875760d10b92e52aae790c776a937fa00c4357289dbc0addec"
BUILD_BASE_REF = f"{BUILD_BASE_IMAGE}@{BUILD_BASE_DIGEST}"
CANONICAL_TEXT_SHA256 = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"

PRODUCTION_PORT = 8000
S3_CANARY_PORT = 18018
PHASE3_PORT = 18011
DEFAULT_CANDIDATE_PORT = 18029
DEFAULT_CANDIDATE_CONTAINER = "hyperloom-r9700-vllm029-gfx1201-candidate"
DEFAULT_CANDIDATE_ROOT = "var/r9700_vllm029_gpu_candidate"

PRODUCTION_CONTAINER = "inneros-vllm-hyperloom-s3-production"
PRODUCTION_SERVICE = "inneros-vllm-hyperloom-s3-production.service"
STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"

FORBIDDEN_PORTS = {PRODUCTION_PORT, S3_CANARY_PORT, PHASE3_PORT}
FORBIDDEN_CONTAINERS = {
    PRODUCTION_CONTAINER,
    "inneros-vllm-hyperloom-s3-canary",
    "hyperloom-r9700-p3-int4-repack-p18011",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(
    *,
    candidate_port: int = DEFAULT_CANDIDATE_PORT,
    candidate_container: str = DEFAULT_CANDIDATE_CONTAINER,
    candidate_root: str = DEFAULT_CANDIDATE_ROOT,
    verify_local_patch: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    root = Path(candidate_root)
    patch_path = ROOT / PATCH_REL

    if not 1 <= candidate_port <= 65535:
        errors.append("candidate_port_out_of_range")
    if candidate_port in FORBIDDEN_PORTS:
        errors.append("candidate_port_collides_with_preserved_runtime")
    if not candidate_container.strip():
        errors.append("candidate_container_empty")
    if candidate_container in FORBIDDEN_CONTAINERS:
        errors.append("candidate_container_collides_with_preserved_runtime")
    if str(root) in {"", ".", "var/r9700_phase5_canary_bundle"}:
        errors.append("candidate_root_not_isolated")

    observed_patch_sha: str | None = None
    if verify_local_patch:
        if not patch_path.is_file():
            errors.append("candidate_patch_missing")
        else:
            observed_patch_sha = sha256_file(patch_path)
            if observed_patch_sha != PATCH_SHA256:
                errors.append("candidate_patch_sha256_mismatch")

    stages: list[dict[str, Any]] = [
        {
            "id": "source_identity",
            "gate": "pin_exact_vllm_and_patch",
            "requirements": {
                "vllm_tag": VLLM_TAG,
                "vllm_sha": VLLM_SHA,
                "patch_rel": PATCH_REL,
                "patch_sha256": PATCH_SHA256,
                "source_apply_prerequisite": "R9700 vLLM Source Apply PASS",
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "build",
            "gate": "isolated_source_build_against_rocm10",
            "requirements": {
                "rocm_generation": ROCM_GENERATION,
                "python": BUILD_PYTHON,
                "torch": BUILD_TORCH,
                "build_base_image": BUILD_BASE_IMAGE,
                "build_base_digest": BUILD_BASE_DIGEST,
                "build_base_ref": BUILD_BASE_REF,
                "pytorch_rocm_arch": GPU_ARCH,
                "strategy": "isolated_source_build",
                "prebuilt_vllm029_wheel": "forbidden_unless_rocm10_abi_is_explicitly_proved",
                "reason": (
                    "vLLM v0.29.0 build metadata pins torch 2.13.0; the ROCm 10 PyTorch "
                    "2.13.0 image provides a coherent source-build base for Ubuntu 24.04."
                ),
                "candidate_root": str(root),
                "production_root_mutation": False,
            },
            "on_failure": "reject_candidate_keep_s3_and_stock_unchanged",
        },
        {
            "id": "gpu_preflight",
            "gate": "prove_physical_target_identity",
            "requirements": {
                "gpu_name": GPU_NAME,
                "gpu_arch": GPU_ARCH,
                "rocm_generation": ROCM_GENERATION,
                "devices": ["/dev/kfd", "/dev/dri"],
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "isolation",
            "gate": "prove_no_preserved_runtime_collision",
            "requirements": {
                "candidate_port": candidate_port,
                "forbidden_ports": sorted(FORBIDDEN_PORTS),
                "candidate_container": candidate_container,
                "forbidden_containers": sorted(FORBIDDEN_CONTAINERS),
                "forbidden_services": [PRODUCTION_SERVICE, STOCK_SERVICE],
                "service_restart": False,
                "boot_policy_change": False,
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "import_abi",
            "gate": "prove_built_runtime_identity",
            "requirements": {
                "python": BUILD_PYTHON,
                "torch": BUILD_TORCH,
                "vllm_version": "0.29.0",
                "vllm_source_sha": VLLM_SHA,
                "torch_hip_present": True,
                "triton_importable": True,
                "current_platform_is_rocm": True,
                "current_platform_arch": GPU_ARCH,
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "load",
            "gate": "load_exact_awq_model_on_r9700",
            "requirements": {
                "model": MODEL,
                "model_path": MODEL_PATH,
                "dtype": "float16",
                "max_model_len": 8192,
                "GPU_MAX_HW_QUEUES": "1",
                "attention_backend": "ROCM_ATTN",
                "aiter": False,
            },
            "on_failure": "reject_candidate_keep_s3_and_stock_unchanged",
        },
        {
            "id": "path_evidence",
            "gate": "prove_interleaved_wna16_path_not_fallback",
            "requirements": {
                "int4_repacked_dtype": "torch.int32",
                "interleave_marker": "tl.interleave",
                "classic_wna16_path_preserved": True,
                "fallback_must_not_be_reported_as_optimized": True,
                "reuse_probe": "scripts/r9700_awq_backend_probe.py",
            },
            "on_failure": "candidate_inconclusive_no_performance_claim",
        },
        {
            "id": "warmup_correctness",
            "gate": "canonical_deterministic_warmup_matches",
            "requirements": {
                "reuse_warmup": "scripts/r9700_readiness_warmup.py",
                "canonical_text_sha256": CANONICAL_TEXT_SHA256,
                "temperature": 0.0,
                "seed": 7,
                "exact_hash_match_required": True,
            },
            "on_failure": "reject_candidate_no_performance_claim",
        },
        {
            "id": "bounded_performance",
            "gate": "measure_only_after_correctness_and_path_proof",
            "requirements": {
                "compare_against": ["controlled_stock", "validated_s3"],
                "metrics": ["decode_tok_s", "ttft_sec"],
                "no_universal_speedup_claim": True,
                "no_phase5_replay": True,
                "no_phase6_replay": True,
            },
            "on_failure": "keep_existing_runtime_policy_unchanged",
        },
        {
            "id": "promotion",
            "gate": "separate_manual_decision",
            "requirements": {
                "automatic_promotion": False,
                "production_restart": False,
                "boot_policy_change": False,
                "stock_fallback_remains_default": True,
                "s3_remains_available": True,
            },
            "on_failure": "keep_existing_runtime_policy_unchanged",
        },
    ]

    return {
        "schema": "hyperloom.r9700.vllm029.gpu_runtime_plan.v1",
        "pass": not errors,
        "errors": errors,
        "truth_boundary": (
            "Planning/gating artifact only. A PASS here does not mean the GPU runtime works; "
            "the physical gfx1201 load, path proof and canonical correctness gates must run."
        ),
        "candidate": {
            "vllm_tag": VLLM_TAG,
            "vllm_sha": VLLM_SHA,
            "patch_sha256": PATCH_SHA256,
            "observed_patch_sha256": observed_patch_sha,
            "model": MODEL,
            "gpu": GPU_NAME,
            "arch": GPU_ARCH,
            "rocm_generation": ROCM_GENERATION,
            "python": BUILD_PYTHON,
            "torch": BUILD_TORCH,
            "build_base_image": BUILD_BASE_IMAGE,
            "build_base_digest": BUILD_BASE_DIGEST,
            "build_base_ref": BUILD_BASE_REF,
            "port": candidate_port,
            "container": candidate_container,
            "root": str(root),
        },
        "preserved_runtime": {
            "production_port": PRODUCTION_PORT,
            "s3_canary_port": S3_CANARY_PORT,
            "stock_service": STOCK_SERVICE,
            "production_service": PRODUCTION_SERVICE,
            "must_not_be_mutated": True,
        },
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_CANDIDATE_PORT)
    parser.add_argument("--candidate-container", default=DEFAULT_CANDIDATE_CONTAINER)
    parser.add_argument("--candidate-root", default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--skip-local-patch-check", action="store_true")
    args = parser.parse_args()

    plan = build_plan(
        candidate_port=args.candidate_port,
        candidate_container=args.candidate_container,
        candidate_root=args.candidate_root,
        verify_local_patch=not args.skip_local_patch_check,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
