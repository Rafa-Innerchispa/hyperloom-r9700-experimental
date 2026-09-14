#!/usr/bin/env python3
"""Emit a fail-closed plan for an isolated current-vLLM R9700 candidate.

This tool is intentionally non-executing. It does not call Docker, systemd,
ROCm, or the network. Its job is to make the next experiment reproducible and
to reject configurations that collide with the proven Phase 6 production path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VLLM_UPSTREAM_COMMIT = "a2685f2cdace04138d10719b9bb612a67bf20886"
HYPERLOOM_UPSTREAM_COMMIT = "ec3b1cbe9da752398388ea3497a3b40a500e0387"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
GPU_ARCH = "gfx1201"
GPU_NAME = "AMD Radeon AI PRO R9700"

PRODUCTION_PORT = 8000
PRODUCTION_CONTAINER = "inneros-vllm-hyperloom-s3-production"
PRODUCTION_SERVICE = "inneros-vllm-hyperloom-s3-production.service"
STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"

DEFAULT_CANDIDATE_PORT = 8011
DEFAULT_CANDIDATE_CONTAINER = "inneros-vllm-r9700-current-upstream-candidate"
DEFAULT_CANDIDATE_ROOT = "var/r9700_current_upstream_candidate"


def build_plan(candidate_port: int, candidate_container: str, candidate_root: str) -> dict[str, object]:
    errors: list[str] = []
    root = Path(candidate_root)

    if candidate_port == PRODUCTION_PORT:
        errors.append("candidate_port_collides_with_production")
    if not 1 <= candidate_port <= 65535:
        errors.append("candidate_port_out_of_range")
    if candidate_container == PRODUCTION_CONTAINER:
        errors.append("candidate_container_collides_with_production")
    if not candidate_container.strip():
        errors.append("candidate_container_empty")
    if str(root) in {".", "", "var/r9700_phase5_canary_bundle"}:
        errors.append("candidate_root_not_isolated")

    stages = [
        {
            "id": "source",
            "gate": "pin_exact_sources",
            "requirements": {
                "vllm_commit": VLLM_UPSTREAM_COMMIT,
                "hyperloom_commit_for_delta_reference": HYPERLOOM_UPSTREAM_COMMIT,
                "phase3_pr43389_overlay": "absent_for_native_candidate",
                "s3_small_m_w1_patch": "absent_for_native_candidate",
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "isolation",
            "gate": "prove_no_production_collision",
            "requirements": {
                "candidate_port": candidate_port,
                "production_port_must_remain": PRODUCTION_PORT,
                "candidate_container": candidate_container,
                "forbidden_container": PRODUCTION_CONTAINER,
                "forbidden_services": [PRODUCTION_SERVICE, STOCK_SERVICE],
                "candidate_root": str(root),
            },
            "on_failure": "reject_candidate",
        },
        {
            "id": "load",
            "gate": "exact_model_loads_on_r9700",
            "requirements": {
                "model": MODEL,
                "gpu": GPU_NAME,
                "arch": GPU_ARCH,
                "backend_policy": "upstream_native_first",
            },
            "on_failure": "reject_candidate_keep_s3_unchanged",
        },
        {
            "id": "correctness",
            "gate": "canonical_hashes_match_stock",
            "requirements": {
                "reuse_existing_canonical_requests": True,
                "exact_hash_match_required": True,
                "relax_correctness": False,
            },
            "on_failure": "reject_candidate_keep_s3_unchanged",
        },
        {
            "id": "path_evidence",
            "gate": "prove_backend_path",
            "requirements": {
                "capture_backend_path": True,
                "fallback_must_not_be_misreported_as_optimized": True,
            },
            "on_failure": "candidate_is_inconclusive_no_performance_claim",
        },
        {
            "id": "performance",
            "gate": "bounded_comparison_after_correctness",
            "requirements": {
                "allowed_only_after": ["correctness", "path_evidence"],
                "compare_against": ["stock", "production_s3"],
                "no_phase5_replay": True,
                "no_phase6_replay": True,
            },
            "on_failure": "keep_s3_unchanged",
        },
        {
            "id": "promotion",
            "gate": "separate_explicit_phase6_control_decision",
            "requirements": {
                "automatic_promotion": False,
                "boot_policy_change": False,
                "fault_injection": False,
                "production_restart": False,
            },
            "on_failure": "keep_s3_unchanged",
        },
    ]

    return {
        "schema": "hyperloom.r9700.current_upstream_candidate.plan.v1",
        "pass": not errors,
        "errors": errors,
        "truth_boundary": (
            "Planning artifact only. No runtime, Docker, systemd, ROCm, network, "
            "benchmark, or production mutation is performed."
        ),
        "candidate": {
            "vllm_commit": VLLM_UPSTREAM_COMMIT,
            "model": MODEL,
            "gpu": GPU_NAME,
            "arch": GPU_ARCH,
            "port": candidate_port,
            "container": candidate_container,
            "root": str(root),
        },
        "production_immutability": {
            "port": PRODUCTION_PORT,
            "container": PRODUCTION_CONTAINER,
            "services": [PRODUCTION_SERVICE, STOCK_SERVICE],
            "must_not_be_mutated": True,
        },
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a non-executing isolated R9700 upstream-candidate plan.")
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_CANDIDATE_PORT)
    parser.add_argument("--candidate-container", default=DEFAULT_CANDIDATE_CONTAINER)
    parser.add_argument("--candidate-root", default=DEFAULT_CANDIDATE_ROOT)
    args = parser.parse_args()

    plan = build_plan(args.candidate_port, args.candidate_container, args.candidate_root)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
