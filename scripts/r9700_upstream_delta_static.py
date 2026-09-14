#!/usr/bin/env python3
"""Deterministic local guard for the R9700 upstream-delta decision.

This script performs no network access and never touches the live inference
service.  It verifies that the repository still contains the exact local
artifacts whose upstream status is documented in R9700_UPSTREAM_DELTA_20260913.

It is deliberately a source/invariant probe, not a claim that current upstream
has been executed on the production R9700 workload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

LOCAL_MAIN: Final = "b442bf11b24d715ef17b26c26deccc86fa59ba22"
HYPERLOOM_UPSTREAM: Final = "ec3b1cbe9da752398388ea3497a3b40a500e0387"
VLLM_UPSTREAM: Final = "a2685f2cdace04138d10719b9bb612a67bf20886"
PHASE3_VLLM_PR: Final = 43389
PHASE3_VLLM_COMMIT: Final = "f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3"

PATCH = ROOT / "patches" / "hyperloom-r9700-gfx1201.patch"
HYBRID = ROOT / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py"
BUILDER = ROOT / "scripts" / "r9700_phase3_build_vllm43389_overlay.py"


def contains(path: Path, *needles: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [needle for needle in needles if needle not in text]
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "missing": missing,
        "pass": path.exists() and not missing,
    }


def main() -> int:
    checks = {
        "hyperloom_identity_patch": contains(
            PATCH,
            '"r9700": {"arch": "gfx1201", "cu": 64}',
            '"gfx1201": "r9700"',
        ),
        "phase3_overlay_lineage": contains(
            BUILDER,
            f"UPSTREAM_COMMIT = \"{PHASE3_VLLM_COMMIT}\"",
            "upstream_pr\": 43389",
            "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
            "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py",
        ),
        "s3_guarded_hybrid": contains(
            HYBRID,
            "class R9700HybridWNA16Experts(TritonWNA16Experts):",
            'HYPERLOOM_R9700_W1_SMALL_TOKEN_LIMIT", "16"',
            'self.last_path = "stock_full_fallback"',
            'self.last_path = "custom_small_w1_stock_w2"',
            "return super().apply(",
            "and hidden_states.size(-1) == 2048",
            "and w1.size(1) == 1536",
            "and topk_ids.size(1) == 8",
        ),
    }

    passed = all(bool(check["pass"]) for check in checks.values())
    report = {
        "schema": "hyperloom.r9700.upstream_delta.static.v1",
        "pass": passed,
        "network_access": False,
        "live_runtime_mutation": False,
        "pinned_review_state": {
            "local_main_at_review": LOCAL_MAIN,
            "hyperloom_upstream": HYPERLOOM_UPSTREAM,
            "vllm_upstream": VLLM_UPSTREAM,
            "phase3_vllm_pr": PHASE3_VLLM_PR,
            "phase3_vllm_commit": PHASE3_VLLM_COMMIT,
        },
        "decisions": {
            "hyperloom_r9700_identity": "retain",
            "generic_rdna4_recognition": "rebase_to_upstream",
            "generic_autoawq_moe_wna16": "rebase_to_upstream",
            "phase3_pr43389_overlay": "retain_until_exact_candidate_validation",
            "s3_small_m_w1_hybrid": "retain",
            "phase6_control_plane": "retain",
        },
        "checks": checks,
        "truth_boundary": (
            "Static repository invariants only; this does not validate current "
            "upstream vLLM numerics, performance, or production readiness."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
