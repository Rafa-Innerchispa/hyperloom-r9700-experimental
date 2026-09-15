#!/usr/bin/env python3
"""Static, fail-closed readiness check for the isolated R9700 vLLM rebase lane.

This tool does not touch the live runtime. It only evaluates captured upstream
source text against the API surface required to port the validated S3 behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "r9700-rebase-readiness-v1"
REQUIRED_AUTOAWQ_MARKERS = (
    "class AutoAWQMoEMethod",
    "select_wna16_moe_backend(",
    "convert_to_wna16_moe_kernel_format(",
)
REQUIRED_ORACLE_MARKERS = (
    "class WNA16MoEBackend",
    "TRITON = \"TRITON\"",
    "TritonWNA16Experts",
    "select_wna16_moe_backend(",
)
AUTOAWQ_TRITON_REJECTION = "the AutoAWQ weight layout is not supported"
HYPERLOOM_SUPPORTED_ACCELERATORS = ("MI300X", "MI325X", "MI355X")


def assess_sources(*, autoawq: str, oracle: str, hyperloom_readme: str) -> dict[str, Any]:
    autoawq_api_ok = all(marker in autoawq for marker in REQUIRED_AUTOAWQ_MARKERS)
    oracle_api_ok = all(marker in oracle for marker in REQUIRED_ORACLE_MARKERS)
    triton_autoawq_blocked = AUTOAWQ_TRITON_REJECTION in oracle
    hyperloom_declares_r9700 = "R9700" in hyperloom_readme or "gfx1201" in hyperloom_readme
    hyperloom_known_platforms_present = all(
        marker in hyperloom_readme for marker in HYPERLOOM_SUPPORTED_ACCELERATORS
    )

    reasons: list[str] = []
    if not autoawq_api_ok:
        reasons.append("autoawq_api_surface_changed")
    if not oracle_api_ok:
        reasons.append("wna16_oracle_api_surface_changed")
    if not hyperloom_known_platforms_present:
        reasons.append("hyperloom_supported_platform_table_changed")

    # This is the exact gap that keeps the validated S3 overlay relevant.
    overlay_equivalent_upstream = autoawq_api_ok and oracle_api_ok and not triton_autoawq_blocked
    if triton_autoawq_blocked:
        reasons.append("autoawq_triton_still_blocked")

    return {
        "schema": SCHEMA,
        "review_required": bool(reasons),
        "runtime_mutation": False,
        "overlay_equivalent_upstream": overlay_equivalent_upstream,
        "decision": "rebase-adapt" if autoawq_api_ok and oracle_api_ok else "retain",
        "observed": {
            "autoawq_api_surface_present": autoawq_api_ok,
            "wna16_oracle_api_surface_present": oracle_api_ok,
            "autoawq_triton_blocked": triton_autoawq_blocked,
            "hyperloom_declares_r9700_or_gfx1201": hyperloom_declares_r9700,
            "hyperloom_known_platforms_present": hyperloom_known_platforms_present,
        },
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoawq", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--hyperloom-readme", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = assess_sources(
            autoawq=args.autoawq.read_text(encoding="utf-8"),
            oracle=args.oracle.read_text(encoding="utf-8"),
            hyperloom_readme=args.hyperloom_readme.read_text(encoding="utf-8"),
        )
    except OSError:
        result = {
            "schema": SCHEMA,
            "review_required": True,
            "runtime_mutation": False,
            "overlay_equivalent_upstream": False,
            "decision": "retain",
            "observed": {},
            "reasons": ["source_unavailable"],
        }

    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result["review_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
