#!/usr/bin/env python3
"""Fail-closed structural rehearsal for the R9700 vLLM/HyperLoom rebase lane.

This tool does not patch vLLM, mutate the live runtime, or authorize removal of
S3. It verifies that the exact upstream revisions still expose the structural
surfaces needed to *adapt* the validated AutoAWQ -> Triton WNA16 overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "r9700-rebase-readiness-v1"
SNAPSHOT_SCHEMA = "r9700-rebase-readiness-snapshot-v1"
OVERLAY_ACTION = "retain"
MAX_TEXT_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0

EXPECTED_REFS = {
    "vllm_stable_tag": "v0.29.0",
    "vllm_stable_sha": "98dff2a81d747d1dba01a47f939f48c3526d4206",
    "vllm_main_sha": "00972dfd72988942138a7a6089eaee08580210b8",
    "vllm_pr_43389_head": "56ef89e1ff4a1552beb3b5c51c00b73ea44daca1",
    "hyperloom_main_sha": "0425bde3f6e76e1588400c37d056dfd3bb75ac11",
}

DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "r9700_rebase_readiness_20260914.json"
)

PR_URL = "https://api.github.com/repos/vllm-project/vllm/pulls/43389"
VLLM_RAW = "https://raw.githubusercontent.com/vllm-project/vllm/{ref}/{path}"
HYPERLOOM_RAW = "https://raw.githubusercontent.com/AMD-AGI/Hyperloom/{ref}/README.md"

ORACLE_PATH = "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"
MOE_WNA16_PATH = "vllm/model_executor/layers/quantization/moe_wna16.py"
FUSED_MOE_PATH = "vllm/model_executor/layers/fused_moe/fused_moe.py"
CONFIG_PATH = "vllm/model_executor/layers/fused_moe/config.py"
TRITON_EXPERT_PATH = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
UTIL_PATH = "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py"

AUTOAWQ_REJECTION = "the AutoAWQ weight layout is not supported"
R9700_SUPPORT_PATTERN = re.compile(
    r"(?:Radeon\s+AI\s+PRO\s+R9700|\bR9700\b|\bgfx1201\b)", re.IGNORECASE
)


class ReadinessError(ValueError):
    """Controlled validation failure without leaking arbitrary upstream text."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReadinessError(code)


def _request(url: str) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hyperloom-r9700-rebase-readiness/1",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers, method="GET")


def _fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        payload = json.loads(response.read())
    _require(isinstance(payload, dict), "live:json_not_object")
    return payload


def _fetch_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        raw = response.read(MAX_TEXT_BYTES + 1)
    _require(len(raw) <= MAX_TEXT_BYTES, "live:text_too_large")
    return raw.decode("utf-8")


def _fetch_optional_text(url: str, *, timeout: float) -> str | None:
    try:
        return _fetch_text(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _lane_snapshot(ref: str, *, timeout: float) -> dict[str, bool]:
    def get(path: str) -> str:
        return _fetch_text(VLLM_RAW.format(ref=ref, path=path), timeout=timeout)

    oracle = get(ORACLE_PATH)
    moe_wna16 = get(MOE_WNA16_PATH)
    fused_moe = get(FUSED_MOE_PATH)
    config = get(CONFIG_PATH)
    triton_expert = get(TRITON_EXPERT_PATH)
    utility = _fetch_optional_text(
        VLLM_RAW.format(ref=ref, path=UTIL_PATH), timeout=timeout
    )

    return {
        "autoawq_triton_rejection_present": AUTOAWQ_REJECTION in oracle,
        "triton_wna16_experts_present": "TritonWNA16Experts" in oracle,
        "select_wna16_moe_backend_present": "def select_wna16_moe_backend(" in oracle,
        "convert_to_wna16_moe_kernel_format_present": (
            "def convert_to_wna16_moe_kernel_format(" in oracle
        ),
        "triton_conversion_branch_present": (
            "elif backend == WNA16MoEBackend.TRITON:" in oracle
        ),
        "process_weights_after_loading_present": (
            "def process_weights_after_loading(" in moe_wna16
            and "convert_to_wna16_moe_kernel_format(" in moe_wna16
        ),
        "rdna3_backend_present": "WNA16MoEBackend.RDNA3" in oracle,
        "overlay_oracle_repack_present": "repack_int4_to_int32" in oracle,
        "overlay_fused_interleave_present": "use_int4_interleave" in fused_moe,
        "overlay_config_layout_marker_present": (
            "is_int4_w4a16_interleaved" in config
        ),
        "overlay_triton_layout_argument_present": (
            "int4_packed_as_int32" in triton_expert
        ),
        "overlay_utility_module_present": utility is not None,
    }


def collect_live_snapshot(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Read exact immutable upstream refs plus the current state of PR #43389."""
    pr = _fetch_json(PR_URL, timeout=timeout)
    state = pr.get("state")
    merged = pr.get("merged")
    head = pr.get("head") or {}
    head_sha = head.get("sha") if isinstance(head, dict) else None
    _require(state in {"open", "closed"}, "live:pr_state")
    _require(type(merged) is bool, "live:pr_merged")
    _require(isinstance(head_sha, str) and bool(head_sha), "live:pr_head")

    stable = _lane_snapshot(EXPECTED_REFS["vllm_stable_sha"], timeout=timeout)
    main = _lane_snapshot(EXPECTED_REFS["vllm_main_sha"], timeout=timeout)
    hyperloom_readme = _fetch_text(
        HYPERLOOM_RAW.format(ref=EXPECTED_REFS["hyperloom_main_sha"]),
        timeout=timeout,
    )

    return {
        "schema": SNAPSHOT_SCHEMA,
        "refs": dict(EXPECTED_REFS),
        "vllm": {
            "pr_43389": {"state": state, "merged": merged, "head_sha": head_sha},
            "stable": stable,
            "main": main,
        },
        "hyperloom": {
            "declares_r9700_or_gfx1201_support": bool(
                R9700_SUPPORT_PATTERN.search(hyperloom_readme)
            )
        },
    }


def _validate_lane(lane: Any, name: str) -> dict[str, bool]:
    _require(isinstance(lane, dict), f"snapshot:{name}")
    required = {
        "autoawq_triton_rejection_present",
        "triton_wna16_experts_present",
        "select_wna16_moe_backend_present",
        "convert_to_wna16_moe_kernel_format_present",
        "triton_conversion_branch_present",
        "process_weights_after_loading_present",
        "rdna3_backend_present",
        "overlay_oracle_repack_present",
        "overlay_fused_interleave_present",
        "overlay_config_layout_marker_present",
        "overlay_triton_layout_argument_present",
        "overlay_utility_module_present",
    }
    for key in required:
        _require(type(lane.get(key)) is bool, f"snapshot:{name}:{key}")
    return lane


def _validated_snapshot(snapshot: Any) -> dict[str, Any]:
    _require(isinstance(snapshot, dict), "snapshot:not_object")
    _require(snapshot.get("schema") == SNAPSHOT_SCHEMA, "snapshot:schema")
    refs = snapshot.get("refs")
    _require(isinstance(refs, dict), "snapshot:refs")
    for key in EXPECTED_REFS:
        _require(isinstance(refs.get(key), str), f"snapshot:refs:{key}")

    vllm = snapshot.get("vllm")
    _require(isinstance(vllm, dict), "snapshot:vllm")
    pr = vllm.get("pr_43389")
    _require(isinstance(pr, dict), "snapshot:pr_43389")
    _require(pr.get("state") in {"open", "closed"}, "snapshot:pr_state")
    _require(type(pr.get("merged")) is bool, "snapshot:pr_merged")
    _require(isinstance(pr.get("head_sha"), str), "snapshot:pr_head")
    _validate_lane(vllm.get("stable"), "stable")
    _validate_lane(vllm.get("main"), "main")

    hyperloom = snapshot.get("hyperloom")
    _require(isinstance(hyperloom, dict), "snapshot:hyperloom")
    _require(
        type(hyperloom.get("declares_r9700_or_gfx1201_support")) is bool,
        "snapshot:hyperloom_support",
    )
    return snapshot


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError("snapshot:invalid_json") from exc
    return _validated_snapshot(payload)


def _lane_backbone_ok(lane: dict[str, bool]) -> bool:
    return all(
        lane[key]
        for key in (
            "autoawq_triton_rejection_present",
            "triton_wna16_experts_present",
            "select_wna16_moe_backend_present",
            "convert_to_wna16_moe_kernel_format_present",
            "triton_conversion_branch_present",
            "process_weights_after_loading_present",
        )
    )


def _overlay_absent(lane: dict[str, bool]) -> bool:
    return not any(
        lane[key]
        for key in (
            "overlay_oracle_repack_present",
            "overlay_fused_interleave_present",
            "overlay_config_layout_marker_present",
            "overlay_triton_layout_argument_present",
            "overlay_utility_module_present",
        )
    )


def evaluate_snapshot(snapshot: Any, *, source: str = "offline") -> dict[str, Any]:
    """Classify whether an isolated port is structurally justified and safe to attempt."""
    try:
        snapshot = _validated_snapshot(snapshot)
    except ReadinessError as exc:
        return {
            "schema": SCHEMA,
            "review_required": True,
            "reasons": [str(exc)],
            "overlay_action": OVERLAY_ACTION,
            "automatic_overlay_removal_allowed": False,
            "runtime_mutation": False,
            "source": source,
        }

    reasons: list[str] = []
    for key, expected in EXPECTED_REFS.items():
        if snapshot["refs"][key] != expected:
            reasons.append(f"ref_changed:{key}")

    pr = snapshot["vllm"]["pr_43389"]
    if pr["state"] != "open":
        reasons.append("vllm_pr_43389_not_open")
    if pr["merged"] is not False:
        reasons.append("vllm_pr_43389_merged")
    if pr["head_sha"] != EXPECTED_REFS["vllm_pr_43389_head"]:
        reasons.append("vllm_pr_43389_head_changed")

    lanes = {
        "stable_v0_29_0": snapshot["vllm"]["stable"],
        "current_main": snapshot["vllm"]["main"],
    }
    hunk_status: dict[str, str] = {}
    for name, lane in lanes.items():
        if not _lane_backbone_ok(lane):
            reasons.append(f"{name}:wna16_backbone_changed")
        if not _overlay_absent(lane):
            reasons.append(f"{name}:overlay_like_code_present_upstream")
        hunk_status[name] = (
            "rebase_adapt"
            if _lane_backbone_ok(lane) and _overlay_absent(lane)
            else "manual_review"
        )

    if snapshot["hyperloom"]["declares_r9700_or_gfx1201_support"]:
        reasons.append("hyperloom_r9700_support_declared")

    return {
        "schema": SCHEMA,
        "review_required": bool(reasons),
        "reasons": reasons,
        "overlay_action": OVERLAY_ACTION,
        "automatic_overlay_removal_allowed": False,
        "runtime_mutation": False,
        "source": source,
        "decisions": {
            "s3_overlay": "retain",
            "vllm_v0_29_0": "rebase/adapt",
            "vllm_current_main": "rebase/adapt",
            "hyperloom_current_main": "rebase/adapt",
        },
        "hunk_status": hunk_status,
        "observed": {
            "pr_43389": pr,
            "stable_rdna3_backend_present": lanes["stable_v0_29_0"][
                "rdna3_backend_present"
            ],
            "main_rdna3_backend_present": lanes["current_main"][
                "rdna3_backend_present"
            ],
            "hyperloom_declares_r9700_or_gfx1201_support": snapshot["hyperloom"][
                "declares_r9700_or_gfx1201_support"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    source = "live" if args.live else "offline"
    try:
        snapshot = (
            collect_live_snapshot(timeout=args.timeout)
            if args.live
            else load_snapshot(args.snapshot)
        )
        result = evaluate_snapshot(snapshot, source=source)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        code = str(exc) if isinstance(exc, ReadinessError) else "rebase_readiness:unavailable"
        result = {
            "schema": SCHEMA,
            "review_required": True,
            "reasons": [code],
            "overlay_action": OVERLAY_ACTION,
            "automatic_overlay_removal_allowed": False,
            "runtime_mutation": False,
            "source": source,
        }

    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result["review_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
