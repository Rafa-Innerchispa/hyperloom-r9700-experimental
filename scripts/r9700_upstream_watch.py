#!/usr/bin/env python3
"""Read-only, fail-closed upstream drift sentinel for the R9700 S3 overlay.

The sentinel distinguishes actionable architectural drift from routine cosmetic
upstream bumps:
- Actionable drift: PR merged upstream, AutoAWQ Triton rejection removed, or
  HyperLoom upstream officially declaring R9700/gfx1201 support.
- Routine / cosmetic drift: new upstream patch releases or PR state transitions
  that confirm the overlay must be retained.

The sentinel never authorizes automatic overlay removal. An actionable change
requests manual revalidation against the exact validated workload in an isolated
canary. Offline snapshot mode is deterministic; ``--live`` performs read-only
HTTP GETs against public endpoints.
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

RESULT_SCHEMA = "r9700-upstream-watch-v2"
SNAPSHOT_SCHEMA = "r9700-upstream-watch-snapshot-v1"
OVERLAY_ACTION = "retain_until_manual_revalidation"
DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1] / "docs" / "evidence" / "r9700_upstream_watch_baseline_20261018.json"
)
LEGACY_BASELINE = (
    Path(__file__).resolve().parents[1] / "docs" / "evidence" / "r9700_upstream_watch_baseline_20260914.json"
)

PR_URL = "https://api.github.com/repos/vllm-project/vllm/pulls/43389"
RELEASE_URL = "https://api.github.com/repos/vllm-project/vllm/releases/latest"
WNA16_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"
)
HYPERLOOM_README_URL = "https://raw.githubusercontent.com/AMD-AGI/Hyperloom/main/README.md"
AUTOAWQ_TRITON_REJECTION = "the AutoAWQ weight layout is not supported"
R9700_SUPPORT_PATTERN = re.compile(r"(?:Radeon\s+AI\s+PRO\s+R9700|\bR9700\b|\bgfx1201\b)", re.IGNORECASE)
MAX_SNAPSHOT_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 15.0

MEANINGFUL_REASONS = {
    "autoawq_triton_rejection_disappeared",
    "hyperloom_r9700_support_declared",
    "vllm_pr_43389_merged",
}

INFORMATIONAL_REASONS = {
    "vllm_latest_release_changed",
    "vllm_pr_43389_state_changed",
    "vllm_pr_43389_merge_flag_changed",
}


class SnapshotError(ValueError):
    """Controlled validation failure without embedding untrusted input."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SnapshotError(code)


def _fail_closed(reason: str, *, source: str) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "review_required": True,
        "actionable_revalidation_required": True,
        "drift_detected": True,
        "reasons": [reason],
        "meaningful_reasons": [reason],
        "informational_reasons": [],
        "automatic_overlay_removal_allowed": False,
        "overlay_action": OVERLAY_ACTION,
        "source": source,
    }


def _validated_snapshot(snapshot: Any) -> dict[str, Any]:
    _require(isinstance(snapshot, dict), "snapshot:not_object")
    _require(snapshot.get("schema") == SNAPSHOT_SCHEMA, "snapshot:schema")

    vllm = snapshot.get("vllm")
    hyperloom = snapshot.get("hyperloom")
    _require(isinstance(vllm, dict), "snapshot:vllm")
    _require(isinstance(hyperloom, dict), "snapshot:hyperloom")

    pr = vllm.get("pr_43389")
    _require(isinstance(pr, dict), "snapshot:pr_43389")
    _require(pr.get("state") in {"open", "closed"}, "snapshot:pr_state")
    _require(type(pr.get("merged")) is bool, "snapshot:pr_merged")

    release = vllm.get("latest_release")
    _require(isinstance(release, str) and bool(release), "snapshot:latest_release")
    marker = vllm.get("autoawq_triton_rejection_present")
    _require(type(marker) is bool, "snapshot:autoawq_marker")

    support = hyperloom.get("declares_r9700_or_gfx1201_support")
    _require(type(support) is bool, "snapshot:hyperloom_support")

    return snapshot


def load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
    _require(len(raw) <= MAX_SNAPSHOT_BYTES, "snapshot:too_large")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SnapshotError("snapshot:invalid_json") from exc
    return _validated_snapshot(payload)


def evaluate_snapshot(
    current: Any,
    baseline: Any,
    *,
    source: str = "offline",
    strict: bool = False,
) -> dict[str, Any]:
    """Compare current facts to the pinned baseline and categorize meaningful vs cosmetic drift."""
    try:
        current = _validated_snapshot(current)
        baseline = _validated_snapshot(baseline)
    except SnapshotError as exc:
        return _fail_closed(str(exc), source=source)

    reasons: list[str] = []
    current_pr = current["vllm"]["pr_43389"]
    baseline_pr = baseline["vllm"]["pr_43389"]

    if current_pr["merged"] is True and baseline_pr["merged"] is False:
        reasons.append("vllm_pr_43389_merged")
    elif current_pr["state"] != baseline_pr["state"]:
        reasons.append("vllm_pr_43389_state_changed")
    elif current_pr["merged"] != baseline_pr["merged"]:
        reasons.append("vllm_pr_43389_merge_flag_changed")

    if current["vllm"]["latest_release"] != baseline["vllm"]["latest_release"]:
        reasons.append("vllm_latest_release_changed")

    if (
        baseline["vllm"]["autoawq_triton_rejection_present"] is True
        and current["vllm"]["autoawq_triton_rejection_present"] is False
    ):
        reasons.append("autoawq_triton_rejection_disappeared")

    if (
        baseline["hyperloom"]["declares_r9700_or_gfx1201_support"] is False
        and current["hyperloom"]["declares_r9700_or_gfx1201_support"] is True
    ):
        reasons.append("hyperloom_r9700_support_declared")

    meaningful = [r for r in reasons if r in MEANINGFUL_REASONS]
    informational = [r for r in reasons if r in INFORMATIONAL_REASONS]

    actionable_revalidation = bool(meaningful)
    # If strict mode is requested, any drift triggers review_required; otherwise only actionable drift does.
    review_required = bool(reasons) if strict else actionable_revalidation

    recommended_action = (
        "revalidate_on_isolated_canary"
        if actionable_revalidation
        else "retain_overlay_no_action_needed"
        if informational
        else "in_sync"
    )

    return {
        "schema": RESULT_SCHEMA,
        "drift_detected": bool(reasons),
        "review_required": review_required,
        "actionable_revalidation_required": actionable_revalidation,
        "reasons": reasons,
        "meaningful_reasons": meaningful,
        "informational_reasons": informational,
        "recommended_action": recommended_action,
        "automatic_overlay_removal_allowed": False,
        "overlay_action": OVERLAY_ACTION,
        "source": source,
        "observed": {
            "vllm_pr_43389_state": current_pr["state"],
            "vllm_pr_43389_merged": current_pr["merged"],
            "vllm_latest_release": current["vllm"]["latest_release"],
            "autoawq_triton_rejection_present": current["vllm"]["autoawq_triton_rejection_present"],
            "hyperloom_declares_r9700_or_gfx1201_support": current["hyperloom"]["declares_r9700_or_gfx1201_support"],
        },
    }


def _request(url: str, *, timeout: float) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hyperloom-r9700-upstream-watch/2",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers, method="GET")


def _fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url, timeout=timeout), timeout=timeout) as response:
        payload = json.loads(response.read())
    _require(isinstance(payload, dict), "live:json_not_object")
    return payload


def _fetch_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(_request(url, timeout=timeout), timeout=timeout) as response:
        raw = response.read(MAX_SNAPSHOT_BYTES + 1)
    _require(len(raw) <= MAX_SNAPSHOT_BYTES, "live:text_too_large")
    return raw.decode("utf-8")


def collect_live_snapshot(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Collect only public, read-only upstream facts needed by the sentinel."""
    pr = _fetch_json(PR_URL, timeout=timeout)
    release = _fetch_json(RELEASE_URL, timeout=timeout)
    wna16 = _fetch_text(WNA16_URL, timeout=timeout)
    hyperloom_readme = _fetch_text(HYPERLOOM_README_URL, timeout=timeout)

    state = pr.get("state")
    merged = pr.get("merged")
    tag_name = release.get("tag_name")
    _require(state in {"open", "closed"}, "live:pr_state")
    _require(type(merged) is bool, "live:pr_merged")
    _require(isinstance(tag_name, str) and bool(tag_name), "live:latest_release")

    return _validated_snapshot(
        {
            "schema": SNAPSHOT_SCHEMA,
            "captured_at": "live",
            "vllm": {
                "pr_43389": {"state": state, "merged": merged},
                "latest_release": tag_name,
                "autoawq_triton_rejection_present": (AUTOAWQ_TRITON_REJECTION in wna16),
            },
            "hyperloom": {"declares_r9700_or_gfx1201_support": bool(R9700_SUPPORT_PATTERN.search(hyperloom_readme))},
        }
    )


def resolve_baseline(path: Path | None) -> Path:
    if path and path.exists():
        return path
    if DEFAULT_BASELINE.exists():
        return DEFAULT_BASELINE
    return LEGACY_BASELINE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Pinned baseline snapshot (default: repository baseline)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--snapshot", type=Path, help="Offline current snapshot to compare")
    group.add_argument("--live", action="store_true", help="Collect current facts with GET only")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--strict", action="store_true", help="Fail closed on any drift including cosmetic")
    args = parser.parse_args()

    baseline_path = resolve_baseline(args.baseline)
    source = "live" if args.live else "offline"
    try:
        baseline = load_snapshot(baseline_path)
        current = (
            collect_live_snapshot(timeout=args.timeout) if args.live else load_snapshot(args.snapshot or baseline_path)
        )
        result = evaluate_snapshot(current, baseline, source=source, strict=args.strict)
    except (OSError, SnapshotError, ValueError, urllib.error.URLError) as exc:
        code = str(exc) if isinstance(exc, SnapshotError) else "upstream_watch:unavailable"
        result = _fail_closed(code, source=source)

    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 2 if result["review_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
