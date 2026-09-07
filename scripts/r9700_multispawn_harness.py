#!/usr/bin/env python3
"""Plan and audit bounded multi-spawn R9700 evidence runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.multispawn_harness.v1"


def build_plan(*, spawn_count: int = 5, base_url: str = "http://127.0.0.1:8000/v1") -> dict[str, Any]:
    if spawn_count < 1 or spawn_count > 8:
        raise ValueError("spawn_count must be between 1 and 8")
    return {
        "schema": SCHEMA,
        "mode": "plan",
        "spawn_count": spawn_count,
        "base_url": base_url.rstrip("/"),
        "runner": "scripts/r9700_upstream_agent_e2e.py",
        "per_spawn_contract": {
            "warmup": "one uncounted request before each measured arm",
            "baseline_rounds": 3,
            "candidate_rounds": 3,
            "requests_per_round": 6,
            "allowed_candidates": [1, 2],
            "required_metrics": [
                "output_tok_s",
                "total_tok_s",
                "mean_e2e_ms",
                "p95_e2e_ms",
                "prompt_tokens",
                "completion_tokens",
                "failures",
            ],
        },
        "live_execution_gate": {
            "default": "blocked",
            "reason": "requires an explicit benchmark window because concurrent client spawns can consume the resident R9700/vLLM service",
            "safe_to_run_without_vllm_restart": True,
            "requires_service_restart": False,
        },
    }


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def audit_spawn_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("spawns")
    if not isinstance(rows, list) or not rows:
        return {"ok": False, "reason": "missing_spawns"}
    evidence_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return {"ok": False, "reason": "invalid_spawn_row", "spawn_index": index}
        if row.get("returncode") != 0:
            return {"ok": False, "reason": "spawn_failed", "spawn_index": index, "returncode": row.get("returncode")}
        summary = row.get("summary")
        if not isinstance(summary, dict) or summary.get("ok") is not True:
            return {"ok": False, "reason": "spawn_summary_not_ok", "spawn_index": index}
        path = summary.get("evidence")
        if not isinstance(path, str) or not path:
            return {"ok": False, "reason": "missing_evidence_path", "spawn_index": index}
        if path in evidence_paths:
            return {"ok": False, "reason": "duplicate_evidence_path", "spawn_index": index}
        evidence_paths.add(path)
        for field in ("baseline_median_output_tok_s", "candidate_median_output_tok_s", "gain_percent", "p95_ratio"):
            if not _finite_number(summary.get(field)):
                return {"ok": False, "reason": "invalid_metric", "spawn_index": index, "field": field}
    return {"ok": True, "spawn_count": len(rows), "evidence_paths": sorted(evidence_paths)}


def run_spawns(*, spawn_count: int, base_url: str, output: Path) -> dict[str, Any]:
    spawns: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    env["HYPERLOOM_EXECUTION_SCOPE"] = "r9700_multispawn_harness"
    started = time.perf_counter()
    for index in range(spawn_count):
        started_at = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "r9700_upstream_agent_e2e.py")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        summary: dict[str, Any] | None = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                maybe = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(maybe, dict):
                summary = maybe
                break
        spawns.append({
            "spawn_index": index,
            "started_at_utc": started_at,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-2000:],
            "summary": summary,
        })
    report = {
        **build_plan(spawn_count=spawn_count, base_url=base_url),
        "mode": "executed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_sec": time.perf_counter() - started,
        "spawns": spawns,
    }
    report["audit"] = audit_spawn_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spawn-count", type=int, default=5)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--output", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / "docs" / "evidence" / "r9700_multispawn_plan.json"
    if args.execute:
        report = run_spawns(spawn_count=args.spawn_count, base_url=args.base_url, output=output)
    else:
        report = build_plan(spawn_count=args.spawn_count, base_url=args.base_url)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "mode": report["mode"],
        "spawn_count": report["spawn_count"],
        "audit_ok": (report.get("audit") or {}).get("ok"),
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
