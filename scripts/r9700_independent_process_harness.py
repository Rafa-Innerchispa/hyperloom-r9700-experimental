#!/usr/bin/env python3
"""Run paired baseline/candidate evidence across independent resident vLLM process spawns.

This harness exists specifically to address ROCm/ROCm#6347-style process-spawn
confounding on Radeon AI PRO R9700. Each iteration restarts the *same* resident
Docker container, waits for the exact served model to become healthy, then runs
one bounded upstream-agent E2E cycle containing baseline and candidate arms.

Default mode is plan-only. --execute intentionally causes bounded service
restarts and should only be used inside a coordinated maintenance window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.independent_vllm_process_harness.v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CONTAINER = "inneros-vllm-canary-rocm10"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return statistics.median(vals) if vals else None


def run(argv: list[str], *, timeout: float = 60.0, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        p = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "argv": argv}
    except Exception as exc:  # noqa: BLE001
        return {"returncode": None, "stdout": "", "stderr": type(exc).__name__ + ":" + str(exc), "argv": argv}


def parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(str(text or "").splitlines()):
        try:
            value = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def models_health(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [row.get("id") for row in payload.get("data", []) if isinstance(row, dict)] if isinstance(payload, dict) else []
        return {"ok": response.status == 200, "status": response.status, "model_ids": ids}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def wait_healthy(base_url: str, *, expected_model: str, timeout_sec: float = 240.0) -> dict[str, Any]:
    started = time.perf_counter()
    attempts = 0
    last: dict[str, Any] = {}
    while time.perf_counter() - started < timeout_sec:
        attempts += 1
        last = models_health(base_url)
        if last.get("ok") and expected_model in (last.get("model_ids") or []):
            return {"ok": True, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}
        time.sleep(2.0)
    return {"ok": False, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}


def container_identity(container: str) -> dict[str, Any]:
    result = run(["docker", "inspect", container], timeout=30.0)
    if result.get("returncode") != 0:
        return {"ok": False, "stderr": str(result.get("stderr") or "")[-2000:]}
    try:
        row = json.loads(result["stdout"])[0]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}
    cmd = [str(x) for x in row.get("Config", {}).get("Cmd", []) or []]
    return {
        "ok": True,
        "container_id": row.get("Id"),
        "image": row.get("Config", {}).get("Image"),
        "started_at": row.get("State", {}).get("StartedAt"),
        "pid": row.get("State", {}).get("Pid"),
        "running": row.get("State", {}).get("Running"),
        "cmd": cmd,
    }


def cli_flag(argv: list[str], name: str, default: str = "") -> str:
    try:
        idx = argv.index(name)
    except ValueError:
        return default
    return argv[idx + 1] if idx + 1 < len(argv) else default


def telemetry_snapshot() -> dict[str, Any]:
    result = run(["rocm-smi", "--showuse", "--showpower", "--showclocks"], timeout=20.0)
    text = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
    use_match = re.search(r"GPU\[0\].*GPU use \(%\):\s*([0-9.]+)", text)
    power_match = re.search(r"GPU\[0\].*Power \(W\):\s*([0-9.]+)", text)
    sclk_lines = [line.strip() for line in text.splitlines() if "GPU[0]" in line and "sclk" in line.lower()]
    return {
        "at_utc": utc_now(),
        "returncode": result.get("returncode"),
        "gpu0_use_percent": float(use_match.group(1)) if use_match else None,
        "gpu0_power_w": float(power_match.group(1)) if power_match else None,
        "gpu0_sclk_lines": sclk_lines[:8],
    }


def monitor_telemetry(stop: threading.Event, bucket: list[dict[str, Any]], interval: float = 2.0) -> None:
    while not stop.is_set():
        bucket.append(telemetry_snapshot())
        stop.wait(interval)


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    use = [x["gpu0_use_percent"] for x in samples if finite(x.get("gpu0_use_percent"))]
    power = [x["gpu0_power_w"] for x in samples if finite(x.get("gpu0_power_w"))]
    sclk = []
    for x in samples:
        for line in x.get("gpu0_sclk_lines") or []:
            if line not in sclk:
                sclk.append(line)
    return {
        "sample_count": len(samples),
        "gpu0_use_percent_max": max(use) if use else None,
        "gpu0_use_percent_median": median(use),
        "gpu0_power_w_max": max(power) if power else None,
        "gpu0_power_w_median": median(power),
        "sclk_observed_lines": sclk[:20],
    }


def build_plan(*, process_count: int = 3, base_url: str = DEFAULT_BASE_URL, container: str = DEFAULT_CONTAINER) -> dict[str, Any]:
    if process_count < 2 or process_count > 5:
        raise ValueError("process_count must be between 2 and 5")
    return {
        "schema": SCHEMA,
        "mode": "plan",
        "process_count": process_count,
        "base_url": base_url.rstrip("/"),
        "container": container,
        "runner": "scripts/r9700_upstream_agent_e2e.py",
        "purpose": "pair baseline and candidate inside independent vLLM process spawns to reduce ROCm #6347 process-state confounding",
        "per_process_contract": {
            "same_container_image_and_launch": True,
            "docker_restart_before_measurement": True,
            "health_wait": True,
            "baseline_candidate_paired_in_same_process": True,
            "runner_baseline_rounds": 3,
            "runner_candidate_rounds": 3,
            "runner_requests_per_round": 6,
            "runtime_telemetry_sampled": True,
        },
        "truth_boundary": {
            "not_kernel_optimization_proof": True,
            "not_official_hyperloom_support": True,
            "restart_changes_process_spawn_state": True,
            "small_sample_warning": "three independent process spawns address the community 3-repeat recommendation but do not statistically characterize every possible ROCm #6347 state",
        },
    }


def audit_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("processes")
    if not isinstance(rows, list) or len(rows) != report.get("process_count"):
        return {"ok": False, "reason": "process_count_mismatch"}
    starts: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return {"ok": False, "reason": "invalid_row", "index": i}
        if row.get("restart_returncode") != 0:
            return {"ok": False, "reason": "restart_failed", "index": i}
        if (row.get("health") or {}).get("ok") is not True:
            return {"ok": False, "reason": "health_failed", "index": i}
        identity = row.get("identity_after_restart") or {}
        started_at = identity.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            return {"ok": False, "reason": "missing_started_at", "index": i}
        if started_at in starts:
            return {"ok": False, "reason": "duplicate_process_started_at", "index": i}
        starts.add(started_at)
        if row.get("runner_returncode") != 0:
            return {"ok": False, "reason": "runner_failed", "index": i}
        summary = row.get("runner_summary")
        if not isinstance(summary, dict) or summary.get("ok") is not True:
            return {"ok": False, "reason": "runner_summary_not_ok", "index": i}
        for field in ("baseline_median_output_tok_s", "candidate_median_output_tok_s", "baseline_median_p95_ttft_ms", "candidate_median_p95_ttft_ms", "gain_percent", "p95_ratio"):
            if not finite(summary.get(field)):
                return {"ok": False, "reason": "invalid_metric", "index": i, "field": field}
    return {"ok": True, "independent_process_count": len(starts), "unique_started_at": sorted(starts)}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row.get("runner_summary") or {} for row in rows]
    baselines = [s.get("baseline_median_output_tok_s") for s in summaries if finite(s.get("baseline_median_output_tok_s"))]
    candidates = [s.get("candidate_median_output_tok_s") for s in summaries if finite(s.get("candidate_median_output_tok_s"))]
    baseline_ttft = [s.get("baseline_median_p95_ttft_ms") for s in summaries if finite(s.get("baseline_median_p95_ttft_ms"))]
    candidate_ttft = [s.get("candidate_median_p95_ttft_ms") for s in summaries if finite(s.get("candidate_median_p95_ttft_ms"))]
    gains = [s.get("gain_percent") for s in summaries if finite(s.get("gain_percent"))]
    p95 = [s.get("p95_ratio") for s in summaries if finite(s.get("p95_ratio"))]
    decisions = [s.get("decision") or s.get("verdict") for s in summaries]
    return {
        "baseline_output_tok_s_median_across_processes": median(baselines),
        "baseline_output_tok_s_min": min(baselines) if baselines else None,
        "baseline_output_tok_s_max": max(baselines) if baselines else None,
        "candidate_output_tok_s_median_across_processes": median(candidates),
        "candidate_output_tok_s_min": min(candidates) if candidates else None,
        "candidate_output_tok_s_max": max(candidates) if candidates else None,
        "baseline_ttft_p95_ms_median_across_processes": median(baseline_ttft),
        "candidate_ttft_p95_ms_median_across_processes": median(candidate_ttft),
        "paired_gain_percent_median": median(gains),
        "paired_gain_percent_min": min(gains) if gains else None,
        "paired_gain_percent_max": max(gains) if gains else None,
        "p95_ratio_median": median(p95),
        "decisions": decisions,
        "keep_count": sum(1 for x in decisions if x == "KEEP"),
        "reject_count": sum(1 for x in decisions if x == "REJECT"),
        "all_processes_candidate_faster": bool(gains) and all(float(x) > 0 for x in gains),
        "all_processes_gain_ge_10pct": bool(gains) and all(float(x) >= 10.0 for x in gains),
        "baseline_process_spread_percent": ((max(baselines) - min(baselines)) / min(baselines) * 100.0) if len(baselines) >= 2 and min(baselines) > 0 else None,
    }


def execute(*, process_count: int, base_url: str, container: str, output: Path) -> dict[str, Any]:
    plan = build_plan(process_count=process_count, base_url=base_url, container=container)
    before = container_identity(container)
    if not before.get("ok") or not before.get("running"):
        raise RuntimeError("resident vLLM container is not running")
    expected_model = cli_flag(before.get("cmd") or [], "--served-model-name") or cli_flag(before.get("cmd") or [], "--model")
    if not expected_model:
        raise RuntimeError("could not bind expected model from container launch")
    if not models_health(base_url).get("ok"):
        raise RuntimeError("resident vLLM endpoint is not healthy before benchmark")

    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    env["HYPERLOOM_EXECUTION_SCOPE"] = "r9700_independent_vllm_process_harness"
    started_wall = time.perf_counter()
    execution_error: str | None = None
    try:
        for index in range(process_count):
            row: dict[str, Any] = {"process_index": index, "started_at_utc": utc_now(), "identity_before_restart": container_identity(container)}
            restart = run(["docker", "restart", "--time", "30", container], timeout=90.0)
            row["restart_returncode"] = restart.get("returncode")
            row["restart_stdout"] = str(restart.get("stdout") or "")[-1000:]
            row["restart_stderr"] = str(restart.get("stderr") or "")[-2000:]
            if restart.get("returncode") != 0:
                rows.append(row)
                raise RuntimeError(f"docker restart failed on process {index}")
            row["identity_after_restart"] = container_identity(container)
            health = wait_healthy(base_url, expected_model=expected_model, timeout_sec=240.0)
            row["health"] = health
            if not health.get("ok"):
                rows.append(row)
                raise RuntimeError(f"vLLM did not become healthy on process {index}")
            row["telemetry_idle"] = telemetry_snapshot()

            samples: list[dict[str, Any]] = []
            stop = threading.Event()
            thread = threading.Thread(target=monitor_telemetry, args=(stop, samples), daemon=True)
            thread.start()
            try:
                runner = run([sys.executable, str(ROOT / "scripts" / "r9700_upstream_agent_e2e.py")], timeout=900.0, env=env)
            finally:
                stop.set()
                thread.join(timeout=5.0)
            summary = parse_last_json(str(runner.get("stdout") or ""))
            row["runner_returncode"] = runner.get("returncode")
            row["runner_stdout_tail"] = str(runner.get("stdout") or "")[-5000:]
            row["runner_stderr_tail"] = str(runner.get("stderr") or "")[-3000:]
            row["runner_summary"] = summary
            row["telemetry_under_load"] = summarize_telemetry(samples)
            row["telemetry_samples"] = samples
            row["completed_at_utc"] = utc_now()
            rows.append(row)
            if runner.get("returncode") != 0 or not isinstance(summary, dict) or summary.get("ok") is not True:
                raise RuntimeError(f"runner failed on process {index}")
    except Exception as exc:  # noqa: BLE001
        execution_error = type(exc).__name__ + ":" + str(exc)
    finally:
        final_health = models_health(base_url)
        recovery = None
        if not final_health.get("ok"):
            recovery_restart = run(["docker", "restart", "--time", "30", container], timeout=90.0)
            recovery_health = wait_healthy(base_url, expected_model=expected_model, timeout_sec=240.0)
            recovery = {"restart_returncode": recovery_restart.get("returncode"), "health": recovery_health}
            final_health = recovery_health.get("health") or {}

    report: dict[str, Any] = {
        **plan,
        "mode": "executed",
        "captured_at_utc": utc_now(),
        "wall_sec": time.perf_counter() - started_wall,
        "container_before": before,
        "expected_model": expected_model,
        "processes": rows,
        "aggregate": aggregate(rows),
        "execution_error": execution_error,
        "final_service_health": final_health,
        "recovery": recovery,
    }
    report["audit"] = audit_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-count", type=int, default=3)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--output", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / "docs" / "evidence" / "r9700_independent_process_plan.json"
    if args.execute:
        report = execute(process_count=args.process_count, base_url=args.base_url, container=args.container, output=output)
    else:
        report = build_plan(process_count=args.process_count, base_url=args.base_url, container=args.container)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report.get("audit", {}).get("ok") if args.execute else True, "mode": report.get("mode"), "process_count": report.get("process_count"), "audit": report.get("audit"), "aggregate": report.get("aggregate"), "execution_error": report.get("execution_error"), "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)}, sort_keys=True))
    return 0 if (not args.execute or (report.get("audit") or {}).get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
