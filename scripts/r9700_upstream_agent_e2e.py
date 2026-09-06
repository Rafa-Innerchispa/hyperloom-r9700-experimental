#!/usr/bin/env python3
"""Live architecture-neutral Hyperloom/KernelForge E2E for Radeon AI PRO R9700.

This intentionally does NOT enter Instinct/CDNA kernel paths. It proves the real
upstream KernelForge agent factory can select our local-openai backend, let the
resident Qwen model choose a bounded concurrency candidate from measured baseline
evidence, benchmark that candidate repeatedly against the same local vLLM server,
and make a deterministic KEEP/REJECT decision.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kernelforge.config import Config  # noqa: E402
from kernelforge.orchestrator.agent import make_agent_fn  # noqa: E402

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
BASELINE_CONCURRENCY = 1
ALLOWED_CANDIDATES = {1, 2}
REQUESTS_PER_ARM = 6
MEASUREMENT_ROUNDS = 3
MAX_TOKENS = 32
AGENT_MAX_TOKENS = 1024
MIN_GAIN = 0.10
MAX_P95_RATIO = 1.25


def _request_json(method: str, path: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _discover_model() -> str:
    payload = _request_json("GET", "/models", timeout=20.0)
    models = payload.get("data") or []
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError("vLLM /models returned no model id")
    return str(models[0]["id"])


def _one_request(model: str, index: int) -> dict:
    prompt = (
        "You are validating a local inference benchmark. "
        f"Request {index}. Reply with one short sentence containing the word AMD."
    )
    started = time.perf_counter()
    try:
        response = _request_json(
            "POST",
            "/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
            timeout=120.0,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(response, dict):
            raise ValueError("invalid_response_object")
        usage = response.get("usage")
        choices = response.get("choices")
        if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
            raise ValueError("missing_choices_or_usage")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty_or_invalid_content")
        if re.search(r"\bAMD\b", text, flags=re.IGNORECASE) is None:
            raise ValueError("benchmark_correctness_failed")
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        if type(completion_tokens) is not int or completion_tokens <= 0:
            raise ValueError("invalid_completion_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("invalid_prompt_tokens")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid_elapsed_time")
        return {
            "ok": True,
            "elapsed_sec": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "text_len": len(text),
            "correctness_passed": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed_sec": time.perf_counter() - started,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _benchmark(model: str, concurrency: int, *, round_index: int = 0) -> dict:
    started = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_one_request, model, round_index * REQUESTS_PER_ARM + i):
                round_index * REQUESTS_PER_ARM + i
            for i in range(REQUESTS_PER_ARM)
        }
        for future in as_completed(futures):
            row = dict(future.result())
            row["request_index"] = futures[future]
            rows.append(row)
    wall = time.perf_counter() - started
    passed = [row for row in rows if row["ok"]]
    completion_tokens = sum(int(row["completion_tokens"]) for row in passed)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in passed)
    latencies = [float(row["elapsed_sec"]) for row in passed]
    return {
        "round": round_index + 1,
        "concurrency": concurrency,
        "requests": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "wall_sec": wall,
        "output_tokens": completion_tokens,
        "total_tokens": completion_tokens + prompt_tokens,
        "output_tok_s": completion_tokens / wall if wall > 0 else 0.0,
        "total_tok_s": (completion_tokens + prompt_tokens) / wall if wall > 0 else 0.0,
        "mean_e2e_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else math.inf,
        "p95_e2e_ms": _percentile(latencies, 0.95) * 1000.0 if latencies else math.inf,
        "errors": [row.get("error") for row in rows if not row["ok"]],
        "samples": sorted(rows, key=lambda row: row["request_index"]),
    }


def _aggregate_rounds(rounds: list[dict]) -> dict:
    if not rounds:
        raise ValueError("at least one benchmark round is required")
    return {
        "round_count": len(rounds),
        "requests": sum(int(row["requests"]) for row in rounds),
        "passed": sum(int(row["passed"]) for row in rounds),
        "failed": sum(int(row["failed"]) for row in rounds),
        "median_output_tok_s": statistics.median(float(row["output_tok_s"]) for row in rounds),
        "mean_output_tok_s": statistics.fmean(float(row["output_tok_s"]) for row in rounds),
        "min_output_tok_s": min(float(row["output_tok_s"]) for row in rounds),
        "max_output_tok_s": max(float(row["output_tok_s"]) for row in rounds),
        "median_total_tok_s": statistics.median(float(row["total_tok_s"]) for row in rounds),
        "median_mean_e2e_ms": statistics.median(float(row["mean_e2e_ms"]) for row in rounds),
        "median_p95_e2e_ms": statistics.median(float(row["p95_e2e_ms"]) for row in rounds),
    }


def _benchmark_rounds(model: str, concurrency: int, rounds: int = MEASUREMENT_ROUNDS) -> dict:
    raw = [_benchmark(model, concurrency, round_index=index) for index in range(rounds)]
    return {"rounds": raw, "aggregate": _aggregate_rounds(raw)}


def _candidate_from_file(path: Path) -> int:
    if path.stat().st_size > 4096:
        raise RuntimeError("candidate exceeds bounded size")
    text = path.read_text(encoding="utf-8")
    statements = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    statements = [line for line in statements if line]
    if len(statements) != 1:
        raise RuntimeError("candidate must contain exactly one CONCURRENCY assignment")
    match = re.fullmatch(r"CONCURRENCY\s*=\s*([0-9]+)", statements[0])
    if not match:
        raise RuntimeError("agent did not leave a parseable CONCURRENCY assignment")
    value = int(match.group(1))
    if value not in ALLOWED_CANDIDATES:
        raise RuntimeError(f"candidate concurrency {value} is outside bounded allowlist {sorted(ALLOWED_CANDIDATES)}")
    return value


def _verdict(baseline: dict, candidate: dict) -> tuple[str, dict]:
    # Fail closed before calculating ratios: a successful HTTP request or a
    # plausible median is not evidence of complete, valid measurements.
    for name, arm in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(arm, dict):
            return "REJECT", {"reason": "invalid_evidence", "arm": name}
        for key in ("round_count", "requests", "passed", "failed"):
            if type(arm.get(key)) is not int or arm[key] < 0:
                return "REJECT", {"reason": "invalid_counts", "arm": name, "field": key}
        if arm["failed"]:
            return "REJECT", {"reason": "request_failure", "arm": name}
        if (
            arm["round_count"] != MEASUREMENT_ROUNDS
            or arm["requests"] != MEASUREMENT_ROUNDS * REQUESTS_PER_ARM
            or arm["passed"] + arm["failed"] != arm["requests"]
        ):
            return "REJECT", {"reason": "incomplete_measurements", "arm": name}
        for key in ("median_output_tok_s", "median_p95_e2e_ms"):
            value = arm.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                return "REJECT", {"reason": "invalid_metric", "arm": name, "field": key}
    gain = candidate["median_output_tok_s"] / baseline["median_output_tok_s"] - 1.0
    p95_ratio = candidate["median_p95_e2e_ms"] / baseline["median_p95_e2e_ms"]
    if not math.isfinite(gain) or not math.isfinite(p95_ratio):
        return "REJECT", {"reason": "invalid_ratio"}
    keep = gain >= MIN_GAIN and p95_ratio <= MAX_P95_RATIO
    return (
        "KEEP" if keep else "REJECT",
        {
            "gain_fraction": gain,
            "gain_percent": gain * 100.0,
            "p95_ratio": p95_ratio,
            "min_gain_fraction": MIN_GAIN,
            "max_p95_ratio": MAX_P95_RATIO,
            "throughput_metric": "median_output_tok_s",
            "latency_metric": "median_p95_e2e_ms",
        },
    )


def _decision_context(baseline: dict) -> str:
    return (
        "A real baseline has already been measured at concurrency 1 over "
        f"{baseline['round_count']} rounds. Median output throughput: "
        f"{baseline['median_output_tok_s']:.4f} tok/s. Median p95 end-to-end latency: "
        f"{baseline['median_p95_e2e_ms']:.2f} ms. Failed requests: {baseline['failed']}. "
        "Choose exactly one candidate concurrency from the bounded set [1, 2]. "
        "Your objective is to maximize aggregate output throughput while hypothesizing that the "
        f"candidate can keep p95 latency within {MAX_P95_RATIO:.2f}x of baseline. "
        "You have not seen candidate benchmark results. Make the choice yourself from the measured "
        "baseline and the bounded search space. Use write_file exactly once to replace the supplied "
        "candidate file with the same two comment lines and a valid CONCURRENCY assignment using "
        "your chosen value. Do not use shell and do not restart any service. Then return a short "
        "PLAN describing your hypothesis and include SUBMIT_CANDIDATE."
    )


async def _agent_choose_candidate(model: str, candidate_path: Path, baseline: dict) -> dict:
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        "# Architecture-neutral inference candidate for R9700.\n"
        "# Allowed values are intentionally bounded to 1 or 2.\n"
        "CONCURRENCY = 0\n",
        encoding="utf-8",
    )
    config = Config(
        workspace=str(candidate_path.parent),
        project_root=ROOT,
        agent_backend="local-openai",
        agent_model=model,
        agent_fallback_provider="",
        agent_timeout_sec=120,
        agent_precheck=True,
        agent_options={
            "base_url": BASE_URL,
            "tool_mode": "json",
            "json_tool_catalog": "minimal",
            "system_prompt_mode": "compact",
            "enabled_tools": ["write_file"],
            "max_tokens": AGENT_MAX_TOKENS,
            "allow_shell": False,
        },
        max_turns=6,
    )
    program = (
        "AMD R9700/gfx1201 local inference optimization experiment. Edit only CONCURRENCY in the "
        "supplied candidate file. Allowed values are 1 or 2. Do not use shell, service restarts, "
        "MI300, gfx942, gfx950, or CDNA-specific kernel paths. Candidate performance will be measured "
        "after your turn by a deterministic gate."
    )
    agent_fn = make_agent_fn(
        config,
        program,
        profiling_enabled=False,
        insession_gate=False,
        source_files=[str(candidate_path)],
        task_type="repository",
        agent_backend="local-openai",
    )
    original_instruction = _decision_context(baseline)
    instruction = original_instruction
    validation_attempts: list[dict] = []
    for attempt in range(1, 3):
        session_sink: dict = {}
        result = await agent_fn(str(candidate_path), instruction, session_sink)
        try:
            selected = _candidate_from_file(candidate_path)
        except (RuntimeError, ValueError, OSError) as exc:
            reason = str(exc)[:240]
            validation_attempts.append({"attempt": attempt, "valid": False, "reason": reason})
            if attempt == 2:
                raise RuntimeError("candidate_format_retry_exhausted") from exc
            instruction = (
                original_instruction
                + " Your previous candidate failed format validation: " + reason + ". "
                "One format-repair attempt remains. Use write_file to write exactly one "
                "uncommented CONCURRENCY assignment with your chosen allowed integer. "
                "Do NOT put the CONCURRENCY assignment in a comment. Comments are optional, "
                "and no other code or markdown fences are permitted. Choose the value yourself; "
                "no candidate measurements have been provided. Then SUBMIT_CANDIDATE."
            )
            continue
        validation_attempts.append({"attempt": attempt, "valid": True})
        return {
            "selected_concurrency": selected,
            "selected_by_model": True,
            "hardcoded_candidate": False,
            "agent_text": str(result)[:500],
            "plan": str(session_sink.get("plan") or "")[:300],
            "progress_log": list(session_sink.get("progress_log") or [])[-20:],
            "decision_context": original_instruction,
            "validation_attempts": validation_attempts,
            "format_repair_used": attempt > 1,
        }
    raise RuntimeError("candidate_validation_exhausted")


def _strict_json_value(value):
    """Preserve failed-run evidence without emitting non-standard NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


async def main() -> int:
    runner_path = Path(__file__).resolve()
    source_digest = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    candidate_path = ROOT / "examples" / "r9700_live" / "candidate.py"
    evidence_dir = ROOT / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "hyperloom-r9700-upstream-autonomous-agent-e2e-v2",
        "sample_schema": "r9700-request-metrics-v1",
        "runner_source_sha256_start": source_digest,
        "runner_source_sha256_end": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "run_status": "failed",
        "failure": None,
        "nonfinite_serialization": "null; invalid measurements never complete successfully",
        "hardware_claim": "target AMD Radeon AI PRO R9700 / gfx1201; requires independent physical attestation",
        "hardware_attested_by_runner": False,
        "orchestrator_host": platform.node(),
        "execution_scope": os.environ.get("HYPERLOOM_EXECUTION_SCOPE", "unattested_runtime"),
        "support_status": "experimental; not official AMD Hyperloom support",
        "path": "KernelForge make_agent_fn -> registered local-openai -> local Qwen/vLLM -> model-selected bounded candidate -> repeated real benchmark -> deterministic KEEP/REJECT",
        "model": None,
        "base_url": BASE_URL,
        "agent_backend": "local-openai",
        "tool_mode": "json",
        "shell_exposed": False,
        "cdna_specific_paths_used": False,
        "measurement_rounds": MEASUREMENT_ROUNDS,
        "requests_per_round": REQUESTS_PER_ARM,
        "baseline_concurrency": BASELINE_CONCURRENCY,
        "allowed_candidates": sorted(ALLOWED_CANDIDATES),
        "baseline": None,
        "agent": None,
        "candidate_concurrency": None,
        "candidate": None,
        "gate": {},
        "verdict": None,
        "truth_boundary": "architecture-neutral experimental R9700 compatibility; no official AMD/upstream Hyperloom R9700 support claim",
    }
    stage = "model_discovery"
    try:
        model = _discover_model()
        evidence["model"] = model
        stage = "baseline_measurement"
        _one_request(model, -1)  # uncounted warmup
        evidence["baseline"] = _benchmark_rounds(model, BASELINE_CONCURRENCY)
        baseline = evidence["baseline"]["aggregate"]
        # Self-comparison validates completeness, not optimization gain. Never
        # ask the model to optimize a baseline that failed measurement checks.
        _, baseline_check = _verdict(baseline, baseline)
        if baseline_check.get("reason"):
            evidence["gate"] = baseline_check
            evidence["failure"] = {"stage": stage, **baseline_check}
        else:
            stage = "candidate_selection"
            agent = await _agent_choose_candidate(model, candidate_path, baseline)
            evidence["agent"] = agent
            selected = agent["selected_concurrency"]
            if type(selected) is not int or selected not in ALLOWED_CANDIDATES:
                raise ValueError("invalid_candidate_selection")
            evidence["candidate_concurrency"] = selected
            stage = "candidate_measurement"
            _one_request(model, -2)  # uncounted warmup
            evidence["candidate"] = _benchmark_rounds(model, selected)
            verdict, gate = _verdict(baseline, evidence["candidate"]["aggregate"])
            evidence["gate"] = gate
            if gate.get("reason"):
                evidence["failure"] = {"stage": stage, **gate}
            else:
                # A valid REJECT is a completed experiment; invalid measurements
                # are not. The process exit code and the report must agree.
                evidence["verdict"] = verdict
                evidence["ok"] = True
                evidence["run_status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        # Do not copy arbitrary exception strings (potential URLs or secrets).
        evidence["failure"] = {"stage": stage, "reason": "execution_error", "error_type": type(exc).__name__}

    evidence["runner_source_sha256_end"] = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    if evidence["runner_source_sha256_end"] != source_digest:
        evidence.update(ok=False, run_status="failed", verdict=None)
        evidence["failure"] = {"stage": "evidence", "reason": "runner_changed_during_run"}
    evidence["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    evidence = _strict_json_value(evidence)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence_path = evidence_dir / f"hyperloom_r9700_upstream_autonomous_e2e_{stamp}.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    baseline = (evidence["baseline"] or {}).get("aggregate", {})
    candidate = (evidence["candidate"] or {}).get("aggregate", {})
    agent = evidence["agent"] or {}
    gate = evidence["gate"]
    print(json.dumps({
        "ok": evidence["ok"],
        "run_status": evidence["run_status"],
        "failure": evidence["failure"],
        "model": evidence["model"],
        "selected_by_model": agent.get("selected_by_model", False),
        "hardcoded_candidate": False,
        "candidate_concurrency": evidence["candidate_concurrency"],
        "measurement_rounds": MEASUREMENT_ROUNDS,
        "baseline_median_output_tok_s": baseline.get("median_output_tok_s"),
        "candidate_median_output_tok_s": candidate.get("median_output_tok_s"),
        "gain_percent": gate.get("gain_percent"),
        "p95_ratio": gate.get("p95_ratio"),
        "verdict": evidence["verdict"],
        "evidence": str(evidence_path.relative_to(ROOT)),
    }, sort_keys=True, allow_nan=False))
    return 0 if evidence["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
