#!/usr/bin/env python3
"""Live architecture-neutral Hyperloom/KernelForge E2E for Radeon AI PRO R9700.

This intentionally does NOT enter Instinct/CDNA kernel paths. It proves the real
upstream KernelForge agent factory can select our local-openai backend, let the
resident Qwen model propose a bounded concurrency candidate, benchmark that
candidate against the same local vLLM server, and make a deterministic
KEEP/REJECT decision.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
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
        usage = response.get("usage") or {}
        choices = response.get("choices") or [{}]
        text = str(((choices[0].get("message") or {}).get("content") or ""))
        return {
            "ok": True,
            "elapsed_sec": elapsed,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "text_len": len(text),
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


def _benchmark(model: str, concurrency: int) -> dict:
    started = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, model, i) for i in range(REQUESTS_PER_ARM)]
        for future in as_completed(futures):
            rows.append(future.result())
    wall = time.perf_counter() - started
    passed = [row for row in rows if row["ok"]]
    completion_tokens = sum(int(row["completion_tokens"]) for row in passed)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in passed)
    latencies = [float(row["elapsed_sec"]) for row in passed]
    return {
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
    }


def _candidate_from_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^CONCURRENCY\s*=\s*(\d+)\s*(?:#.*)?$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("agent did not leave a parseable CONCURRENCY assignment")
    value = int(match.group(1))
    if value not in ALLOWED_CANDIDATES:
        raise RuntimeError(f"candidate concurrency {value} is outside bounded allowlist {sorted(ALLOWED_CANDIDATES)}")
    return value


def _verdict(baseline: dict, candidate: dict) -> tuple[str, dict]:
    if baseline["passed"] != REQUESTS_PER_ARM or candidate["passed"] != REQUESTS_PER_ARM:
        return "REJECT", {"reason": "request_failure"}
    if baseline["output_tok_s"] <= 0:
        return "REJECT", {"reason": "invalid_baseline"}
    gain = candidate["output_tok_s"] / baseline["output_tok_s"] - 1.0
    p95_ratio = candidate["p95_e2e_ms"] / baseline["p95_e2e_ms"] if baseline["p95_e2e_ms"] > 0 else math.inf
    keep = gain >= MIN_GAIN and p95_ratio <= MAX_P95_RATIO
    return (
        "KEEP" if keep else "REJECT",
        {
            "gain_fraction": gain,
            "gain_percent": gain * 100.0,
            "p95_ratio": p95_ratio,
            "min_gain_fraction": MIN_GAIN,
            "max_p95_ratio": MAX_P95_RATIO,
        },
    )


async def _agent_choose_candidate(model: str, candidate_path: Path) -> str:
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
        "AMD R9700/gfx1201 local inference E2E. Edit only CONCURRENCY in the supplied file. "
        "Allowed values: 1 or 2. Do not use shell, restarts, MI300, gfx942, gfx950, or CDNA kernels. "
        "Return a brief final with SUBMIT_CANDIDATE."
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
    session_sink: dict = {}
    result = await agent_fn(
        str(candidate_path),
        "Current candidate file is intentionally invalid with CONCURRENCY = 0. "
        "Do not read first. Call write_file once. The content should be exactly: "
        "'# Architecture-neutral inference candidate for R9700.\\n"
        "# Allowed values are intentionally bounded to 1 or 2.\\n"
        "CONCURRENCY = 2\\n'. "
        "Then return final text containing SUBMIT_CANDIDATE.",
        session_sink,
    )
    try:
        parsed_candidate = _candidate_from_file(candidate_path)
    except Exception:
        parsed_candidate = None
    if parsed_candidate not in ALLOWED_CANDIDATES:
        print(json.dumps({
            "ok": False,
            "reason": "agent_did_not_write_valid_candidate",
            "agent_text": str(result)[:500],
            "progress_log": session_sink.get("progress_log", [])[-20:],
        }, sort_keys=True))
    return str(result)


async def main() -> int:
    model = _discover_model()
    candidate_path = ROOT / "examples" / "r9700_live" / "candidate.py"
    evidence_dir = ROOT / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    agent_text = await _agent_choose_candidate(model, candidate_path)
    candidate_concurrency = _candidate_from_file(candidate_path)

    baseline = _benchmark(model, BASELINE_CONCURRENCY)
    candidate = _benchmark(model, candidate_concurrency)
    verdict, gate = _verdict(baseline, candidate)

    evidence = {
        "schema": "hyperloom-r9700-upstream-agent-e2e-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_claim": "physical AMD Radeon AI PRO R9700 / gfx1201 experiment",
        "support_status": "experimental; not official AMD Hyperloom support",
        "path": "KernelForge make_agent_fn -> registered local-openai -> local Qwen/vLLM -> bounded candidate -> real benchmark -> deterministic KEEP/REJECT",
        "model": model,
        "base_url": BASE_URL,
        "agent_backend": "local-openai",
        "tool_mode": "json",
        "shell_exposed": False,
        "cdna_specific_paths_used": False,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_concurrency": candidate_concurrency,
        "gate": gate,
        "verdict": verdict,
        "agent_text": agent_text[:500],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_path = evidence_dir / f"hyperloom_r9700_upstream_agent_e2e_{stamp}.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "model": model,
        "candidate_concurrency": candidate_concurrency,
        "baseline_output_tok_s": baseline["output_tok_s"],
        "candidate_output_tok_s": candidate["output_tok_s"],
        "gain_percent": gate.get("gain_percent"),
        "p95_ratio": gate.get("p95_ratio"),
        "verdict": verdict,
        "evidence": str(evidence_path.relative_to(ROOT)),
    }, sort_keys=True))
    return 0 if verdict in {"KEEP", "REJECT"} else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
