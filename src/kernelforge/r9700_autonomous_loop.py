# SPDX-License-Identifier: MIT
"""Bounded autonomous optimization loop for the experimental R9700 path.

The loop deliberately separates *proposal* from *acceptance*:

1. Measure a baseline against an existing OpenAI-compatible serving endpoint.
2. Ask the registered local-openai agent to write one bounded candidate.
3. Validate the candidate against an explicit allow-list.
4. Measure the candidate with the same workload.
5. KEEP or REJECT using deterministic gates.

The model is never allowed to decide whether its own change is accepted.  That
last decision stays deterministic and auditable, which is the useful part of an
autonomous optimizer rather than merely letting an LLM declare victory.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from kernelforge.agent_backends import AgentRunSpec, AgentToolPolicy


@dataclass(frozen=True)
class BenchmarkResult:
    """One serving benchmark arm."""

    concurrency: int
    requested: int
    succeeded: int
    failed: int
    wall_time_sec: float
    completion_tokens: int
    output_throughput_tps: float
    mean_latency_ms: float
    p95_latency_ms: float
    errors: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.requested if self.requested else 0.0


@dataclass(frozen=True)
class CandidateProposal:
    concurrency: int
    reason: str
    source: str = "agent"


@dataclass(frozen=True)
class AcceptanceDecision:
    decision: str
    throughput_gain_pct: float
    mean_latency_change_pct: float
    p95_latency_change_pct: float
    reason: str


@dataclass(frozen=True)
class AutonomousLoopResult:
    schema: str
    baseline: BenchmarkResult
    proposal: CandidateProposal
    candidate: BenchmarkResult
    acceptance: AcceptanceDecision
    agent_text: str
    agent_turns: int | None
    agent_tool_calls: tuple[str, ...]
    generated_at_unix: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline"]["success_rate"] = self.baseline.success_rate
        payload["candidate"]["success_rate"] = self.candidate.success_rate
        return payload


class CandidateValidationError(ValueError):
    """The agent proposed something outside the bounded experiment."""


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return math.inf if new > 0 else 0.0
    return ((new - old) / old) * 100.0


def decide_keep_or_reject(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    min_throughput_gain_pct: float = 10.0,
    max_mean_latency_regression_pct: float = 60.0,
    max_p95_latency_regression_pct: float = 75.0,
    require_full_success: bool = True,
) -> AcceptanceDecision:
    """Apply deterministic acceptance gates to one candidate.

    The thresholds are intentionally explicit rather than hidden in a model
    prompt.  A candidate must improve aggregate throughput materially while
    keeping latency growth bounded and preserving request success.
    """

    gain = _pct_change(candidate.output_throughput_tps, baseline.output_throughput_tps)
    mean_change = _pct_change(candidate.mean_latency_ms, baseline.mean_latency_ms)
    p95_change = _pct_change(candidate.p95_latency_ms, baseline.p95_latency_ms)

    failures: list[str] = []
    if require_full_success and candidate.succeeded != candidate.requested:
        failures.append(f"candidate success {candidate.succeeded}/{candidate.requested}")
    if candidate.output_throughput_tps <= 0:
        failures.append("candidate throughput is zero")
    if gain < min_throughput_gain_pct:
        failures.append(f"throughput gain {gain:.2f}% < {min_throughput_gain_pct:.2f}%")
    if mean_change > max_mean_latency_regression_pct:
        failures.append(
            f"mean latency regression {mean_change:.2f}% > {max_mean_latency_regression_pct:.2f}%"
        )
    if p95_change > max_p95_latency_regression_pct:
        failures.append(f"p95 latency regression {p95_change:.2f}% > {max_p95_latency_regression_pct:.2f}%")

    if failures:
        return AcceptanceDecision(
            decision="REJECT",
            throughput_gain_pct=gain,
            mean_latency_change_pct=mean_change,
            p95_latency_change_pct=p95_change,
            reason="; ".join(failures),
        )
    return AcceptanceDecision(
        decision="KEEP",
        throughput_gain_pct=gain,
        mean_latency_change_pct=mean_change,
        p95_latency_change_pct=p95_change,
        reason=(
            f"throughput +{gain:.2f}% with mean latency {mean_change:+.2f}% and "
            f"p95 latency {p95_change:+.2f}% inside deterministic gates"
        ),
    )


class OpenAIEndpointBenchmark:
    """Small dependency-free benchmark against a resident OpenAI endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        prompt: str = "Return exactly: R9700_OK",
        max_tokens: int = 32,
        request_timeout_sec: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.max_tokens = max(1, int(max_tokens))
        self.request_timeout_sec = float(request_timeout_sec)

    def _request_once(self) -> tuple[float, int, str]:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
            usage = body.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)
            if completion_tokens <= 0:
                text = str((((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
                completion_tokens = max(1, len(text.split())) if text else 0
            return (time.perf_counter() - started, completion_tokens, "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-800:]
            return (time.perf_counter() - started, 0, f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001 - benchmark records failures instead of hiding them.
            return (time.perf_counter() - started, 0, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
        return ordered[rank]

    def run(self, concurrency: int, requests: int) -> BenchmarkResult:
        concurrency = max(1, int(concurrency))
        requests = max(concurrency, int(requests))
        started = time.perf_counter()
        latencies: list[float] = []
        completion_tokens = 0
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self._request_once) for _ in range(requests)]
            for future in as_completed(futures):
                latency, tokens, error = future.result()
                latencies.append(latency)
                completion_tokens += tokens
                if error:
                    errors.append(error)

        wall = max(1e-9, time.perf_counter() - started)
        succeeded = requests - len(errors)
        mean_latency = statistics.fmean(latencies) if latencies else 0.0
        p95 = self._percentile(latencies, 0.95)
        return BenchmarkResult(
            concurrency=concurrency,
            requested=requests,
            succeeded=succeeded,
            failed=len(errors),
            wall_time_sec=wall,
            completion_tokens=completion_tokens,
            output_throughput_tps=(completion_tokens / wall),
            mean_latency_ms=(mean_latency * 1000.0),
            p95_latency_ms=(p95 * 1000.0),
            errors=tuple(errors[:5]),
        )


async def propose_candidate(
    backend: Any,
    *,
    baseline: BenchmarkResult,
    allowed_concurrencies: Iterable[int],
    model: str = "",
    workspace: Path | None = None,
) -> tuple[CandidateProposal, Any]:
    """Ask the agent for exactly one bounded concurrency candidate."""

    allowed = sorted({int(value) for value in allowed_concurrencies if int(value) > 0})
    allowed = [value for value in allowed if value != baseline.concurrency]
    if not allowed:
        raise CandidateValidationError("no candidate concurrency remains after excluding baseline")

    temp_context = tempfile.TemporaryDirectory(prefix="hyperloom-r9700-agent-") if workspace is None else None
    root = Path(temp_context.name if temp_context is not None else workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "candidate.json"
    if target.exists():
        target.unlink()

    prompt = (
        "You are choosing ONE architecture-neutral serving candidate for an AMD Radeon AI PRO R9700 "
        "experiment. Do not change model weights, kernels, ROCm, vLLM process flags or the running server. "
        f"Baseline concurrency={baseline.concurrency}, throughput={baseline.output_throughput_tps:.4f} output tok/s, "
        f"mean_latency_ms={baseline.mean_latency_ms:.3f}, p95_latency_ms={baseline.p95_latency_ms:.3f}, "
        f"success={baseline.succeeded}/{baseline.requested}. Allowed candidate concurrencies: {allowed}. "
        "Use write_file to create candidate.json containing exactly a JSON object with keys concurrency (integer) "
        "and reason (short string). Pick exactly one allowed value. Then return final."
    )
    progress: list[str] = []
    spec = AgentRunSpec(
        system_prompt=(
            "Operate conservatively. The candidate is only a proposal; deterministic benchmark gates decide KEEP/REJECT. "
            "Never claim measured improvement before the candidate benchmark exists."
        ),
        user_prompt=prompt,
        cwd=str(root),
        model=model,
        writable=True,
        timeout_sec=180,
        target_files=["candidate.json"],
        tool_policy=AgentToolPolicy(read=True, search=False, write=True, shell=False, max_turns=6),
        progress_log=progress,
    )
    result = await backend.run(spec)
    if not target.is_file():
        if temp_context is not None:
            temp_context.cleanup()
        raise CandidateValidationError(
            "local-openai agent completed without writing candidate.json; progress=" + " | ".join(progress[-6:])
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if temp_context is not None:
            temp_context.cleanup()
        raise CandidateValidationError(f"candidate.json is not valid JSON: {exc}") from exc

    try:
        concurrency = int(payload.get("concurrency"))
    except (TypeError, ValueError) as exc:
        if temp_context is not None:
            temp_context.cleanup()
        raise CandidateValidationError("candidate concurrency must be an integer") from exc
    if concurrency not in allowed:
        if temp_context is not None:
            temp_context.cleanup()
        raise CandidateValidationError(f"candidate concurrency {concurrency} is outside allow-list {allowed}")

    proposal = CandidateProposal(
        concurrency=concurrency,
        reason=str(payload.get("reason") or "agent proposed bounded concurrency candidate")[:500],
        source="agent",
    )
    if temp_context is not None:
        temp_context.cleanup()
    return proposal, result


async def run_autonomous_loop(
    *,
    backend: Any,
    benchmark: Callable[[int, int], BenchmarkResult],
    baseline_concurrency: int = 1,
    candidate_concurrencies: Iterable[int] = (2, 3, 4),
    requests_per_arm: int = 10,
    model: str = "",
    min_throughput_gain_pct: float = 10.0,
    max_mean_latency_regression_pct: float = 60.0,
    max_p95_latency_regression_pct: float = 75.0,
) -> AutonomousLoopResult:
    """Run one bounded baseline -> agent proposal -> candidate -> verdict cycle."""

    baseline = await asyncio.to_thread(benchmark, int(baseline_concurrency), int(requests_per_arm))
    if baseline.succeeded != baseline.requested:
        raise RuntimeError(
            f"baseline is not healthy: {baseline.succeeded}/{baseline.requested} requests succeeded; refusing optimization"
        )
    proposal, agent_result = await propose_candidate(
        backend,
        baseline=baseline,
        allowed_concurrencies=candidate_concurrencies,
        model=model,
    )
    candidate = await asyncio.to_thread(benchmark, proposal.concurrency, int(requests_per_arm))
    acceptance = decide_keep_or_reject(
        baseline,
        candidate,
        min_throughput_gain_pct=min_throughput_gain_pct,
        max_mean_latency_regression_pct=max_mean_latency_regression_pct,
        max_p95_latency_regression_pct=max_p95_latency_regression_pct,
    )
    return AutonomousLoopResult(
        schema="hyperloom.r9700.autonomous_loop.v1",
        baseline=baseline,
        proposal=proposal,
        candidate=candidate,
        acceptance=acceptance,
        agent_text=str(getattr(agent_result, "text", "")),
        agent_turns=getattr(agent_result, "num_turns", None),
        agent_tool_calls=tuple(name for name, _ in (getattr(agent_result, "tool_calls", None) or [])),
        generated_at_unix=time.time(),
    )


def write_loop_evidence(result: AutonomousLoopResult, path: Path) -> Path:
    """Write an auditable JSON artifact atomically enough for a single-process run."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


__all__ = [
    "AcceptanceDecision",
    "AutonomousLoopResult",
    "BenchmarkResult",
    "CandidateProposal",
    "CandidateValidationError",
    "OpenAIEndpointBenchmark",
    "decide_keep_or_reject",
    "propose_candidate",
    "run_autonomous_loop",
    "write_loop_evidence",
]
