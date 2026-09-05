# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunResult
from kernelforge.r9700_autonomous_loop import (
    BenchmarkResult,
    CandidateValidationError,
    decide_keep_or_reject,
    propose_candidate,
    run_autonomous_loop,
    write_loop_evidence,
)


def _bench(
    concurrency: int,
    *,
    throughput: float,
    mean_ms: float,
    p95_ms: float,
    requested: int = 10,
    succeeded: int = 10,
) -> BenchmarkResult:
    return BenchmarkResult(
        concurrency=concurrency,
        requested=requested,
        succeeded=succeeded,
        failed=requested - succeeded,
        wall_time_sec=1.0,
        completion_tokens=int(throughput),
        output_throughput_tps=throughput,
        mean_latency_ms=mean_ms,
        p95_latency_ms=p95_ms,
        errors=() if succeeded == requested else ("synthetic failure",),
    )


def test_keep_requires_material_throughput_and_bounded_latency() -> None:
    baseline = _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1200)
    candidate = _bench(2, throughput=34.0, mean_ms=1200, p95_ms=1450)
    decision = decide_keep_or_reject(baseline, candidate)
    assert decision.decision == "KEEP"
    assert decision.throughput_gain_pct == pytest.approx(70.0)
    assert decision.mean_latency_change_pct == pytest.approx(20.0)


def test_rejects_latency_regression_even_when_throughput_improves() -> None:
    baseline = _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1100)
    candidate = _bench(4, throughput=45.0, mean_ms=1900, p95_ms=2200)
    decision = decide_keep_or_reject(baseline, candidate)
    assert decision.decision == "REJECT"
    assert "latency regression" in decision.reason


def test_rejects_request_failures() -> None:
    baseline = _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1100)
    candidate = _bench(2, throughput=40.0, mean_ms=1200, p95_ms=1300, succeeded=9)
    decision = decide_keep_or_reject(baseline, candidate)
    assert decision.decision == "REJECT"
    assert "candidate success 9/10" in decision.reason


class _WritingBackend:
    def __init__(self, concurrency: int) -> None:
        self.concurrency = concurrency
        self.seen_spec = None

    async def run(self, spec):
        self.seen_spec = spec
        target = Path(spec.cwd) / "candidate.json"
        target.write_text(
            json.dumps({"concurrency": self.concurrency, "reason": "increase bounded serving parallelism"}),
            encoding="utf-8",
        )
        return AgentRunResult(
            text="candidate written",
            num_turns=2,
            tool_calls=[("write_file", {"path": "candidate.json"})],
            file_changes=["candidate.json"],
            edit_count=1,
        )


class _InvalidBackend:
    async def run(self, spec):
        (Path(spec.cwd) / "candidate.json").write_text(
            json.dumps({"concurrency": 99, "reason": "not bounded"}),
            encoding="utf-8",
        )
        return AgentRunResult(text="bad candidate")


def test_agent_proposal_is_bounded_and_write_only(tmp_path: Path) -> None:
    backend = _WritingBackend(2)
    baseline = _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1200)
    proposal, result = asyncio.run(
        propose_candidate(
            backend,
            baseline=baseline,
            allowed_concurrencies=(2, 3, 4),
            workspace=tmp_path,
        )
    )
    assert proposal.concurrency == 2
    assert proposal.source == "agent"
    assert result.edit_count == 1
    assert backend.seen_spec.target_files == ["candidate.json"]
    assert backend.seen_spec.tool_policy.write is True
    assert backend.seen_spec.tool_policy.shell is False


def test_agent_proposal_outside_allow_list_is_rejected(tmp_path: Path) -> None:
    baseline = _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1200)
    with pytest.raises(CandidateValidationError, match="outside allow-list"):
        asyncio.run(
            propose_candidate(
                _InvalidBackend(),
                baseline=baseline,
                allowed_concurrencies=(2, 3, 4),
                workspace=tmp_path,
            )
        )


def test_full_loop_agent_proposes_benchmark_decides_keep() -> None:
    backend = _WritingBackend(2)

    def benchmark(concurrency: int, requests: int) -> BenchmarkResult:
        assert requests == 10
        if concurrency == 1:
            return _bench(1, throughput=21.8, mean_ms=1467.8, p95_ms=1600)
        if concurrency == 2:
            return _bench(2, throughput=36.59, mean_ms=1746.4, p95_ms=1900)
        raise AssertionError(f"unexpected concurrency {concurrency}")

    result = asyncio.run(
        run_autonomous_loop(
            backend=backend,
            benchmark=benchmark,
            baseline_concurrency=1,
            candidate_concurrencies=(2, 3, 4),
            requests_per_arm=10,
        )
    )
    assert result.schema == "hyperloom.r9700.autonomous_loop.v1"
    assert result.proposal.concurrency == 2
    assert result.acceptance.decision == "KEEP"
    assert result.acceptance.throughput_gain_pct == pytest.approx(67.8440366972)
    assert result.agent_tool_calls == ("write_file",)


def test_unhealthy_baseline_stops_before_agent() -> None:
    backend = _WritingBackend(2)

    def benchmark(concurrency: int, requests: int) -> BenchmarkResult:
        return _bench(concurrency, throughput=10.0, mean_ms=1000, p95_ms=1200, requested=requests, succeeded=9)

    with pytest.raises(RuntimeError, match="baseline is not healthy"):
        asyncio.run(
            run_autonomous_loop(
                backend=backend,
                benchmark=benchmark,
                baseline_concurrency=1,
                candidate_concurrencies=(2,),
                requests_per_arm=10,
            )
        )
    assert backend.seen_spec is None


def test_evidence_writer_records_verdict(tmp_path: Path) -> None:
    backend = _WritingBackend(2)

    def benchmark(concurrency: int, requests: int) -> BenchmarkResult:
        if concurrency == 1:
            return _bench(1, throughput=20.0, mean_ms=1000, p95_ms=1100, requested=requests)
        return _bench(2, throughput=30.0, mean_ms=1100, p95_ms=1250, requested=requests)

    result = asyncio.run(
        run_autonomous_loop(
            backend=backend,
            benchmark=benchmark,
            candidate_concurrencies=(2,),
            requests_per_arm=10,
        )
    )
    path = write_loop_evidence(result, tmp_path / "evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "hyperloom.r9700.autonomous_loop.v1"
    assert payload["acceptance"]["decision"] == "KEEP"
    assert payload["candidate"]["success_rate"] == 1.0
