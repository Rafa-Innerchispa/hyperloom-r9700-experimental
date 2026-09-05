from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "r9700_upstream_agent_e2e.py"
SPEC = importlib.util.spec_from_file_location("r9700_upstream_agent_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
E2E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E2E)


def _round(output_tok_s: float, p95_ms: float, *, failed: int = 0) -> dict:
    passed = E2E.REQUESTS_PER_ARM - failed
    return {
        "round": 1,
        "concurrency": 1,
        "requests": E2E.REQUESTS_PER_ARM,
        "passed": passed,
        "failed": failed,
        "wall_sec": 1.0,
        "output_tokens": 100,
        "total_tokens": 200,
        "output_tok_s": output_tok_s,
        "total_tok_s": output_tok_s * 2,
        "mean_e2e_ms": p95_ms * 0.8,
        "p95_e2e_ms": p95_ms,
        "errors": (["boom"] if failed else []),
    }


def test_aggregate_rounds_uses_medians_and_preserves_failures():
    aggregate = E2E._aggregate_rounds(
        [
            _round(20.0, 900.0),
            _round(22.0, 950.0),
            _round(100.0, 5000.0, failed=1),
        ]
    )
    assert aggregate["round_count"] == 3
    assert aggregate["median_output_tok_s"] == 22.0
    assert aggregate["median_p95_e2e_ms"] == 950.0
    assert aggregate["failed"] == 1
    assert aggregate["requests"] == 3 * E2E.REQUESTS_PER_ARM


def test_verdict_keeps_only_when_throughput_and_latency_gates_pass():
    baseline = {
        "failed": 0,
        "median_output_tok_s": 20.0,
        "median_p95_e2e_ms": 1000.0,
    }
    candidate = {
        "failed": 0,
        "median_output_tok_s": 30.0,
        "median_p95_e2e_ms": 1200.0,
    }
    verdict, gate = E2E._verdict(baseline, candidate)
    assert verdict == "KEEP"
    assert gate["gain_percent"] == 50.0
    assert gate["p95_ratio"] == 1.2

    candidate["median_p95_e2e_ms"] = 1300.0
    verdict, gate = E2E._verdict(baseline, candidate)
    assert verdict == "REJECT"
    assert gate["p95_ratio"] == 1.3


def test_verdict_fails_closed_on_any_request_failure():
    baseline = {
        "failed": 0,
        "median_output_tok_s": 20.0,
        "median_p95_e2e_ms": 1000.0,
    }
    candidate = {
        "failed": 1,
        "median_output_tok_s": 40.0,
        "median_p95_e2e_ms": 900.0,
    }
    verdict, gate = E2E._verdict(baseline, candidate)
    assert verdict == "REJECT"
    assert gate["reason"] == "request_failure"


def test_decision_context_contains_baseline_but_does_not_force_candidate_two():
    baseline = {
        "round_count": 3,
        "median_output_tok_s": 21.2345,
        "median_p95_e2e_ms": 987.65,
        "failed": 0,
    }
    text = E2E._decision_context(baseline)
    assert "21.2345 tok/s" in text
    assert "987.65 ms" in text
    assert "bounded set [1, 2]" in text
    assert "Make the choice yourself" in text
    assert "CONCURRENCY = 2" not in text
    assert "content should be exactly" not in text.lower()


def test_benchmark_rounds_calls_each_round_and_aggregates(monkeypatch):
    seen = []

    def fake_benchmark(model: str, concurrency: int, *, round_index: int = 0):
        seen.append((model, concurrency, round_index))
        row = _round(20.0 + round_index, 900.0 + 10.0 * round_index)
        row["round"] = round_index + 1
        row["concurrency"] = concurrency
        return row

    monkeypatch.setattr(E2E, "_benchmark", fake_benchmark)
    result = E2E._benchmark_rounds("qwen-test", 2, rounds=3)
    assert seen == [
        ("qwen-test", 2, 0),
        ("qwen-test", 2, 1),
        ("qwen-test", 2, 2),
    ]
    assert result["aggregate"]["median_output_tok_s"] == 21.0
    assert result["aggregate"]["median_p95_e2e_ms"] == 910.0
