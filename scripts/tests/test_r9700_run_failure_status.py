"""A completed experiment may REJECT; invalid measurements must not succeed.

Every response and timing in this file is a synthetic offline test fixture.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "r9700_upstream_agent_e2e.py"
spec = importlib.util.spec_from_file_location("r9700_failure_status_target", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def bundle(throughput=20.0, latency=1000.0, failed=False):
    rows = [{
        "round": i + 1, "concurrency": 1 if throughput == 20 else 2,
        "requests": 6, "passed": 0 if failed else 6, "failed": 6 if failed else 0,
        "wall_sec": 1.0, "output_tokens": 0 if failed else 20,
        "total_tokens": 0 if failed else 40,
        "output_tok_s": 0.0 if failed else throughput,
        "total_tok_s": 0.0 if failed else throughput * 2,
        "mean_e2e_ms": math.inf if failed else latency * 0.8,
        "p95_e2e_ms": math.inf if failed else latency,
        "errors": ["synthetic_timeout"] * 6 if failed else [],
    } for i in range(3)]
    return {"rounds": rows, "aggregate": runner._aggregate_rounds(rows)}


def reject_constant(value):
    raise AssertionError("Non-standard JSON constant in evidence: " + value)


def setup_offline(monkeypatch, tmp_path, problem="", latency=1100.0):
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def discover():
        calls.append("discover")
        if problem == "discovery_error":
            raise RuntimeError("synthetic_discovery_failure")
        return "synthetic-offline-model"

    def no_network(*a, **k):
        raise AssertionError("No network is allowed in these offline tests")

    def measure(model, concurrency):
        name = "baseline" if "baseline" not in calls else "candidate"
        calls.append(name)
        result = bundle(20.0 if name == "baseline" else 36.0, 1000.0 if name == "baseline" else latency,
                        failed=problem == name + "_requests")
        if problem == "candidate_nonfinite" and name == "candidate":
            result["aggregate"]["median_output_tok_s"] = math.inf
        if problem == "candidate_counts" and name == "candidate":
            result["aggregate"]["passed"] = 17
        return result

    async def choose(*a, **k):
        calls.append("agent")
        if problem == "agent_error":
            raise RuntimeError("candidate_format_retry_exhausted")
        return {"selected_concurrency": 2, "selected_by_model": True, "hardcoded_candidate": False}

    monkeypatch.setattr(runner, "_discover_model", discover)
    monkeypatch.setattr(runner, "_one_request", lambda *a: {"ok": True})
    monkeypatch.setattr(runner, "_request_json", no_network)
    monkeypatch.setattr(runner, "_benchmark_rounds", measure)
    monkeypatch.setattr(runner, "_agent_choose_candidate", choose)
    return calls


@pytest.mark.parametrize("problem", [
    "baseline_requests", "candidate_requests", "candidate_nonfinite",
    "candidate_counts", "discovery_error", "agent_error",
])
def test_invalid_run_never_reports_success(monkeypatch, tmp_path, capsys, problem):
    calls = setup_offline(monkeypatch, tmp_path, problem)
    assert asyncio.run(runner.main()) == 2
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1], parse_constant=reject_constant)
    assert summary["ok"] is False
    assert summary["run_status"] == "failed"
    files = list((tmp_path / "docs" / "evidence").glob("*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert report["ok"] is False
    assert report["run_status"] == "failed"
    assert report["failure"]["stage"]
    if problem == "baseline_requests":
        assert "agent" not in calls
        assert "candidate" not in calls
    if problem == "discovery_error":
        assert calls == ["discover"]
    if problem == "candidate_nonfinite":
        assert report["candidate"]["aggregate"]["median_output_tok_s"] is None


@pytest.mark.parametrize("latency,verdict", [(1100.0, "KEEP"), (1400.0, "REJECT")])
def test_valid_experiment_completion_is_distinct_from_candidate_verdict(monkeypatch, tmp_path, capsys, latency, verdict):
    setup_offline(monkeypatch, tmp_path, latency=latency)
    assert asyncio.run(runner.main()) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["ok"] is True
    assert summary["run_status"] == "completed"
    assert summary["verdict"] == verdict
    report = json.loads(next((tmp_path / "docs" / "evidence").glob("*.json")).read_text(encoding="utf-8"))
    assert report["failure"] is None
