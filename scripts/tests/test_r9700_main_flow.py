"""Offline control-flow tests. Synthetic timings here are NOT live GPU evidence."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "r9700_upstream_agent_e2e.py"
spec = importlib.util.spec_from_file_location("r9700_main_flow_target", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("selection,expected", [(1, "REJECT"), (2, "KEEP")])
def test_actual_main_baseline_then_autonomous_choice_then_candidate(monkeypatch, tmp_path, selection, expected):
    events = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_discover_model", lambda: "synthetic-offline-test-only")
    monkeypatch.setattr(runner, "_one_request", lambda *a: {"ok": True})

    def no_network(*args, **kwargs):
        raise AssertionError("offline tests must not contact an inference service")

    monkeypatch.setattr(runner, "_request_json", no_network)

    def benchmark(model, concurrency, *, round_index=0):
        events.append(("benchmark", concurrency, round_index))
        throughput = 20.0 if concurrency == 1 else 36.0
        latency = 1000.0 if concurrency == 1 else 1200.0
        return {
            "round": round_index + 1, "concurrency": concurrency,
            "requests": 6, "passed": 6, "failed": 0,
            "wall_sec": 1.0, "output_tokens": int(throughput),
            "total_tokens": int(throughput * 2),
            "output_tok_s": throughput, "total_tok_s": throughput * 2,
            "mean_e2e_ms": latency * 0.8, "p95_e2e_ms": latency,
            "errors": [],
        }

    monkeypatch.setattr(runner, "_benchmark", benchmark)

    def factory(config, program, **kwargs):
        assert config.agent_backend == "local-openai"
        assert config.agent_fallback_provider == ""
        assert config.agent_options["allow_shell"] is False
        assert config.agent_options["enabled_tools"] == ["write_file"]
        assert config.agent_options["tool_mode"] == "json"
        assert kwargs["profiling_enabled"] is False
        assert kwargs["insession_gate"] is False

        async def choose(candidate_path, instruction, sink):
            assert events == [("benchmark", 1, i) for i in range(3)]
            assert "20.0000 tok/s" in instruction
            assert "36.0000" not in instruction
            assert "CONCURRENCY = 2" not in instruction
            events.append(("agent", selection))
            Path(candidate_path).write_text(
                f"# offline synthetic test candidate\nCONCURRENCY = {selection}\n",
                encoding="utf-8",
            )
            sink["plan"] = "Synthetic unit test, not model inference."
            return "PLAN: offline test SUBMIT_CANDIDATE"

        return choose

    monkeypatch.setattr(runner, "make_agent_fn", factory)
    assert asyncio.run(runner.main()) == 0
    assert events == (
        [("benchmark", 1, i) for i in range(3)]
        + [("agent", selection)]
        + [("benchmark", selection, i) for i in range(3)]
    )
    paths = list((tmp_path / "docs" / "evidence").glob("*.json"))
    assert len(paths) == 1
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    assert report["verdict"] == expected
    assert report["candidate_concurrency"] == selection
    assert report["baseline"]["aggregate"]["requests"] == 18
    assert report["candidate"]["aggregate"]["requests"] == 18
    assert report["shell_exposed"] is False
    assert report["cdna_specific_paths_used"] is False
    assert report["hardware_attested_by_runner"] is False
    assert "independent physical attestation" in report["hardware_claim"]
    assert report["orchestrator_host"]
    assert "not official" in report["support_status"]
