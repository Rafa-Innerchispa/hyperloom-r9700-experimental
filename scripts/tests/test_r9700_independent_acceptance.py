"""Independent, offline acceptance regressions for the real R9700 E2E runner."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "r9700_upstream_agent_e2e.py"
spec = importlib.util.spec_from_file_location("r9700_independent_target", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(runner)


def response(text="AMD validates this local response.", completion_tokens=8, prompt_tokens=12):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
    }


def arm(throughput=20.0, latency=1000.0):
    return {
        "round_count": 3, "requests": 18, "passed": 18, "failed": 0,
        "median_output_tok_s": throughput, "median_p95_e2e_ms": latency,
    }


@pytest.mark.parametrize("payload", [
    {},
    response(text=""),
    response(text="   "),
    response(text="An unrelated answer."),
    response(text=["AMD"]),
    response(completion_tokens=0),
    response(completion_tokens=-2),
    response(completion_tokens=True),
    response(completion_tokens=2.5),
    response(prompt_tokens=-1),
    {"choices": [], "usage": {"completion_tokens": 8, "prompt_tokens": 12}},
])
def test_invalid_model_responses_are_failures(monkeypatch, payload):
    monkeypatch.setattr(runner, "_request_json", lambda *a, **k: payload)
    row = runner._one_request("local-model", 0)
    assert row["ok"] is False, "HTTP success is not valid benchmark output"
    assert row["completion_tokens"] == 0


def test_valid_model_response_is_counted(monkeypatch):
    monkeypatch.setattr(runner, "_request_json", lambda *a, **k: response())
    row = runner._one_request("local-model", 0)
    assert row["ok"] is True
    assert row["completion_tokens"] == 8


@pytest.mark.parametrize("which,key,value", [
    ("candidate", "median_output_tok_s", math.inf),
    ("candidate", "median_output_tok_s", math.nan),
    ("baseline", "median_output_tok_s", math.nan),
    ("baseline", "median_p95_e2e_ms", math.inf),
    ("candidate", "median_p95_e2e_ms", -1.0),
    ("candidate", "median_p95_e2e_ms", 0.0),
    ("candidate", "median_p95_e2e_ms", math.nan),
    ("candidate", "passed", 17),
    ("candidate", "requests", 0),
    ("baseline", "round_count", 1),
])
def test_gate_rejects_invalid_or_incomplete_evidence(which, key, value):
    baseline, candidate = arm(), arm(36.0, 1100.0)
    (baseline if which == "baseline" else candidate)[key] = value
    verdict, _ = runner._verdict(baseline, candidate)
    assert verdict == "REJECT"


def test_gate_rejects_missing_evidence_without_crashing():
    assert runner._verdict({}, {})[0] == "REJECT"


def test_gate_keeps_measured_gain_inside_latency_bound():
    assert runner._verdict(arm(), arm(36.0, 1200.0))[0] == "KEEP"


def test_gate_rejects_fast_candidate_outside_latency_bound():
    assert runner._verdict(arm(), arm(36.0, 1347.5))[0] == "REJECT"


def test_gate_rejects_any_failed_request():
    candidate = arm(36.0, 1100.0)
    candidate.update(passed=17, failed=1)
    assert runner._verdict(arm(), candidate)[0] == "REJECT"


@pytest.mark.parametrize("text", [
    "CONCURRENCY = 1\nCONCURRENCY = 2\n",
    "CONCURRENCY = 2\nprint('unexpected executable content')\n",
    "CONCURRENCY = 0\n",
    "CONCURRENCY = 3\n",
])
def test_candidate_must_be_one_bounded_assignment(tmp_path, text):
    target = tmp_path / "candidate.py"
    target.write_text(text, encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        runner._candidate_from_file(target)


@pytest.mark.parametrize("value", [1, 2])
def test_both_model_choices_are_allowed(tmp_path, value):
    target = tmp_path / "candidate.py"
    target.write_text(f"# experiment\nCONCURRENCY = {value}\n", encoding="utf-8")
    assert runner._candidate_from_file(target) == value


def test_context_exposes_baseline_without_prescribing_candidate():
    context = runner._decision_context(arm())
    assert "20.0000" in context
    assert "[1, 2]" in context
    assert "CONCURRENCY = 2" not in context
    assert "CONCURRENCY=2" not in context


def test_default_measurement_has_three_rounds_and_shell_disabled():
    assert runner.MEASUREMENT_ROUNDS == 3
    assert runner.ALLOWED_CANDIDATES == {1, 2}
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"allow_shell": False' in source
    assert '"enabled_tools": ["write_file"]' in source
