"""Bounded format repair must not replace the model's optimization choice."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "r9700_upstream_agent_e2e.py"
spec = importlib.util.spec_from_file_location("r9700_recovery_target", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
BASELINE = {"round_count": 3, "median_output_tok_s": 20.0, "median_p95_e2e_ms": 1000.0, "failed": 0}


@pytest.mark.parametrize("chosen", [1, 2])
def test_one_format_repair_preserves_model_choice(monkeypatch, tmp_path, chosen):
    calls = []

    def factory(*args, **kwargs):
        async def agent(path, instruction, sink):
            calls.append(instruction)
            text = f"# CONCURRENCY={chosen}\n" if len(calls) == 1 else f"CONCURRENCY = {chosen}\n"
            Path(path).write_text(text, encoding="utf-8")
            return "SUBMIT_CANDIDATE"
        return agent

    monkeypatch.setattr(runner, "make_agent_fn", factory)
    result = asyncio.run(runner._agent_choose_candidate("offline", tmp_path / "candidate.py", BASELINE))
    assert len(calls) == 2
    assert result["selected_concurrency"] == chosen
    assert result["selected_by_model"] is True
    assert result["hardcoded_candidate"] is False
    assert result["format_repair_used"] is True
    assert [row["valid"] for row in result["validation_attempts"]] == [False, True]
    assert "format-repair attempt remains" in calls[1]
    assert "CONCURRENCY = 2" not in calls[1]
    assert "20.0000 tok/s" in calls[1]


def test_repeated_invalid_candidate_stops_after_two_attempts(monkeypatch, tmp_path):
    calls = []

    def factory(*args, **kwargs):
        async def agent(path, instruction, sink):
            calls.append(instruction)
            Path(path).write_text("# no executable assignment\n", encoding="utf-8")
            return "SUBMIT_CANDIDATE"
        return agent

    monkeypatch.setattr(runner, "make_agent_fn", factory)
    with pytest.raises(RuntimeError, match="candidate_format_retry_exhausted"):
        asyncio.run(runner._agent_choose_candidate("offline", tmp_path / "candidate.py", BASELINE))
    assert len(calls) == 2


@pytest.mark.parametrize("chosen", [1, 2])
def test_valid_first_choice_needs_no_repair(monkeypatch, tmp_path, chosen):
    calls = []

    def factory(*args, **kwargs):
        async def agent(path, instruction, sink):
            calls.append(instruction)
            Path(path).write_text(f"CONCURRENCY = {chosen}\n", encoding="utf-8")
            return "SUBMIT_CANDIDATE"
        return agent

    monkeypatch.setattr(runner, "make_agent_fn", factory)
    result = asyncio.run(runner._agent_choose_candidate("offline", tmp_path / "candidate.py", BASELINE))
    assert len(calls) == 1
    assert result["selected_concurrency"] == chosen
    assert result["format_repair_used"] is False
    assert result["validation_attempts"] == [{"attempt": 1, "valid": True}]
