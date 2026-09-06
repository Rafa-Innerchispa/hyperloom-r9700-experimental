"""Recalculate two recorded live runs; these tests do not perform inference."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCES = (
    "hyperloom_r9700_upstream_autonomous_e2e_20260906T155715Z.json",
    "hyperloom_r9700_upstream_autonomous_e2e_20260906T155832Z.json",
)


@pytest.fixture(params=EVIDENCES)
def report(request):
    # Explicit filenames intentionally fail when evidence is missing.
    return json.loads((ROOT / "docs" / "evidence" / request.param).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["baseline", "candidate"])
def test_rounds_and_rates_are_internally_consistent(report, name):
    arm = report[name]
    rounds = arm["rounds"]
    aggregate = arm["aggregate"]
    expected_concurrency = 1 if name == "baseline" else report["candidate_concurrency"]
    assert len(rounds) == aggregate["round_count"] == 3
    assert [row["round"] for row in rounds] == [1, 2, 3]
    for row in rounds:
        for key in ("requests", "passed", "failed", "output_tokens", "total_tokens"):
            assert type(row[key]) is int and row[key] >= 0
        assert row["requests"] == row["passed"] == 6
        assert row["failed"] == 0 and row["errors"] == []
        assert row["concurrency"] == expected_concurrency
        assert 0 < row["output_tokens"] <= 6 * 32
        assert row["total_tokens"] >= row["output_tokens"]
        for key in ("wall_sec", "output_tok_s", "total_tok_s", "mean_e2e_ms", "p95_e2e_ms"):
            assert type(row[key]) in (int, float)
            assert math.isfinite(row[key]) and row[key] > 0
        assert math.isclose(row["output_tok_s"], row["output_tokens"] / row["wall_sec"], rel_tol=1e-12)
        assert math.isclose(row["total_tok_s"], row["total_tokens"] / row["wall_sec"], rel_tol=1e-12)
    for key, expected in (("requests", 18), ("passed", 18), ("failed", 0)):
        assert aggregate[key] == sum(row[key] for row in rounds) == expected
    expected_metrics = {
        "median_output_tok_s": statistics.median(row["output_tok_s"] for row in rounds),
        "mean_output_tok_s": statistics.fmean(row["output_tok_s"] for row in rounds),
        "min_output_tok_s": min(row["output_tok_s"] for row in rounds),
        "max_output_tok_s": max(row["output_tok_s"] for row in rounds),
        "median_total_tok_s": statistics.median(row["total_tok_s"] for row in rounds),
        "median_mean_e2e_ms": statistics.median(row["mean_e2e_ms"] for row in rounds),
        "median_p95_e2e_ms": statistics.median(row["p95_e2e_ms"] for row in rounds),
    }
    for key, value in expected_metrics.items():
        assert math.isclose(aggregate[key], value, rel_tol=1e-12)


def test_recorded_gate_is_recalculated_independently(report):
    baseline = report["baseline"]["aggregate"]
    candidate = report["candidate"]["aggregate"]
    gain = candidate["median_output_tok_s"] / baseline["median_output_tok_s"] - 1.0
    ratio = candidate["median_p95_e2e_ms"] / baseline["median_p95_e2e_ms"]
    assert report["gate"]["min_gain_fraction"] == 0.10
    assert report["gate"]["max_p95_ratio"] == 1.25
    assert math.isclose(report["gate"]["gain_percent"], gain * 100, rel_tol=1e-12)
    assert math.isclose(report["gate"]["gain_fraction"], gain, rel_tol=1e-12)
    assert math.isclose(report["gate"]["p95_ratio"], ratio, rel_tol=1e-12)
    assert report["verdict"] == ("KEEP" if gain >= 0.10 and ratio <= 1.25 else "REJECT")


def test_recorded_candidate_has_tool_progress_and_bounded_repair(report):
    agent = report["agent"]
    assert agent["selected_concurrency"] == report["candidate_concurrency"]
    assert report["candidate_concurrency"] in (1, 2)
    assert agent["selected_by_model"] is True
    assert agent["hardcoded_candidate"] is False
    assert "tool: write_file" in agent["progress_log"]
    assert [row["valid"] for row in agent["validation_attempts"]] == [False, True]
    baseline_value = report["baseline"]["aggregate"]["median_output_tok_s"]
    candidate_value = report["candidate"]["aggregate"]["median_output_tok_s"]
    assert f"{baseline_value:.4f} tok/s" in agent["decision_context"]
    assert f"{candidate_value:.4f} tok/s" not in agent["decision_context"]
    assert "CONCURRENCY=2" not in agent["decision_context"]
    assert "CONCURRENCY = 2" not in agent["decision_context"]


def test_proxy_run_is_not_misrepresented_as_amd_worktree_acceptance(report):
    assert report["execution_scope"] == "primary_orchestrated_existing_proxy_not_amd_worktree_acceptance"
    assert report["hardware_attested_by_runner"] is False
    assert report["shell_exposed"] is False
    assert report["cdna_specific_paths_used"] is False
    assert "not official" in report["support_status"]
