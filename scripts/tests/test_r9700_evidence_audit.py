"""Synthetic adversarial fixtures for offline metric replay, not GPU evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import socket
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r9700_audit_test_target", SCRIPTS / "r9700_evidence_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_arm(concurrency, tokens=10):
    rounds = []
    for index in range(3):
        samples = [
            {"request_index": index * 6 + i, "ok": True, "correctness_passed": True,
             "elapsed_sec": (i + 1) / 10, "completion_tokens": tokens,
             "prompt_tokens": 5, "text_len": 3}
            for i in range(6)
        ]
        rounds.append({
            "round": index + 1, "concurrency": concurrency, "samples": samples,
            "requests": 6, "passed": 6, "failed": 0, "errors": [], "wall_sec": 4.0,
            "output_tokens": tokens * 6, "total_tokens": tokens * 6 + 30,
            "output_tok_s": tokens * 1.5, "total_tok_s": tokens * 1.5 + 7.5,
            "mean_e2e_ms": 350.0, "p95_e2e_ms": 575.0,
        })
    return {"rounds": rounds, "aggregate": {
        "round_count": 3, "requests": 18, "passed": 18, "failed": 0,
        "median_output_tok_s": tokens * 1.5, "mean_output_tok_s": tokens * 1.5,
        "min_output_tok_s": tokens * 1.5, "max_output_tok_s": tokens * 1.5,
        "median_total_tok_s": tokens * 1.5 + 7.5,
        "median_mean_e2e_ms": 350.0, "median_p95_e2e_ms": 575.0,
    }}


def fixture_report():
    return {
        "sample_schema": audit.SAMPLE_SCHEMA, "ok": True, "run_status": "completed", "failure": None,
        "timestamp_utc": "2026-09-06T10:00:00+00:00", "finished_at_utc": "2026-09-06T10:01:00+00:00",
        "runner_source_sha256_start": "a" * 64, "runner_source_sha256_end": "a" * 64,
        "hardware_attested_by_runner": False, "shell_exposed": False, "cdna_specific_paths_used": False,
        "agent_backend": "local-openai", "tool_mode": "json", "model": "synthetic-not-a-real-model",
        "execution_scope": "unattested_runtime", "measurement_rounds": 3, "requests_per_round": 6,
        "baseline_concurrency": 1, "candidate_concurrency": 2, "allowed_candidates": [1, 2],
        "baseline": fixture_arm(1), "candidate": fixture_arm(2, tokens=20),
        "agent": {"selected_by_model": True, "hardcoded_candidate": False, "selected_concurrency": 2,
                  "progress_log": ["tool: write_file"], "decision_context": "15.0000 tok/s. Choose [1, 2]."},
        "gate": {"gain_fraction": 1.0, "gain_percent": 100.0, "p95_ratio": 1.0,
                 "min_gain_fraction": 0.10, "max_p95_ratio": 1.25,
                 "throughput_metric": "median_output_tok_s", "latency_metric": "median_p95_e2e_ms"},
        "verdict": "KEEP",
    }


def test_valid_samples_replay_without_network(monkeypatch):
    def prohibited(*a, **k):
        raise AssertionError("offline audit must not access network")
    monkeypatch.setattr(socket, "socket", prohibited)
    result = audit.audit_report(fixture_report(), expected_runner_sha256="a" * 64)
    assert result["ok"] is True
    assert result["samples_verified"] == 36
    assert result["recomputed_verdict"] == "KEEP"
    assert result["physical_execution_verified"] is False
    assert result["authentication_verified"] is False
    assert result["global_fabric_verified"] is False


def test_valid_reject_is_successfully_audited():
    report = fixture_report()
    report["candidate"] = fixture_arm(1)
    report["candidate_concurrency"] = report["agent"]["selected_concurrency"] = 1
    report["gate"].update(gain_fraction=0.0, gain_percent=0.0)
    report["verdict"] = "REJECT"
    result = audit.audit_report(report)
    assert result["ok"] is True
    assert result["recomputed_verdict"] == "REJECT"


@pytest.mark.parametrize("path,value", [
    (("sample_schema",), "legacy-without-samples"),
    (("ok",), False), (("run_status",), "failed"), (("failure",), {"reason": "failed"}),
    (("hardware_attested_by_runner",), True), (("shell_exposed",), True),
    (("cdna_specific_paths_used",), True), (("execution_scope",), "claimed_amd_worktree_pass"),
    (("runner_source_sha256_end",), "b" * 64), (("runner_source_sha256_start",), "invalid"),
    (("finished_at_utc",), "2026-09-06T09:00:00+00:00"),
    (("finished_at_utc",), "2026-09-06T10:00:01+00:00"),
    (("timestamp_utc",), "2026-09-06T10:00:00"),
    (("measurement_rounds",), True), (("requests_per_round",), 0),
    (("allowed_candidates",), [True, 2]), (("candidate_concurrency",), 3),
    (("baseline", "rounds", 0, "samples"), []),
    (("baseline", "rounds", 0, "samples", 0, "request_index"), 1),
    (("baseline", "rounds", 0, "samples", 0, "elapsed_sec"), math.nan),
    (("baseline", "rounds", 0, "samples", 0, "elapsed_sec"), math.inf),
    (("baseline", "rounds", 0, "samples", 0, "elapsed_sec"), 0),
    (("baseline", "rounds", 0, "samples", 0, "elapsed_sec"), 5.0),
    (("baseline", "rounds", 0, "samples", 0, "ok"), False),
    (("baseline", "rounds", 0, "samples", 0, "correctness_passed"), False),
    (("baseline", "rounds", 0, "samples", 0, "completion_tokens"), True),
    (("baseline", "rounds", 0, "samples", 0, "completion_tokens"), -1),
    (("baseline", "rounds", 0, "samples", 0, "prompt_tokens"), 1.2),
    (("baseline", "rounds", 0, "p95_e2e_ms"), 1.0),
    (("baseline", "rounds", 0, "output_tok_s"), 999.0),
    (("candidate", "aggregate", "median_p95_e2e_ms"), 1.0),
    (("candidate", "aggregate", "median_output_tok_s"), 999.0),
    (("candidate", "aggregate", "requests"), 17),
    (("agent", "hardcoded_candidate"), True), (("agent", "progress_log"), []),
    (("agent", "selected_concurrency"), 1), (("agent", "decision_context"), "999.0000 tok/s"),
    (("gate", "max_p95_ratio"), 2.0), (("gate", "min_gain_fraction"), 0.0),
    (("gate", "gain_percent"), 999.0), (("verdict",), "REJECT"),
])
def test_tampered_or_incomplete_evidence_is_rejected(path, value):
    report = fixture_report()
    cursor = report
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    result = audit.audit_report(report)
    assert result["ok"] is False
    assert result["metric_replay_verified"] is False
    assert result["reason"]


def test_expected_source_digest_is_checked():
    assert audit.audit_report(fixture_report(), expected_runner_sha256="b" * 64)["ok"] is False


@pytest.mark.parametrize("value", [None, {}, [], {"sample_schema": audit.SAMPLE_SCHEMA}])
def test_missing_documents_fail_closed(value):
    assert audit.audit_report(value)["ok"] is False


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_strict_json_load_rejects_ambiguous_numbers_or_keys(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        audit.load_report(path)


def test_trusted_digest_detects_file_change(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture_report()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report, actual = audit.load_report(path, expected_sha256=digest)
    assert actual == digest
    assert audit.audit_report(report)["ok"] is True
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="sha256_mismatch"):
        audit.load_report(path, expected_sha256=digest)


def test_report_size_is_bounded(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (audit.MAX_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match="too_large"):
        audit.load_report(path)


def test_runner_retains_every_sample_index(monkeypatch):
    target_spec = importlib.util.spec_from_file_location("r9700_capture_test_target", SCRIPTS / "r9700_upstream_agent_e2e.py")
    runner = importlib.util.module_from_spec(target_spec)
    target_spec.loader.exec_module(runner)
    def one_request(model, index):
        return {"ok": True, "correctness_passed": True, "elapsed_sec": 0.01,
                "completion_tokens": 10, "prompt_tokens": 5, "text_len": 3}
    monkeypatch.setattr(runner, "_one_request", one_request)
    row = runner._benchmark("synthetic", 2, round_index=2)
    assert [sample["request_index"] for sample in row["samples"]] == list(range(12, 18))
    assert len(row["samples"]) == row["passed"] == row["requests"] == 6
