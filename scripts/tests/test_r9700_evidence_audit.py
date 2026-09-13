from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = SCRIPTS / "r9700_evidence_audit.py"
spec = importlib.util.spec_from_file_location("r9700_evidence_audit", AUDIT_SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sample(index=0, elapsed=0.01, ttft=0.005):
    return {
        "ok": True,
        "correctness_passed": True,
        "elapsed_sec": elapsed,
        "ttft_sec": ttft,
        "completion_tokens": 10,
        "prompt_tokens": 5,
        "text_len": 3,
        "request_index": index,
    }


def round_row(round_index=0, concurrency=1):
    samples = [sample(round_index * 6 + i) for i in range(6)]
    return {
        "round": round_index + 1,
        "concurrency": concurrency,
        "requests": 6,
        "passed": 6,
        "failed": 0,
        "wall_sec": 0.1,
        "output_tokens": 60,
        "total_tokens": 90,
        "output_tok_s": 600.0,
        "total_tok_s": 900.0,
        "mean_e2e_ms": 10.0,
        "p95_e2e_ms": 10.0,
        "mean_ttft_ms": 5.0,
        "p95_ttft_ms": 5.0,
        "errors": [],
        "samples": samples,
    }


def arm(concurrency=1):
    rounds = [round_row(i, concurrency) for i in range(3)]
    return {
        "rounds": rounds,
        "aggregate": {
            "round_count": 3,
            "requests": 18,
            "passed": 18,
            "failed": 0,
            "median_output_tok_s": 600.0,
            "mean_output_tok_s": 600.0,
            "min_output_tok_s": 600.0,
            "max_output_tok_s": 600.0,
            "median_total_tok_s": 900.0,
            "median_mean_e2e_ms": 10.0,
            "median_p95_e2e_ms": 10.0,
            "median_mean_ttft_ms": 5.0,
            "median_p95_ttft_ms": 5.0,
        },
    }


def fixture_report():
    return {
        "sample_schema": audit.SAMPLE_SCHEMA,
        "runner_sha256": "a" * 64,
        "baseline": arm(1),
        "candidate": arm(2),
    }


def test_valid_fixture_passes():
    assert audit.audit_report(fixture_report())["ok"] is True


def test_duplicate_or_missing_sample_index_fails_closed():
    report = fixture_report()
    report["baseline"]["rounds"][0]["samples"][0]["request_index"] = 1
    assert audit.audit_report(report)["ok"] is False


def test_invalid_ttft_fails_closed():
    report = fixture_report()
    report["candidate"]["rounds"][0]["samples"][0]["ttft_sec"] = 0.02
    assert audit.audit_report(report)["ok"] is False


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
        return {
            "ok": True,
            "correctness_passed": True,
            "elapsed_sec": 0.01,
            "ttft_sec": 0.005,
            "completion_tokens": 10,
            "prompt_tokens": 5,
            "text_len": 3,
        }

    monkeypatch.setattr(runner, "_one_request", one_request)
    row = runner._benchmark("synthetic", 2, round_index=2)
    assert [sample["request_index"] for sample in row["samples"]] == list(range(12, 18))
    assert len(row["samples"]) == row["passed"] == row["requests"] == 6
