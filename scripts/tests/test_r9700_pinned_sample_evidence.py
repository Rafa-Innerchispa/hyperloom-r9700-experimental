"""Recheck captured live evidence without any model, GPU, or network calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("r9700_pinned_audit", ROOT / "scripts/r9700_evidence_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
REPORT = ROOT / "docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260906T170726136772Z.json"
REPORT_DIGEST = "0e8c626ee1c69a84f454e373aea5ae0edd17d25ccbfe4a909375a00ef8347034"
RUNNER_DIGEST = "311be5b6bf34c3baa0d77e49f43cb1edff565a386855dbab0d66510649d6374a"


def test_captured_run_hash_and_all_36_samples_replay():
    report, digest = audit.load_report(REPORT, expected_sha256=REPORT_DIGEST)
    result = audit.audit_report(report, expected_runner_sha256=RUNNER_DIGEST)
    assert digest == REPORT_DIGEST
    assert result["ok"] is True
    assert result["samples_verified"] == 36
    assert result["recomputed_verdict"] == "KEEP"
    assert result["expected_runner_digest_checked"] is True
    assert result["physical_execution_verified"] is False


def test_captured_run_rejects_in_memory_latency_tampering():
    report, _ = audit.load_report(REPORT, expected_sha256=REPORT_DIGEST)
    report["candidate"]["rounds"][0]["samples"][0]["elapsed_sec"] *= 2
    assert audit.audit_report(report)["ok"] is False


def test_legacy_aggregate_only_report_is_not_sample_level_proof():
    path = ROOT / "docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260906T165116889777Z.json"
    report, _ = audit.load_report(path)
    result = audit.audit_report(report)
    assert result["ok"] is False
    assert result["reason"] == "report:sample_schema_unsupported"
