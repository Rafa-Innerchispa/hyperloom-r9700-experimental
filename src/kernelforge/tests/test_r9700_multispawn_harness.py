from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "r9700_multispawn_harness.py"
SPEC = importlib.util.spec_from_file_location("r9700_multispawn_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _spawn(path: str, *, rc: int = 0, ok: bool = True, gain: float = 20.0) -> dict:
    return {
        "returncode": rc,
        "summary": {
            "ok": ok,
            "evidence": path,
            "baseline_median_output_tok_s": 20.0,
            "candidate_median_output_tok_s": 30.0,
            "gain_percent": gain,
            "p95_ratio": 1.1,
        },
    }


def test_build_plan_is_blocked_by_default_until_benchmark_window() -> None:
    plan = HARNESS.build_plan(spawn_count=5)
    assert plan["schema"] == HARNESS.SCHEMA
    assert plan["spawn_count"] == 5
    assert plan["live_execution_gate"]["default"] == "blocked"
    assert plan["live_execution_gate"]["requires_service_restart"] is False


def test_audit_spawn_report_accepts_unique_complete_successes() -> None:
    report = {"spawns": [_spawn("a.json"), _spawn("b.json")]}
    audit = HARNESS.audit_spawn_report(report)
    assert audit == {"ok": True, "spawn_count": 2, "evidence_paths": ["a.json", "b.json"]}


def test_audit_spawn_report_fails_closed_on_duplicate_or_invalid_metrics() -> None:
    assert HARNESS.audit_spawn_report({"spawns": [_spawn("a.json"), _spawn("a.json")]})["reason"] == "duplicate_evidence_path"
    assert HARNESS.audit_spawn_report({"spawns": [_spawn("a.json", rc=2)]})["reason"] == "spawn_failed"
    bad = _spawn("bad.json")
    bad["summary"]["gain_percent"] = float("nan")
    assert HARNESS.audit_spawn_report({"spawns": [bad]})["reason"] == "invalid_metric"
