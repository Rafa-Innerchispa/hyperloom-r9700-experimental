#!/usr/bin/env python3
"""Offline replay of R9700 request metrics. Never executes code or uses a network.

PASS means arithmetic/contract consistency, not hardware attestation, statistical
significance, or authenticity. A caller-supplied trusted digest adds integrity
checking; a digest computed from the same file alone cannot prove provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

SAMPLE_SCHEMA = "r9700-request-metrics-v1"
MAX_REPORT_BYTES = 1024 * 1024
ROUNDS = 3
REQUESTS = 6
MIN_GAIN = 0.10
MAX_P95_RATIO = 1.25


class InvalidEvidence(ValueError):
    """Contains only a controlled field code, never arbitrary report content."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise InvalidEvidence(code)


def number(value, code: str, *, positive: bool = True) -> float:
    require(type(value) in (int, float), code + ":type")
    result = float(value)
    require(math.isfinite(result), code + ":finite")
    if positive:
        require(result > 0, code + ":positive")
    return result


def integer(value, code: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, code + ":integer")
    return value


def equal_number(actual, expected: float, code: str) -> None:
    value = number(actual, code, positive=False)
    require(math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9), code + ":mismatch")


def timestamp(value, code: str) -> datetime:
    require(isinstance(value, str), code + ":type")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None, code + ":timezone")
    return result


def replay_arm(bundle: dict, concurrency: int, name: str) -> dict:
    require(isinstance(bundle, dict), name + ":object")
    rounds = bundle.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == ROUNDS, name + ":rounds")
    computed = []
    for index, row in enumerate(rounds):
        key = f"{name}.round{index + 1}"
        require(isinstance(row, dict), key + ":object")
        require(integer(row.get("round"), key) == index + 1, key + ":order")
        require(integer(row.get("concurrency"), key) == concurrency, key + ":concurrency")
        samples = row.get("samples")
        require(isinstance(samples, list) and len(samples) == REQUESTS, key + ":samples_missing_or_incomplete")
        latencies, indexes = [], []
        output_tokens = prompt_tokens = 0
        wall = number(row.get("wall_sec"), key + ".wall_sec")
        for sample in samples:
            require(isinstance(sample, dict), key + ":sample_object")
            indexes.append(integer(sample.get("request_index"), key + ".request_index"))
            require(sample.get("ok") is True and sample.get("correctness_passed") is True, key + ":failed_sample")
            require("error" not in sample, key + ":sample_error")
            elapsed = number(sample.get("elapsed_sec"), key + ".elapsed_sec")
            require(elapsed <= wall + 1e-6, key + ":latency_exceeds_round_wall")
            latencies.append(elapsed * 1000)
            output_tokens += integer(sample.get("completion_tokens"), key + ".completion_tokens", minimum=1)
            prompt_tokens += integer(sample.get("prompt_tokens"), key + ".prompt_tokens")
            integer(sample.get("text_len"), key + ".text_len", minimum=1)
        require(sorted(indexes) == list(range(index * REQUESTS, (index + 1) * REQUESTS)), key + ":duplicate_or_missing_index")
        for field, expected in (("requests", REQUESTS), ("passed", REQUESTS), ("failed", 0),
                                ("output_tokens", output_tokens), ("total_tokens", output_tokens + prompt_tokens)):
            require(integer(row.get(field), key + "." + field) == expected, key + "." + field + ":mismatch")
        require(row.get("errors") == [], key + ":errors")
        values = {
            "output_tok_s": output_tokens / wall,
            "total_tok_s": (output_tokens + prompt_tokens) / wall,
            "mean_e2e_ms": statistics.fmean(latencies),
            "p95_e2e_ms": statistics.quantiles(latencies, n=100, method="inclusive")[94],
        }
        for field, expected in values.items():
            equal_number(row.get(field), expected, key + "." + field)
        computed.append(values)
    rates = [row["output_tok_s"] for row in computed]
    rebuilt = {
        "round_count": ROUNDS, "requests": ROUNDS * REQUESTS,
        "passed": ROUNDS * REQUESTS, "failed": 0,
        "median_output_tok_s": statistics.median(rates),
        "mean_output_tok_s": statistics.fmean(rates),
        "min_output_tok_s": min(rates), "max_output_tok_s": max(rates),
        "median_total_tok_s": statistics.median(row["total_tok_s"] for row in computed),
        "median_mean_e2e_ms": statistics.median(row["mean_e2e_ms"] for row in computed),
        "median_p95_e2e_ms": statistics.median(row["p95_e2e_ms"] for row in computed),
    }
    aggregate = bundle.get("aggregate")
    require(isinstance(aggregate, dict), name + ":aggregate")
    for field, expected in rebuilt.items():
        if type(expected) is int:
            require(integer(aggregate.get(field), name + "." + field) == expected, name + "." + field + ":mismatch")
        else:
            equal_number(aggregate.get(field), expected, name + "." + field)
    return rebuilt


def audit_report(report: dict, *, expected_runner_sha256: str | None = None) -> dict:
    result = {
        "schema": "r9700-offline-evidence-audit-v1", "ok": False,
        "metric_replay_verified": False, "physical_execution_verified": False,
        "global_fabric_verified": False,
        "authentication_verified": False,
        "scope": "offline consistency check only; no network or model invocation",
    }
    try:
        require(isinstance(report, dict), "report:object")
        require(report.get("sample_schema") == SAMPLE_SCHEMA, "report:sample_schema_unsupported")
        require(report.get("ok") is True and report.get("run_status") == "completed", "report:run_not_completed")
        require(report.get("failure") is None, "report:failure_present")
        require(report.get("hardware_attested_by_runner") is False, "report:unsupported_hardware_claim")
        require(report.get("shell_exposed") is False and report.get("cdna_specific_paths_used") is False, "report:unsafe_execution_flags")
        require(report.get("agent_backend") == "local-openai" and report.get("tool_mode") == "json", "report:backend_contract")
        require(isinstance(report.get("model"), str) and bool(report["model"]), "report:model_missing")
        require(report.get("execution_scope") in {"primary_orchestrated_existing_proxy_not_amd_worktree_acceptance", "unattested_runtime"}, "report:scope_not_supported")
        start = timestamp(report.get("timestamp_utc"), "report.start")
        end = timestamp(report.get("finished_at_utc"), "report.end")
        require(end >= start, "report:time_order")
        source = report.get("runner_source_sha256_start")
        require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{64}", source) is not None, "report:source_digest")
        require(report.get("runner_source_sha256_end") == source, "report:runner_changed")
        if expected_runner_sha256 is not None:
            require(source == expected_runner_sha256, "report:unexpected_runner_digest")
        require(integer(report.get("measurement_rounds"), "report.rounds") == ROUNDS, "report:round_count")
        require(integer(report.get("requests_per_round"), "report.requests") == REQUESTS, "report:request_count")
        require(integer(report.get("baseline_concurrency"), "report.baseline_concurrency") == 1, "report:baseline_concurrency")
        allowed = report.get("allowed_candidates")
        require(isinstance(allowed, list) and all(type(x) is int for x in allowed) and allowed == [1, 2], "report:candidate_allowlist")
        selected = integer(report.get("candidate_concurrency"), "report.candidate_concurrency")
        require(selected in (1, 2), "report:candidate_out_of_bounds")
        baseline = replay_arm(report.get("baseline"), 1, "baseline")
        candidate = replay_arm(report.get("candidate"), selected, "candidate")
        require((end - start).total_seconds() + 1 >= sum(row["wall_sec"] for name in ("baseline", "candidate") for row in report[name]["rounds"]), "report:insufficient_elapsed_time")
        agent = report.get("agent")
        require(isinstance(agent, dict), "agent:missing")
        require(agent.get("selected_by_model") is True and agent.get("hardcoded_candidate") is False, "agent:choice_contract")
        require(type(agent.get("selected_concurrency")) is int and agent["selected_concurrency"] == selected, "agent:selection_mismatch")
        progress = agent.get("progress_log")
        require(isinstance(progress, list) and "tool: write_file" in progress, "agent:write_not_recorded")
        context = agent.get("decision_context")
        require(isinstance(context, str) and f"{baseline['median_output_tok_s']:.4f} tok/s" in context, "agent:baseline_context")
        require("CONCURRENCY = 2" not in context and "CONCURRENCY=2" not in context, "agent:forced_candidate_literal")
        gain = candidate["median_output_tok_s"] / baseline["median_output_tok_s"] - 1
        ratio = candidate["median_p95_e2e_ms"] / baseline["median_p95_e2e_ms"]
        require(math.isfinite(gain) and math.isfinite(ratio), "gate:invalid_ratio")
        gate = report.get("gate")
        require(isinstance(gate, dict) and not gate.get("reason"), "gate:invalid")
        for field, value in (("gain_fraction", gain), ("gain_percent", gain * 100), ("p95_ratio", ratio),
                             ("min_gain_fraction", MIN_GAIN), ("max_p95_ratio", MAX_P95_RATIO)):
            equal_number(gate.get(field), value, "gate." + field)
        require(gate.get("throughput_metric") == "median_output_tok_s" and gate.get("latency_metric") == "median_p95_e2e_ms", "gate:metric_contract")
        verdict = "KEEP" if gain >= MIN_GAIN and ratio <= MAX_P95_RATIO else "REJECT"
        require(report.get("verdict") == verdict, "gate:verdict_mismatch")
        result.update(ok=True, metric_replay_verified=True, samples_verified=36,
                      recomputed_verdict=verdict, gain_percent=gain * 100, p95_ratio=ratio,
                      runner_source_sha256=source, expected_runner_digest_checked=expected_runner_sha256 is not None,
                      declared_execution_scope=report["execution_scope"])
    except InvalidEvidence as exc:
        result["reason"] = str(exc)
    except (KeyError, TypeError, ValueError, OverflowError, ArithmeticError):
        result["reason"] = "report:malformed"
    return result


def load_report(path: Path, *, expected_sha256: str | None = None) -> tuple[dict, str]:
    with path.open("rb") as handle:
        raw = handle.read(MAX_REPORT_BYTES + 1)
    require(len(raw) <= MAX_REPORT_BYTES, "file:too_large")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, "file:sha256_mismatch")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "json:duplicate_key")
            value[key] = item
        return value

    def reject_constant(value):
        raise InvalidEvidence("json:nonfinite_literal")

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant), digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args()
    try:
        path = args.report
        if path is None:
            directory = Path(__file__).resolve().parents[1] / "docs" / "evidence"
            paths = sorted(directory.glob("hyperloom_r9700_upstream_autonomous_e2e_*.json"))
            require(bool(paths), "file:no_evidence")
            path = paths[-1]
        report, digest = load_report(path, expected_sha256=args.expected_sha256)
        result = audit_report(report, expected_runner_sha256=args.expected_runner_sha256)
        result.update(evidence_sha256=digest, expected_evidence_digest_checked=args.expected_sha256 is not None,
                      evidence_filename=path.name)
    except (OSError, ValueError, TypeError, UnicodeError):
        result = {"ok": False, "reason": "file:invalid_or_unavailable", "physical_execution_verified": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
