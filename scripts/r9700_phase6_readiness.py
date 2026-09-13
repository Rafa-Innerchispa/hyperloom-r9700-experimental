#!/usr/bin/env python3
"""Phase 6 production readiness gate for the R9700 S3 backend.

Phase 5 proved a stock-exact canonical output hash after the known cold/JIT
warmup path. Phase 6 keeps that correctness requirement, but does not promote on
one lucky sample: it drains bounded cold-JIT behavior and requires consecutive
canonical outputs before production is considered ready.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error

from r9700_readiness_warmup import DEFAULT_MODEL, wait_models, warmup


def gate(
    base_url: str,
    model: str,
    ready_timeout: float,
    max_attempts: int = 4,
    required_consecutive: int = 2,
) -> dict:
    readiness = wait_models(base_url, ready_timeout)
    if not readiness.get("ok"):
        return {
            "schema": "hyperloom.r9700.phase6.readiness.v1",
            "ready": readiness,
            "attempts": [],
            "required_consecutive": required_consecutive,
            "pass": False,
            "reason": "models_endpoint_not_ready",
        }

    attempts: list[dict] = []
    consecutive = 0
    for index in range(1, max_attempts + 1):
        try:
            result = warmup(base_url, model)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        attempt = {"attempt": index, **result}
        attempts.append(attempt)
        if result.get("ok") and result.get("canonical_match"):
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= required_consecutive:
            return {
                "schema": "hyperloom.r9700.phase6.readiness.v1",
                "ready": readiness,
                "attempts": attempts,
                "required_consecutive": required_consecutive,
                "consecutive_canonical": consecutive,
                "pass": True,
                "reason": "canonical_steady_state_proven",
            }

    return {
        "schema": "hyperloom.r9700.phase6.readiness.v1",
        "ready": readiness,
        "attempts": attempts,
        "required_consecutive": required_consecutive,
        "consecutive_canonical": consecutive,
        "pass": False,
        "reason": "canonical_steady_state_not_proven",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded steady-state correctness gate for Phase 6 S3 promotion.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--required-consecutive", type=int, default=2)
    args = parser.parse_args()

    if args.max_attempts < 1 or args.required_consecutive < 1 or args.required_consecutive > args.max_attempts:
        print(json.dumps({"schema": "hyperloom.r9700.phase6.readiness.error.v1", "error": "invalid_attempt_policy"}, indent=2), file=sys.stderr)
        return 2

    result = gate(
        args.base_url,
        args.model,
        args.ready_timeout,
        max_attempts=args.max_attempts,
        required_consecutive=args.required_consecutive,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
