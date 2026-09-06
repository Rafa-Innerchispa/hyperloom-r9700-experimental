#!/usr/bin/env python3
"""Bounded local-proxy inference probe; not a GPU benchmark or node attestation.

Uses the loopback proxy already reported healthy by InnerOS. It does not change
routing, restart services, load a different model, execute shell, or read secrets.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

BASE_URL = "http://127.0.0.1:18000/v1"
EXPECTED_MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("invalid_response_object")
    return result


def probe() -> dict:
    models = request_json("/models").get("data") or []
    if EXPECTED_MODEL not in [row.get("id") for row in models if isinstance(row, dict)]:
        raise ValueError("expected_resident_model_missing")
    started = time.perf_counter()
    response = request_json("/chat/completions", {
        "model": EXPECTED_MODEL,
        "messages": [{"role": "user", "content": "Reply exactly: AMD LOCAL OK"}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    })
    elapsed = time.perf_counter() - started
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("missing_completion")
    message = choices[0].get("message") or {}
    text = message.get("content") if isinstance(message, dict) else None
    usage = response.get("usage") or {}
    tokens = usage.get("completion_tokens")
    if not isinstance(text, str) or text.strip() != "AMD LOCAL OK":
        raise ValueError("unexpected_probe_content")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("invalid_completion_usage")
    return {
        "schema": "inneros-local-proxy-probe-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "model": EXPECTED_MODEL,
        "endpoint_kind": "existing_loopback_proxy_18000",
        "response_id": str(response.get("id") or ""),
        "content": text.strip(),
        "completion_tokens": tokens,
        "elapsed_sec": elapsed,
        "physical_node_execution_verified": False,
        "benchmark": False,
        "scope": "one real inference request only; not the Hyperloom six-round E2E",
    }


def main() -> int:
    try:
        report = probe()
    except Exception as exc:
        report = {"ok": False, "error_type": type(exc).__name__, "benchmark": False}
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
