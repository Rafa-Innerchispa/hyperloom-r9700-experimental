#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Zero-argument peer entrypoint for the live R9700 autonomous loop.

This exists because AG-41 peer_python_runtime executes one allowlisted Python
script without arbitrary CLI arguments. It autodiscovers the resident vLLM
model, constructs the real registered ``local-openai`` KernelForge backend and
runs one bounded baseline -> proposal -> candidate -> KEEP/REJECT cycle.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kernelforge.agent_backends import AgentRuntimeConfig, create_registered_backend  # noqa: E402
from kernelforge.r9700_autonomous_loop import (  # noqa: E402
    OpenAIEndpointBenchmark,
    run_autonomous_loop,
    write_loop_evidence,
)

BASE_URL = "http://127.0.0.1:8000/v1"


def discover_model() -> str:
    request = urllib.request.Request(BASE_URL + "/models", method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") or []
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError("resident vLLM /models returned no model id")
    return str(models[0]["id"])


def main() -> int:
    model = discover_model()
    runtime = AgentRuntimeConfig(
        provider="local-openai",
        model=model,
        timeout_sec=240,
        options={
            "base_url": BASE_URL,
            "api_key": "local",
            "tool_mode": "json",
            "max_tokens": 1024,
        },
    )
    backend = create_registered_backend(runtime)
    benchmarker = OpenAIEndpointBenchmark(
        base_url=BASE_URL,
        model=model,
        api_key="local",
        max_tokens=32,
    )
    result = asyncio.run(
        run_autonomous_loop(
            backend=backend,
            benchmark=benchmarker.run,
            baseline_concurrency=1,
            candidate_concurrencies=(2, 3, 4),
            requests_per_arm=6,
            model=model,
            min_throughput_gain_pct=3.0,
            max_mean_latency_regression_pct=150.0,
            max_p95_latency_regression_pct=150.0,
        )
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = _REPO_ROOT / "docs" / "evidence" / f"r9700_exact_backend_{stamp}.json"
    write_loop_evidence(result, output)
    payload = result.to_dict()
    payload["execution_path"] = "full Hyperloom-derived branch + registered LocalOpenAIBackend"
    payload["model"] = model
    print(json.dumps(payload, sort_keys=True))
    print(f"EVIDENCE={output.relative_to(_REPO_ROOT)}")
    print(f"VERDICT={result.acceptance.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
