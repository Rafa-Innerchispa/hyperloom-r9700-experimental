#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one bounded autonomous optimization cycle on the experimental R9700 path.

Example:

    python scripts/r9700_autonomous_optimizer.py \
      --base-url http://127.0.0.1:8000/v1 \
      --model QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
      --baseline-concurrency 1 \
      --candidates 2,3,4 \
      --requests 10 \
      --output docs/evidence/r9700_autonomous_loop.json

This script does not restart vLLM, mutate ROCm, change kernels, or touch cloud
resources.  It only benchmarks the already-running OpenAI-compatible endpoint,
lets the local-openai agent propose one bounded concurrency value, then applies
an explicit deterministic KEEP/REJECT gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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


def _parse_candidates(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("candidate concurrencies must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one candidate concurrency is required")
    return tuple(dict.fromkeys(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--baseline-concurrency", type=int, default=1)
    parser.add_argument("--candidates", type=_parse_candidates, default=(2, 3, 4))
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-throughput-gain-pct", type=float, default=10.0)
    parser.add_argument("--max-mean-latency-regression-pct", type=float, default=60.0)
    parser.add_argument("--max-p95-latency-regression-pct", type=float, default=75.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/r9700_autonomous_loop.json"),
    )
    args = parser.parse_args(argv)

    runtime = AgentRuntimeConfig(
        provider="local-openai",
        model=args.model,
        timeout_sec=240,
        options={
            "base_url": args.base_url,
            "api_key": args.api_key,
            # The resident vLLM may not have --tool-call-parser enabled. JSON
            # mode exercises the same bounded tool loop without a server restart.
            "tool_mode": "json",
            "max_tokens": 1024,
        },
    )
    backend = create_registered_backend(runtime)
    benchmarker = OpenAIEndpointBenchmark(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )

    result = asyncio.run(
        run_autonomous_loop(
            backend=backend,
            benchmark=benchmarker.run,
            baseline_concurrency=args.baseline_concurrency,
            candidate_concurrencies=args.candidates,
            requests_per_arm=args.requests,
            model=args.model,
            min_throughput_gain_pct=args.min_throughput_gain_pct,
            max_mean_latency_regression_pct=args.max_mean_latency_regression_pct,
            max_p95_latency_regression_pct=args.max_p95_latency_regression_pct,
        )
    )
    output = write_loop_evidence(result, args.output)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"EVIDENCE={output}")
    print(f"VERDICT={result.acceptance.decision}")
    return 0 if result.acceptance.decision in {"KEEP", "REJECT"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
