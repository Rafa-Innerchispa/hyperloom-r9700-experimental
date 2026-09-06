#!/usr/bin/env python3
"""Run the real upstream agent experiment through the existing local proxy.

This is a primary-orchestrated diagnostic, NOT evidence of worktree execution on
AMD. The normal runner records that distinction and does not attest hardware.
No service configuration, permissions, models, or shell exposure are changed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:18000/v1"
EXECUTION_SCOPE = "primary_orchestrated_existing_proxy_not_amd_worktree_acceptance"
EXPECTED_MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"


def main() -> int:
    # Only this process receives these settings. No host config is modified.
    os.environ["OPENAI_BASE_URL"] = BASE_URL
    os.environ["HYPERLOOM_EXECUTION_SCOPE"] = EXECUTION_SCOPE
    target = Path(__file__).resolve().with_name("r9700_upstream_agent_e2e.py")
    spec = importlib.util.spec_from_file_location("r9700_upstream_proxy_target", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("upstream_runner_unavailable")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if runner._discover_model() != EXPECTED_MODEL:
        raise RuntimeError("unexpected_model_at_existing_proxy")
    return asyncio.run(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
