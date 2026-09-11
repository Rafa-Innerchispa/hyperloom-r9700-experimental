#!/usr/bin/env python3
"""Explicit process-local server entrypoint for the R9700 HyperLoom v7 candidate.

Install the versioned patch at module import time so Python multiprocessing
spawned children receive the same process-local monkeypatch. Start the vLLM API
server only in the original __main__ process, avoiding recursive server startup
when multiprocessing re-imports this file as __mp_main__.
"""
from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, "/tmp")
os.environ.setdefault("HYPERLOOM_R9700_EVIDENCE_FILE", "/tmp/r9700_candidate_paths.jsonl")
import r9700_wna16_hybrid_patch as _r9700_patch

# Intentionally top-level: spawned Python processes must install the same class
# substitutions before vLLM reconstructs the EngineCore state.
_r9700_patch.install_patch(force=True)


def main() -> None:
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
