#!/usr/bin/env python3
"""Explicit process-local server entrypoint for the R9700 HyperLoom v7 candidate.

Installs the versioned v7 patch before importing/starting vLLM. This avoids
implicit .pth/sitecustomize execution and makes the candidate bootstrap visible
in Docker inspect and reproducible from Git.
"""
from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, "/tmp")
os.environ.setdefault("HYPERLOOM_R9700_EVIDENCE_FILE", "/tmp/r9700_candidate_paths.jsonl")
import r9700_wna16_hybrid_patch as _r9700_patch

_r9700_patch.install_patch(force=True)
sys.argv[0] = "vllm.entrypoints.openai.api_server"
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
