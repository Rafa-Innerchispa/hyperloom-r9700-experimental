from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from kernelforge.agent_backends import create_registered_backend, resolve_agent_runtime
from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy


HOST = "192.168.1.5"
PORT = 8000
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"


def _r9700_vllm_reachable() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1.5):
            return True
    except OSError:
        return False


def test_registered_local_openai_backend_drives_live_r9700_qwen(tmp_path: Path):
    if not _r9700_vllm_reachable():
        pytest.skip("R9700 vLLM endpoint is not reachable from this runner")

    runtime = resolve_agent_runtime(
        "local-openai",
        model=MODEL,
        timeout_sec=90,
        precheck=False,
        options={
            "base_url": f"http://{HOST}:{PORT}/v1",
            "tool_mode": "auto",
            "max_tokens": 256,
        },
    )
    backend = create_registered_backend(runtime, preflight=False)
    probe = backend.probe(cwd=str(tmp_path), timeout_sec=30)
    assert "OK" in probe.text

    progress: list[str] = []
    spec = AgentRunSpec(
        system_prompt=(
            "You are testing a bounded Hyperloom local agent backend. "
            "Use the available workspace tool and obey the requested content exactly."
        ),
        user_prompt=(
            "Create candidate.conf containing exactly this single line: concurrency=2. "
            "After the tool result confirms the write, finish successfully."
        ),
        cwd=str(tmp_path),
        writable=True,
        target_files=["candidate.conf"],
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=False, max_turns=6),
        progress_log=progress,
        timeout_sec=90,
    )
    result = asyncio.run(backend.run(spec))

    assert (tmp_path / "candidate.conf").read_text(encoding="utf-8").strip() == "concurrency=2"
    assert result.edit_count == 1
    assert result.target_edit_count == 1
    assert result.file_changes == ["candidate.conf"]
    assert result.end_reason == "agent_stopped"
    assert any(call[0] == "write_file" for call in result.tool_calls)
    assert "local-openai: native tools unavailable; falling back to json-action" in progress
    assert "local-openai: json-action tool mode" in progress
