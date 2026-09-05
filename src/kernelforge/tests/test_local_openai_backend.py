from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kernelforge.agent_backends import get_agent_provider, list_agent_providers
from kernelforge.agent_backends.base import AgentRunSpec, AgentRuntimeConfig, AgentToolPolicy
from kernelforge.agent_backends.local_openai import LocalOpenAIBackend, LocalOpenAIUnavailableError


def runtime(**options) -> AgentRuntimeConfig:
    merged = {"base_url": "http://127.0.0.1:8000/v1", "max_tokens": 128}
    merged.update(options)
    return AgentRuntimeConfig(
        provider="local-openai",
        model="qwen-test",
        timeout_sec=30,
        options=merged,
    )


def test_provider_is_registered():
    assert "local-openai" in list_agent_providers()
    provider = get_agent_provider("local-openai")
    assert provider.capabilities.probe is True
    assert provider.capabilities.resumable is False
    assert provider.owns_model("Qwen3-Coder-30B") is True


def test_path_escape_is_denied(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes workspace"):
        LocalOpenAIBackend._resolve(tmp_path, "../secret.txt")


def test_write_tool_respects_target_and_protected_paths(tmp_path: Path):
    backend = LocalOpenAIBackend(runtime())
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        target_files=["allowed.txt"],
        protected_paths=["protected.txt"],
        tool_policy=AgentToolPolicy(write=True),
    ).resolved(runtime())
    result, edited, changed = backend._execute_tool("write_file", {"path": "allowed.txt", "content": "ok"}, spec)
    assert edited is True
    assert changed == "allowed.txt"
    assert (tmp_path / "allowed.txt").read_text() == "ok"
    result, edited, changed = backend._execute_tool("write_file", {"path": "other.txt", "content": "no"}, spec)
    assert "not in target_files" in result
    assert edited is False
    assert not (tmp_path / "other.txt").exists()


def test_protected_path_is_denied_even_if_target(tmp_path: Path):
    backend = LocalOpenAIBackend(runtime())
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        target_files=["protected.txt"],
        protected_paths=["protected.txt"],
        tool_policy=AgentToolPolicy(write=True),
    ).resolved(runtime())
    result, edited, _ = backend._execute_tool("write_file", {"path": "protected.txt", "content": "no"}, spec)
    assert "protected path" in result
    assert edited is False


def test_command_tool_has_narrow_allowlist(tmp_path: Path):
    backend = LocalOpenAIBackend(runtime())
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        tool_policy=AgentToolPolicy(shell=True),
    ).resolved(runtime())
    result, edited, _ = backend._execute_tool("run_command", {"argv": ["bash", "-lc", "echo bad"]}, spec)
    assert "command not allowed" in result
    assert edited is False


def test_minimal_json_tool_catalog_omits_large_schemas(tmp_path: Path):
    backend = LocalOpenAIBackend(runtime(json_tool_catalog="minimal"))
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True),
    ).resolved(backend.runtime)
    catalog = json.loads(
        backend._json_tool_catalog(
            backend._tool_definitions(spec),
            minimal=backend.json_tool_catalog_mode == "minimal",
        )
    )
    assert {"name": "write_file", "args": ["content", "path"]} in catalog
    assert all("parameters" not in item for item in catalog)
    assert all("description" not in item for item in catalog)


def test_enabled_tools_filters_runtime_capabilities(tmp_path: Path):
    backend = LocalOpenAIBackend(runtime(enabled_tools=["write_file"], allow_shell=True))
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True),
    ).resolved(backend.runtime)
    names = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in backend._tool_definitions(spec)
    }
    assert names == {"write_file"}


def test_compact_system_prompt_mode_keeps_large_prompt_out_of_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = LocalOpenAIBackend(runtime(tool_mode="json", system_prompt_mode="compact"))
    calls = []

    def fake_post(path, payload, timeout):
        calls.append(payload)
        assert payload["messages"][1]["content"].startswith("You are a bounded local coding agent.")
        assert "VERY LARGE PROMPT" not in payload["messages"][1]["content"]
        return {"choices": [{"message": {"content": '{"final":"ok"}'}}], "usage": {}}

    monkeypatch.setattr(backend, "_post", fake_post)
    spec = AgentRunSpec(
        system_prompt="VERY LARGE PROMPT " * 1000,
        user_prompt="Finish.",
        cwd=str(tmp_path),
        writable=False,
        tool_policy=AgentToolPolicy(read=True, search=False, write=False, shell=False, max_turns=1),
    )
    result = asyncio.run(backend.run(spec))
    assert result.text == "ok"
    assert len(calls) == 1


def test_run_executes_native_tool_call_then_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = LocalOpenAIBackend(runtime(tool_mode="native"))
    calls = []

    def fake_post(path, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": "I will write the requested candidate.",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": json.dumps({"path": "candidate.txt", "content": "candidate=2"})},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        assert payload["messages"][-1]["role"] == "tool"
        return {
            "choices": [{"message": {"content": "Done."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        }

    monkeypatch.setattr(backend, "_post", fake_post)
    progress = []
    spec = AgentRunSpec(
        system_prompt="Optimize the candidate safely.",
        user_prompt="Write candidate=2",
        cwd=str(tmp_path),
        writable=True,
        target_files=["candidate.txt"],
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=False, max_turns=4),
        progress_log=progress,
    )
    result = asyncio.run(backend.run(spec))
    assert result.text == "Done."
    assert result.edit_count == 1
    assert result.target_edit_count == 1
    assert result.file_changes == ["candidate.txt"]
    assert result.num_turns == 2
    assert (tmp_path / "candidate.txt").read_text() == "candidate=2"
    assert ("write_file", {"path": "candidate.txt", "content": "candidate=2"}) in result.tool_calls
    assert "tool: write_file" in progress
    assert result.usage["total_tokens"] == 29


def test_auto_mode_falls_back_to_json_actions_when_vllm_has_no_tool_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = LocalOpenAIBackend(runtime(tool_mode="auto"))
    calls = []

    def fake_post(path, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            assert payload["tool_choice"] == "auto"
            raise LocalOpenAIUnavailableError(
                'local OpenAI HTTP 400: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
            )
        if len(calls) == 2:
            assert "tools" not in payload
            return {
                "choices": [{"message": {"content": '{"action":"write_file","arguments":{"path":"candidate.txt","content":"candidate=3"}}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            }
        assert "TOOL_RESULT action=write_file: WROTE candidate.txt" in payload["messages"][-1]["content"]
        return {
            "choices": [{"message": {"content": '{"final":"Candidate written and verified."}'}}],
            "usage": {"prompt_tokens": 22, "completion_tokens": 6, "total_tokens": 28},
        }

    monkeypatch.setattr(backend, "_post", fake_post)
    progress = []
    spec = AgentRunSpec(
        system_prompt="Optimize safely.",
        user_prompt="Write candidate=3",
        cwd=str(tmp_path),
        writable=True,
        target_files=["candidate.txt"],
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=False, max_turns=5),
        progress_log=progress,
    )
    result = asyncio.run(backend.run(spec))
    assert result.text == "Candidate written and verified."
    assert result.edit_count == 1
    assert result.file_changes == ["candidate.txt"]
    assert (tmp_path / "candidate.txt").read_text() == "candidate=3"
    assert "local-openai: native tools unavailable; falling back to json-action" in progress
    assert "local-openai: json-action tool mode" in progress
    assert result.usage["total_tokens"] == 56


def test_json_action_accepts_name_args_dialect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = LocalOpenAIBackend(runtime(tool_mode="json", json_tool_catalog="minimal"))
    calls = []

    def fake_post(path, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"name":"write_file","args":["CONCURRENCY = 1\\n","candidate.py"]}'
                        }
                    }
                ],
                "usage": {},
            }
        assert "TOOL_RESULT action=write_file: WROTE candidate.py" in payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": '{"final":"SUBMIT_CANDIDATE"}'}}], "usage": {}}

    monkeypatch.setattr(backend, "_post", fake_post)
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="Write the candidate.",
        cwd=str(tmp_path),
        writable=True,
        target_files=["candidate.py"],
        tool_policy=AgentToolPolicy(write=True, max_turns=3),
    )
    result = asyncio.run(backend.run(spec))
    assert result.text == "SUBMIT_CANDIDATE"
    assert (tmp_path / "candidate.py").read_text(encoding="utf-8") == "CONCURRENCY = 1\n"
    assert result.edit_count == 1


def test_json_protocol_retries_invalid_non_json_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = LocalOpenAIBackend(runtime(tool_mode="json"))
    calls = []

    def fake_post(path, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "I should use a tool."}}], "usage": {}}
        assert "PROTOCOL_ERROR" in payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": '{"final":"Recovered."}'}}], "usage": {}}

    monkeypatch.setattr(backend, "_post", fake_post)
    progress = []
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="Inspect only.",
        cwd=str(tmp_path),
        writable=False,
        tool_policy=AgentToolPolicy(read=True, search=False, write=False, shell=False, max_turns=3),
        progress_log=progress,
    )
    result = asyncio.run(backend.run(spec))
    assert result.text == "Recovered."
    assert "local-openai: invalid json-action response; retrying" in progress
