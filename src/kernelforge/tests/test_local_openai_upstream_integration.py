from __future__ import annotations

import asyncio
from pathlib import Path

from kernelforge.agent_backends.base import AgentRunResult, AgentRunSpec, AgentRuntimeConfig, AgentToolPolicy
from kernelforge.agent_backends.local_openai import LocalOpenAIBackend
from kernelforge.config import Config
from kernelforge.orchestrator.agent import make_agent_fn


def _runtime(**options) -> AgentRuntimeConfig:
    merged = {
        "base_url": "http://127.0.0.1:8000/v1",
        "tool_mode": "json",
        "max_tokens": 128,
    }
    merged.update(options)
    return AgentRuntimeConfig(
        provider="local-openai",
        model="qwen-test",
        timeout_sec=30,
        options=merged,
    )


def test_absolute_target_file_is_normalized_inside_workspace(tmp_path: Path):
    target = tmp_path / "kernel.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    backend = LocalOpenAIBackend(_runtime())
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        target_files=[str(target)],
        tool_policy=AgentToolPolicy(write=True),
    ).resolved(_runtime())

    result, edited, changed = backend._execute_tool(
        "write_file",
        {"path": "kernel.py", "content": "VALUE = 2\n"},
        spec,
    )

    assert result == "WROTE kernel.py"
    assert edited is True
    assert changed == "kernel.py"
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_shell_tool_is_fail_closed_without_explicit_runtime_opt_in(tmp_path: Path):
    backend = LocalOpenAIBackend(_runtime())
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=True,
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True),
    ).resolved(_runtime())
    names = {
        tool["function"]["name"]
        for tool in backend._tool_definitions(spec)
    }
    assert "write_file" in names
    assert "run_command" not in names

    opted_in = LocalOpenAIBackend(_runtime(allow_shell=True))
    opted_spec = spec.resolved(opted_in.runtime)
    opted_names = {
        tool["function"]["name"]
        for tool in opted_in._tool_definitions(opted_spec)
    }
    assert "run_command" in opted_names


def test_upstream_make_agent_fn_runs_registered_local_openai_backend(
    tmp_path: Path,
    monkeypatch,
):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        LocalOpenAIBackend,
        "probe",
        lambda self, **kwargs: AgentRunResult(text="OK", end_reason="agent_stopped"),
    )

    calls: list[dict] = []

    def fake_post(self, path, payload, timeout):
        assert path == "/chat/completions"
        calls.append(payload)
        if len(calls) == 1:
            # make_agent_fn asks for shell access, but local-openai must remain
            # fail-closed unless allow_shell=true was explicitly configured.
            tool_names = {
                tool["name"]
                for tool in __import__("json").loads(payload["messages"][0]["content"].split("Available tools: ", 1)[1])
            }
            assert "write_file" in tool_names
            assert "run_command" not in tool_names
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"write_file","arguments":{"path":"kernel.py","content":"VALUE = 2\\n"}}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        assert "TOOL_RESULT action=write_file: WROTE kernel.py" in payload["messages"][-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"final":"PLAN: update the bounded candidate value\\nSUBMIT_CANDIDATE"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }

    monkeypatch.setattr(LocalOpenAIBackend, "_post", fake_post)

    config = Config(
        workspace=str(tmp_path),
        project_root=tmp_path,
        agent_backend="local-openai",
        agent_model="qwen-test",
        agent_fallback_provider="",
        agent_timeout_sec=30,
        agent_precheck=True,
        agent_options={
            "base_url": "http://127.0.0.1:8000/v1",
            "tool_mode": "json",
            "max_tokens": 128,
        },
        max_turns=4,
    )

    agent_fn = make_agent_fn(
        config,
        "Apply one bounded architecture-neutral candidate change.",
        profiling_enabled=False,
        insession_gate=False,
    )

    assert getattr(agent_fn, "backend_name") == "local-openai"
    assert getattr(agent_fn, "backend_model") == "qwen-test"

    result = asyncio.run(agent_fn(str(kernel), ""))

    assert kernel.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert "PLAN: update the bounded candidate value" in result
    assert len(calls) == 2
