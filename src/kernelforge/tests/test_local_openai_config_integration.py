# SPDX-License-Identifier: MIT

from __future__ import annotations

from kernelforge.agent_backends import LocalOpenAIBackend, create_registered_backend
from kernelforge.config import Config


def test_config_env_resolves_registered_local_openai_backend(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_AGENT_BACKEND", "local-openai")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "Qwen3-Coder-local")
    monkeypatch.setenv(
        "FORGE_AGENT_OPTIONS_JSON",
        '{"base_url":"http://127.0.0.1:8000/v1","tool_mode":"json","max_tokens":1024}',
    )
    monkeypatch.setenv("FORGE_AGENT_FALLBACK_PROVIDER", "")
    monkeypatch.setenv("FORGE_AGENT_PRECHECK", "0")

    config = Config.from_env(gpu_target="gfx1201", gpu_type="r9700")
    runtime = config.agent_runtime()
    backend = create_registered_backend(runtime, preflight=False)

    assert runtime.provider == "local-openai"
    assert runtime.model == "Qwen3-Coder-local"
    assert runtime.options["base_url"] == "http://127.0.0.1:8000/v1"
    assert runtime.options["tool_mode"] == "json"
    assert runtime.options["max_tokens"] == 1024
    assert isinstance(backend, LocalOpenAIBackend)
    assert backend.base_url == "http://127.0.0.1:8000/v1"
    assert backend.tool_mode == "json"


def test_r9700_config_keeps_gpu_identity_separate_from_agent_provider(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_AGENT_BACKEND", "local-openai")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "Qwen3-Coder-local")
    monkeypatch.setenv("FORGE_AGENT_OPTIONS_JSON", '{"tool_mode":"json"}')
    monkeypatch.setenv("FORGE_AGENT_FALLBACK_PROVIDER", "")

    config = Config.from_env(gpu_target="gfx1201", gpu_type="r9700")

    assert config.gpu_target == "gfx1201"
    assert config.gpu_type == "r9700"
    assert config.agent_backend == "local-openai"
    assert config.agent_model == "Qwen3-Coder-local"
