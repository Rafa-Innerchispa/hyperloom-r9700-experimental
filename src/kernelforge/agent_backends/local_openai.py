# SPDX-License-Identifier: MIT
"""Experimental local OpenAI-compatible KernelForge agent backend.

This backend is intentionally local-first. It can drive an OpenAI-compatible
vLLM/SGLang endpoint without Claude/Codex SDKs and, when the serving runtime was
started without native tool-calling flags, it falls back to a strict JSON action
protocol rather than requiring a production model restart.

It implements only bounded capabilities needed by the Hyperloom experiment:
probe, workspace-local read/search/write, narrow argv execution, timeout/turn
limits, progress telemetry and file-change evidence. It does not claim resume,
MCP, subagents or provider-level sandboxing.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
)


class LocalOpenAIUnavailableError(AgentProviderUnavailableError):
    """Report an unavailable or incompatible local OpenAI-compatible endpoint."""


class LocalOpenAIBackend:
    """Run bounded KernelForge agent turns through a local OpenAI endpoint."""

    name = "local-openai"
    capabilities = AgentCapabilities(
        writable=True,
        resumable=False,
        stop_hooks=False,
        native_subagents=False,
        mcp=False,
        sandbox=False,
        probe=True,
        requires_workspace_cwd=False,
        session_env=True,
        workspace_guard=False,
    )

    _COMMANDS = {"python", "python3", "pytest", "git", "grep", "rg"}
    _JSON_PROTOCOL = (
        "When tools are available, respond with exactly one JSON object and no markdown. "
        "To call a tool: {\"action\":\"tool_name\",\"arguments\":{...}}. "
        "When the task is complete: {\"final\":\"short result\"}. "
        "Never invent a tool result; wait for TOOL_RESULT before continuing."
    )

    def __init__(self, runtime: AgentRuntimeConfig | None = None) -> None:
        self.runtime = runtime or AgentRuntimeConfig(provider=self.name, model="local-model")
        self.fallback_reason = ""
        self._native_tools_disabled = False

    @property
    def base_url(self) -> str:
        configured = str(self.runtime.options.get("base_url") or "").strip()
        return (configured or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")

    @property
    def api_key(self) -> str:
        return str(self.runtime.options.get("api_key") or os.environ.get("OPENAI_API_KEY") or "local")

    @property
    def tool_mode(self) -> str:
        mode = str(self.runtime.options.get("tool_mode") or "auto").strip().lower()
        if mode not in {"auto", "native", "json"}:
            raise ValueError("local-openai tool_mode must be one of: auto, native, json")
        return mode

    @property
    def allow_shell(self) -> bool:
        """Require an explicit opt-in before exposing subprocess execution."""
        return bool(self.runtime.options.get("allow_shell", False))

    @property
    def json_tool_catalog_mode(self) -> str:
        mode = str(self.runtime.options.get("json_tool_catalog") or "full").strip().lower()
        if mode not in {"full", "minimal"}:
            raise ValueError("local-openai json_tool_catalog must be one of: full, minimal")
        return mode

    @property
    def system_prompt_mode(self) -> str:
        mode = str(self.runtime.options.get("system_prompt_mode") or "full").strip().lower()
        if mode not in {"full", "compact"}:
            raise ValueError("local-openai system_prompt_mode must be one of: full, compact")
        return mode

    @property
    def enabled_tools(self) -> set[str] | None:
        raw = self.runtime.options.get("enabled_tools")
        if raw in (None, "", []):
            return None
        if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
            raise ValueError("local-openai enabled_tools must be a list of tool names")
        return set(raw)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise LocalOpenAIUnavailableError(f"local OpenAI HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LocalOpenAIUnavailableError(f"local OpenAI request failed: {type(exc).__name__}: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return self._request("POST", path, payload, timeout)

    def _get(self, path: str, timeout: float) -> dict[str, Any]:
        return self._request("GET", path, None, timeout)

    def _resolve_model(self, requested: str) -> str:
        selected = (requested or self.runtime.model).strip()
        if selected and selected not in {"local-model", "auto"}:
            return selected
        payload = self._get("/models", min(20.0, float(self.runtime.timeout_sec)))
        models = payload.get("data") or []
        if not models or not isinstance(models[0], dict) or not models[0].get("id"):
            raise LocalOpenAIUnavailableError("local OpenAI /models returned no model id")
        return str(models[0]["id"])

    def probe(
        self,
        *,
        cwd: str,
        model: str = "",
        reasoning_effort: str = "",
        timeout_sec: int | None = None,
        usage: Any = None,
    ) -> AgentRunResult:
        del cwd, reasoning_effort, usage
        selected = self._resolve_model(model)
        response = self._post(
            "/chat/completions",
            {
                "model": selected,
                "messages": [{"role": "user", "content": "Reply with exactly OK"}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            },
            float(timeout_sec or min(60, self.runtime.timeout_sec)),
        )
        text = str((((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
        if "OK" not in text:
            raise LocalOpenAIUnavailableError(f"local model probe returned {text[:160]!r}")
        return AgentRunResult(text=text, end_reason="agent_stopped", usage=dict(response.get("usage") or {}))

    @staticmethod
    def _resolve(cwd: Path, raw: str) -> Path:
        candidate = (cwd / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        root = cwd.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes workspace: {raw}") from exc
        return candidate

    @staticmethod
    def _policy_paths(cwd: Path, paths: list[str]) -> set[str]:
        """Normalize absolute or relative policy paths to workspace-relative form."""
        root = cwd.resolve()
        normalized: set[str] = set()
        for raw in paths:
            path = Path(raw)
            if path.is_absolute():
                try:
                    path = path.resolve().relative_to(root)
                except ValueError:
                    continue
            normalized.add(str(path).replace(os.sep, "/"))
        return normalized

    @staticmethod
    def _protected(spec: AgentRunSpec, path: Path, cwd: Path) -> bool:
        import fnmatch

        rel = str(path.relative_to(cwd)).replace(os.sep, "/")
        if rel in LocalOpenAIBackend._policy_paths(cwd, spec.protected_paths):
            return True
        return any(fnmatch.fnmatch(rel, pattern) for pattern in spec.protected_globs)

    def _tool_definitions(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        policy = spec.tool_policy
        if policy is None:
            return []
        tools: list[dict[str, Any]] = []

        def add(name: str, description: str, properties: dict[str, Any], required: list[str]) -> None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if policy.read:
            add("read_file", "Read a UTF-8 text file inside the workspace.", {"path": {"type": "string"}}, ["path"])
        if policy.search:
            add(
                "search_text",
                "Search text recursively inside workspace files.",
                {"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
                ["query"],
            )
        if policy.write and spec.writable:
            add(
                "write_file",
                "Replace a UTF-8 text file inside the workspace. Protected paths are denied.",
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            )
        if policy.shell and self.allow_shell:
            add(
                "run_command",
                "Run a bounded argv command in the workspace. Only python/python3/pytest/git/grep/rg are accepted.",
                {"argv": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
                ["argv"],
            )
        enabled = self.enabled_tools
        if enabled is not None:
            tools = [
                tool
                for tool in tools
                if str((tool.get("function") or {}).get("name") or "") in enabled
            ]
        return tools

    @staticmethod
    def _json_tool_catalog(tools: list[dict[str, Any]], *, minimal: bool = False) -> str:
        compact = []
        for tool in tools:
            fn = tool.get("function") or {}
            if minimal:
                params = ((fn.get("parameters") or {}).get("properties") or {}).keys()
                compact.append({"name": fn.get("name"), "args": sorted(params)})
            else:
                compact.append(
                    {
                        "name": fn.get("name"),
                        "description": fn.get("description"),
                        "parameters": fn.get("parameters"),
                    }
                )
        return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(cleaned):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
        return None

    @staticmethod
    def _tool_argument_names(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
        names: dict[str, list[str]] = {}
        for tool in tools:
            fn = tool.get("function") or {}
            name = str(fn.get("name") or "")
            params = ((fn.get("parameters") or {}).get("properties") or {}).keys()
            if name:
                names[name] = sorted(str(param) for param in params)
        return names

    @staticmethod
    def _normalize_json_action(payload: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if "action" in payload or "final" in payload:
            return payload
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            return payload
        args = payload.get("arguments", payload.get("args", {}))
        if isinstance(args, list):
            arg_names = LocalOpenAIBackend._tool_argument_names(tools).get(name, [])
            args = {key: value for key, value in zip(arg_names, args, strict=False)}
        elif not isinstance(args, dict):
            args = {"_raw": args}
        return {"action": name, "arguments": args}

    @staticmethod
    def _native_tooling_unavailable(error: Exception) -> bool:
        text = str(error).lower()
        return "tool-call-parser" in text or "enable-auto-tool-choice" in text or "tool choice requires" in text

    def _execute_tool(self, name: str, args: dict[str, Any], spec: AgentRunSpec) -> tuple[str, bool, str]:
        cwd = Path(spec.cwd).resolve()
        if name == "read_file":
            path = self._resolve(cwd, str(args.get("path") or ""))
            if not path.is_file():
                return f"ERROR: file not found: {path.relative_to(cwd)}", False, ""
            return path.read_text(encoding="utf-8", errors="replace")[:120000], False, ""
        if name == "search_text":
            query = str(args.get("query") or "")
            root = self._resolve(cwd, str(args.get("path") or "."))
            matches: list[str] = []
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if len(matches) >= 120 or not path.is_file() or ".git" in path.parts:
                    continue
                try:
                    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if query.lower() in line.lower():
                            matches.append(f"{path.relative_to(cwd)}:{lineno}:{line[:300]}")
                            if len(matches) >= 120:
                                break
                except OSError:
                    continue
            return "\n".join(matches) or "NO_MATCHES", False, ""
        if name == "write_file":
            policy = spec.tool_policy
            if policy is None or not policy.write or not spec.writable:
                return "ERROR: write not allowed", False, ""
            path = self._resolve(cwd, str(args.get("path") or ""))
            if self._protected(spec, path, cwd):
                return f"ERROR: protected path: {path.relative_to(cwd)}", False, ""
            if spec.target_files:
                allowed = self._policy_paths(cwd, spec.target_files)
                rel = str(path.relative_to(cwd)).replace(os.sep, "/")
                if rel not in allowed:
                    return f"ERROR: path not in target_files: {rel}", False, ""
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content") or ""), encoding="utf-8")
            return f"WROTE {path.relative_to(cwd)}", True, str(path.relative_to(cwd))
        if name == "run_command":
            policy = spec.tool_policy
            if policy is None or not policy.shell:
                return "ERROR: shell not allowed", False, ""
            argv = args.get("argv") or []
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                return "ERROR: argv must be a non-empty string list", False, ""
            executable = Path(argv[0]).name
            if executable not in self._COMMANDS:
                return f"ERROR: command not allowed: {executable}", False, ""
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=min(120, int(spec.timeout_sec or 120)),
                check=False,
                env={**os.environ, **spec.env},
            )
            return (
                f"exit={completed.returncode}\nstdout:\n{completed.stdout[-20000:]}\nstderr:\n{completed.stderr[-10000:]}",
                False,
                "",
            )
        return f"ERROR: unknown tool {name}", False, ""

    @staticmethod
    def _result(
        *,
        text: str,
        subtype: str = "",
        num_turns: int,
        end_reason: str,
        tool_calls: list[tuple[str, dict[str, Any]]],
        changed: list[str],
        edit_count: int,
        usage: dict[str, int],
        spec: AgentRunSpec,
    ) -> AgentRunResult:
        unique_changes = list(dict.fromkeys(changed))
        return AgentRunResult(
            text=text,
            subtype=subtype,
            num_turns=num_turns,
            end_reason=end_reason,
            tool_calls=tool_calls,
            file_changes=unique_changes,
            edit_count=edit_count,
            target_edit_count=(edit_count if spec.target_files else None),
            usage=usage,
        )

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        spec = spec.resolved(self.runtime)
        selected_model = self._resolve_model(spec.model)
        if spec.progress_log is not None:
            spec.progress_log.append(f"local-openai: session started model={selected_model}")

        messages: list[dict[str, Any]] = []
        if spec.system_prompt:
            system_prompt = spec.system_prompt
            if self.system_prompt_mode == "compact":
                system_prompt = (
                    "You are a bounded local coding agent. Use only supplied JSON tools. "
                    "Edit only target files. Do not use shell unless that tool is explicitly available."
                )
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": spec.user_prompt})

        tools = self._tool_definitions(spec)
        max_turns = spec.tool_policy.max_turns if spec.tool_policy and spec.tool_policy.max_turns is not None else 8
        max_turns = max(1, min(int(max_turns), 32))
        deadline = time.monotonic() + float(spec.timeout_sec or self.runtime.timeout_sec)
        tool_calls_seen: list[tuple[str, dict[str, Any]]] = []
        changed: list[str] = []
        edit_count = 0
        final_text = ""
        aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        json_mode = self.tool_mode == "json" or self._native_tools_disabled
        json_protocol_injected = False

        for turn in range(max_turns):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._result(
                    text=final_text,
                    subtype="timeout",
                    num_turns=turn,
                    end_reason="timeout",
                    tool_calls=tool_calls_seen,
                    changed=changed,
                    edit_count=edit_count,
                    usage=aggregate_usage,
                    spec=spec,
                )

            payload_messages = messages
            if tools and json_mode and not json_protocol_injected:
                protocol = self._JSON_PROTOCOL + " Available tools: " + self._json_tool_catalog(
                    tools,
                    minimal=self.json_tool_catalog_mode == "minimal",
                )
                payload_messages = [{"role": "system", "content": protocol}, *messages]
                messages = payload_messages
                json_protocol_injected = True
                if spec.progress_log is not None:
                    spec.progress_log.append("local-openai: json-action tool mode")

            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": payload_messages,
                "temperature": 0,
                "max_tokens": int(self.runtime.options.get("max_tokens", 2048)),
                "stream": False,
            }
            if tools and not json_mode:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            try:
                response = await asyncio.to_thread(self._post, "/chat/completions", payload, min(remaining, 300.0))
            except LocalOpenAIUnavailableError as exc:
                if tools and self.tool_mode == "auto" and not json_mode and self._native_tooling_unavailable(exc):
                    self._native_tools_disabled = True
                    json_mode = True
                    if spec.progress_log is not None:
                        spec.progress_log.append("local-openai: native tools unavailable; falling back to json-action")
                    continue
                raise

            response_usage = response.get("usage") or {}
            for key in aggregate_usage:
                aggregate_usage[key] += int(response_usage.get(key) or 0)
            if usage is not None and hasattr(usage, "add_from_usage"):
                try:
                    usage.add_from_usage(response_usage)
                except Exception:
                    pass

            choice = (response.get("choices") or [{}])[0]
            message = dict(choice.get("message") or {})
            content = str(message.get("content") or "")
            if content:
                final_text = content

            if json_mode and tools:
                action_payload = self._extract_json_object(content)
                if not action_payload:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "PROTOCOL_ERROR: reply with exactly one JSON object containing either action+arguments or final.",
                        }
                    )
                    if spec.progress_log is not None:
                        spec.progress_log.append("local-openai: invalid json-action response; retrying")
                    continue
                action_payload = self._normalize_json_action(action_payload, tools)
                if "final" in action_payload:
                    final_text = str(action_payload.get("final") or "")
                    if spec.progress_log is not None:
                        spec.progress_log.append(f"local-openai: completed turn {turn + 1}")
                    return self._result(
                        text=final_text,
                        num_turns=turn + 1,
                        end_reason="agent_stopped",
                        tool_calls=tool_calls_seen,
                        changed=changed,
                        edit_count=edit_count,
                        usage=aggregate_usage,
                        spec=spec,
                    )
                name = str(action_payload.get("action") or "")
                args = action_payload.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"_raw": args}
                allowed_names = {str((tool.get("function") or {}).get("name") or "") for tool in tools}
                if name not in allowed_names:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"TOOL_RESULT action={name}: ERROR: unknown or disallowed action"})
                    continue
                tool_calls_seen.append((name, args))
                if spec.progress_log is not None:
                    spec.progress_log.append(f"tool: {name}")
                result, edited, changed_path = self._execute_tool(name, args, spec)
                if edited:
                    edit_count += 1
                    if changed_path:
                        changed.append(changed_path)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT action={name}: {result}\nContinue with the next JSON action or return final.",
                    }
                )
                continue

            calls = message.get("tool_calls") or []
            if not calls:
                if spec.progress_log is not None:
                    spec.progress_log.append(f"local-openai: completed turn {turn + 1}")
                return self._result(
                    text=final_text,
                    num_turns=turn + 1,
                    end_reason="agent_stopped",
                    tool_calls=tool_calls_seen,
                    changed=changed,
                    edit_count=edit_count,
                    usage=aggregate_usage,
                    spec=spec,
                )

            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except Exception:
                    args = {"_raw": str(raw)}
                tool_calls_seen.append((name, args))
                if spec.progress_log is not None:
                    spec.progress_log.append(f"tool: {name}")
                result, edited, changed_path = self._execute_tool(name, args, spec)
                if edited:
                    edit_count += 1
                    if changed_path:
                        changed.append(changed_path)
                messages.append({"role": "tool", "tool_call_id": str(call.get("id") or name), "content": result})

        return self._result(
            text=final_text,
            subtype="max_turns",
            num_turns=max_turns,
            end_reason="max_turns",
            tool_calls=tool_calls_seen,
            changed=changed,
            edit_count=edit_count,
            usage=aggregate_usage,
            spec=spec,
        )


def _local_openai_owns_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("qwen", "local-", "llama", "mistral", "gemma", "deepseek"))


def register_local_openai_provider() -> None:
    """Register the experimental built-in provider after the core registry loads."""
    from kernelforge.agent_backends.registry import AgentProvider, register_agent_provider

    provider = AgentProvider(
        name="local-openai",
        factory=lambda runtime: LocalOpenAIBackend(runtime=runtime),
        default_model=os.environ.get("LOCAL_OPENAI_MODEL", "local-model"),
        capabilities=LocalOpenAIBackend.capabilities,
        availability=lambda: True,
        owns_model=_local_openai_owns_model,
    )
    try:
        register_agent_provider(provider)
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


__all__ = [
    "LocalOpenAIBackend",
    "LocalOpenAIUnavailableError",
    "register_local_openai_provider",
]
