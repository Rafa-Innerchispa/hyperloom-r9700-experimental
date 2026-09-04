# Experimental local-openai backend on Radeon AI PRO R9700

This branch adds an **experimental** KernelForge agent backend named `local-openai` for local OpenAI-compatible serving endpoints such as vLLM. It is intended for the InnerChispa R9700 research path and does **not** claim that upstream Hyperloom officially supports Radeon AI PRO R9700.

## Current validated local stack

- GPU: AMD Radeon AI PRO R9700 (`gfx1201`)
- ROCm: 10.x local canary/runtime
- serving: vLLM OpenAI-compatible endpoint
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- execution policy: local-first, workspace-scoped writes, no cloud required

## Provider configuration

Use `local-openai` as the KernelForge agent provider and point it to the existing local vLLM endpoint. The provider supports runtime options:

- `base_url`: OpenAI-compatible base URL, for example `http://127.0.0.1:8000/v1`
- `tool_mode`: `auto`, `native`, or `json`
- `max_tokens`: bounded response budget

The default `tool_mode=auto` first attempts normal OpenAI tool calls. If the serving runtime reports that `--tool-call-parser` or `--enable-auto-tool-choice` is missing, the backend automatically falls back to a strict JSON action protocol. This avoids restarting a production vLLM instance merely to run the experiment.

## JSON action fallback

When native tool calling is unavailable, the model receives a strict contract and may return one action at a time, for example:

```json
{"action":"write_file","arguments":{"path":"candidate.conf","content":"concurrency=2"}}
```

The backend executes only tools allowed by `AgentToolPolicy`, returns the real tool result to the model, and requires a final JSON response before the session ends.

## Safety boundaries

The experimental backend deliberately implements fewer capabilities than Claude/Codex backends:

- no session resume claim
- no MCP claim
- no native subagent claim
- no arbitrary shell
- workspace path traversal is denied
- protected paths/globs are denied
- `target_files` is enforced for writes
- shell-like execution is available only when `AgentToolPolicy.shell` is true and the executable is in a narrow argv allowlist
- timeout and max-turn ceilings are enforced

## Evidence already obtained

A live R9700/Qwen smoke outside the KernelForge class proved the JSON-action primitive end to end: Qwen selected `write_file`, the guard wrote `candidate.conf` with `concurrency=2`, the real tool result was returned, and Qwen completed the turn. The registered backend has focused mocked coverage for the same native-to-JSON fallback path.

The next acceptance gate is the registered `LocalOpenAIBackend` running on node `.5` against the resident Qwen/vLLM service, followed by a bounded optimization cycle:

`baseline -> Qwen decision -> candidate edit -> benchmark -> KEEP/REJECT`

Any performance conclusion from that cycle must be reported as R9700 experimental evidence, not as official upstream Hyperloom support.
