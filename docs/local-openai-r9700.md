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

## Bounded autonomous loop

The branch now includes `kernelforge.r9700_autonomous_loop` plus `scripts/r9700_autonomous_optimizer.py`. The loop intentionally separates model proposal from deterministic acceptance:

`baseline -> Qwen proposal -> bounded candidate -> benchmark -> KEEP/REJECT`

The local model may propose only an allow-listed serving concurrency and writes that proposal into `candidate.json`. It cannot accept its own result. The final gate is deterministic and checks throughput gain, request success and bounded mean/p95 latency regression.

Focused backend + registry + autonomous-loop tests currently pass **34/34**.

## Live R9700 autonomous evidence

A real bounded cycle ran on AMD node `.5` against the resident R9700/Qwen/vLLM service without restarting vLLM and without cloud resources.

Baseline (`concurrency=1`):

- 6/6 requests successful
- 21.079 output tokens/s
- mean latency 1.3834 s
- p95 latency 1.6055 s

Qwen proposed `concurrency=2` using the bounded `write_file` action. The candidate produced:

- 6/6 requests successful
- 36.2918 output tokens/s
- mean latency 1.5531 s
- p95 latency 1.7177 s

The deterministic gate measured **+72.17% aggregate output throughput**, p95 ratio **1.07**, and returned **KEEP**. No rollback was required.

Canonical sanitized evidence is stored in:

`docs/evidence/r9700_autonomous_loop_live_20260905.json`

## Truth boundary

The live cycle proves the autonomous proposal/benchmark/KEEP-REJECT primitive on the physical R9700. The `LocalOpenAIBackend` implementation is committed and tested in the full Hyperloom-derived branch. The remaining integration gate is to execute that exact full branch directly on node `.5` and then connect it to the upstream optimizer entry point. Until that exact checkout runs end to end, this project must not claim official or complete upstream Hyperloom R9700 support.
