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
- `json_tool_catalog`: `full` by default, or `minimal` for local models with small context windows
- `system_prompt_mode`: `full` by default, or `compact` for local upstream E2E runs that would otherwise exceed context
- `enabled_tools`: optional explicit list of exposed tools for a specific bounded run

The default `tool_mode=auto` first attempts normal OpenAI tool calls. If the serving runtime reports that `--tool-call-parser` or `--enable-auto-tool-choice` is missing, the backend automatically falls back to a strict JSON action protocol. This avoids restarting a production vLLM instance merely to run the experiment.

## JSON action fallback

When native tool calling is unavailable, the model receives a strict contract and may return one action at a time, for example:

```json
{"action":"write_file","arguments":{"path":"candidate.conf","content":"concurrency=2"}}
```

The backend executes only tools allowed by `AgentToolPolicy`, returns the real tool result to the model, and requires a final JSON response before the session ends.

Some local models emit a shorter dialect:

```json
{"name":"write_file","args":["concurrency=2","candidate.conf"]}
```

The backend normalizes that form into the same allowlisted action path; it does not add capabilities or bypass path checks.

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

## Live upstream KernelForge evidence

A second live run exercised the upstream construction path:

`KernelForge make_agent_fn -> registered local-openai -> local Qwen/vLLM -> bounded candidate -> real benchmark -> deterministic KEEP/REJECT`

The E2E run used `system_prompt_mode=compact`, `json_tool_catalog=minimal`, `enabled_tools=["write_file"]`, and `allow_shell=False` so the local 8K-context model could complete the JSON action without exposing shell or broad tools. The candidate file starts with invalid `CONCURRENCY = 0`; the run fails unless the agent writes a valid allowlisted value.

On AMD node `.5`, Qwen wrote `CONCURRENCY = 2` in two turns through `write_file`. The measured candidate increased output throughput from **20.2773** to **34.7577** output tokens/s, but p95 latency ratio was **1.3475**, above the **1.25** gate. The deterministic verdict was therefore **REJECT**. This is a successful control-plane E2E because it proves local upstream agent selection, bounded write, real benchmark, and honest gate rejection.

Canonical sanitized evidence is stored in:

`docs/evidence/hyperloom_r9700_upstream_agent_e2e_20260905T044117Z.json`

## Truth boundary

The live cycles prove the autonomous proposal/benchmark/KEEP-REJECT primitive and the upstream KernelForge `make_agent_fn` path on the physical R9700. This remains an experimental InnerChispa compatibility path, not official AMD or upstream Hyperloom R9700 support. CDNA/Instinct-specific kernel paths remain blocked on `gfx1201` unless separately ported and verified.
