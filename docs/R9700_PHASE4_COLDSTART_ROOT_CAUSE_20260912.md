# R9700 Phase 4 — First-request cold-start root cause

Captured: 2026-09-12 UTC

Parent Phase 3 closure SHA: `321d5d0cff9020e70ba11cc8929e09b80dc3d655`
Phase 4 task: `ops_d0c3d4eebd67`
Phase 4 branch: `chatgpt/r9700-phase4-coldstart-parity-20260912`

## Finding

The roughly 4.6–5.1 second first-request TTFT observed after fresh vLLM process starts is directly associated with an **unexpected Triton JIT compilation after the API server has already reported ready**.

This was reproduced in the freshly restored stock ROCm10 runtime, so the phenomenon is not specific to the Phase 3 S3 MoE candidate.

## Direct runtime timeline

Stock container: `inneros-vllm-canary-rocm10`

- API server HTTP started at `2026-09-12T03:46:27.115584769Z`.
- `/v1/models` returned HTTP 200 at `03:46:31.591Z` and again at `03:46:38.776Z`.
- During the first real inference request, EngineCore emitted at `2026-09-12T03:46:42.919340225Z`:

> `Triton kernel JIT compilation during inference: _fwd_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`

- That first request completed at roughly `03:46:43.541Z`.
- Subsequent requests did not emit the same JIT warning and hot TTFT normalized to the ~50 ms class.

The warning is emitted by vLLM's post-warmup JIT monitor. Installed `gpu_worker.py` explicitly activates that monitor only after its compile/warmup sequence, kernel warmup, graph capture, sampler warmup and Inductor lazy initialization. Therefore this event is an uncovered runtime specialization, not planned startup compilation.

## Kernel-path identification

The stock backend is `ROCM_ATTN`, mapped to:

`vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend`

For the decoder-only Qwen workload, `RocmAttentionImpl.forward` calls:

`chunked_prefill_paged_decode(...)`

That module imports:

`from .prefix_prefill import context_attention_fwd`

The installed `prefix_prefill.py` contains the plain Triton kernel:

`@triton.jit def _fwd_kernel(...)`

and launches it from `context_attention_fwd` with the prefill specialization parameters. This plain function name exactly matches the JIT warning.

The separate `triton_prefill_attention.py::_fwd_kernel` path is called by ROCM_ATTN's encoder-attention method, which does not describe the decoder-only Qwen request that reproduced the event.

The evidence therefore strongly identifies the missed specialization as the ROCM_ATTN / prefix-prefill Triton `_fwd_kernel` path rather than the S3 MoE repack kernels. The latter use different function names such as `_fwd_kernel_ep_*`.

## Relevant installed warmup behavior

Installed `gpu_worker.py` performs:

1. compile-range / warmup-size selection;
2. `_dummy_run(size)` for selected warmup sizes;
3. `kernel_warmup(self)`;
4. graph capture;
5. sampler/model-runner warmup;
6. Inductor lazy initialization where applicable;
7. only then activates `jit_monitor` for unexpected inference-time compilation.

Because `_fwd_kernel` compiles after step 7, the current warmup does not cover the exact prefill shape/config used by the first real request.

## JIT monitor verbose capability

The installed runtime exposes CLI flags:

- `--jit-monitor-mode {error,warn}`
- `--jit-monitor-verbose`

When verbose mode is enabled, the Triton monitor can emit:

- constexprs;
- signature;
- extra compile info;
- specialization/cache key information.

The next diagnostic gate is therefore a fresh, isolated stock-equivalent process with `--jit-monitor-verbose`, followed by the exact first-request probe. This should reveal the missing specialization sufficiently to design a directed startup warmup.

## Current claim boundary

Proven:

- the first-request latency spike occurs after HTTP readiness;
- vLLM identifies a Triton `_fwd_kernel` JIT during that request;
- the Qwen ROCM_ATTN decoder path reaches `prefix_prefill.context_attention_fwd`, which launches a plain `_fwd_kernel`;
- existing startup warmup did not compile this specialization.

Not yet proven:

- the exact specialization key/constexpr set responsible;
- the minimal internal warmup shape required to cover it;
- whether persistent compile-cache reuse is sufficient across fresh processes;
- whether the final mitigation preserves Phase 3 S3 C1/C4/long-context behavior across independent starts.

Do not describe the cold-start issue as solved until a fresh-process A/B removes the inference-time JIT warning and normalizes first-user TTFT without breaking correctness or the Phase 3 performance gate.
