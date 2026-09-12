# HyperLoom R9700 — Phase 4 Active Handoff

Last updated: 2026-09-12 (America/Guayaquil)

## READ THIS FIRST IN ANY NEW CHAT

Do **not** repeat Phase 2 or Phase 3 experiments. They are closed and preserved.

Current active work is **Phase 4: cold-start/parity optimization**.

### Canonical identity

- Repo: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Active branch: `chatgpt/r9700-phase4-coldstart-parity-20260912`
- Phase 4 task: `ops_d0c3d4eebd67`
- Parent Phase 3 closure SHA: `321d5d0cff9020e70ba11cc8929e09b80dc3d655`
- Phase 4 root-cause commit before this handoff: `7c8627ae6c132f83d2b487cc066406e053dd7e88`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Operational default remains stock ROCm10 on port 8000.

## Phase 3 is CLOSED — do not redo it

Selected experimental full-model S3 config passed the engineering gate for batched C4/concurrent serving.

Three independent fresh S3 processes:

- conservative first-measure C4 median: `188.598166 tok/s`
- hot-repeat C4 median: `192.460522 tok/s`
- combined six-observation C4 median: `190.529344 tok/s`
- correctness hash matched stock: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- all four canonical C4 output hashes matched stock
- C1 was near parity with healthy stock
- long context was within a few percent of healthy stock
- stock was restored healthy after closure

Safe headline: `188.6 tok/s` conservative independent-start C4 median, about `+19%` versus stable stock+queue1 and about `+13.9%` versus freshly restored healthy-hot stock.

Do not use the single best `193.948 tok/s` observation as the universal result.

Important Phase 3 artifacts:

1. `FINAL_STATUS.md`
2. `docs/R9700_PHASE3_S3_CLOSURE_20260912.md`
3. `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`
4. `docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`
5. `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
6. `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`

Phase 3 runtime patch SHA256:
`3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`

S3 config SHA256:
`8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`

## Phase 4 current finding — root cause identified

The recurring fresh-process first-request TTFT spike of roughly `4.6-5.1 s` is **not specific to S3**. It was reproduced on freshly restored stock ROCm10.

Direct evidence:

- vLLM API reached HTTP-ready state;
- `/v1/models` returned HTTP 200;
- on the first real inference request, vLLM emitted an unexpected Triton JIT warning for `_fwd_kernel`;
- subsequent requests did not emit the same warning and hot TTFT normalized to roughly `50 ms` class.

The evidence strongly identifies the missed specialization as the decoder ROCm attention path:

`ROCM_ATTN -> RocmAttentionImpl.forward -> chunked_prefill_paged_decode -> prefix_prefill.context_attention_fwd -> @triton.jit _fwd_kernel`

This is **not** the Phase 3 S3 MoE repack kernel path. S3 MoE kernels use different function names.

Canonical Phase 4 root-cause document:

`docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`

Root-cause commit:

`7c8627ae6c132f83d2b487cc066406e053dd7e88`

## Exact next technical gate

Resume from here. Do not rerun Phase 3 performance campaigns.

1. Start one fresh isolated stock-equivalent process with `--jit-monitor-verbose`.
2. Send the exact first-request probe that reproduces the ~5 s latency.
3. Capture Triton verbose information for the unexpected `_fwd_kernel` compile: constexprs, signature, compile info and/or specialization/cache key.
4. Identify the exact missing `prefix_prefill` specialization.
5. Implement the smallest reversible startup warmup/precompile that covers that specialization.
6. Fresh-process A/B:
   - no unexpected inference-time JIT warning;
   - first-user TTFT moves toward hot class rather than ~5 s;
   - stock-exact correctness remains intact.
7. Apply the same mitigation to the S3 candidate and verify that it preserves:
   - conservative S3 C4 result materially near the closed Phase 3 gate;
   - C1/long-context parity;
   - canonical correctness and C4 hashes.
8. Repeat independent fresh starts as needed for reproducibility.
9. Restore stock ROCm10 operational default and verify `/v1/models` HTTP 200.
10. Preserve raw evidence, negative results, final verdict and updated continuity before closing Phase 4.

## Current claim boundary

Proven:

- Phase 3 S3 full-model C4 engineering promotion passed.
- Stock remains operational default.
- The cold first-request spike occurs on stock too.
- The spike is associated with an unexpected Triton `_fwd_kernel` JIT after HTTP readiness.
- The decoder ROCm prefix-prefill path strongly matches the warned kernel.

Not yet proven:

- exact specialization key/constexpr set;
- minimal warmup shape required;
- whether persistent compile-cache reuse alone is enough;
- final fresh-process mitigation;
- that the mitigation preserves Phase 3 S3 performance across independent starts.

Do not claim cold-start solved until fresh-process A/B proves it.

## Worktrees — NEVER CLEAN BLINDLY

Phase 3 canonical worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`

AMD runtime/raw evidence worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

Historical AMD research worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset or delete these before inventory/archive.

## Restart rule

A new ChatGPT/Codex session should:

1. read this file first;
2. read `docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`;
3. inspect current remote HEAD of `chatgpt/r9700-phase4-coldstart-parity-20260912`;
4. inspect ops task `ops_d0c3d4eebd67`;
5. verify runtime state before mutation;
6. continue from the exact next technical gate above.

Do not repeat closed Phase 2/3 experiments merely to reconstruct context.