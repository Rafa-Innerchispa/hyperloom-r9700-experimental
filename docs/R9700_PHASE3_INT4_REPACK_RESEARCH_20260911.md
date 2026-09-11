# HyperLoom R9700 Phase 3 — RDNA INT4 MoE repack research

Date: 2026-09-11
Base milestone: `d7aa6ca77a98731ece08e859925fd94ecfa111d5`
Branch: `chatgpt/r9700-phase3-int4-repack-20260911`
Task: `ops_d07579f84116`

## Why Phase 3 exists

Phase 2 proved that the custom packed-INT4 W1 microkernel is genuinely faster than stock Triton WNA16 W1 on the physical Radeon AI PRO R9700 / gfx1201, but the `v7_clean` hybrid full-model integration remained about 4.5–5.1% below the stable stock+`GPU_MAX_HW_QUEUES=1` C4 serving baseline. The correct response is not to claim a false full-model win and not to stop: investigate newer RDNA4 work that attacks the integrated WNA16/MoE path rather than only W1.

## Critical new upstream finding: vLLM PR #43389

Upstream: https://github.com/vllm-project/vllm/pull/43389

State observed 2026-09-11: OPEN, not merged.

Primary upstream implementation commit:
`f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3`

Follow-up comment-only commit:
`56ef89e1ff4a1552beb3b5c51c00b73ea44daca1`

Author: `amd-xavierwang` / AMD.

The PR optimizes the ROCm fused MoE INT4 W4A16 path by changing the packed weight layout at load time:

- old: K-packed `uint8 [E, N, K//2]`;
- new: N-packed `int32 [E, K, N//8]`;
- scales are transposed for the new per-N layout;
- AWQ zero points are unpacked/prepared at load time;
- the Triton fused MoE kernel uses `tl.interleave` and constant shifts rather than repeatedly loading packed bytes and using variable per-element shifts;
- W1/W13 and W2 are repacked independently when their N dimension is compatible;
- non-compatible weights fall back to the existing scalar layout/path;
- the implementation is guarded to AMD RDNA rather than CDNA.

The motivation is especially relevant to this project because the real Qwen workload uses FP16 activations. The PR reports that the current FP16 int4 dequant path causes high VGPR pressure/spilling, while the interleaved layout reduces those live intermediates.

### Upstream benchmark directly relevant to us

For `Qwen3-30B-A3B-AWQ`, FP16, on `gfx1201`:

- batched output throughput: `685 -> 928 tok/s`, reported `+35%`;
- median TPOT: `233 -> 164 ms`, about `-30%`;
- median ITL: `199 -> 139 ms`, about `-30%`;
- median TTFT: `21263 -> 18093 ms`, about `-15%`;
- single-batch decode: `40.19 -> 54.22 tok/s`, about `+35%`;
- single-batch TPOT: `24.14 -> 17.95 ms`, about `-26%`.

These are upstream measurements, not InnerChispa measurements. They must not be quoted as our result until reproduced on our physical R9700 and our Qwen3-Coder AWQ checkpoint.

The upstream reported margin is materially larger than the roughly 5% deficit of our Phase-2 hybrid and therefore justifies a Phase-3 backport experiment.

## Accuracy/review status of #43389

Reviewers initially questioned an apparent ~0.5% accuracy movement and the safety of changing a shared path. The author subsequently reran full-dataset lm_eval across gfx1100, gfx1151 and gfx1201, guarded the change to RDNA, and reported no generalized cross-architecture regression. The PR remains open and therefore remains experimental input, not accepted upstream truth.

Phase 3 will require our own deterministic/token parity checks and will not rely on the upstream accuracy conclusion.

## Other current upstream findings

### `RDNAHybridW4A16LinearKernel`

Current vLLM main contains `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py`, a hybrid dense W4A16 kernel with explicit gfx1201/R9700-tuned tile heuristics. It uses a HIP skinny kernel for decode and Triton for larger-M prefill. This did not exist in our 2026-08-27 vLLM snapshot.

Relevant original PR: https://github.com/vllm-project/vllm/pull/40977

This is dense-linear work, while our main deficit is in routed MoE WNA16. Its architecture is still useful as a template: choose a specialized kernel by runtime M while keeping a single packed representation and a safe fallback.

### Native RDNA3 W4A16 fused MoE

PR: https://github.com/vllm-project/vllm/pull/44075

vLLM has a native HIP W4A16 fused-MoE backend for gfx1100 and its design explicitly leaves dispatch extensible for later architectures. This is a deeper Phase-3B option if the Triton repack from #43389 is insufficient. Do not port this first because the gfx1100 kernel uses architecture-specific assumptions and our immediate candidate already has direct gfx1201 measurements.

### Compile-safe dispatch matters

PR: https://github.com/vllm-project/vllm/pull/51453

Recent vLLM work registers W4A16 Triton GEMM as an opaque custom op so `torch.compile` does not freeze Python tile-selection logic at capture time. This aligns with our own rejected mmap/signal runtime-gate experiment: dynamic Python switching after graph capture is not a valid path. Phase 3 must select/repack before capture and keep runtime dispatch compile-safe.

### Current RDNA4 hardware status

ROCm 10 documentation lists Radeon AI PRO R9700 as RDNA4/gfx1201. Current vLLM main includes gfx1201 in supported HIP architectures and recognizes PCI ID `0x7551` as `AMD_Radeon_R9700`.

AITER remains useful for some RDNA4 operators but its fused-MoE coverage is not the strongest immediate route for this AWQ INT4 workload. Phase 3 prioritizes vLLM's Triton WNA16 repack path.

## Phase-3 hypothesis

The Phase-2 v7 design accelerated only the custom small-M W1 route but paid integration/routing overhead and retained stock W2. PR #43389 instead optimizes the existing fused MoE INT4 layout at load time and applies independently to both W1/W13 and W2 launches. If the reported reduction in VGPR pressure exists on our gfx1201, this should have a better chance of converting a kernel improvement into full-model E2E throughput.

## Experiment order

1. Keep the presentation milestone `d7aa6ca77...` untouched.
2. Inspect exact vLLM 0.27.1 ROCm10 source in the current container and map the #43389 changes onto that version.
3. Backport the minimal repack/interleave path as an isolated, auditable overlay. Do not patch the stable container in place.
4. Validate repack reversibility/layout and real Qwen AWQ W1/W2 numerical correctness.
5. Benchmark the complete routed W1+W2 MoE path, not W1 alone.
6. Only if the real-weight gate passes, run full Qwen with `ROCM_ATTN` + `GPU_MAX_HW_QUEUES=1`.
7. Compare against stable stock queue1 ~158.49 tok/s and credible fast-stock 162.10–165.70 tok/s. Never use a pathological ~72 tok/s spawn as the promotion baseline.
8. If promising, repeat with independent process starts.
9. Promote only with preserved raw evidence and correctness. Otherwise preserve the result and move to the deeper RDNA4 native-HIP/oracle path.

## Current truth

Phase 2 is still a valid closed milestone. Phase 3 reopens optimization based on newly identified upstream RDNA work. Nothing in this document changes previously measured evidence; it defines the next evidence-generating experiment.
