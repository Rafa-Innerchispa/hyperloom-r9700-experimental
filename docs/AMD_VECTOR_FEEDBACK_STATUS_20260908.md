# AMD Community Feedback Status — Vector.sys Review

**Date:** 2026-09-08  
**Project:** Hyperloom R9700 Experimental  
**Hardware:** AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4)  
**Status:** public technical progress snapshot

This document maps the detailed technical feedback received from `Vector.sys [AMD]` and the ROCm AI Assistant in the AMD Developer Discord to the work completed so far. It is intentionally explicit about what is proven, partial, deferred, or still pending.

## 1. Repeat baseline/candidate to avoid single-run / bimodal confounding

**Status: PROCESS-SPAWN VALIDATION COMPLETE; GPU CLOCK/STATE TELEMETRY STILL OPEN**

The original single-run result was superseded by a three-process campaign using independent vLLM process starts.

- process verdicts: `KEEP / KEEP / KEEP`
- baseline output throughput min/median/max: approximately `19.89 / 19.94 / 19.99 tok/s`
- candidate output throughput min/median/max: approximately `36.00 / 36.08 / 36.19 tok/s`
- paired gain min/median/max: approximately `80.51% / 80.99% / 81.49%`
- baseline cross-process spread: approximately `0.47%`
- `audit.ok = true`

This closes the most important process-spawn repeatability concern. However, the public evidence does **not** yet prove that SCLK/GFXCLK or equivalent GPU runtime state was captured under load for every independent spawn. That sub-point remains open for a future validation campaign.

Truth boundary: this is a serving/concurrency optimization result, not a kernel-level GPU speedup.

Primary evidence is preserved on the experimental branch in `docs/evidence/r9700_independent_process_final_20260908.json`.

## 2. AITER gate / `ROCM_AITER_UNIFIED_ATTN` stress test

**Status: PARTIAL / NOT YET CLOSED**

We investigated the live attention path and AITER support boundary on the R9700. The current tested runtime selected `ROCM_ATTN`. AITER is installed, but the observed support path remains constrained for `gfx1201`.

A bounded AITER/FlyDSL sorting-bypass experiment was also performed in an isolated child process. It produced `HSA_STATUS_ERROR_MEMORY_FAULT` in `moe_sorting_oneshot_kernel_0`; the failure is preserved and the path was not promoted.

We have **not** yet claimed the exact long-context `ROCM_AITER_UNIFIED_ATTN` gate-relaxation experiment suggested by Vector as complete. That remains a separate stress-test target after the current WNA16 integration gate.

## 3. Missing / untuned Triton or MoE configurations for R9700

**Status: PARTIAL, WITH DEEPER CUSTOM KERNEL WORK COMPLETED**

Rather than only generating a stock per-shape config, Phase 2 isolated the AutoAWQ MoE WNA16 fallback and built a dedicated packed-INT4 Triton kernel for `gfx1201`.

The strongest current W1 variant precomputes `correction = zero_point × scale` at load/conversion time and uses a fixed RDNA4 configuration:

- `BM=16`
- `BN=128`
- `BK=32`
- `num_warps=4`
- `num_stages=1`
- `waves_per_eu=4`

In 21 alternating paired HIP-event rounds per M versus the BF16 pre-dequantized routed W1 reference:

- M=1: `1.107x`, 17/21 wins
- M=2: `1.100x`, 15/21 wins
- M=4: `1.133x`, 21/21 wins
- M=8: `1.123x`, 21/21 wins
- M=16: `1.100x`, 20/21 wins

Numerical gates remained approximately cosine `0.999995`, max absolute error `0.0078125`.

This does **not** replace the value of contributing upstream vLLM/AITER tuned configuration files. A stock upstream tuning/config contribution remains open work.

## 4. FP8 KV cache / attention on gfx1201

**Status: DEFERRED BY DESIGN**

Vector recommended not making FP8 KV-cache/attention the primary Arbor target because dequantization overhead may erase gains on gfx1201. We followed that prioritization and focused Phase 2 on the observed AutoAWQ WNA16 MoE bottleneck instead.

A current-build FP8 KV quick check is still optional future work and is not claimed complete.

## 5. Study ROCm/AITER gfx1201 enablement

**Status: INVESTIGATED**

The AITER gfx1201 tracking issue and community enablement work were used as references. The current public AITER discussion describes gfx1201 support as work in progress, with a Triton-first functional path and HIP/FlyDSL performance work following.

This aligns with the project's decision to pursue a Triton-first RDNA4 WNA16 path rather than claim unsupported AITER production enablement.

## 6. Study deeper community RDNA4 vLLM builds

**Status: INVESTIGATED / NO DEPENDENCY CLAIMED**

Community projects that rebuild or patch vLLM/AITER/attention for `gfx1201` are useful references for deeper enablement. We have not presented those community patches as part of our implementation and do not claim to have upstreamed or adopted them wholesale.

Our current differentiating path is specifically:

`R9700 / gfx1201 -> ROCm 10 -> vLLM -> Qwen3-Coder AWQ MoE -> WNA16 -> packed INT4 Triton -> hybrid Experts backend`

## 7. Use AMD's official vLLM optimization guidance

**Status: INCORPORATED AS REFERENCE**

The current official AMD vLLM optimization guide documents AITER controls, attention backend selection and Radeon fallback paths. The project treats those docs as authoritative guidance while keeping a strict distinction between validated Instinct configurations and experimental Radeon/gfx1201 behavior.

## 8. Test autonomous rediscovery rather than manually seed known optimizations

**Status: OPEN**

Vector's most interesting Phase 2 question remains unanswered: can the HyperLoom-style agent loop independently rediscover a known gfx1201 optimization such as Unified Attention or a missing tuned config, rather than being told exactly what to enable?

The current WNA16 kernel work was driven by measured backend analysis and iterative engineering. It should not be described as proof that GEAK/Arbor autonomously rediscovered the known AITER optimization.

## 9. `VLLM_ROCM_USE_AITER_RMSNORM=0`

**Status: NOT VERIFIED IN THIS REPOSITORY**

The ROCm AI Assistant suggested that disabling AITER RMSNorm is appropriate for gfx12 targets. The current public repository does not contain evidence that this exact environment flag is part of the validated runtime configuration, so it is not marked complete here.

## Current integration gate

The next gate before any kernel-level end-to-end serving claim is:

1. launch an isolated vLLM process with instrumentation;
2. record the actual MoE activation dtype/layout on the loaded Qwen3-Coder AWQ model;
3. load the experimental hybrid WNA16 Experts path with real weights;
4. verify numerical correctness and fallback behavior;
5. repeat independent serving A/B runs and capture throughput, TTFT and E2E evidence;
6. add GPU clock/runtime-state telemetry per spawn where practical.

## Claim boundary

We do not claim official AMD HyperLoom support for R9700, an upstream-merged RDNA4 backend, an autonomous rediscovery of the known AITER optimization, or an end-to-end serving gain caused by the new WNA16 kernel until the relevant isolated real-model campaigns prove those claims.
