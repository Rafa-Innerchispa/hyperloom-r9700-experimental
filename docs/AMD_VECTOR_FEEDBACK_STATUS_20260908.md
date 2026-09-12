# AMD Community Feedback Status — Vector.sys Review

**Date:** 2026-09-08  
**Project:** Hyperloom R9700 Experimental  
**Hardware:** AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4)  
**Status:** historical public technical progress snapshot

> Historical note: preserved from the original repository main line during the 2026-09-12 consolidation. Later Phase 3/4 documents supersede the current engineering status. Do not use this file as the current verdict.

This document maps the detailed technical feedback received from `Vector.sys [AMD]` and the ROCm AI Assistant in the AMD Developer Discord to the work completed at that time. It is intentionally explicit about what was proven, partial, deferred, or pending.

## 1. Repeat baseline/candidate to avoid single-run / bimodal confounding

**Status AT CAPTURE: PROCESS-SPAWN VALIDATION COMPLETE; GPU CLOCK/STATE TELEMETRY STILL OPEN**

The original single-run result was superseded by a three-process campaign using independent vLLM process starts.

- process verdicts: `KEEP / KEEP / KEEP`
- baseline output throughput min/median/max: approximately `19.89 / 19.94 / 19.99 tok/s`
- candidate output throughput min/median/max: approximately `36.00 / 36.08 / 36.19 tok/s`
- paired gain min/median/max: approximately `80.51% / 80.99% / 81.49%`
- baseline cross-process spread: approximately `0.47%`
- `audit.ok = true`

Truth boundary: this was a serving/concurrency optimization result, not a kernel-level GPU speedup.

## 2. AITER gate / `ROCM_AITER_UNIFIED_ATTN` stress test

**Status AT CAPTURE: PARTIAL / NOT YET CLOSED**

The tested runtime selected `ROCM_ATTN`. A bounded AITER/FlyDSL sorting-bypass experiment produced `HSA_STATUS_ERROR_MEMORY_FAULT` in `moe_sorting_oneshot_kernel_0`; the path was not promoted.

## 3. Missing / untuned Triton or MoE configurations for R9700

**Status AT CAPTURE: PARTIAL, WITH CUSTOM KERNEL WORK COMPLETED**

Phase 2 isolated the AutoAWQ MoE WNA16 fallback and built a packed-INT4 Triton kernel for `gfx1201`.

Fixed configuration at that checkpoint:

- `BM=16`
- `BN=128`
- `BK=32`
- `num_warps=4`
- `num_stages=1`
- `waves_per_eu=4`

Representative paired W1 results versus the BF16 pre-dequantized routed reference:

- M=1: `1.107x`, 17/21 wins
- M=2: `1.100x`, 15/21 wins
- M=4: `1.133x`, 21/21 wins
- M=8: `1.123x`, 21/21 wins
- M=16: `1.100x`, 20/21 wins

Numerical gates remained approximately cosine `0.999995`, max absolute error `0.0078125`.

## 4. FP8 KV cache / attention on gfx1201

**Status AT CAPTURE: DEFERRED BY DESIGN**

The project prioritized the observed AutoAWQ WNA16 MoE bottleneck rather than FP8 KV-cache/attention.

## 5. Study ROCm/AITER gfx1201 enablement

**Status AT CAPTURE: INVESTIGATED**

The project used the AITER gfx1201 tracking work as reference and pursued a Triton-first RDNA4 path rather than claiming unsupported production enablement.

## 6. Study deeper community RDNA4 vLLM builds

**Status AT CAPTURE: INVESTIGATED / NO DEPENDENCY CLAIMED**

The differentiating path at the time was:

`R9700 / gfx1201 -> ROCm 10 -> vLLM -> Qwen3-Coder AWQ MoE -> WNA16 -> packed INT4 Triton -> hybrid Experts backend`

## 7. Use AMD's official vLLM optimization guidance

**Status AT CAPTURE: INCORPORATED AS REFERENCE**

Official AMD vLLM optimization guidance was treated as authoritative while keeping a strict distinction between validated Instinct configurations and experimental Radeon/gfx1201 behavior.

## 8. Autonomous rediscovery

**Status AT CAPTURE: OPEN**

The project had not proven autonomous rediscovery of the known AITER optimization.

## 9. `VLLM_ROCM_USE_AITER_RMSNORM=0`

**Status AT CAPTURE: NOT VERIFIED IN THIS REPOSITORY**

The exact environment flag was not marked complete without repository evidence.

## Historical claim boundary

This document never claimed official AMD HyperLoom support for R9700, an upstream-merged RDNA4 backend, autonomous rediscovery of AITER optimization, or an end-to-end serving gain caused by the then-new WNA16 kernel without isolated real-model evidence.

For current truth, read `docs/R9700_PROJECT_CONTINUITY.md`, `FINAL_STATUS.md`, and the active Phase 4 handoff.
