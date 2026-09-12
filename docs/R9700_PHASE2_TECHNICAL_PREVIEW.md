# R9700 Phase 2 Technical Preview

**Date:** 2026-09-08  
**Status:** Historical experimental engineering snapshot  
**Hardware:** AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GiB-class VRAM)  
**Runtime:** ROCm 10, vLLM, Qwen3-Coder-30B-A3B-Instruct-AWQ

> Preserved during the 2026-09-12 main-line consolidation. Phase 3 and Phase 4 supersede this as the current engineering state. This file is historical evidence, not the current verdict.

## What was proven at this checkpoint

### 1. HyperLoom architecture-neutral execution on physical R9700

The experimental identity path recognized the R9700 as `gfx1201` / 64 CU and executed the architecture-neutral HyperLoom benchmark flow against the physical GPU.

### 2. Independent serving validation

A three-process validation campaign used independent vLLM starts and produced `KEEP / KEEP / KEEP`.

Median measurements across the three independent starts:

- baseline output throughput: **19.94 tok/s**
- candidate output throughput: **36.08 tok/s**
- paired throughput gain: **+80.99%**
- baseline TTFT p95: **119.26 ms**
- candidate TTFT p95: **168.93 ms**

This was a serving/concurrency result, not a kernel-level GPU speedup.

### 3. WNA16 / AutoAWQ MoE backend root cause

The tested Qwen3-Coder AWQ MoE configuration fell back to INT4 emulation. Investigation isolated backend gates in Marlin, Humming and FlashInfer paths and made a dedicated packed-INT4 Triton path a concrete optimization target.

### 4. Real RDNA4 Triton WNA16 kernel

A real Triton WNA16 kernel executed on the physical R9700 and passed numerical validation.

Representative numerical gates:

- cosine similarity: approximately **0.999995**
- max absolute error: **0.0078125**

### 5. Precomputed-correction optimization

The strongest variant at that checkpoint kept W1 expert weights packed INT4 and precomputed `correction = zero_point × scale` at load/conversion time.

Fixed configuration:

- `BM=16`
- `BN=128`
- `BK=32`
- `num_warps=4`
- `num_stages=1`
- `waves_per_eu=4`

Results versus the BF16 pre-dequantized routed W1 reference:

| M | Paired median speedup | Wins |
|---:|---:|---:|
| 1 | **1.107x** | 17/21 |
| 2 | **1.100x** | 15/21 |
| 4 | **1.133x** | 21/21 |
| 8 | **1.123x** | 21/21 |
| 16 | **1.100x** | 20/21 |

These were kernel-level small-M W1 results, not end-to-end serving results.

### 6. Hybrid vLLM Experts contract

An experimental `R9700HybridWNA16Experts` path instantiated through the real vLLM modular MoE contract. Synthetic M=1 custom-path and M=20 fallback smokes passed.

## Negative results intentionally preserved

- standalone activation-group pre-sum was slower and was not integrated;
- bounded AITER/FlyDSL sorting bypass produced an isolated HSA memory fault and was not promoted;
- direct FP16 runtime-mirroring exposed an unresolved `fp16 × bf16` boundary in the synthetic reference path at that time.

## Historical truth boundary

This checkpoint did not claim official AMD HyperLoom support for R9700, an upstream-merged RDNA4 backend, a universal GPU speedup, an end-to-end serving gain caused by the kernel, or production readiness.

The later Phase 2 final result, Phase 3 S3 result and active Phase 4 state are documented in `FINAL_STATUS.md` and `docs/R9700_PROJECT_CONTINUITY.md`.
