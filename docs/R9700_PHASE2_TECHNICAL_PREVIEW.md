# R9700 Phase 2 Technical Preview

**Date:** 2026-09-08  
**Status:** Active experimental engineering  
**Hardware:** AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GiB-class VRAM)  
**Runtime:** ROCm 10, vLLM, Qwen3-Coder-30B-A3B-Instruct-AWQ  

This document is a public progress snapshot, not a claim of official AMD or HyperLoom support for the R9700.

## What is already proven

### 1. HyperLoom architecture-neutral execution on physical R9700

The experimental identity path recognizes the R9700 as `gfx1201` / 64 CU and can execute the architecture-neutral HyperLoom benchmark flow against the physical GPU.

### 2. Independent serving validation

A three-process validation campaign used independent vLLM starts and produced `KEEP / KEEP / KEEP`.

Median measurements across the three independent starts:

- baseline output throughput: **19.94 tok/s**
- candidate output throughput: **36.08 tok/s**
- paired throughput gain: **+80.99%**
- baseline TTFT p95: **119.26 ms**
- candidate TTFT p95: **168.93 ms**

This is a serving/concurrency result. It is **not** attributed to GEAK, Arbor or an RDNA4 kernel.

### 3. WNA16 / AutoAWQ MoE backend root cause

For the tested Qwen3-Coder AWQ MoE configuration, the current path falls back to INT4 emulation. The investigation isolated several backend gates:

- Marlin rejects ROCm in the relevant support check.
- Humming is CUDA-oriented for the tested path.
- FlashInfer TRT-LLM conflicts with zero-point / bias constraints for this configuration.
- Triton is device-compatible with ROCm/gfx1201, but AutoAWQ layout / eligibility prevents the stock path from being selected.

This made a dedicated packed-INT4 Triton path a concrete optimization target rather than a speculative rewrite.

### 4. Real RDNA4 Triton WNA16 kernel

A real Triton WNA16 kernel now executes on the physical R9700 and passes numerical validation.

Representative numerical gates:

- cosine similarity: approximately **0.999995**
- max absolute error: **0.0078125**

### 5. Precomputed-correction optimization

The strongest current kernel variant keeps W1 expert weights packed INT4 and precomputes:

`correction = zero_point × scale`

at weight-conversion/load time. This removes zero-point unpack and the `zero_point × scale` multiply from the request hot path.

A fixed configuration was then measured for **21 alternating paired HIP-event rounds per M**, with no retuning between rounds.

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

These are kernel-level small-M W1 results, not end-to-end serving results.

### 6. Hybrid vLLM Experts contract

An experimental `R9700HybridWNA16Experts` path now instantiates through the real vLLM modular MoE contract.

Synthetic contract smoke:

- M=1: custom packed/correction W1 path -> **PASS**
- M=20: generic fallback path -> **PASS**
- `FusedMoEKernel` construction -> **PASS**
- outputs finite -> **PASS**

The production/resident vLLM server was not patched or restarted for this smoke test.

## Negative results intentionally preserved

The project keeps failed and negative experiments because they narrow the search space:

- a standalone activation-group pre-sum kernel was slower and is not being integrated;
- a bounded AITER/FlyDSL sorting-bypass experiment produced an isolated HSA memory fault and is not being promoted;
- a direct FP16 runtime-mirroring probe exposed an unresolved `fp16 × bf16` Triton dtype boundary in the synthetic reference path.

The last item is the current integration gate. It does not invalidate the packed kernel or hybrid contract smoke; it means the exact activation dtype/layout entering the live MoE path must be measured before enabling the hybrid backend on the real model.

## Current truth boundary

We are **not** claiming:

- official AMD HyperLoom support for Radeon AI PRO R9700;
- an upstream-merged RDNA4 backend;
- a universal GPU speedup;
- an end-to-end serving gain caused by the new kernel;
- production readiness.

## Immediate next gate

1. Launch an isolated vLLM process with instrumentation.
2. Record the real MoE `hidden_states.dtype` / layout on Qwen3-Coder AWQ.
3. Load the hybrid backend with actual model weights.
4. Validate output correctness and fallback behavior.
5. Repeat independent serving A/B runs with throughput, TTFT and E2E evidence.
6. Only if the kernel advantage survives end-to-end, prepare an upstream-style patch/PR candidate.

## Reproducible technical branch

Current technical checkpoint:

- branch: `chatgpt/r9700-rdna4-algebraic-checkpoint-20260908`
- commit: `42219754f7a7f61b8714fc5c3f63d7927a46b836`

Technical branch:

https://github.com/Rafa-Innerchispa/hyperloom-r9700-experimental/tree/chatgpt/r9700-rdna4-algebraic-checkpoint-20260908

Key ledger:

https://github.com/Rafa-Innerchispa/hyperloom-r9700-experimental/blob/chatgpt/r9700-rdna4-algebraic-checkpoint-20260908/docs/DEVELOPMENT_LEDGER.md

## Evidence discipline

The branch preserves successful runs, negative results, source probes, harnesses and evidence JSONs. Every claim above is intentionally scoped to the evidence that actually exists. The goal is an upstream-quality experimental contribution, not a support claim that AMD has not made.
