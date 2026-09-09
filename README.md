# Hyperloom R9700 Experimental

Experimental AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 64 CUs, 32 GiB-class VRAM) work focused on bringing HyperLoom-style optimization and evidence-driven vLLM experimentation to a physical RDNA4 workstation GPU.

> **Public Technical Preview — 2026-09-08**
>
> This repository documents active experimental engineering. It is **not official AMD HyperLoom support for the R9700** and it is not a production-readiness claim.

## Current status

**Phase 1: PROVEN** — architecture-neutral HyperLoom execution and independent vLLM serving validation on the physical R9700.

**Phase 2: ACTIVE / PARTIALLY PROVEN** — RDNA4-specific AutoAWQ MoE / WNA16 investigation, real Triton kernel execution, stable small-M W1 optimization, and an experimental hybrid vLLM Experts contract.

The current technical checkpoint is public on:

- branch: `chatgpt/r9700-rdna4-algebraic-checkpoint-20260908`
- commit: `42219754f7a7f61b8714fc5c3f63d7927a46b836`

Detailed public progress report:

[`docs/R9700_PHASE2_TECHNICAL_PREVIEW.md`](docs/R9700_PHASE2_TECHNICAL_PREVIEW.md)

Technical ledger on the experimental branch:

https://github.com/Rafa-Innerchispa/hyperloom-r9700-experimental/blob/chatgpt/r9700-rdna4-algebraic-checkpoint-20260908/docs/DEVELOPMENT_LEDGER.md

## Hardware and runtime

- GPU: **AMD Radeon AI PRO R9700**
- architecture: **RDNA4 / gfx1201**
- VRAM: **32 GiB-class**
- ROCm: **10.x experimental runtime path**
- serving: **vLLM**
- model used for the current MoE work: **Qwen3-Coder-30B-A3B-Instruct-AWQ**

## Experimental HyperLoom identity port

Base work started from an upstream HyperLoom snapshot and adds the minimal R9700 identity mapping:

```python
"r9700": ("gfx1201", 64)
```

plus:

```python
"gfx1201": "r9700"
```

Reproducible patch:

`patches/hyperloom-r9700-gfx1201.patch`

## Independent serving validation

A later validation campaign used **three independent vLLM starts** rather than repeated requests against a single process.

All three process-level verdicts were:

`KEEP / KEEP / KEEP`

Median results:

- baseline output throughput: **19.94 tok/s**
- candidate output throughput: **36.08 tok/s**
- paired throughput gain: **+80.99%**
- baseline TTFT p95: **119.26 ms**
- candidate TTFT p95: **168.93 ms**

Important: this is a **serving/concurrency optimization result**. It is not presented as a kernel-level or universal GPU speedup.

## Phase 2: AutoAWQ MoE / WNA16 on gfx1201

The current Qwen3-Coder AWQ MoE path exposed a concrete WNA16 backend gap on ROCm/gfx1201. The investigation showed that the tested configuration falls back to INT4 emulation while other available backends are blocked by platform or layout constraints.

That led to a dedicated packed-INT4 Triton path for RDNA4.

### Real Triton WNA16 kernel

A real Triton kernel now runs on the physical R9700 and passes numerical validation.

Representative gates:

- cosine similarity: approximately **0.999995**
- max absolute error: **0.0078125**

### Stable precomputed-correction result

The strongest current W1 kernel variant keeps expert weights packed INT4 and precomputes:

`correction = zero_point × scale`

at weight conversion / load time.

A fixed configuration was measured for **21 alternating paired HIP-event rounds per M** without retuning between rounds.

| M | Paired median speedup vs BF16 routed W1 | Wins |
|---:|---:|---:|
| 1 | **1.107x** | 17/21 |
| 2 | **1.100x** | 15/21 |
| 4 | **1.133x** | 21/21 |
| 8 | **1.123x** | 21/21 |
| 16 | **1.100x** | 20/21 |

This is the first stable kernel-level small-M W1 result that clears the BF16 pre-dequantized reference in the paired routed harness.

It is **not yet an end-to-end serving claim**.

## Experimental hybrid vLLM backend

An experimental `R9700HybridWNA16Experts` path now instantiates through the real vLLM modular MoE contract.

Synthetic smoke status:

- M=1 custom packed/correction W1 path: **PASS**
- M=20 generic fallback path: **PASS**
- real `FusedMoEKernel` construction: **PASS**
- finite outputs: **PASS**

The resident production-like vLLM process was not patched or restarted for this smoke test.

## Negative results are part of the project

The repository intentionally preserves failed or negative experiments because they narrow the search space and make the work auditable.

Examples:

- activation-group pre-sum: slower, not integrated;
- bounded AITER/FlyDSL sorting-bypass probe: isolated HSA memory fault, not promoted;
- direct FP16 runtime-mirroring probe: exposed an unresolved `fp16 × bf16` dtype boundary in the synthetic reference path.

The last item is the current integration gate.

## Immediate next step

Before enabling the hybrid backend in a real model server we will:

1. instrument an **isolated vLLM process**;
2. record the actual MoE activation dtype/layout on Qwen3-Coder AWQ;
3. load the hybrid backend against actual model weights;
4. verify output correctness and fallback behavior;
5. repeat independent process A/B runs for throughput, TTFT and E2E latency;
6. prepare an upstream-style patch/PR only if the kernel advantage survives end-to-end.

## Truth boundary

We do **not** claim:

- official AMD HyperLoom support for R9700;
- an upstream-merged RDNA4 backend;
- a universal GPU speedup;
- end-to-end serving gain caused by the new kernel;
- production readiness.

The goal is an upstream-quality experimental contribution with reproducible evidence on real hardware.

## Challenge / related integration

Judge-facing and broader InnerOS integration work lives separately so this repository can remain focused on the R9700 / HyperLoom / RDNA4 engineering path.
