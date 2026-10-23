# ROCm 10.1 & Upstream Compatibility Matrix: Radeon AI PRO R9700 (gfx1201)

## Executive Summary
This document defines the verified compatibility matrix between ROCm 10.1, upstream vLLM (v0.32.0), HyperLoom (1.1.0), and the physical AMD Radeon AI PRO R9700 (`gfx1201` / DID `0x7551`).

### Non-Negotiable Operational Boundaries
- **ACT III Freeze Untouched**: Branch `freeze/act3-live-verified-20261017` at `a1b4520921c5198e1444edcb1b8569e00eb4f7a7` is 100% read-only and preserved.
- **Stock S3 Production Protected**: Stock service `inneros-vllm-qwen3-coder-30b-awq` on port 8000 remains uninterrupted and running.
- **Isolated Branch**: All work executed strictly in `Rafa-Innerchispa/hyperloom-r9700-experimental` on branch `antigravity/hyperloom-rocm101-refresh-20261018`.

---

## Hardware Identity & Host Profile
- **GPU**: AMD Radeon AI PRO R9700 (RDNA4 / `gfx1201`, 64 CUs, Device ID `0x7551`, 32GB VRAM).
- **Host Nodes**: `/dev/kfd` and `/dev/dri` accessible with video/render groups.
- **Model**: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` located locally at `/home/rlopez/inneros/inneros_core/var/local_models/`.

---

## Component Compatibility Matrix

| Component | Upstream / Baseline Version | R9700 (`gfx1201`) Support Status | Required Decision |
|---|---|---|---|
| **ROCm Driver/Runtime** | ROCm 10.0 / 10.1 | Official native recognition for `gfx1200` & `gfx1201`. | **Adopt upstream native recognition**; retire historical device spoofing. |
| **PyTorch ROCm** | `torch 2.13.0 / 2.14.0+rocm10` | Verified functional via Docker smoke on physical GPU (`cuda_available: True`). | **Candidate Ready** (`rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0`). |
| **vLLM Upstream** | `v0.32.0` (baseline was `v0.29.0`) | PR #43389 closed unmerged; `int_wna16.py` still explicitly rejects AutoAWQ layout on Triton. | **Retain S3 Overlay** (`vllm_v0_29_0_r9700_awq_triton_candidate.patch`); upstream still lacks out-of-the-box AutoAWQ Triton MoE. |
| **HyperLoom Upstream** | `1.1.0` | Supported platforms list only MI300X, MI325X, MI355X; no declared Radeon support. | **Retain HyperLoom R9700 experimental fork**. |
| **Triton AMD Backend** | Gluon / AMD `gfx1200/gfx1201` | Architecture supported, but requires readiness warmup to eliminate JIT latency penalty on first user turn. | **Retain deterministic prefix warmup**. |

---

## Patch Lifecycle & Retiring of Obsolete Hacks

1. **Retire (Obsolete)**:
   - Standalone GPU device-naming / detection patches: ROCm 10+ natively enumerates `gfx1201`.
2. **Retain (Mandatory)**:
   - `wna16_hybrid_patch` / `vllm_v0_29_0_r9700_awq_triton_candidate.patch`: Required until upstream vLLM provides native AutoAWQ Triton MoE support for RDNA4.
   - `GPU_MAX_HW_QUEUES=1`: Performance tuning ensuring single-stream determinism and wave scheduling stability.
   - Prefix prefill readiness warmup: Prevents ~500ms initial JIT compilation overhead on live agent queries.

---

## Agent Runtime & Benchmark Evidence

### Truth boundary

The autonomous E2E and C1-C16 serving benchmarks prove that the local R9700 serving stack and Qwen3-Coder workload are stable and scale under concurrency. They **do not by themselves prove that the ROCm 10.1 vLLM candidate image was the serving runtime**. The recorded E2E harness targets `http://127.0.0.1:8000/v1`, while the production/S3 service was intentionally preserved during the ROCm 10.1 canary work. Treat these results as R9700 serving evidence, not as ROCm 10.1 promotion evidence.

The ROCm 10.1 evidence currently establishes:
- physical R9700/gfx1201 preflight;
- ROCm 10.1 PyTorch framework/GPU smoke;
- compatibility planning for ROCm 10.1 + vLLM;
- upstream/patch lifecycle analysis.

Still required before ROCm 10.1 can be promoted:
- raw runtime manifest from the actual ROCm 10.1 candidate container;
- exact 30B model load on that candidate;
- observed AWQ/WNA16 backend/path proof;
- deterministic correctness hash;
- C1/C4/long-context benchmark on the candidate itself;
- VRAM, cold/warm TTFT, soak, and clean stock restore evidence.

### Preserved R9700 serving baseline evidence

- **Upstream Autonomous E2E Agent Harness (`r9700_upstream_agent_e2e.py`)**:
  - Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
  - Baseline Concurrency 1: 18.92 tok/s (median e2e 757ms)
  - Candidate Concurrency 2: 35.07 tok/s (median e2e 800ms)
  - **Throughput Gain**: **+85.36%** (Verdict: **KEEP**)
  - Evidence File: `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20261019T043055453703Z.json`
  - Classification: serving/concurrency evidence; **not ROCm 10.1 candidate proof**.
- **Tool-Call / Agent Workload Finding**:
  - For native OpenAI function-calling (`tools` parameter), future vLLM candidate container must be launched with:
    `--enable-auto-tool-choice` and `--tool-call-parser hermes` (or `qwen`).
  - Standard prompt-based coding agent inference verified at 21.11 tok/s with correct code output.
