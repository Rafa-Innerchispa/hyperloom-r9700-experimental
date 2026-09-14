# R9700 upstream delta — 2026-09-13

Status: **engineering review complete; no live migration authorized**

This document compares the production-proven HyperLoom R9700 path with current upstream HyperLoom, vLLM, ROCm, and Triton state. It is intentionally conservative: upstream feature presence is not treated as proof that the exact production workload is equivalent.

## Pinned baselines

| Component | Pinned state used for this review |
| --- | --- |
| This repository | `b442bf11b24d715ef17b26c26deccc86fa59ba22` |
| HyperLoom upstream `main` | `ec3b1cbe9da752398388ea3497a3b40a500e0387` |
| vLLM upstream `main` | `a2685f2cdace04138d10719b9bb612a67bf20886` |
| Production workload | `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` on Radeon AI PRO R9700 / `gfx1201` |
| Proven production overlay | S3 WNA16 hybrid, Phase 5 + Phase 6 acceptance already closed |

The live service is not changed by this review.

## Current upstream facts

### ROCm and Triton now understand RDNA4 directly

AMD ROCm documentation lists Radeon AI PRO R9700 as RDNA4, LLVM target `gfx1201`, 32 GiB VRAM, and 64 compute units. Current Triton sources also model RDNA4 explicitly (`gfx1200` / `gfx1201`). Basic architecture recognition is therefore no longer the interesting engineering problem.

References:

- https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html
- https://github.com/triton-lang/triton/blob/main/python/triton/experimental/gluon/language/amd/_layouts.py

### HyperLoom still does not map R9700

At pinned HyperLoom upstream `ec3b1cbe...`, `gpu_identity.py` has no `r9700 -> gfx1201, 64 CU` mapping and `gpu_types.py` has no `gfx1201 -> r9700` reverse mapping.

Our experimental patch still adds exactly those two mappings:

```text
"r9700": {"arch": "gfx1201", "cu": 64}
"gfx1201": "r9700"
```

Therefore the HyperLoom identity extension remains a real local delta and must be retained unless/until upstream accepts an equivalent mapping.

### vLLM has absorbed much of the generic RDNA4 plumbing

Current vLLM has direct RDNA4 hardware support and a generic AWQ-to-WNA16 MoE path. `AutoAWQConfig` can fall back to `MoeWNA16Config`; `moe_wna16.py` handles AWQ/GPTQ packed MoE weights and selects a WNA16 backend; and the dense W4A16 linear path has an RDNA hybrid kernel.

The dense hybrid work is already upstream: vLLM PR #40977, `[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode`, merged 2026-07-14. It explicitly targets gfx11/gfx12 and contains gfx1201 tuning.

Reference:

- https://github.com/vllm-project/vllm/pull/40977

### The exact MoE optimization we backported is not fully upstream

Our Phase 3 overlay was built from vLLM PR #43389, commit `f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3`, and applied five files over runtime vLLM `f46a9dfe2c5f57bebbd29556cbbb25eabd874226`.

As of this review PR #43389 is still **open**, not merged. Its current description explicitly benchmarks gfx1201 and shows that its ROCm int4 MoE repacking materially changes fp16 behavior. That is important: current vLLM contains newer generic WNA16 plumbing, but the exact optimization lineage used by our proven overlay is not simply obsolete upstream code.

Reference:

- https://github.com/vllm-project/vllm/pull/43389

### Our S3 hybrid remains more specific than upstream

`scripts/r9700_wna16_hybrid_patch_v7_clean.py` is not a generic “enable gfx1201” shim. It intentionally:

- inherits stock AutoAWQ conversion/layout;
- subclasses stock `TritonWNA16Experts`;
- substitutes only W1 for the validated small-token region (`M <= 16` by default);
- requires the exact validated shape/routing contract;
- keeps W2 on stock packed Triton WNA16;
- falls back to `super().apply(...)` for every unproven shape, dtype, activation, or routing contract;
- records which path executed for evidence.

That control boundary remains local value even as upstream improves.

## Delta classification

| Area | Upstream state | Local decision | Reason |
| --- | --- | --- | --- |
| ROCm R9700 / `gfx1201` hardware recognition | Native/current | **REMOVE local dependency** | Do not maintain generic ROCm architecture hacks that upstream already owns. |
| Triton RDNA4 target/layout recognition | Native/current | **REMOVE local dependency** | Triton itself now models RDNA4/gfx1201. |
| HyperLoom `r9700 -> gfx1201, 64 CU` | Missing | **RETAIN** | Still absent from pinned HyperLoom upstream. |
| HyperLoom `gfx1201 -> r9700` reverse mapping | Missing | **RETAIN** | Still absent from pinned HyperLoom upstream. |
| Dense W4A16 ROCm hybrid linear kernel | PR #40977 merged | **REBASE TO UPSTREAM** | No reason to carry a separate dense compatibility implementation when validating a future vLLM candidate. |
| Generic AutoAWQ -> MoE WNA16 dispatch | Present in current vLLM | **REBASE / DROP OLD SHIMS** | Current upstream already owns the generic conversion/dispatch contract. |
| PR #43389 ROCm int4 MoE repacking lineage | PR still open | **RETAIN PINNED OVERLAY** | Our production evidence depends on this lineage; it is not yet fully represented by a merged upstream PR. |
| R9700 S3 small-M W1 correction kernel | Local, workload-specific | **RETAIN** | Exact guarded optimization with stock W2 and full stock fallback; no equivalent upstream proof for our workload. |
| AITER on gfx1201 | Evolving / experimental | **PROBE ONLY** | Do not make it the production default without exact-model correctness evidence. |
| Phase 6 readiness / promotion / guard / fallback | Local control plane | **RETAIN** | Kernel upstreaming does not replace safe operational promotion and automatic recovery. |
| Correctness hashes / path evidence / soak harness | Local evidence plane | **RETAIN** | Needed to decide whether an upstream candidate is actually equivalent. |
| Old exploratory probes | Historical | **ARCHIVE LATER** | Keep until a current-upstream candidate closes; do not mix cleanup with migration validation. |

## What changed in our strategic position

The public claim should no longer be “we made vLLM recognize RDNA4.” That would be stale.

The defensible value is:

> HyperLoom R9700 Experimental validates and operates an exact AWQ/MoE inference path on Radeon AI PRO R9700, with a guarded small-token WNA16 optimization, strict correctness gates, measured evidence, controlled promotion, and automatic stock fallback.

HyperLoom upstream still does not validate R9700, and vLLM upstream progress does not replace the end-to-end safety/evidence layer.

## Safe next experiment: current-upstream candidate

The next experiment is **isolated**. It must not replace or restart the current S3 production service.

1. Build a separate candidate from pinned/current vLLM upstream.
2. Use the exact production model and R9700/gfx1201 hardware.
3. Prefer upstream native dense W4A16 and generic AWQ/WNA16 plumbing; do not blindly copy the five-file Phase 3 overlay.
4. Validate model load and readiness first.
5. Run the existing canonical correctness request set against stock and candidate.
6. Require the exact canonical output hashes before any performance claim.
7. Capture backend/path evidence so an apparently-correct fallback is not mistaken for optimized execution.
8. Only after correctness and path identity pass, run a bounded performance comparison.
9. Promotion remains a separate Phase 6-controlled decision. A successful experiment does not automatically change the live service.

### Candidate outcomes

- **PASS and equivalent/better:** prepare a new controlled candidate and identify which old overlay files can be retired.
- **Correct but slower:** keep S3; upstream candidate remains informational.
- **Wrong hash, wrong path, load failure, or instability:** reject candidate and keep the production-proven S3 unchanged.

## Explicit non-actions

This review does **not** authorize any of the following:

- restarting the current S3 service;
- changing boot policy;
- inducing another failure/fallback event;
- rerunning Phase 5 or Phase 6 acceptance;
- deleting the pinned Phase 3 overlay;
- deleting the S3 hybrid patch;
- claiming official HyperLoom R9700 support;
- claiming universal acceleration from the measured workload-specific result.

## Bottom line

Upstream has caught up on **basic RDNA4 support and generic W4A16/AWQ infrastructure**, which is good news. It has **not** erased the parts of this project that matter most: HyperLoom R9700 identity support, the production-proven MoE optimization lineage, the workload-specific small-M W1 guard, and the correctness/promotion/fallback control plane.

The correct engineering move is **rebase selectively, not rewrite and not delete**.