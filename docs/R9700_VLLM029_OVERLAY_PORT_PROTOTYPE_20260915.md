# R9700 vLLM 0.29/current AWQ -> Triton overlay port prototype — 2026-09-15

Status: **prototype only, not applied to any runtime**.

This lane turns the earlier upstream-delta and rebase-readiness work into a narrow, reproducible port plan for the exact validated workload:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ + ROCm 10 + Radeon AI PRO R9700/gfx1201 + MoE + W4A16`

No service, S3 artifact, stock fallback, systemd unit, model process, port, benchmark result, promotion state, or Phase 2–6 evidence is changed by this work.

## Pinned state

HyperLoom experimental base:

- `main`: `9ed433f371c9b68f231d2b6d36ba051e0b8bd925`

vLLM references:

- stable target: `v0.29.0` / `98dff2a81d747d1dba01a47f939f48c3526d4206`
- previous rebase-rehearsal `main`: `00972dfd72988942138a7a6089eaee08580210b8`
- current `main` observed for this prototype: `e6960af33b379d502f409e3e2241bbf2b2c2f68d`
- PR #43389 head: `56ef89e1ff4a1552beb3b5c51c00b73ea44daca1`
- PR #43389: still open and unmerged at prototype time.

The important distinction is deliberate: `00972df...` is a historical rehearsal baseline, while `e6960af...` is the current source target for this prototype. A moving upstream `main` is not a versioning strategy, despite humanity's recurring experiments to the contrary.

## Key architecture correction

The port **must not remove** the current direct `AutoAWQConfig -> Triton` rejection.

Current vLLM already has a safer path for the exact workload:

`AutoAWQConfig -> MoeWNA16Config fallback/normalization -> MoeWNA16Method -> WNA16 oracle -> Triton`

`MoeWNA16Method` calls the WNA16 oracle with `may_have_bias=False`, and its AWQ loader already converts the checkpoint into the common N-first WNA16 representation before `convert_to_wna16_moe_kernel_format()` is called.

Therefore #43389's repack/interleave behavior can be adapted **after normalization**. That is narrower than enabling arbitrary direct AutoAWQ layouts in Triton and matches the exact workload boundary we need to preserve.

## Integration-point matrix

| current integration point | current state at `e6960af...` | prototype action | reason |
|---|---|---|---|
| `AutoAWQConfig.get_quant_method()` | Can fall back to `MoeWNA16Config` for RoutedExperts when Marlin is unsuitable. | **retain / no edit** | This is the normalization entrance for the exact AWQ workload. |
| `AutoAWQMoEMethod` direct WNA16 path | Exists, but the oracle rejects direct `AutoAWQConfig` for Triton; constructor also declares possible bias. | **retain / no edit** | Broadening this path is unnecessary for the exact normalized AWQ lane and would widen risk. |
| `MoeWNA16Method` | Builds N-first uint8 W4A16 weights/scales and calls `convert_to_wna16_moe_kernel_format()`. | **retain API** | It is already the correct hand-off point. No loader rewrite is needed for the prototype. |
| `select_wna16_moe_backend()` | Triton supports the normalized `MoeWNA16Config` lane; direct `AutoAWQConfig` remains rejected. | **retain selector behavior** | The selector is not the missing performance primitive. |
| `convert_to_wna16_moe_kernel_format()` Triton branch | Leaves normalized `MoeWNA16Config` int4 tensors in N-first uint8 layout. | **rebase/adapt** | Add RDNA int4 repack after normalization, not before it. |
| `moe_wna16_utils.py` | Absent in current main. | **add** | Supply the pure-tensor weight and zero-point repack helpers from the #43389 design. |
| `FusedMoEQuantConfig` | Has `use_int4_w4a16`, no explicit interleaved-layout predicate. | **rebase/adapt** | The kernel needs an explicit layout signal rather than assuming layout from platform. |
| `TritonWNA16Experts` | Assumes classic `[E,N,K/2]` uint8 layout for int4 problem size/assertions/config. | **rebase/adapt** | Repacked `[E,K,N/8]` int32 needs N derived from scales and corrected K assertions. |
| `fused_moe_kernel_gptq_awq` + invoke wrapper | Classic scalar-shift int4 unpack path only. | **rebase/adapt** | Add `tl.interleave` path and layout-aware N/stride/zero-point handling while preserving classic path. |
| `vllm/platforms/rocm.py` | Already identifies R9700/gfx1201 and exposes RDNA helpers. | **retain / no edit** | Hardware recognition is already upstream; this is not a device-enablement port. |

## Prototype edit set

The candidate runtime diff is intentionally limited to five paths:

1. `vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py`
2. `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
3. `vllm/model_executor/layers/fused_moe/config.py`
4. `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`
5. `vllm/model_executor/layers/fused_moe/fused_moe.py`

This repository does **not** apply those edits to vLLM. Their exact scope, source blob identities, required anchors, safety boundaries and remaining GPU gates are encoded in:

`docs/evidence/r9700_vllm029_overlay_port_manifest_20260915.json`

The machine-readable fail-closed validator is:

`python scripts/r9700_overlay_port_contract.py`

## What is intentionally not part of this port

The prototype does not change:

- `vllm/model_executor/layers/quantization/auto_awq.py`;
- `vllm/platforms/rocm.py`;
- S3 or stock serving defaults;
- `ROCM_ATTN`;
- `GPU_MAX_HW_QUEUES=1`;
- deterministic readiness warmup;
- fail-closed hashes or fallback;
- any systemd unit or serving port;
- Phases 2–6 or their historical evidence.

The native RDNA3 WNA16 backend now present upstream is also **not** a replacement for this R9700 lane. It is separately gated and does not establish exact gfx1201 AutoAWQ/W4A16 equivalence.

## Remaining GPU gates before a runtime candidate exists

Static compatibility is not permission to promote anything. A future isolated vLLM checkout/candidate must still prove, in this order:

1. Candidate builds/imports on ROCm 10 + gfx1201.
2. The exact QuantTrio AWQ model reaches the normalized `MoeWNA16Config` path and selects Triton as intended.
3. Repacked weight, scale and asymmetric zero-point layouts are verified before first inference.
4. Exact correctness/hash regression passes against the preserved S3 reference contract.
5. Only after correctness, measure new-version TTFT, decode and throughput. These are migration gates, not a replay of closed Phases 2–6.
6. Fail-closed startup and stock fallback remain effective before any consideration of replacing the preserved S3 service.

## Decision

- current S3: **retain**
- vLLM v0.29.0: **rebase/adapt candidate**
- current vLLM main: **rebase/adapt candidate**
- direct `AutoAWQConfig -> Triton` guard: **retain**
- normalized `MoeWNA16Config -> Triton` repack/interleave prototype: **proceed in isolation**
- live runtime mutation: **false**
