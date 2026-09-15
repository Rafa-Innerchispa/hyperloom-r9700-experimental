# R9700 vLLM 0.29 GPU runtime gate - 2026-09-15

## Purpose

This is the next proof lane after the clean-source application gate. It answers one narrow question:

> Does the exact vLLM 0.29.0 candidate, with the validated RDNA4 WNA16 patch, load the exact Qwen3-Coder AWQ model and execute correctly on the physical AMD Radeon AI PRO R9700 / gfx1201 under the proven ROCm 10 generation?

A source-apply PASS is necessary but is **not** GPU-runtime proof.

## Exact candidate identity

- vLLM tag: `v0.29.0`
- vLLM commit: `98dff2a81d747d1dba01a47f939f48c3526d4206`
- candidate patch SHA-256: `372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- target GPU: AMD Radeon AI PRO R9700
- target architecture: `gfx1201`
- ROCm generation: `10.0`
- source-build Python: `3.13`
- source-build Torch: `2.13.0`
- source-build base: `rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.13.0`
- isolated candidate port: `18029`
- isolated candidate container: `hyperloom-r9700-vllm029-gfx1201-candidate`
- candidate root: `var/r9700_vllm029_gpu_candidate`

The build-base choice is deliberate. Exact vLLM 0.29.0 build metadata pins Torch 2.13.0, while the AMD ROCm 10 image above provides Torch 2.13.0 on Ubuntu 24.04 with Python 3.13. The earlier validated vLLM 0.27 serving image uses Torch 2.12.0 and is preserved as prior-runtime evidence, not reused as the 0.29 build ABI.

## Preserved runtime boundary

The following are not modified by this lane:

- production port `8000`
- S3 canary port `18018`
- prior Phase 3 port `18011`
- stock fallback service `inneros-vllm-canary-rocm10.service`
- production service `inneros-vllm-hyperloom-s3-production.service`
- production boot policy
- Phase 2-6 evidence

S3 and stock remain available. Promotion is a separate manual decision.

## Proof sequence

1. **Source identity**
   - Require exact vLLM commit and exact patch SHA.
   - Require the hosted clean-source apply gate to be PASS.

2. **Isolated ROCm 10 build**
   - Build vLLM 0.29.0 from exact source in an isolated candidate root.
   - Base the build on `rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.13.0`.
   - Require Python 3.13, Torch 2.13.0, HIP present, Rust 1.95 and `PYTORCH_ROCM_ARCH=gfx1201`.
   - Use `docker/Dockerfile.r9700-vllm029-rocm10` as the reproducible build recipe.
   - Do not silently substitute an upstream prebuilt wheel built for a different ROCm generation.

3. **Physical GPU preflight**
   - Prove R9700 identity and `gfx1201`.
   - Prove `/dev/kfd` and `/dev/dri` are available.
   - Record ROCm/driver/runtime identity.
   - Use `scripts/r9700_vllm029_gpu_preflight.py`; it is read-only and fails closed.

4. **Import / ABI gate**
   - Import Torch and vLLM from the candidate runtime.
   - Record Python, Torch, HIP and vLLM versions.
   - Require Python 3.13, Torch 2.13.0, vLLM `0.29.0` and ROCm platform detection.

5. **Exact model load**
   - Load the local `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` model.
   - Keep `GPU_MAX_HW_QUEUES=1`.
   - Keep `ROCM_ATTN` unless isolated evidence explicitly justifies a different choice.
   - Do not switch to AITER in this gate.

6. **WNA16 path proof**
   - Prove the intended interleaved INT4 layout is actually selected.
   - Capture the repacked INT4 dtype/layout evidence.
   - Capture `tl.interleave` / WNA16 path evidence.
   - A fallback path may run, but it must never be reported as the optimized path.
   - Reuse `scripts/r9700_awq_backend_probe.py` where applicable.

7. **Deterministic warmup and correctness**
   - Reuse `scripts/r9700_readiness_warmup.py`.
   - Canonical rendered-text SHA-256 remains `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`.
   - Exact canonical equality is required before performance measurement.

8. **Bounded performance measurement**
   - Only after path proof and correctness PASS.
   - Record TTFT and decode tok/s.
   - Compare to controlled stock and validated S3 evidence.
   - Do not claim universal speedup.
   - Do not replay Phase 5 or Phase 6.

9. **Separate promotion decision**
   - No automatic promotion.
   - No production restart.
   - No boot-policy change.
   - Stock fallback remains the default until an explicit later decision.

## Existing components to reuse

Prefer adapting the existing evidence and launcher components instead of creating parallel frameworks:

- `scripts/r9700_phase3_isolated_port_launcher.py`
- `scripts/r9700_phase3_isolated_measure.py`
- `scripts/r9700_phase3_wait_isolated.py`
- `scripts/r9700_awq_backend_probe.py`
- `scripts/r9700_runtime_manifest.py`
- `scripts/r9700_readiness_warmup.py`
- `scripts/r9700_wna16_real_weight_layer_probe.py`
- `scripts/r9700_wna16_real_weight_vs_stock.py`
- `scripts/r9700_v7_full_model_collect.py`
- `scripts/r9700_v7_remove_candidate.py`

Do not reuse the old vLLM 0.27 overlay identity as if it were the 0.29 candidate. The validated 0.29 source patch is the canonical migration artifact for this lane.

## Definition of done

We may say **"the new stack works on the R9700"** only when all of these are recorded for the exact candidate identity:

- isolated ROCm 10 build PASS
- candidate import/ABI PASS with Python 3.13 + Torch 2.13.0 + HIP
- exact AWQ model load PASS on physical `gfx1201`
- intended WNA16/interleaved path proven
- deterministic canonical correctness PASS
- no preserved runtime mutated

Performance is an additional measured result, not part of the correctness definition.
