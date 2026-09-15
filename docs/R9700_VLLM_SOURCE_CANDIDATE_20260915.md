# R9700 isolated vLLM source candidate — 2026-09-15

Status: **static source candidate verified; not applied to any runtime**.

This lane materializes the next migration artifact for the exact workload:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ + vLLM + ROCm 10 + Radeon AI PRO R9700/gfx1201 + MoE + W4A16`

It does not start, stop, patch, replace or reconfigure the preserved S3 service, stock fallback, systemd, serving ports or Phases 2–6.

## Candidate identity

HyperLoom base:

- `92bb944e44b62573b3a759815efe5a47b9fda3f2`

Primary vLLM source target:

- tag: `v0.29.0`
- commit: `98dff2a81d747d1dba01a47f939f48c3526d4206`

Patch artifact:

- `docs/evidence/vllm_v0_29_0_r9700_awq_triton_candidate.patch`
- SHA-256: `3e366ff0c8dff5b1e697d227250bce1f1fd6e06e70e78cc8d6672c59ba6a7089`
- size: 17,384 bytes
- scope: exactly five vLLM paths

The candidate is immutable-by-test: the repository test suite pins this exact patch hash, so future patch edits require an explicit review rather than silently changing the migration target.

## Exact five-file scope

1. add `vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py`
2. adapt `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
3. adapt `vllm/model_executor/layers/fused_moe/config.py`
4. adapt `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`
5. adapt `vllm/model_executor/layers/fused_moe/fused_moe.py`

Explicit non-edits remain:

- `vllm/model_executor/layers/quantization/auto_awq.py`
- `vllm/platforms/rocm.py`

The direct `AutoAWQConfig -> Triton` incompatibility is intentionally preserved. This candidate targets only the normalized lane:

`AutoAWQConfig -> MoeWNA16Config -> MoeWNA16Method -> WNA16 oracle -> Triton`

That keeps the experiment bounded to the layout actually needed by the validated workload instead of broadening Triton support speculatively.

## Source identity verification

The four files that already exist in vLLM v0.29.0 were verified against immutable GitHub blob identities and match the candidate validator exactly:

- `int_wna16.py`: `c56712ac58bc9933165ccf04753b9ee1a6b9d45a`
- `config.py`: `8736b1393d82590c1adb9a8ca91841ab2337e555`
- `triton_moe.py`: `9f982644b5691f8bccde0aa5229c404d394882f9`
- `fused_moe.py`: `dc619b2b12d27d2955f38a656e7c524fd6042887`

`moe_wna16_utils.py` is absent at the v0.29.0 target and is therefore a deliberate new file.

## Current-main drift check

vLLM `main` advanced during this work from the earlier observed `e6960af...` to:

`dffbb714e4e8e4b95ccc888df98d47c7d2cef78d`

The new head is an unrelated gfx950 decode top-k optimization. The source surfaces relevant to this port retain the same blob identities seen in the prototype for the oracle, config, Triton experts, fused MoE kernel, AutoAWQ and MoeWNA16 integration. Therefore the movement of `main` does not invalidate this v0.29.0 candidate.

PR #43389 also remains open and unmerged. Upstream still has not supplied a merged replacement that would justify removing the local S3 behavior.

## Static and algebraic validation

Candidate-specific tests:

- `11 passed`

Complete repository `scripts/tests` suite:

- `377 passed`

Additional checks:

- `compileall`: PASS
- `git diff --check`: PASS
- local Ruff: delegated to repository CI because the local `python-tests` execution profile does not permit direct Ruff invocation

The pure-Python reference checks prove:

- nibble-preserving weight transformation from `[E,N,K/2]` uint8 to `[E,K,N/8]` int32;
- packed zero-point order to `[E,K_groups,N]`;
- expected gate projection shapes for `N=768, K=2048, group_size=128`;
- expected down projection shapes for `N=2048, K=768, group_size=128`;
- the classic non-interleaved scalar-shift path remains in the patch;
- the interleaved layout predicate is bounded to W4A16 plus `torch.int32` weights.

## What has not been proven yet

A full clean vLLM checkout was not available inside the restricted local execution lane, and the isolated container cannot reach GitHub. Therefore this work does **not** claim that a full `git apply --check`, vLLM import, ROCm build or GPU execution has passed.

Those are deliberate next gates, not hidden omissions:

1. apply the candidate to a clean checkout of vLLM `98dff2a8...`;
2. build/import it with ROCm 10 on gfx1201 in a non-serving environment;
3. load the exact QuantTrio AWQ checkpoint and prove the normalized `MoeWNA16Config -> Triton` route;
4. verify real repacked w13/w2, scales and asymmetric zero points before inference;
5. run the exact correctness/hash contract against preserved S3;
6. only after correctness, measure migration TTFT/decode/throughput and test fail-closed startup plus stock fallback.

## Decision

- static source candidate: **PASS**
- patch application to a clean vLLM tree: **next gate**
- GPU/runtime equivalence: **not yet claimed**
- preserved S3: **retain**
- stock fallback: **retain**
- direct AutoAWQ Triton rejection: **retain**
- classic WNA16 path: **retain**
- runtime mutation: **false**
- automatic promotion: **not allowed**

Machine-readable evidence is stored in:

`docs/evidence/r9700_vllm_source_candidate_20260915.json`
