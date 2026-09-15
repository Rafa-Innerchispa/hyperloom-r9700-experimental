# R9700 isolated rebase rehearsal — 2026-09-14

Status: structural rehearsal only. No GPU serving runtime, S3 service, stock fallback, systemd unit, model, port, correctness hash, promotion state, benchmark, soak, or Phase 2–6 evidence was changed or rerun.

## Pinned inputs

- local canonical base: `2ab90d4c297eea70944c98bece02d1bf6d16c41a`
- validated live S3 remains on vLLM `0.27.0`; this rehearsal does **not** upgrade it
- vLLM stable candidate: `v0.29.0` → `98dff2a81d747d1dba01a47f939f48c3526d4206`
- vLLM current-main candidate: `00972dfd72988942138a7a6089eaee08580210b8`
- vLLM PR #43389 head: `56ef89e1ff4a1552beb3b5c51c00b73ea44daca1`, state `open`, `merged=false`
- HyperLoom current main / README 1.1.0: `0425bde3f6e76e1588400c37d056dfd3bb75ac11`

Exact workload whose behavior must remain equivalent:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ + ROCm 10 + Radeon AI PRO R9700/gfx1201 + MoE + AutoAWQ + WNA16 + Triton + local S3 tuning/guards`

## Result

Both vLLM `v0.29.0` and pinned current `main` are structurally suitable for a **rebase/adapt experiment**, but neither makes the S3 overlay redundant.

The decisive facts are:

1. Both candidates still contain `TritonWNA16Experts`, `select_wna16_moe_backend()`, `convert_to_wna16_moe_kernel_format()`, the Triton conversion branch, and the `MoeWNA16Method.process_weights_after_loading()` integration point.
2. Both candidates still explicitly reject `AutoAWQConfig` for the Triton WNA16 backend with `the AutoAWQ weight layout is not supported`.
3. The #43389-specific markers `repack_int4_to_int32`, `use_int4_interleave`, `is_int4_w4a16_interleaved`, `int4_packed_as_int32`, and `moe_wna16_utils.py` are absent from both candidates.
4. #43389 remains open and unmerged. Its patch still adds the exact ROCm-only N-packed int32 repack/interleave behavior that overlaps the validated S3 concept.
5. Current vLLM main has added a native `RDNA3` WNA16 backend, but its implementation is explicitly for `gfx1100`. It does not replace the R9700 / `gfx1201` path.
6. HyperLoom 1.1.0 still does not declare Radeon R9700 / `gfx1201` as a supported platform. Its recent harness fixes remain useful selective-rebase candidates, not a reason to delete Radeon-specific integration.

## Compatibility matrix

| surface | vLLM v0.29.0 | pinned vLLM main | #43389 hunk/readiness | risk | decision |
|---|---|---|---|---|---|
| `select_wna16_moe_backend()` / modular WNA16 oracle | present | present; adds RDNA3/gfx1100 path | structural insertion point remains, but oracle evolved | blindly replaying an older patch can bypass new backend-selection semantics | rebase/adapt |
| AutoAWQ → Triton compatibility | explicit rejection remains | explicit rejection remains | exact upstream gap still exists | removing S3 would route the exact workload away from the validated path | retain |
| `convert_to_wna16_moe_kernel_format()` Triton branch | present | present | #43389 repack hunk is structurally placeable, but raw clean-apply is **not claimed** | return/layout handling has evolved; port behavior, not line offsets | rebase/adapt |
| `MoeWNA16Method.process_weights_after_loading()` | present and calls converter | present and calls converter | stable hook for adapted post-load conversion | altering checkpoint conversion can silently corrupt layout | rebase/adapt |
| `FusedMoEQuantConfig.use_int4_w4a16` | present | present | #43389 adds an interleaved-layout discriminator; marker absent upstream | kernel can misread `[E,K,N/8] int32` as legacy packed layout without this distinction | rebase/adapt |
| Triton `fused_moe_kernel_gptq_awq` / invoke path | present | present | #43389 interleave constexpr/stride path absent upstream | wrong K/N/stride interpretation can produce correctness or crash failures | rebase/adapt |
| `TritonWNA16Experts` problem-size/apply path | present | present | #43389 problem-size/layout argument absent upstream | repacked N cannot be inferred from legacy weight shape | rebase/adapt |
| `moe_wna16_utils.py` repack helpers | absent | absent | additive helper still required for #43389-style behavior | helper removal without equivalent upstream transformation breaks format conversion | retain/adapt |
| native RDNA backend | no RDNA3 lane in pinned stable snapshot | RDNA3 W4A16 HIP path added, explicitly gfx1100-only | not equivalent to gfx1201 AutoAWQ Triton path | confusing RDNA3 enablement with RDNA4 support creates a false removal signal | retain |
| S3 R9700-specific tuning: `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, deterministic warmup | no demonstrated replacement | no demonstrated replacement | outside #43389 API port | removing measured knobs during an API migration mixes two independent changes | retain |
| S3 fail-closed hashes / stock-canary fallback / rollback | no equivalent deployment contract | no equivalent deployment contract | outside kernel patch | losing rollback/correctness gates invalidates operational evidence | retain |
| HyperLoom R9700 fork integration | upstream harness improvements available | same pinned HyperLoom main `0425bde...` | selective harness rebase only | wholesale merge could destabilize validated Radeon integration | rebase/adapt |

No component is classified `remove` in this rehearsal.

## Patch applicability interpretation

`applicable` here means the upstream architecture still exposes a compatible insertion surface. It does **not** mean `git apply` of PR #43389 is certified clean against v0.29.0 or current main.

That distinction is deliberate. PR #43389 was authored against an evolving vLLM tree, while current main already contains additional WNA16/RDNA backend work. The safe migration is to port these behaviors intentionally:

- ROCm/RDNA int4 repack from legacy N-first uint8 to N-packed int32;
- scale permutation paired with that layout;
- asymmetric zero-point unpack when required;
- interleaved Triton unpack path and stride handling;
- repacked-layout-aware problem size;
- explicit fail-closed backend compatibility rather than silently selecting a different kernel.

## New executable guard

This branch adds `scripts/r9700_rebase_readiness.py` plus a pinned evidence fixture and tests.

The checker is deliberately read-only and fail-closed. It verifies:

- immutable candidate SHAs and #43389 head SHA;
- #43389 remains open/unmerged;
- required WNA16/Triton architecture points still exist;
- the AutoAWQ→Triton rejection still exists;
- #43389-equivalent overlay markers are still absent upstream;
- HyperLoom still does not claim R9700/gfx1201 support;
- any future appearance of overlay-like upstream code requests manual review but **never** authorizes automatic S3 removal.

Expected baseline decisions:

- S3 overlay: `retain`
- vLLM v0.29.0: `rebase/adapt`
- vLLM current main: `rebase/adapt`
- HyperLoom current main: `rebase/adapt`
- runtime mutation: `false`
- automatic overlay removal: `false`

## Validation performed

Local deterministic validation on the isolated branch:

- `python -m pytest scripts/tests/test_r9700_rebase_readiness.py scripts/tests/test_r9700_upstream_watch.py -q`
  - `21 passed`
- `python -m compileall scripts/r9700_rebase_readiness.py scripts/tests/test_r9700_rebase_readiness.py`
  - PASS
- `git diff --check`
  - PASS

The local execution policy does not allow a direct Ruff command under its `python-tests` profile, so Ruff is intentionally delegated to repository CI rather than bypassing policy.

## Upstream evidence

- vLLM `v0.29.0` tag ref: `https://github.com/vllm-project/vllm/releases/tag/v0.29.0`
- vLLM current oracle: `https://github.com/vllm-project/vllm/blob/00972dfd72988942138a7a6089eaee08580210b8/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- vLLM current `moe_wna16.py`: `https://github.com/vllm-project/vllm/blob/00972dfd72988942138a7a6089eaee08580210b8/vllm/model_executor/layers/quantization/moe_wna16.py`
- vLLM current fused MoE kernel: `https://github.com/vllm-project/vllm/blob/00972dfd72988942138a7a6089eaee08580210b8/vllm/model_executor/layers/fused_moe/fused_moe.py`
- current RDNA3/gfx1100 expert: `https://github.com/vllm-project/vllm/blob/00972dfd72988942138a7a6089eaee08580210b8/vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py`
- vLLM PR #43389: `https://github.com/vllm-project/vllm/pull/43389`
- HyperLoom pinned main: `https://github.com/AMD-AGI/Hyperloom/tree/0425bde3f6e76e1588400c37d056dfd3bb75ac11`

## Next safe migration step

The next code-bearing migration should be a **separate non-live port branch** that adapts the #43389 behavior to one candidate vLLM revision, starting with v0.29.0 because it is a release tag, while preserving all S3 operational contracts. That branch should stop at build/unit/static compatibility until a deliberately scheduled GPU regression gate is warranted by a concrete port candidate.
