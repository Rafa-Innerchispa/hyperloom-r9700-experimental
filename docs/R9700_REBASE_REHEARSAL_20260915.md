# R9700 isolated rebase rehearsal — 2026-09-15

Status: isolated structural rehearsal only. No live runtime, S3, stock, systemd, serving port, model process, promotion state, rollback state, Phase 2–6 benchmark, soak or fallback behavior was changed.

## Pinned identities

- local base `main`: `2ab90d4c297eea70944c98bece02d1bf6d16c41a`
- work branch: `chatgpt/r9700-rebase-rehearsal-20260914`
- validated live S3 remains on vLLM `0.27.0`; this rehearsal does not replace it
- vLLM stable candidate: `v0.29.0`
- vLLM current-main snapshot inspected: `00972dfd72988942138a7a6089eaee08580210b8`
- HyperLoom current-main snapshot inspected: `0425bde3f6e76e1588400c37d056dfd3bb75ac11`
- relevant upstream patch remains vLLM PR `#43389`, head previously pinned as `56ef89e1ff4a1552beb3b5c51c00b73ea44daca1`

Exact workload boundary remains:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ + vLLM + ROCm 10 + R9700/gfx1201 + MoE + AutoAWQ + WNA16`

## Rebase-readiness matrix

| component | v0.29.0 / current upstream shape | exact overlap with S3 | risk | decision |
|---|---|---|---|---|
| `AutoAWQMoEMethod` | Present in v0.29.0 and current main. It calls `select_wna16_moe_backend(...)`, allocates AWQ MoE tensors and later calls `convert_to_wna16_moe_kernel_format(...)`. | High structural overlap with the old integration entry point, but not proof of the exact ROCm Triton layout working. | Mistaking API presence for exact backend equivalence. | rebase/adapt |
| WNA16 modular-kernel oracle | Present. `WNA16MoEBackend.TRITON`, `TritonWNA16Experts`, `select_wna16_moe_backend(...)` and conversion helpers exist. | High structural overlap; this is now the correct integration surface for a future port. | Replaying the old 0.27 patch directly would target stale architecture. | rebase/adapt |
| AutoAWQ -> Triton WNA16 backend selection | Still explicitly rejected by the oracle with `the AutoAWQ weight layout is not supported`. | This is the decisive non-equivalence. S3's validated #43389-style path is not redundant. | Removing S3 overlay would remove the only validated exact-path behavior. | retain |
| AWQ standard-format conversion helper | v0.29.0 already contains `_convert_awq_to_standard_format(...)` and standardizes AWQ packing for generic kernel consumers. | Useful overlap, but this generic conversion is not the same claim as the RDNA MoE Triton repack/interleave path from #43389. | Assuming generic AWQ normalization automatically solves RDNA MoE Triton layout. | rebase/adapt |
| PR #43389 RDNA INT4 MoE repack/interleave | Patch adds RDNA-specific int4 repack/interleave plumbing, changes Triton WNA16 weight interpretation and adjusts scales/zero-points. It remains the closest upstream analogue to S3. | Very high conceptual overlap with S3. | Patch hunks target moving files/APIs and cannot be replayed blindly. | retain behavior; rebase implementation |
| `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, deterministic warmup | No structural upstream evidence reviewed here proves these validated runtime controls obsolete. | Local operational/tuning controls, not just API compatibility. | Removing them during a code rebase would mix migration with retuning and invalidate evidence. | retain |
| correctness hashes, fail-closed preflight, stock/canary separation, fallback | No upstream source inspected replaces the deployment contract. | Orthogonal to vLLM source refactor. | A code migration must not weaken rollback/correctness controls. | retain |
| HyperLoom R9700 integration | HyperLoom current main contains newer controller/patch/liveness fixes but still does not establish exact R9700 + AutoAWQ MoE/WNA16 equivalence. | Harness fixes are reusable; Radeon/runtime integration remains local. | Wholesale rebase could entangle harness updates with runtime migration. | rebase/adapt selectively |

## Exact compatibility finding

vLLM `v0.29.0` is materially closer to the validated S3 design than `0.27.0`:

1. AutoAWQ has a first-class `AutoAWQMoEMethod`.
2. That method delegates backend choice to the WNA16 oracle.
3. It delegates loaded-weight transformation through `convert_to_wna16_moe_kernel_format(...)`.
4. The oracle exposes a Triton WNA16 experts backend.
5. Despite all of that plumbing, the oracle still rejects `AutoAWQConfig` for Triton.

That means the future migration is **not** "port the whole old patch" and is also **not** "delete the overlay because upstream supports AWQ". The correct shape is much narrower:

- retain the validated S3 behavior;
- target the current modular WNA16 oracle/conversion interfaces;
- adapt only the RDNA AutoAWQ-to-Triton layout/repack behavior still missing upstream;
- keep runtime/tuning/rollback contracts unchanged until the isolated new-version lane earns its own evidence.

## New static readiness guard

Added `scripts/r9700_rebase_readiness.py` plus tests.

The guard is deliberately source-only and fail-closed. It checks for the current integration surfaces:

- `AutoAWQMoEMethod`
- `select_wna16_moe_backend(...)`
- `convert_to_wna16_moe_kernel_format(...)`
- `WNA16MoEBackend`
- `TritonWNA16Experts`
- the explicit AutoAWQ/Triton rejection
- the known HyperLoom supported-platform markers

It never mutates runtime and never authorizes automatic overlay removal. If the expected API surface disappears, the decision falls back to `retain` rather than guessing that upstream became equivalent.

## Validation

Focused local validation:

`python -m pytest scripts/tests/test_r9700_rebase_readiness.py scripts/tests/test_r9700_upstream_watch.py -q`

Result: `17 passed`.

Local Ruff invocation is not permitted by the repository's `python-tests` execution profile, so Ruff is intentionally left to repository CI rather than reported as locally passing.

## Decision

**Proceed with a future isolated vLLM 0.29/current-main migration lane, but do not change the live S3 runtime yet.**

The migration target should be the current WNA16 modular-kernel/oracle architecture. The S3 #43389-style behavior remains required until exact AutoAWQ -> Triton WNA16 equivalence is demonstrated for the R9700 workload. No reviewed evidence justifies removing `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, deterministic warmup, correctness hashes, fail-closed deployment guards or automatic fallback.
