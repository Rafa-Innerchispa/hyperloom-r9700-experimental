# Main consolidation — 2026-09-12

## Purpose

Make `main` the single canonical integrated project state. Working branches remain useful as historical evidence and isolated development lanes, but validated reusable work must not remain exclusive to them.

## Histories found

Before consolidation, GitHub `main` was at:

`c03f0d492bc174a709f4ccce3fdf8bd0a69c7e3b`

That history had no common ancestor with the later Phase 3/Phase 4 engineering history. The old main line was therefore preserved intact before any canonical ref change as:

`archive/main-pre-consolidation-20260912`

The current integrated engineering line comes from:

`chatgpt/r9700-phase4-coldstart-parity-20260912`

Phase 3 closure SHA:

`321d5d0cff9020e70ba11cc8929e09b80dc3d655`

Phase 4 root-cause checkpoint:

`7c8627ae6c132f83d2b487cc066406e053dd7e88`

## Useful old-main artifacts recovered into the integrated line

The old README was superseded by the newer project README and was not reintroduced. The following useful artifacts were recovered:

- `config/litellm-r9700.yaml`
- `requirements-bridge.txt`
- `scripts/bootstrap_r9700_preflight.py`
- `scripts/run_r9700_bypass_baseline.py`
- `patches/hyperloom-r9700-gfx1201.patch`
- `evidence/live-local-agent-loop-20260904.json`
- `evidence/live-r9700-results-20260904.json`
- `docs/AMD_VECTOR_FEEDBACK_STATUS_20260908.md`
- `docs/R9700_PHASE2_TECHNICAL_PREVIEW.md`

The two recovered docs are explicitly marked historical so they cannot be mistaken for the active verdict.

## Canonical state rule

After this consolidation:

1. `main` is the product/research truth that a new contributor should read first.
2. `docs/R9700_PROJECT_CONTINUITY.md` is the canonical restart document.
3. `docs/R9700_PHASE4_ACTIVE_HANDOFF_20260912.md` is the active engineering handoff while Phase 4 remains open.
4. Phase-specific branches are development/evidence lanes, not the only home of reusable validated code.
5. Old unrelated `main` history remains reachable through `archive/main-pre-consolidation-20260912`.
6. Do not force-delete historical branches or raw-evidence worktrees merely to make the branch list pretty.

## Active engineering state

Phase 3 S3 is closed and preserved. Phase 4 is active and targets the fresh-process first-request latency spike. The spike has already been reproduced on stock and associated with an unexpected Triton JIT compile in the ROCm prefix-prefill `_fwd_kernel` path after HTTP readiness.

Exact next gate is documented in `docs/R9700_PROJECT_CONTINUITY.md` and `docs/R9700_PHASE4_ACTIVE_HANDOFF_20260912.md`.
