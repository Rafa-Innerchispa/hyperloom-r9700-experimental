# R9700 Phase 6 — Live Promotion Acceptance

Date: 2026-09-13 UTC / 2026-09-13 America/Guayaquil

Status: **LIVE PROMOTION PASS — S3 ACTIVE, GUARD ACTIVE, STOCK AVAILABLE FOR FALLBACK**

Canonical Phase 5 base remains `eee6b5b4ecb19be9ad700aed0c8d98b171e77df7`. Phases 2–5 were not re-benchmarked. Phase 6 only changed operational promotion / rollback / readiness control.

## 1. Current live state

At the end of this session on AMD `.5`:

- active backend: `hyperloom_s3`
- desired backend: `hyperloom_s3`
- state reason: `promotion_validated`
- state validated: `true`
- S3 service: `inneros-vllm-hyperloom-s3-production.service` → `active (running)`
- stock service: `inneros-vllm-canary-rocm10.service` → `inactive`
- public/private API contract: `http://127.0.0.1:8000`
- `/v1/models`: HTTP 200
- served model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- transient guard timer: active
- guard service: oneshot, latest execution PASS
- latest recorded guard: `2026-09-13T06:43:13Z`, `pass=true`, `guard_failures=0`
- R9700 VRAM used at final status: approximately `28.47 GB`

The successful promotion transaction completed at `2026-09-13 06:36:55 UTC`. Systemd recorded the production service `ExecStartPre` and Phase 6 `ExecStartPost` readiness gate as `status=0/SUCCESS`.

## 2. Production S3 identity

The promoted backend runs:

- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- runtime vLLM reported: `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`
- PyTorch: `2.12.0+rocm10.0.0`
- ROCm/HIP reported: `7.15.26333`
- GPU: AMD Radeon AI PRO R9700 / `gfx1201`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- `max_model_len=8192`
- `gpu_memory_utilization=0.82`
- `dtype=float16`
- `GPU_MAX_HW_QUEUES=1`
- exact Phase 5 overlay and S3 configuration bundle
- runtime path observed: AutoAWQ MoE fallback → `TRITON` WNA16 → `TritonWNA16Experts`

## 3. Phase 6 safety model now implemented

Phase 6 keeps the operational API contract on `127.0.0.1:8000` while enforcing a single GPU owner.

Promotion transaction:

1. exclusive Phase 6 lock;
2. verify stock service + model identity;
3. stop stock;
4. wait stock inactive;
5. wait R9700 VRAM clean;
6. wait until no real TCP listener exists on `:8000`;
7. start S3 production unit;
8. fail-closed preflight verifies exact Phase 5 bundle/config hashes;
9. Phase 6 readiness gate proves canonical steady-state correctness;
10. verify S3 service + model identity;
11. start transient 30-second guard timer;
12. atomically record `promotion_validated`.

Rollback / fallback:

- stop transient guard;
- stop S3;
- clean the exact named S3 container even if systemd startup failed;
- wait S3 inactive;
- wait VRAM clean;
- wait until no listener owns `:8000`;
- start stock;
- wait up to 600 seconds for stock service + exact model identity;
- only then persist `stock` as validated;
- otherwise fail closed with active backend `unknown` rather than claiming recovery.

S3 production remains manual-only and is not boot-enabled. Stock remains the boot/default service, so a reboot does not silently re-promote S3.

## 4. New live bugs found during Phase 6 and fixes

### A. Initial port release race

First live cutover reached stock inactive + clean VRAM, but production preflight saw `:8000` still unavailable and rejected S3. Automatic recovery returned stock.

Fix: controller explicitly waits for route availability before starting S3.

### B. TCP TIME_WAIT was falsely treated as an active listener

The first implementation tested availability by `bind()` to `127.0.0.1:8000`. After stock stopped, TCP teardown state could keep the bind probe failing even though no server was listening. This caused a false `timeout_waiting_for_port_8000_free` and unnecessary stock fallback.

Fix: both controller and production preflight now use a TCP connect probe. `connect_ex != 0` means no listener owns the route; TIME_WAIT no longer blocks promotion.

Live proof on the successful attempt: preflight reported:

- `port_8000_free=true`
- `port_8000_listener_present=false`
- `port_8000_probe_connect_ex=111`

### C. Failed ExecStartPost could leave a named Docker container alive

During an earlier failed readiness gate, systemd killed the `docker run` client but the named S3 container remained and retained `:8000`. The fallback correctly failed closed rather than lying about stock recovery.

Fix: production service now has an `ExecStopPost` exact-name cleanup for `inneros-vllm-hyperloom-s3-production`.

### D. First cold S3 correctness sample diverged

An earlier production attempt reached HTTP 200 and returned 64 tokens, but its first cold correctness sample hash was:

`d2baa23b6df003eaadd817ad33206de10b18bd1ab71417e16c095a0d51e8080b`

Canonical stock hash is:

`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

Stock was re-launched and independently reproduced the canonical `7931...` hash, proving the gate itself was not stale.

Fix: Phase 6 does **not** weaken correctness. `scripts/r9700_phase6_readiness.py` now allows the cold-JIT transient to be consumed but requires **two canonical hashes consecutively** before declaring S3 production ready. Defaults used by the service:

- max attempts: 4
- required consecutive canonical results: 2

If S3 never converges to the stock-exact canonical output, promotion fails closed and returns to stock.

The successful live promotion completed with this stricter Phase 6 readiness gate returning `status=0/SUCCESS`.

## 5. Tests / validation

On AMD `.5` after the Phase 6 readiness and TCP-listener fixes:

- `tests/test_r9700_phase6_control.py`: **11/11 PASS**
- `tests/test_r9700_phase6_readiness.py`: **4/4 PASS**
- combined focused Phase 6 suite: **15/15 PASS**
- Python compile checks: PASS
- read-only pre-promotion gate: PASS
- exact Phase 5 overlay/config hash preflight: PASS
- live S3 service start: PASS
- Phase 6 canonical steady-state readiness: PASS
- `/v1/models` expected model identity: PASS
- transient guard scheduling: PASS
- repeated live guard health checks: PASS

No Phase 2–5 benchmark was repeated.

## 6. Phase 5 immutable evidence still authoritative

Do not rewrite Phase 5 results. Existing canonical evidence remains:

- clean exclusive-lock soak: 5/5 PASS
- C4 median: `191.92 tok/s`
- C1 median: `68.34 tok/s`
- fresh-prefix ~6K: ~`62 tok/s`
- correctness hashes: stock exact
- benchmark concurrency bug: preserved and fixed with exclusive benchmark lease

## 7. Branch / PR

Phase 6 branch:

`chatgpt/r9700-phase6-canonical-20260912`

PR:

`#1 Phase 6: controlled S3 promotion, readiness guard and stock fallback`

The PR must remain **draft / unmerged** until the final live automatic-fallback acceptance is executed and documented.

## 8. Remaining acceptance item

The remaining live Phase 6 acceptance item is intentionally disruptive and was not run after the successful final promotion in this session:

1. verify S3 + guard remain healthy;
2. perform one bounded controlled S3 failure while the guard timer is active;
3. prove guard initiates automatic fallback;
4. prove stock returns `active` and `/v1/models` returns HTTP 200 with the exact model;
5. confirm no orphan S3 container and VRAM/routing ownership is clean;
6. document evidence;
7. decide desired final operational state;
8. if desired final state is S3, perform one final promotion and verify guard again;
9. only then update PR #1 from draft and consider merge.

Do not use performance benchmarking for this acceptance item. This is control-plane/fallback validation only.

## 9. New-chat rules

Do not repeat Phases 2, 3, 4 or 5. Do not redo Phase 6 discovery. Start from the live state described here and the internal continuity file. Treat only a newly observed verifiable regression as justification for deeper investigation.
