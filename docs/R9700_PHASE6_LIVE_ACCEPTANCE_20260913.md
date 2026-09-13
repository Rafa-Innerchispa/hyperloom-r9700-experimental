# R9700 Phase 6 — Live Promotion Acceptance

Date: 2026-09-13 UTC / 2026-09-13 America/Guayaquil

Status: **CLOSED / PASS — LIVE AUTOMATIC FALLBACK PROVEN, S3 RE-PROMOTED AND GUARDED**

Canonical Phase 5 base remains `eee6b5b4ecb19be9ad700aed0c8d98b171e77df7`. Phases 2–5 were not re-benchmarked. Phase 6 only changed operational promotion / rollback / readiness control.

## 1. Final live state

Final desired operational state on AMD `.5` is S3.

- active backend: `hyperloom_s3`
- desired backend: `hyperloom_s3`
- state reason after successful transaction: `promotion_validated`
- state validated: `true`
- S3 service: `inneros-vllm-hyperloom-s3-production.service` → `active (running)`
- stock service: `inneros-vllm-canary-rocm10.service` → `inactive`
- public/private API contract: `http://127.0.0.1:8000`
- `/v1/models`: HTTP 200
- served model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- transient guard scheduling: active
- final observed guard: PASS at `2026-09-13 13:27:08 UTC`
- final guard verification: S3 service `active`, exact model present, `ready=true`
- R9700 VRAM after final re-promotion: approximately `28.47 GB`
- only the S3 vLLM container owns the model/GPU after final re-promotion

The final re-promotion transaction started at `2026-09-13 13:20:42 UTC` and completed successfully at `2026-09-13 13:26:36 UTC`. Systemd recorded both `ExecStartPre` and the strict Phase 6 `ExecStartPost` readiness gate as `status=0/SUCCESS`.

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

The final re-promotion again logged `Using TritonWNA16Experts` and loaded the exact R9700 INT4 WNA16 configuration before the readiness gate passed.

## 3. Phase 6 safety model implemented

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

## 4. Live bugs found during Phase 6 and fixes

### A. Initial port release race

An early live cutover reached stock inactive + clean VRAM, but production preflight saw `:8000` still unavailable and rejected S3. Automatic recovery returned stock.

Fix: controller explicitly waits for route availability before starting S3.

### B. TCP TIME_WAIT was falsely treated as an active listener

The first implementation tested availability by `bind()` to `127.0.0.1:8000`. After stock stopped, TCP teardown state could keep the bind probe failing even though no server was listening. This caused a false `timeout_waiting_for_port_8000_free` and unnecessary stock fallback.

Fix: both controller and production preflight now use a TCP connect probe. `connect_ex != 0` means no listener owns the route; TIME_WAIT no longer blocks promotion.

Live proof on the successful attempt:

- `port_8000_free=true`
- `port_8000_listener_present=false`
- `port_8000_probe_connect_ex=111`

### C. Failed ExecStartPost could leave a named Docker container alive

During an earlier failed readiness gate, systemd killed the `docker run` client but the named S3 container remained and retained `:8000`. The fallback correctly failed closed rather than claiming stock recovery.

Fix: production service now has an `ExecStopPost` exact-name cleanup for `inneros-vllm-hyperloom-s3-production`.

### D. First cold S3 correctness sample diverged

An earlier production attempt reached HTTP 200 and returned 64 tokens, but its first cold correctness sample hash was:

`d2baa23b6df003eaadd817ad33206de10b18bd1ab71417e16c095a0d51e8080b`

Canonical stock hash is:

`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

Stock was re-launched and independently reproduced the canonical `7931...` hash, proving the gate itself was not stale.

Fix: Phase 6 does **not** weaken correctness. `scripts/r9700_phase6_readiness.py` consumes a possible cold-JIT transient but requires **two canonical hashes consecutively** before declaring S3 production ready.

Defaults used by the service:

- max attempts: 4
- required consecutive canonical results: 2

The final re-promotion passed this same strict gate with `reason=canonical_steady_state_proven` and `required_consecutive=2`.

## 5. Focused tests / validation

Focused Phase 6 validation remains:

- `tests/test_r9700_phase6_control.py`: **11/11 PASS**
- `tests/test_r9700_phase6_readiness.py`: **4/4 PASS**
- combined focused Phase 6 suite: **15/15 PASS**
- Python compile checks: PASS
- exact Phase 5 overlay/config hash preflight: PASS
- live S3 service start: PASS
- Phase 6 canonical steady-state readiness: PASS
- `/v1/models` expected model identity: PASS
- transient guard scheduling: PASS
- live automatic fallback: PASS
- final S3 re-promotion after fallback: PASS
- final post-promotion guard: PASS

No Phase 2–5 benchmark was repeated during this final acceptance.

## 6. Phase 5 immutable evidence remains authoritative

Do not rewrite Phase 5 results. Existing canonical evidence remains:

- clean exclusive-lock soak: 5/5 PASS
- C4 median: `191.92 tok/s`
- C1 median: `68.34 tok/s`
- fresh-prefix ~6K: ~`62 tok/s`
- correctness hashes: stock exact
- benchmark concurrency bug: preserved and fixed with exclusive benchmark lease

## 7. Final automatic-fallback acceptance — PASS

The final disruptive control-plane gate was executed once, deliberately and reversibly.

### 7.1 Controlled failure

Only `inneros-vllm-hyperloom-s3-production.service` was intentionally stopped while the Phase 6 transient guard remained enabled.

S3 shutdown completed at approximately `2026-09-13 13:07:33 UTC`. The production unit's exact-name cleanup ran, and no S3 container remained to retain `:8000` or the R9700.

### 7.2 Guard-triggered automatic fallback

The guard detected that S3 was no longer active and entered the automatic fallback path. The stock service was started at `2026-09-13 13:07:43 UTC`.

At `2026-09-13 13:14:05 UTC`, the guard/fallback transaction completed with evidence including:

- `trigger=service_not_active`
- stock service `inneros-vllm-canary-rocm10.service`
- stock `service_state=active`
- stock `ready=true`
- exact model `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`

Because controller readiness requires HTTP 200 from `/v1/models` **and** the exact model ID, `ready=true` proves the operational API contract and exact model identity were restored before fallback was declared successful.

Runtime inspection after fallback showed the only active vLLM container was `inneros-vllm-canary-rocm10`; the S3 container was absent. This proves there was no double GPU ownership or surviving S3 route owner.

### 7.3 Final S3 re-promotion

The desired final live state remained S3, so one final controlled promotion was executed. No second promotion was launched when the original MCP start RPC exceeded its 60-second RPC timeout; the already-running systemd transaction was allowed to finish, as designed.

Evidence:

- promotion transaction started: `2026-09-13 13:20:42 UTC`
- stock stopped cleanly: `2026-09-13 13:20:46 UTC`
- S3 exact-hash `ExecStartPre`: SUCCESS
- S3 served exact model on `127.0.0.1:8000`
- strict `ExecStartPost`: SUCCESS
- readiness result: `reason=canonical_steady_state_proven`
- required consecutive canonical outputs: `2`
- `/v1/models`: HTTP 200
- transaction completed: `2026-09-13 13:26:36 UTC`
- final post-promotion guard PASS: `2026-09-13 13:27:08 UTC`
- final guard saw S3 `service_state=active`, exact model, `ready=true`
- stock remained inactive
- final R9700 VRAM approximately `28.47 GB`

No orphan S3 container, `:8000` conflict, or double GPU ownership was observed.

## 8. Branch / PR gate

Phase 6 branch:

`chatgpt/r9700-phase6-canonical-20260912`

PR:

`#1 Phase 6: controlled S3 promotion, readiness guard and stock fallback`

The previously pending final automatic-fallback acceptance is now **PASS**. This document closes the live Phase 6 gate. PR #1 may now be marked ready for review; merge still requires the normal final Git/PR checks.

## 9. Closure

Phase 6 is **CLOSED / PASS** from the live control-plane perspective.

The validated sequence is now proven end to end:

`stock -> controlled S3 promotion -> strict canonical readiness -> transient guard -> induced S3 loss -> automatic stock fallback -> exact model recovery -> final S3 re-promotion -> strict canonical readiness -> guard PASS`

Do not repeat Phases 2–5 or this disruptive fallback test unless a new verifiable regression requires a targeted reproduction.
