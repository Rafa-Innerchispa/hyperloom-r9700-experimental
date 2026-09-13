# R9700 Phase 6 — Controlled Operational Promotion

Date: 2026-09-13 UTC / 2026-09-12 America/Guayaquil

Status: **CONTROL PLANE IMPLEMENTED — STOCK REMAINS DEFAULT — LIVE CUTOVER NOT YET AUTHORIZED**

Canonical base: `eee6b5b4ecb19be9ad700aed0c8d98b171e77df7`.

Phase 5 remains closed and immutable. Phase 6 does not repeat Phase 2–5 benchmarking.

## Objective

Turn the validated S3 recipe into a controlled operational promotion path with explicit ownership, readiness gates, automatic fallback and manual rollback. The operational endpoint remains `127.0.0.1:8000` so callers do not need a new application endpoint.

Because the R9700 is a single-GPU runtime, stock and S3 cannot safely own the model simultaneously. Phase 6 therefore uses a transactional stop/clean/start cutover rather than pretending there is zero-downtime blue/green capacity where no second GPU exists.

## Default and boot policy

- `inneros-vllm-canary-rocm10.service` remains the operational default.
- S3 production service is manual-only and is not install-enabled for boot.
- Promotion is explicit through `r9700_phase6_control.py promote`.
- A reboot does not silently re-promote S3.
- The guard timer exists only while a validated S3 promotion is active.

## Promotion transaction

1. Acquire exclusive Phase 6 lock.
2. Verify stock service is active and `/v1/models` returns the expected model identity.
3. Persist `promotion_in_progress` while stock remains the recorded active backend.
4. Stop stock.
5. Confirm stock inactive.
6. Confirm VRAM usage is below 1 GB.
7. Start `inneros-vllm-hyperloom-s3-production.service` on port 8000.
8. Production preflight must pass exact Phase 5 manifest, overlay and S3 config hashes.
9. systemd `ExecStartPost` must pass the existing deterministic readiness warmup on port 8000.
10. Verify systemd active plus expected `/v1/models` identity.
11. Enable/start the 30-second guard timer.
12. Atomically persist `active_backend=hyperloom_s3`, `validated=true`.

Any failure after stock is stopped invokes the stock fallback transaction.

## Automatic fallback

Immediate fallback triggers:

- S3 production systemd state is not `active`.
- Promotion start/readiness/warmup fails.

Debounced fallback trigger:

- three consecutive lightweight `/v1/models` failures or wrong model identity at 30-second guard intervals.

Fallback sequence:

1. Disable/stop guard timer.
2. Stop S3 production service.
3. Confirm S3 inactive.
4. Wait for clean VRAM under 1 GB.
5. Start stock service.
6. Verify stock systemd state and expected `/v1/models` identity.
7. Atomically persist stock as active/default.

If clean GPU or stock readiness cannot be proven, the controller fails closed and records `fallback_failed` instead of claiming recovery.

## Manual rollback

`r9700_phase6_control.py rollback` executes the same verified fallback path. It does not depend on benchmark results.

## Concurrency protection

Promotion, rollback and guard execution share the exclusive lock:

`$XDG_RUNTIME_DIR/hyperloom-r9700-phase6.lock`

A second controller exits with `phase6_lock_busy` before it can mutate services.

## State and observability

Atomic state file:

`~/.local/state/hyperloom-r9700/phase6_state.json`

`status` reports:

- desired/active backend and validation state;
- stock/S3/guard service states;
- public endpoint readiness and model identity;
- current R9700 VRAM usage.

## Validation completed without runtime mutation

Local simulation suite: **7/7 PASS**.

Covered:

- stock is default;
- promotion dry-run never mutates services;
- successful promotion commits S3 only after readiness;
- failed promotion invokes verified stock fallback;
- dead S3 service triggers immediate fallback;
- HTTP/model readiness uses a three-failure threshold;
- controller lock is exclusive.

No Phase 2–5 benchmark was rerun. No live S3 cutover was performed during this validation.

## Current blocker before live dry-run/install

The AMD peer-ops helper currently returns `permission denied` for `/home/rlopez/bin/ralfia-peer-node-helper` and requires its execute bit restored (`chmod 0755`). This is a control-plane permission issue, not evidence of a HyperLoom regression. Until peer operations are restored, Phase 6 remains code/simulation validated only and stock must remain untouched.

## Live promotion acceptance gate

A live promotion may be declared validated only after all of the following pass in one controlled window:

- stock healthy immediately before cutover;
- production preflight exact hashes PASS;
- clean VRAM gate PASS;
- S3 systemd start PASS;
- deterministic readiness warmup PASS;
- `/v1/models` expected identity PASS on port 8000;
- guard timer active;
- controller status reports validated S3;
- synthetic guard failure test proves automatic fallback to stock, followed by stock `/v1/models` PASS;
- a final deliberate promotion may then be performed if operational ownership approves it.

This acceptance gate validates routing and recovery behavior, not performance. Closed Phase 5 performance evidence remains the performance basis unless a new independently verifiable regression appears.
