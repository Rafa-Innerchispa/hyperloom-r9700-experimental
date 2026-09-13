# R9700 Phase 5 — Reversible S3 Canary Closure

Date: 2026-09-13 UTC / 2026-09-12 America/Guayaquil

Status: **CLOSED — CANARY PACKAGING + SINGLE-OWNER SOAK PASS**

Operational default after closure: **stock ROCm10 service remains active on port 8000**. Phase 5 does not silently promote S3 into production routing.

## 1. What Phase 5 proved

Phase 5 converted the Phase 3/4 experimental S3 recipe into a reproducible, reversible canary service:

- separate systemd user unit: `inneros-vllm-hyperloom-s3-canary.service`
- isolated canary port: `18018`
- separate Docker container: `inneros-vllm-hyperloom-s3-canary`
- fail-closed preflight checks stock is inactive, R9700 VRAM is clean, and all overlay/config hashes match
- overlay rebuilt from the pinned versioned builder instead of depending on a historical worktree
- deterministic readiness warmup must pass the canonical output hash before systemd reports the canary ready
- service does not stop or replace the stock service automatically
- explicit rollback restores `inneros-vllm-canary-rocm10.service`

Pinned identity:

- runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`

## 2. First packaged-canary window

The first real Phase 5 service window passed its runtime gate:

- `/v1/models`: HTTP 200
- correctness hash: stock exact
- C1: `68.8094 tok/s`
- C4: `191.4875 tok/s`
- fresh-prefix ~6K: `61.8344 / 62.1267 tok/s`
- hot correctness TTFT: `52.0 ms`
- canonical four C4 hashes: stock exact

Evidence:

- `docs/evidence/r9700_phase5_canary_bundle_manifest.json`
- `docs/evidence/r9700_phase5_canary_measure_20260913T023328Z.json`

## 3. Benchmark-concurrency incident and correction

A later soak attempt initially appeared to regress badly, including a C4 sample near `88.64 tok/s`. That sample is **not valid candidate-stability evidence**.

Forensics showed two separate ChatGPT/coordinated tasks launched benchmark suites against the same port `18018` within roughly one second:

- `r9700_phase5_canary_soak_20260913T025312Z.json`
- `r9700_phase5_canary_soak_20260913T025313Z.json`

During overlap:

- one measurement produced C4 `151.30 tok/s`
- the second produced C4 `88.64 tok/s`
- one first C4 output hash changed to the known `e79a200b...` anomaly
- correctness-short remained stock exact

After the overlapping clients stopped, the **same canary process** immediately recovered without restart to:

- C1 `67.98 tok/s`
- C4 `193.46 tok/s`
- fresh-prefix ~6K `62.73 / 62.55 tok/s`
- stock-exact correctness and C4 hashes

Therefore the overlapping measurements are preserved as a negative harness result, not interpreted as persistent S3 degradation.

Evidence:

`docs/evidence/r9700_phase5_concurrency_contamination_20260913.json`

## 4. Concurrency guard added

Phase 5 now serializes benchmark traffic with one exclusive local file lease:

`var/r9700_phase5_benchmark.lock`

- standalone `r9700_phase5_canary_measure.py` takes the lease before health/inference traffic
- `r9700_phase5_canary_soak.py` owns the same lease for its complete multi-round run
- a second client exits with rc `4`, `benchmark_lock_busy`, before sending inference traffic
- soak children inherit the existing lease explicitly and do not compete with the owner

Dynamic lock self-test: **PASS** for both measure and soak clients.

Evidence:

`docs/evidence/r9700_phase5_benchmark_lock_selftest_20260913.json`

## 5. Clean single-owner soak

After adding and testing the benchmark lease, a new clean canary was started from a clean GPU state. Its systemd preflight passed, readiness warmup passed, and one single-owner five-round soak ran under the exclusive lease.

Result: **5 / 5 rounds PASS; zero aborts.**

Aggregate:

| metric | min | median | mean | max |
|---|---:|---:|---:|---:|
| C1 tok/s | 68.3227 | **68.3391** | 68.3704 | 68.5126 |
| C4 tok/s | 184.0334 | **191.9242** | 190.2179 | 192.1595 |
| fresh long Y tok/s | 62.1840 | **62.8224** | 62.7132 | 62.9794 |
| fresh long Z tok/s | 61.2839 | **61.9523** | 61.8156 | 61.9874 |
| correctness TTFT s | 0.05135 | **0.05240** | 0.05315 | 0.05607 |

All five rounds required and passed:

- canonical correctness hash
- canonical four C4 output hashes
- C1 >= 60 tok/s
- C4 >= 180 tok/s
- both fresh-prefix 6K measurements >= 55 tok/s
- hot correctness TTFT < 500 ms

Canonical evidence manifest:

`docs/evidence/r9700_phase5_clean_soak_manifest_20260913.json`

Compressed raw bundle:

`docs/evidence/r9700_phase5_clean_soak_raw_bundle_20260913.gz.b64`

The manifest records exact SHA-256 and byte size for the five measurement files, full soak JSON, and final stock-restore JSON.

## 6. Final rollback / operational state

After the clean soak:

1. the experimental canary was stopped;
2. the stock service was started;
3. stock reached `active` state;
4. `/v1/models` returned HTTP 200;
5. the expected Qwen model was present;
6. the stock container/image identity was correct.

Final restore evidence:

`r9700_phase4_stock_restore_20260913T031729Z.json`

Its SHA-256 is recorded in the clean-soak manifest as:

`94dee76980c5df074c87bd7bedfcd19bf862f32a56d1201dc4c3bd740ac3914a`

## 7. Phase 5 verdict

**PASS for reproducible reversible canary packaging and bounded single-owner soak validation.**

This is stronger than the Phase 4 benchmark result because it proves the selected recipe survives packaging into an actual systemd-managed service with preflight, readiness, measurement gates, rollback, and benchmark-concurrency protection.

It does **not** by itself authorize replacing the stock operational default. Any production promotion should be a separate explicit change with routing/rollback policy and a clear owner decision.

## 8. Safe claim boundary

Safe:

- the packaged S3 canary reproduced approximately `191.49 tok/s` C4 in its first service window;
- a later clean exclusive-lock five-round soak passed 5/5 with C4 median `191.92 tok/s`, C1 median `68.34 tok/s`, and fresh-prefix ~6K medians around `62 tok/s`;
- deterministic correctness and the canonical C4 hashes remained stock exact in the valid canary gates;
- the canary can be started and rolled back without silently mutating the operational stock runtime;
- a cross-chat benchmark-concurrency bug was found, preserved, and fixed with an exclusive benchmark lease.

Do not claim:

- universal +20% acceleration;
- official AMD R9700 support;
- first port in the world;
- production default has already been changed;
- overlapping benchmark samples are valid stability measurements.

## 9. Next phase

The next legitimate step is an **explicit promotion decision**, not more reconstruction benchmarking. If promotion is approved, create a separate Phase 6 change that defines production routing, rollout/rollback, and health gates while keeping this Phase 5 canary evidence immutable.
