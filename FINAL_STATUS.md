# HyperLoom Radeon AI PRO R9700 / RDNA4 — Current Experimental Status

Last reconciled: 2026-09-13 UTC / 2026-09-12 America/Guayaquil
Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
Hardware: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
Workload: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
Runtime: ROCm 10 + vLLM + Triton

## Current verdict

**PHASE 5 CLOSED — REVERSIBLE S3 CANARY PACKAGING + SINGLE-OWNER 5-ROUND SOAK: PASS**

**The stock ROCm10 service remains the operational default. S3 has not been silently promoted into production routing.**

Canonical current continuity:

`docs/R9700_PROJECT_CONTINUITY.md`

Canonical Phase 5 closure:

`docs/R9700_PHASE5_CANARY_CLOSURE_20260913.md`

## What is now concrete

The selected experimental S3 recipe combines:

- the relevant vLLM #43389-style RDNA INT4/W4A16 MoE repack/interleave path;
- R9700-specific `int4_w4a16` MoE tuning;
- stock `ROCM_ATTN`;
- `GPU_MAX_HW_QUEUES=1`;
- deterministic readiness-stage warmup to absorb the late ROCm prefix-prefill Triton JIT before user traffic.

Pinned identities:

- runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`

## Phase 4 performance gate

Three independent fresh processes passed correctness and reproducibility after readiness warmup.

Controlled stock+queue1 C4 baseline:

`158.489996 tok/s`

Phase 4 first-measure C4:

- `190.5969 tok/s`
- `194.0823 tok/s`
- `189.9816 tok/s`

Median:

**`190.5969 tok/s`**, about **`+20.26%`** versus that controlled stock+queue1 baseline on the tested C4 workload.

Other Phase 4 observations:

- C1 median: `69.679 tok/s`
- fresh-prefix ~6K: about `61-62 tok/s`
- first-user TTFT after readiness warmup: about `51-54 ms`
- correctness and canonical four C4 hashes: stock exact in all three processes

This is a tested-workload result, not a universal acceleration claim.

## Phase 5 packaged service

Phase 5 converted the experimental recipe into a real reversible canary:

- unit: `inneros-vllm-hyperloom-s3-canary.service`
- port: `18018`
- fail-closed preflight
- exact overlay/config hash verification
- deterministic readiness warmup required before service becomes active
- explicit stock/canary separation
- explicit rollback to `inneros-vllm-canary-rocm10.service`

First packaged-canary runtime gate:

- C1 `68.8094 tok/s`
- C4 `191.4875 tok/s`
- fresh-prefix ~6K `61.8344 / 62.1267 tok/s`
- correctness and C4 hashes stock exact
- hot correctness TTFT `52.0 ms`

## Benchmark-concurrency bug found and fixed

A later soak initially produced degraded samples because **two coordinated benchmark clients hit the same canary simultaneously**. Those overlapping samples are preserved but explicitly excluded from candidate-stability decisions.

Evidence:

`docs/evidence/r9700_phase5_concurrency_contamination_20260913.json`

The same process recovered after the overlapping traffic stopped to C4 `193.4585 tok/s`, C1 `67.9771 tok/s`, and fresh-prefix ~6K `62.7305 / 62.5542 tok/s` with stock-exact hashes.

The harness now uses one exclusive benchmark lease:

`var/r9700_phase5_benchmark.lock`

A second measure/soak client exits rc `4` with `benchmark_lock_busy` **before sending inference traffic**. Dynamic lock self-test passed.

## Clean single-owner soak

After the concurrency fix, a fresh canary ran a clean five-round exclusive-lock soak.

Result: **5/5 PASS**.

Aggregate:

- C1 median: **`68.3391 tok/s`**
- C4 median: **`191.9242 tok/s`**
- C4 min/max: `184.0334 / 192.1595 tok/s`
- fresh long Y median: **`62.8224 tok/s`**
- fresh long Z median: **`61.9523 tok/s`**
- correctness TTFT median: **`52.40 ms`**

Every valid round required canonical correctness, canonical four C4 hashes, C1 >= 60, C4 >= 180, both fresh-prefix 6K results >= 55, and hot correctness TTFT < 500 ms.

Durable evidence:

- `docs/evidence/r9700_phase5_clean_soak_manifest_20260913.json`
- `docs/evidence/r9700_phase5_clean_soak_raw_bundle_20260913.gz.b64`

The raw bundle preserves the five round JSONs, full soak JSON, and final stock-restore JSON. The manifest records their exact SHA-256 values and byte sizes.

## Final operational state

After the clean Phase 5 soak, the experimental canary was stopped and stock was restored.

Final verified stock state:

- service: `inneros-vllm-canary-rocm10.service` active
- container: `inneros-vllm-canary-rocm10`
- correct ROCm10 image
- `/v1/models`: HTTP 200
- expected Qwen model present

Final restore evidence SHA-256:

`94dee76980c5df074c87bd7bedfcd19bf862f32a56d1201dc4c3bd740ac3914a`

## Historical truth remains preserved

Phase 2 remains a valid negative full-model result: its W1 microkernel was genuinely faster in isolation, but the v7 full-model integration was slower than stock and was correctly rejected.

Phase 3 remains the stage where the directly relevant RDNA INT4 MoE repack path and S3 tuning first produced the full-model improvement.

Phase 4 remains the stage that isolated and mitigated the cold first-user Triton JIT penalty.

Phase 5 proves the selected recipe can be reconstructed, packaged as a reversible systemd canary, gated by hashes/readiness, soaked under single-owner measurement, and rolled back cleanly.

## Claim boundary

Safe to say:

- the tested C4 workload reproduced roughly `+20.3%` median improvement versus the controlled `158.489996 tok/s` stock+queue1 baseline across three fresh Phase 4 processes;
- the packaged Phase 5 canary reproduced C4 `191.49 tok/s` in its first service window;
- the clean five-round Phase 5 soak passed 5/5 with C4 median `191.92 tok/s`, C1 median `68.34 tok/s`, fresh-prefix ~6K around `62 tok/s`, and stock-exact deterministic hashes;
- first-user TTFT for the tested canonical request is in the ~50 ms class after readiness warmup;
- the canary is reproducible and reversible;
- a benchmark-concurrency contamination issue was found and fixed rather than hidden.

Do not say:

- universal +20% acceleration;
- official AMD R9700 support;
- first port in the world;
- full Qwen is `1.477x` faster;
- every workload is faster;
- overlapping benchmark samples are valid stability evidence;
- S3 is already the production default.

## Next step

No Phase 2-5 benchmark rerun is required for reconstruction.

The next step is a **separate explicit Phase 6 production-promotion decision** defining routing, rollout, health gates, automatic fallback and rollback. Until that decision is made, stock remains the operational default.
