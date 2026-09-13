# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-13 UTC / 2026-09-12 America/Guayaquil.

This is the canonical restart file for a fresh ChatGPT/Codex session. **Read this before touching the R9700 runtime.**

## 0. CURRENT STATE — READ FIRST

**PHASE 5 IS CLOSED: REVERSIBLE CANARY PACKAGING + SINGLE-OWNER SOAK PASS.**

The operational default after closure is still the **stock ROCm10 service on port 8000**. S3 has **not** been silently promoted into production routing.

Do not repeat Phase 2, Phase 3, Phase 4, or Phase 5 experiments merely to reconstruct context.

Canonical repo:

`Rafa-Innerchispa/hyperloom-r9700-experimental`

Canonical validated Phase 5 branch:

`chatgpt/r9700-phase5-canary-v3-20260913`

Canonical Phase 5 task:

`ops_1fdc08a1b732`

Important Phase 5 closure artifacts:

- `docs/R9700_PHASE5_CANARY_CLOSURE_20260913.md`
- `docs/evidence/r9700_phase5_canary_bundle_manifest.json`
- `docs/evidence/r9700_phase5_canary_measure_20260913T023328Z.json`
- `docs/evidence/r9700_phase5_concurrency_contamination_20260913.json`
- `docs/evidence/r9700_phase5_benchmark_lock_selftest_20260913.json`
- `docs/evidence/r9700_phase5_clean_soak_manifest_20260913.json`
- `docs/evidence/r9700_phase5_clean_soak_raw_bundle_20260913.gz.b64`
- `scripts/systemd/inneros-vllm-hyperloom-s3-canary.service`
- `scripts/r9700_phase5_prepare_canary.py`
- `scripts/r9700_phase5_canary_preflight.py`
- `scripts/r9700_phase5_wait_canary_active.py`
- `scripts/r9700_phase5_canary_measure.py`
- `scripts/r9700_phase5_canary_soak.py`
- `scripts/r9700_phase5_benchmark_lock_selftest.py`
- `scripts/r9700_readiness_warmup.py`

The next legitimate engineering action is a **separate explicit promotion decision / Phase 6**, not another reconstruction benchmark.

## 1. Exact hardware/runtime identity

- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- stable endpoint: `http://127.0.0.1:8000/v1`
- stable service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- runtime vLLM lineage: `f46a9dfe2c5f57bebbd29556cbbb25eabd874226`
- experimental patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 MoE config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- S3 config mount name: `E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`
- serving controls used by the selected candidate: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`

## 2. Closed history — preserve, do not redo

### Phase 2

The custom W1 microkernel proof was genuinely faster in isolation, with median small-M speedup around `1.47682x`, but the final v7 full-model hybrid remained roughly 4.5-5.1% below stable stock C4. It was correctly **not promoted**.

### Phase 3

The upstream vLLM #43389-style INT4 MoE repack/interleave path materially improved the full Qwen3-Coder serving workload on this exact R9700 stack.

An earlier three-start candidate produced about `191.43 tok/s` C4 median versus controlled stock+queue1 `158.489996 tok/s`, about `+20.8%` on that tested workload.

Phase 3 then added the R9700-specific S3 MoE config to recover C1 and long-context performance rather than optimizing C4 alone.

Canonical Phase 3 artifacts:

- `docs/R9700_PHASE3_S3_CLOSURE_20260912.md`
- `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`
- `docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`
- `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
- `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`

### Phase 4

Phase 4 identified the multi-second first-request penalty as a late Triton JIT in:

`vllm.v1.attention.ops.prefix_prefill._fwd_kernel`

The same class of JIT appeared on freshly restored stock, so it was not S3-specific.

The reversible mitigation is a deterministic readiness-stage inference. It intentionally pays the late JIT before exposing the process to user traffic and requires the canonical output hash.

Three independent fresh Phase 4 processes passed:

- C4: `190.5969`, `194.0823`, `189.9816 tok/s`
- median C4: **`190.5969 tok/s`**
- controlled stock+queue1 baseline: `158.489996 tok/s`
- tested-workload median gain: **`+20.258%`**
- C1 median: `69.6790 tok/s`
- fresh-prefix ~6K: about `61-62 tok/s`
- first-user TTFT after readiness warmup: about `51-54 ms`
- correctness and canonical four C4 hashes: stock exact

Canonical Phase 4 artifacts:

- `docs/R9700_PHASE4_FINAL_GATE_20260913.md`
- `docs/evidence/r9700_phase4_three_start_aggregate_20260913.json`
- `docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`

Phase 4 task `ops_d0c3d4eebd67` is closed PASS.

## 3. Phase 5 packaged canary

Phase 5 converted S3 into a real reversible service rather than a hand-launched benchmark recipe.

Canary identity:

- systemd unit: `inneros-vllm-hyperloom-s3-canary.service`
- Docker container: `inneros-vllm-hyperloom-s3-canary`
- isolated port: `18018`
- stock service is never stopped automatically by the canary unit
- preflight fails closed unless stock is inactive, R9700 VRAM is clean, bundle hashes are exact, and config hash is exact
- `ExecStartPost` runs the readiness warmup and requires the canonical hash before the unit becomes active

The bundle is rebuilt from versioned/pinned repo sources. It no longer depends on a historical Phase 3 worktree for deployment.

First packaged-canary runtime gate:

- `/v1/models`: HTTP 200
- correctness: stock exact
- C1: `68.8094 tok/s`
- C4: `191.4875 tok/s`
- fresh-prefix ~6K: `61.8344 / 62.1267 tok/s`
- hot correctness TTFT: `52.0 ms`
- canonical C4 hashes: stock exact

## 4. Phase 5 concurrency incident — negative harness result, not candidate regression

Two coordinated ChatGPT tasks accidentally benchmarked the same canary on port 18018 at nearly the same time.

Overlapping soak starts:

- `r9700_phase5_canary_soak_20260913T025312Z.json`
- `r9700_phase5_canary_soak_20260913T025313Z.json`

During overlap, samples degraded to C4 `151.30` and `88.64 tok/s`; one first C4 hash became the known `e79a200b...` anomaly.

This evidence is **invalid for single-client stability decisions** because two benchmark suites were competing for the same GPU.

The key forensic control is that the same canary process, without restart, recovered after overlap stopped to:

- C1 `67.9771 tok/s`
- C4 `193.4585 tok/s`
- fresh-prefix ~6K `62.7305 / 62.5542 tok/s`
- correctness and C4 hashes stock exact
- PASS

Canonical incident evidence:

`docs/evidence/r9700_phase5_concurrency_contamination_20260913.json`

## 5. Benchmark-concurrency fix

Phase 5 now uses one exclusive local benchmark lease:

`var/r9700_phase5_benchmark.lock`

- standalone measure takes it before any health/inference request
- soak owns it for the entire multi-round campaign
- a second client exits rc `4` with `benchmark_lock_busy` before sending inference traffic
- soak child measurements inherit the owner lease explicitly

Dynamic self-test passed for both measure and soak clients.

Evidence:

`docs/evidence/r9700_phase5_benchmark_lock_selftest_20260913.json`

## 6. Clean single-owner five-round soak

After the concurrency fix, a fresh canary was started from a clean GPU state. Preflight and readiness passed. One exclusive-lock soak then ran five rounds.

Result: **5/5 PASS, zero aborts.**

Aggregate:

- C1 min/median/mean/max: `68.3227 / 68.3391 / 68.3704 / 68.5126 tok/s`
- C4 min/median/mean/max: `184.0334 / 191.9242 / 190.2179 / 192.1595 tok/s`
- fresh long Y min/median/mean/max: `62.1840 / 62.8224 / 62.7132 / 62.9794 tok/s`
- fresh long Z min/median/mean/max: `61.2839 / 61.9523 / 61.8156 / 61.9874 tok/s`
- correctness TTFT min/median/mean/max: `51.35 / 52.40 / 53.15 / 56.07 ms`

Every round required:

- canonical correctness hash
- canonical four C4 hashes
- C1 >= 60 tok/s
- C4 >= 180 tok/s
- both fresh-prefix 6K measurements >= 55 tok/s
- hot correctness TTFT < 500 ms

All five rounds passed all gates.

Durable evidence:

- `docs/evidence/r9700_phase5_clean_soak_manifest_20260913.json`
- `docs/evidence/r9700_phase5_clean_soak_raw_bundle_20260913.gz.b64`

The manifest records exact byte sizes and SHA-256 for all five raw round JSONs, the full soak JSON, and the final stock restore JSON. The compressed raw bundle preserves their complete contents.

## 7. Final operational state after Phase 5

After the clean soak:

1. experimental canary was stopped;
2. stock service was started;
3. stock service reached `active`;
4. stock container/image identity was correct;
5. `/v1/models` returned **HTTP 200**;
6. expected Qwen model was present.

Final restore evidence file:

`r9700_phase4_stock_restore_20260913T031729Z.json`

SHA-256:

`94dee76980c5df074c87bd7bedfcd19bf862f32a56d1201dc4c3bd740ac3914a`

Operational default therefore remains stock ROCm10.

## 8. Claim boundary

Safe:

- Phase 4 reproduced roughly `+20.3%` median C4 improvement against the controlled `158.489996 tok/s` stock+queue1 baseline on the tested workload across three fresh processes;
- the packaged Phase 5 service reproduced `191.49 tok/s` C4 in its first window;
- the clean single-owner Phase 5 five-round soak passed 5/5 with C4 median `191.92 tok/s`, C1 median `68.34 tok/s`, and fresh-prefix 6K medians around `62 tok/s`;
- deterministic correctness and canonical C4 hashes remained stock exact in valid gates;
- readiness warmup keeps first-user TTFT in the ~50 ms class for the tested canonical request;
- the canary packaging is reversible and stock was restored after validation;
- the cross-chat benchmark-concurrency failure was detected, preserved, and fixed with an exclusive lease.

Do not claim:

- universal +20% acceleration;
- official AMD R9700 support;
- first port in the world;
- every workload is faster;
- overlapping benchmark samples are valid stability evidence;
- S3 is already the production default.

## 9. Exact next action

**No Phase 2-5 benchmark rerun is required to reconstruct state.**

The next legitimate step is a separate **Phase 6 / explicit production-promotion decision**. If approved, define:

- routing/cutover mechanism;
- health and readiness requirements;
- production rollback trigger;
- whether S3 becomes default or remains an opt-in local provider;
- observability and automatic fallback to `inneros-vllm-canary-rocm10.service`.

Do not silently replace the stock default as part of cleanup.

## 10. Worktree/raw preservation

Historical/raw worktrees still contain useful negative and forensic evidence:

- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`
- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`
- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset, or delete them blindly. Phase 5 clean-soak raw evidence is additionally preserved in the compressed GitHub bundle, so Phase 5 no longer depends on those worktrees for continuity.

## 11. Restart rule

A new session must:

1. read this file first;
2. read `docs/R9700_PHASE5_CANARY_CLOSURE_20260913.md`;
3. read `docs/evidence/r9700_phase5_clean_soak_manifest_20260913.json`;
4. read `docs/evidence/r9700_phase5_concurrency_contamination_20260913.json` so the invalid overlapping soak is not mistaken for a candidate regression;
5. verify current remote `main` and the stock runtime before mutation;
6. continue from the Phase 6 promotion decision only;
7. do not repeat closed Phase 2/3/4/5 experiments merely to rebuild context.
