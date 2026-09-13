# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-12 America/Guayaquil / 2026-09-13 UTC.

This is the canonical restart file for a fresh ChatGPT/Codex session. **Read this before touching the R9700 runtime.**

## 0. CURRENT STATE

**PHASE 4 IS CLOSED: PASS ACROSS 3/3 INDEPENDENT FRESH PROCESSES, AND STOCK ROCm10 HAS BEEN RESTORED HEALTHY.**

Do not repeat Phase 2, Phase 3, or the Phase 4 cold-start discovery simply to reconstruct context.

Canonical repo:

`Rafa-Innerchispa/hyperloom-r9700-experimental`

Canonical development line:

`main` and `chatgpt/r9700-phase4-coldstart-parity-20260912` share the consolidated history. `main` must be advanced to the latest validated Phase 4 closure commit whenever Phase 4 receives a final documentation/evidence commit.

Canonical Phase 4 closure artifacts:

- `docs/R9700_PHASE4_FINAL_GATE_20260913.md`
- `docs/evidence/r9700_phase4_three_start_aggregate_20260913.json`
- `docs/evidence/r9700_phase4_stock_restore_20260913.json`
- `scripts/r9700_readiness_warmup.py`
- `scripts/r9700_phase4_s3_verbose_launcher.py`
- `scripts/r9700_phase4_s3_warmup_probe.py`
- `scripts/r9700_phase4_s3_measure.py`
- `scripts/r9700_phase4_s3_long_fresh_probe.py`

Historical discovery notes remain useful but are no longer the active next gate:

- `docs/R9700_PHASE4_ACTIVE_HANDOFF_20260912.md`
- `docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`

Phase 4 ops task:

`ops_d0c3d4eebd67`

## 1. Exact hardware/runtime identity

- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- stable endpoint: `http://127.0.0.1:8000/v1`
- stable service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- runtime vLLM lineage: `f46a9dfe2c5f57bebbd29556cbbb25eabd874226`
- experimental Phase 3/4 patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 MoE config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- serving controls used for experiment: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, clean fresh process

S3 config mount name:

`E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

## 2. Phase 3 result preserved

Phase 3 proved that the upstream vLLM #43389-style INT4 MoE repack/interleave path can materially improve the full Qwen3-Coder serving workload on this exact R9700 stack.

Earlier Phase 3 three-start candidate median:

- C4: about `191.43 tok/s`
- controlled stock+queue1 baseline: `158.489996 tok/s`
- gain on that tested C4 workload: about `+20.8%`

Phase 3 then added the R9700-specific S3 config to recover small-M and long-context performance rather than optimizing C4 alone.

Canonical Phase 3 closure artifacts:

- `docs/R9700_PHASE3_S3_CLOSURE_20260912.md`
- `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`
- `docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`
- `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
- `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`

Phase 2 remains a valid negative result: its custom W1 microkernel was faster in isolation but the final v7 full-model candidate was slower than stock and was correctly not promoted.

## 3. Phase 4 root cause

Fresh stock-equivalent and S3 processes showed that `/v1/models` HTTP 200 did not mean the first real inference was latency-ready.

The first inference could trigger a late Triton JIT compile in:

`vllm.v1.attention.ops.prefix_prefill._fwd_kernel`

Captured specialization included:

- `BLOCK_DMODEL=128`
- `BLOCK_DMODEL_PADDED=128`
- `BLOCK_M=128`
- `BLOCK_N=64`
- `BLOCK_SIZE=32`
- `CAUSAL=True`
- `KV_FROM_CACHE=False`
- `SKIP_DECODE=True`
- FP16
- `num_queries_per_kv=8`

This was **not S3-specific**. Stock reproduced the same class of late first-request JIT after HTTP readiness.

## 4. Phase 4 mitigation

The selected reversible mitigation is a **readiness-stage deterministic inference**, not a risky modification to vLLM internals.

Operational pattern:

1. launch the serving process from a clean GPU state;
2. wait for HTTP 200;
3. run `scripts/r9700_readiness_warmup.py` or equivalent deterministic request;
4. require the canonical output hash;
5. only after the warmup passes expose the process to user traffic.

The readiness request intentionally pays the 4-5 second JIT cost. Subsequent first-user requests showed no new inference JIT for that specialization.

## 5. Phase 4 three-fresh-process gate

Controlled stock+queue1 C4 baseline:

`158.489996 tok/s`

The aggregate uses the **first post-readiness-warmup C4 measurement from each independent process**, avoiding cherry-picking later hot repeats.

### Start 1

- warmup JIT TTFT: `4.9047 s`
- first user TTFT: `50.9 ms`
- C1: `69.96 tok/s`
- C4: `190.60 tok/s`
- fresh-prefix ~6K: `62.00 / 61.97 tok/s`
- correctness: PASS
- C4 hashes: stock exact
- new JIT during first user inference: none

### Start 2

- warmup JIT TTFT: `4.7205 s`
- first user TTFT: `53.6 ms`
- C1: `69.68 tok/s`
- C4: `194.08 tok/s`
- ~6K: `61.96 tok/s`
- correctness: PASS
- C4 hashes: stock exact
- new JIT during first user inference: none

### Start 3

- warmup JIT TTFT: `4.6236 s`
- first user TTFT: `52.9 ms`
- C1: `68.15 tok/s`
- C4: `189.98 tok/s`
- ~6K standard: `61.45 tok/s`
- fresh-prefix ~6K controls: `61.57 / 60.79 tok/s`
- correctness: PASS
- C4 hashes: stock exact
- new JIT during first user inference: none

### Aggregate

- fresh-process passes: **3/3**
- C4 median: **190.5969 tok/s**
- C4 mean: `191.5536 tok/s`
- C4 range: `189.9816-194.0823 tok/s`
- median C4 gain vs controlled stock: **+20.258%**
- C1 median: **69.6790 tok/s**
- first-user TTFT median after readiness warmup: **52.94 ms**
- readiness JIT TTFT median: `4.720 s`
- fresh-prefix long-context median by process: **61.955 tok/s**
- canonical correctness: stock exact in all three processes
- canonical C4 hash vector: stock exact in all three processes

One transient Start 1 long-context observation near 47 tok/s was explicitly rejected as a stable result. Fresh-prefix controls and the two later independent processes reproduced the ~61-62 tok/s class.

## 6. Stock restore after Phase 4

The experimental S3 container `hyperloom-r9700-p4-s3-verbose-p18017` was removed after the third gate.

Immediately after removal, R9700 VRAM returned to the clean idle class:

`59,994,112 bytes`

The stable service was then restored:

`inneros-vllm-canary-rocm10.service`

Verified post-restore state:

- service state: `active`
- container: `inneros-vllm-canary-rocm10`
- correct ROCm10/vLLM image
- `/v1/models`: **HTTP 200**
- expected Qwen3-Coder model present
- VRAM after ready: about `28.61 GB`

Evidence:

`docs/evidence/r9700_phase4_stock_restore_20260913.json`

The restored stock process again emitted the same class of `_fwd_kernel` JIT warning on a first inference after HTTP readiness, reinforcing the Phase 4 conclusion that the cold first-request issue is not introduced by S3.

## 7. Claim boundary

Safe claims:

- the tested full-model C4 workload reproduced a roughly **+20.3% median improvement** versus the controlled `158.489996 tok/s` stock+queue1 baseline across three independent fresh Phase 4 processes;
- the readiness warmup moved the first user request from the multi-second JIT class to roughly **51-54 ms TTFT** in all three fresh processes;
- C1 remained around `68-70 tok/s` and fresh-prefix ~6K decode around `61-62 tok/s`;
- deterministic correctness and the canonical four C4 hashes matched stock;
- operational stock was restored after the experiment.

Do not claim:

- universal +20% acceleration;
- official AMD R9700 support;
- first port in the world;
- `1.47682x` as a full-model result;
- every workload or prompt shape is faster;
- the S3 candidate is already the production default.

## 8. Exact next action

**No benchmark rerun is required. Phase 4 is closed.**

The next engineering decision is separate from Phase 4:

- decide whether to promote S3 + readiness warmup into a persistent service/canary;
- if promoted, implement it as a controlled deployment with explicit rollback to `inneros-vllm-canary-rocm10.service` stock recipe;
- preserve the current stock operational default until that promotion change is deliberately approved and validated.

Do not silently mutate production while treating it as benchmark cleanup.

## 9. Worktree preservation

Important historical/raw worktrees:

- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`
- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`
- `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset, or delete them before deliberate inventory/archive.

## 10. Restart rule

A new session must:

1. read this file first;
2. read `docs/R9700_PHASE4_FINAL_GATE_20260913.md`;
3. read `docs/evidence/r9700_phase4_three_start_aggregate_20260913.json`;
4. read `docs/evidence/r9700_phase4_stock_restore_20260913.json`;
5. verify current remote `main` and Phase 4 branch HEADs;
6. verify stock runtime is still healthy before any mutation;
7. continue only with the separate promotion/canary decision, not by repeating closed Phase 2/3/4 experiments.
