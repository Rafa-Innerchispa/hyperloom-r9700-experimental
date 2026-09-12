# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-11 22:50 America/Guayaquil / 2026-09-12 03:50 UTC

This is the canonical restart file for a fresh ChatGPT/Codex session.

## 0. Current final state — read this first

**PHASE 3 S3 FULL-MODEL EXPERIMENTAL PROMOTION: PASS FOR BATCHED C4 / CONCURRENT SERVING.**

**Stock ROCm10 remains the operational default and is currently restored healthy.**

Read, in this order:

1. `FINAL_STATUS.md`
2. `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`
3. `docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`
4. `docs/R9700_PHASE3_ACTIVE_HANDOFF_20260912.md` for the detailed pre-closure trail
5. `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
6. `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`
7. `docs/DEVELOPMENT_LEDGER.md`

Do not restart the investigation from zero. Phase 2 is historical context; Phase 3 is the current engineering result.

## 1. Canonical identity

- repo: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- active engineering branch: `chatgpt/r9700-phase3-int4-repack-20260911`
- active Phase3 ops task: `ops_d07579f84116`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- stock endpoint: `http://127.0.0.1:8000/v1`
- stock service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Phase3 runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- serving controls: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, clean fresh process

Important preservation commits from the closure sequence:

- long-soak evidence: `db41aaf4d5da26340c1afb520101b5577f9b5cbc`
- S3 config: `f11328fe5d58ca6997b423b584c51cfa2a3a5907`
- S3 initial gate: `58c13fb29ffb4693bdf168d47a399d654e81115d`
- detailed active handoff: `4c96b44b955626d4608dc6eb2a6c99051dd553d1`
- pre-closure continuity checkpoint: `222c47501e3f91271efa1b0c8f92bb3145695b2b`
- S3 three-start aggregate: `8459866f9961f0bae7bedd120c99dce8ce9a797b`
- start5 launch evidence: `9b2f7690aec94ecdc99c06a255fbd82a58e25be6`
- start5 first measure: `bfe8a2a11b4d49d6553c985b78c732f6b9344865`
- start5 hot measure: `59705bef2f2914ad3b27ce31a9bc2d1cf0fcd841`
- final S3 machine summary: `2d22dd72f0a938d033cb845fe8c35093f4b54f7c`
- final status closure: `5521740f0a58891265c7d47c4b99590a8307e1d5`

Always verify current remote HEAD before editing further.

## 2. Final Phase3 S3 result

The selected candidate combines the relevant vLLM #43389 RDNA INT4/W4A16 MoE repack/interleave path with the R9700-specific S3 MoE config.

Three independent fresh S3 processes passed correctness and reproducibility gates.

### Conservative first-measurement aggregate

- C1 median: `69.496803 tok/s`
- C4 median: `188.598166 tok/s`
- C4 min/max: `186.601539 / 193.662339`
- C4 range/median: about `3.74%`
- ~6K-context median: `61.801768 tok/s`

### Hot-repeat aggregate

- C1 median: `67.749036 tok/s`
- C4 median: `192.460522 tok/s`
- C4 min/max: `186.492550 / 193.948262`
- C4 range/median: about `3.87%`
- ~6K-context median: `62.675115 tok/s`

Combined first+hot C4 median: `190.529344 tok/s`.

Correctness:

- canonical stock hash: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- selected S3 correctness requests returned that stock hash
- all four canonical C4 response hashes matched stock in the selected S3 gate

## 3. Fresh stock restore control

After S3 evidence capture, the candidate was removed and R9700 VRAM returned to about 60 MB before stock restart.

The stock service was restored successfully:

- `inneros-vllm-canary-rocm10.service`: active/running
- `/v1/models`: HTTP 200
- model loaded successfully
- correctness hash remained canonical

Fresh restored-stock first measurement:

- C1 `68.964437`
- C4 `162.909696`
- long `63.161537`

Fresh restored-stock hot measurement:

- C1 `69.455893`
- C4 `165.577595`
- long `63.752833`

Against that strong same-session healthy-hot stock control:

Conservative S3 first-measure median:

- C1: about `+0.06%`
- C4: about `+13.90%`
- long: about `-3.06%`

S3 hot-repeat median:

- C1: about `-2.46%`
- C4: about `+16.24%`
- long: about `-1.69%`

Therefore the final Phase3 claim is a substantial full-model C4/concurrent-serving improvement with C1 near parity and long-context within a few percent of healthy stock. It is not a universal acceleration claim.

## 4. Stable historical stock control

The previous clean factorial selected:

`stock ROCM_ATTN + GPU_MAX_HW_QUEUES=1`

- clean stock C4 median: `158.489996 tok/s`

Against that stable control:

- S3 conservative first-measure C4 median gain: about `+19.0%`
- S3 hot-repeat C4 median gain: about `+21.4%`

The newly restored healthy-hot stock at `165.577595` is a stronger same-session comparison and should be included whenever presenting the final result.

## 5. Cold first-request limitation

Fresh S3 processes repeatedly showed:

- first dedicated correctness TTFT: roughly `4.61-5.08 s`
- hot correctness TTFT: roughly `49-54 ms`

The first request still returned the correct stock hash. C1/C4/long measurements remained healthy.

The restored stock runtime also showed a similar cold first-request delay before normalizing, so the evidence supports a cold/lazy compile/cache effect rather than a candidate-only steady-state correctness failure.

This remains a real deployment limitation for latency-sensitive use and must be disclosed.

## 6. Long-soak evidence

A same-process untuned Phase3 observation once dropped to about `145.12 tok/s`; it is not used in the canonical independent-start result.

A later roughly 10-hour soak did not reproduce the extreme 145 event:

- late C4: `188.870452`
- 8-round median: `178.180645`
- min/max: `175.520296 / 189.760035`
- deterministic C4 hash vector remained stable

Telemetry did not support thermal throttle, power collapse or sclk collapse as the explanation for the earlier anomaly.

Evidence: `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`.

## 7. S3 config

Canonical config:

`docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`

Mounted as:

`E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Selected config:

- M1: BM16 BN64 BK32 GROUP1 SPLIT1 warps4 stages2 waves4
- M2/M4/M8/M16: BM16 GROUP1 SPLIT1
- M32: BM32 GROUP1 SPLIT1
- M64: BM64 GROUP1 SPLIT1

This config trades a small amount of the untuned Phase3 C4 peak for a much stronger cross-regime balance.

## 8. Untuned Phase3 milestone

Three independent fresh untuned starts:

- `191.450567`
- `189.645424`
- `191.426697`

Median: `191.426697 tok/s`, about `+20.8%` versus the established stable stock+queue1 baseline.

But untuned C1 median was about `60.367` and long-context about `55.199`, so it was not selected as the final cross-regime configuration.

## 9. Phase 2 remains a valid negative result

Phase 2 custom W1 proof is real:

- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- 63/63 wins per tested shape aggregate
- cosine effectively 1

This is a W1 microkernel result only.

The clean Phase2 v7 full-model hybrid remained roughly 4.5-5.1% below stable stock C4 and was correctly **not promoted**. That negative result is preserved. Phase3 is a different full-model path based on the later RDNA INT4 repack/interleave lead.

## 10. Worktree preservation

Canonical Phase3 worktree:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`

AMD final/runtime worktree containing current launchers and additional runtime evidence:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

Historical AMD research worktree:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset, or delete these before inventory/archive.

A known diagnostic bug also remains documented: `scripts/r9700_phase3_wait_isolated.py` can misleadingly print `container: still_running` if `docker inspect` fails because the named container does not exist. Verify inspect rc/listener/GPU ownership instead of trusting that string.

## 11. Safe claim boundary

Safe:

- experimental full-model R9700 Phase3 candidate reproduced about `188.6 tok/s` C4 median across three independent fresh processes;
- this is about `+19%` versus the established stable stock+queue1 baseline and about `+13.9%` versus a freshly restored healthy-hot stock observation from the same closure campaign;
- hot S3 median was about `192.5 tok/s`;
- C1 was near parity and long-context remained within a few percent of healthy stock;
- correctness and canonical C4 hashes matched stock in the selected gate;
- stock was restored healthy after the campaign.

Do not claim:

- official AMD/upstream R9700 support
- first port in the world
- `1.47682x` full-model acceleration
- universal 19-21% acceleration
- best single `193.948` observation as the universal result
- candidate is already deployed as operational default
- cold first-request latency is solved

## 12. Current operational state

At closure:

- no Phase3 candidate is intended to remain running;
- stock ROCm10 service is active/running;
- stock `/v1/models` is HTTP 200;
- healthy stock measurements were captured after restore;
- the Phase3 S3 configuration is preserved as the selected **experimental** full-model configuration, not silently installed into the stock service.

Future work should start from this state. Highest-value next engineering target: reduce cold first-request latency and tighten long-context/C1 parity without sacrificing the independently reproduced C4 advantage.
