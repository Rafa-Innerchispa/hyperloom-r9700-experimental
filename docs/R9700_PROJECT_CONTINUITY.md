# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-11 22:33 America/Guayaquil / 2026-09-12 03:33 UTC

This is the canonical restart file for a fresh ChatGPT/Codex session.

## 0. CURRENT ACTIVE STATE — READ FIRST

**PHASE 3 IS ACTIVE. DO NOT RESUME FROM THE OLD PHASE 2 CLOSURE.**

Read, in this order:

1. `docs/R9700_PHASE3_ACTIVE_HANDOFF_20260912.md`
2. `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
3. `docs/evidence/r9700_phase3_tuned_s3_gate_20260912.json`
4. `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`
5. `docs/evidence/r9700_phase3_three_start_aggregate_20260911.json`
6. `docs/DEVELOPMENT_LEDGER.md`
7. This file's Phase 2 history below only as historical context.

Active repo/branch:
- repo `Rafa-Innerchispa/hyperloom-r9700-experimental`
- branch `chatgpt/r9700-phase3-int4-repack-20260911`
- Phase3 tuned handoff commit: `4c96b44b955626d4608dc6eb2a6c99051dd553d1`
- S3 config commit: `f11328fe5d58ca6997b423b584c51cfa2a3a5907`
- S3 initial gate commit: `58c13fb29ffb4693bdf168d47a399d654e81115d`
- long-soak checkpoint: `db41aaf4d5da26340c1afb520101b5577f9b5cbc`
- active ops task: `ops_d07579f84116`

Current truth boundary:

**PHASE3 S3 TUNED CANDIDATE KEEP FOR FINAL GATE / PROMOTION PENDING START5 + AGGREGATE + STOCK RESTORE**

Do not claim final production promotion yet.

## 1. Canonical identity and runtime

- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- normal stock endpoint: `http://127.0.0.1:8000/v1`
- stock service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Phase3 serving controls: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, fresh process, stock systemd inactive, clean VRAM before launch
- Phase3 runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`

Never reset or clean a research/runtime worktree merely because it is dirty.

## 2. Phase 3 untuned result — reproduced C4 win

Phase3 backports the relevant vLLM #43389 RDNA INT4/W4A16 MoE repack/interleave path to the ROCm10 runtime. Real Qwen layer-0 AutoAWQ repack, qzeros, scales and sampled dequantization gates passed exactly.

Three independent fresh Phase3 starts produced:

- `191.45056709531252 tok/s`
- `189.64542351365026 tok/s`
- `191.42669659550938 tok/s`

Median C4: `191.42669659550938 tok/s`.

Gain:
- vs stable stock+queue1 C4 `158.489996`: `+20.7816%`
- vs healthy-fast stock `162.102053`: `+18.0902%`
- vs historical fast stock `165.699`: `+15.5268%`

Correctness:
- canonical stock hash `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- canonical comparable Phase3 requests returned the stock hash
- canonical C4 output hashes matched stock in the independent-start campaign

Untuned limitation:
- C1 median ~`60.367 tok/s`
- ~6K decode median ~`55.199 tok/s`
- stock+queue1 reference ~63-65 C1 and ~58.87 long-context

Therefore untuned Phase3 is a reproducible batched-C4 win, not yet a universal serving promotion.

## 3. Same-process 145 anomaly and long-soak result

A later same-process observation once fell to ~145.12 tok/s. It is excluded from the canonical independent-start campaign.

A later ~10h soak did not reproduce the extreme drop:
- single late C4 `188.87045177070354`
- C1 `60.64490792010163`
- long `55.75631711428731`
- 8-round C4 median `178.1806449814934`, min `175.52029636921574`, max `189.76003455123833`
- deterministic C4 hash vector remained stable
- no evidence supporting thermal throttle or power/sclk collapse

Evidence: `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`.

Interpretation: C4 advantage persists over long uptime; intra-process variability remains but the strongest next lever became R9700-specific INT4 MoE tuning.

## 4. S3 tuned R9700 INT4 config

Canonical config:
`docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`

Observed config SHA:
`8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`

Mounted as:
`vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Exact config:
- M1: BM16 BN64 BK32 GROUP1 SPLIT1 warps4 stages2 waves4
- M2/M4/M8/M16: BM16 GROUP1 SPLIT1
- M32: BM32 GROUP1 SPLIT1
- M64: BM64 GROUP1 SPLIT1

### S3 tuned start3

First/cold:
- C1 `66.90866653149924`
- C4 `186.60153918601006`
- long `58.18529057634683`
- correctness stock-exact
- correctness TTFT `5.081645346s`

Hot repeat:
- C1 `67.74903556188706`
- C4 `186.49254976456643`
- long `62.67511482162822`
- correctness stock-exact
- correctness TTFT `0.053394466s`

Additional same-start canonical hybrid measure:
- C1 `61.755682610028146`
- C4 `189.767407023313`
- long `62.120630097987394`

### Independent S3 tuned start4

First:
- C1 `69.49680303871574`
- C4 `188.59816600815128`
- long `61.801768267545576`
- correctness stock-exact
- first correctness TTFT `4.612734004s`

Hot repeat:
- C1 `63.25507427770107`
- C4 `193.94826236494046`
- long `62.20834147639644`
- correctness stock-exact
- correctness TTFT `0.049290768s`

All four C4 request hashes remained the canonical stock hashes.

Interpretation: S3 materially improves the small-M/long-context balance while retaining the large C4 advantage. The first request after fresh process startup repeatedly shows a 4-5s cold/lazy artifact, while hot TTFT returns to ~50ms.

## 5. LIVE START5 STATE AT HANDOFF

Do not relaunch blindly. A fresh independent start5 has already been launched:

- label `hybrid_start5`
- container `hyperloom-r9700-p3-hybrid-s5-p18015`
- port `18015`
- container id `e45e8e1301ba19029feb9faea78ac69b547b18983addf09e3daaedb8c18aa2b5`
- launched `2026-09-12T03:04:16Z`
- VRAM before launch ~59,994,112 bytes
- stock systemd inactive before launch
- config SHA `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- runtime patch SHA `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- model loaded 16.05 GiB in ~128.45 s
- torch.compile took ~133.75 s
- log explicitly confirmed use of the R9700 `int4_w4a16` config file
- last observed pre-handoff state: still completing startup/graph capture, with no qzeros/layout/kernel crash

Exact next action: inspect start5 / port 18015 and keep it if alive. Wait for HTTP 200; run the hybrid measure once cold/first and once hot. Preserve the raw launch and both raw measurement JSONs.

If start5 agrees with start3/start4, aggregate three independent S3 starts and close the promotion gate. If start5 materially contradicts them, run one additional clean start6 before deciding.

After evidence capture, remove candidate, restore `inneros-vllm-canary-rocm10.service`, verify stock `/v1/models` HTTP 200, then update `FINAL_STATUS.md`, `docs/DEVELOPMENT_LEDGER.md` and public claims.

## 6. Diagnostic traps discovered

- `scripts/r9700_phase3_wait_isolated.py` can say `container: still_running` when `docker inspect` actually failed because the container does not exist. Verify inspect rc, listener and GPU owner explicitly.
- systemd can be inactive while a manually launched vLLM candidate occupies ~28 GiB VRAM. Do not infer GPU ownership from systemd state alone.
- Never launch a benchmark candidate with non-clean VRAM.

## 7. Worktree preservation

Canonical Phase3 worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`

AMD final/runtime worktree with current raw Phase2/3 evidence and launchers:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

Historical AMD research worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, reset, or delete before inventory/archive.

---

# Historical Phase 2 truth

The sections below remain relevant as historical evidence and claim boundaries, but they no longer describe the active engineering phase.

## 8. Phase 2 clean v7 gate — NOT promoted

Candidate source: `scripts/r9700_wna16_hybrid_patch_v7_clean.py`

SHA-256: `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`

The corrected v7 loaded the complete model and observed:
- 48 x `custom_small_w1_stock_w2`
- 48 x `stock_full_fallback`

Candidate C4:
- `149.891101 tok/s`
- `150.842143 tok/s`
- median `150.366622 tok/s`

Fair stock+queue1 control:
- current clean control `157.489588 tok/s`
- prior clean factorial median `158.489996 tok/s`

Therefore the Phase2 v7 full-model path remained roughly 4.5-5.1% below stable stock and was not promoted.

## 9. Stable serving baseline

Selected stock control:
`stock attention + GPU_MAX_HW_QUEUES=1`

- clean C4 runs `158.567959 / 158.412033 tok/s`
- median `158.489996 tok/s`
- spread about `0.098%`

Unified Attention:
- default queues median `138.764971`
- queue1 median `156.753550`

Unified Attention did not beat stock+queue1 and remains excluded.

## 10. Phase 2 W1 kernel proof

Real-weight custom W1 versus stock Triton WNA16:
- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- `63/63` wins per tested shape aggregate
- cosine effectively 1

Evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`.

This is a W1 microkernel result only.

## 11. Rejected/invalid historical paths to preserve

Do not delete or reinterpret these as successes:
- AITER/FlyDSL sorting: HSA memory fault
- activation group pre-sum: correct but slower
- early FP16/BF16 mirror: dtype mismatch
- custom algebraic W2: slower than stock
- mmap/SIGUSR1/SIGUSR2 same-process gate: invalid under captured graphs
- recovered live tuned-config override: not promotable
- Unified Attention: valid test, but not better than stock+queue1
- Phase2 final v7 full model: functional but C4 below stable stock

## 12. Claim boundary

Never claim:
- official AMD support
- “first port in the world”
- `1.47682x` as a full-model speedup
- universal exact-output parity unless the exact tested gate supports it
- universal serving speedup from one workload regime
- promotion against a pathological slow stock start

The active Phase3 result is stronger, but final S3 production promotion remains gated on start5, aggregate reproducibility and stock restore.
