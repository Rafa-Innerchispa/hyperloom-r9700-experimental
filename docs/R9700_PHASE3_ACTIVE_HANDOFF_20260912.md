# HyperLoom R9700 — Phase 3 S3 Active Handoff

Last updated: 2026-09-11 22:33 America/Guayaquil / 2026-09-12 03:33 UTC

Status: **PHASE 3 ACTIVE — TUNED S3 FULL-MODEL CANDIDATE REPRODUCING; START5 IN FLIGHT; PROMOTION NOT YET CLOSED**

This file exists so a fresh ChatGPT/Codex session can resume without depending on chat history.

## Canonical identity

- Repo: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Active branch: `chatgpt/r9700-phase3-int4-repack-20260911`
- Branch HEAD before this handoff commit: `58c13fb29ffb4693bdf168d47a399d654e81115d`
- Active ops task: `ops_d07579f84116`
- AMD node: `ralfiia-amd` / `.5`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Stock service: `inneros-vllm-canary-rocm10.service`, normal endpoint port 8000
- Candidate serving controls: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, fresh process, stock systemd inactive and VRAM released before launch

## Worktrees — do not clean/reset blindly

Canonical Phase3 project/runtime worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`

AMD final/runtime worktree containing current launchers and raw evidence:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

Historical AMD research worktree:
`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset or delete these before inventory/archive.

## Historical truth that remains valid

Phase 2 custom W1 microkernel proof remains real:
- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- 63/63 wins per tested shape aggregate
- cosine effectively 1

This is **W1 microkernel only**, not a full-model speedup.

The Phase2 clean v7 full-model integration remained roughly 4.5-5.1% below stable stock C4 and was correctly **NOT promoted**. Preserve that rejection.

## Phase 3 repacked INT4 path

Phase 3 backports the relevant vLLM #43389 RDNA INT4/W4A16 MoE repack/interleave work to the ROCm10 runtime.

Runtime patch SHA-256:
`3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`

Real Qwen layer-0 AutoAWQ repack/qzeros/scales/dequant gates passed exactly. The five vLLM patch files are mounted read-only into disposable candidate containers.

### Untuned Phase3 independent-start result

Three independent fresh starts:
- `191.45056709531252 tok/s`
- `189.64542351365026 tok/s`
- `191.42669659550938 tok/s`

Median C4: `191.42669659550938 tok/s`

Gain:
- vs stock+queue1 median 158.489996: `+20.7816%`
- vs same-day healthy-fast stock 162.102053: `+18.0902%`
- vs historical fast stock 165.699: `+15.5268%`

Untuned limitation:
- C1 median ~60.367 tok/s
- ~6K decode median ~55.199 tok/s
- stock+queue1 reference ~63-65 C1 and ~58.87 long

Therefore untuned Phase3 is a reproducible C4 win but not universal-regime promotion.

## Long-soak / 145 tok/s anomaly

A later same-process observation once fell to ~145.12 tok/s. It is excluded from the canonical independent-start campaign.

A ~10h soak did not reproduce that extreme drop:
- late single C4 `188.87045177070354`
- C1 `60.64490792010163`
- long `55.75631711428731`
- 8-round C4 values `[176.28501142481116,177.58045561323954,189.76003455123833,177.57259018704798,175.52029636921574,181.71630827433628,178.7808343497473,189.2325916901254]`
- median `178.1806449814934`
- deterministic C4 hash vector
- no evidence supporting thermal throttle or power/sclk collapse

Canonical evidence: `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`.

## S3 R9700-specific INT4 MoE config

Config evidence:
`docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`

Config commit:
`f11328fe5d58ca6997b423b584c51cfa2a3a5907`

Config SHA observed by launcher:
`8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`

Mounted as:
`vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Exact shape config:
- M1: BM16 BN64 BK32 GROUP1 SPLIT1 warps4 stages2 waves4
- M2/M4/M8/M16: BM16 GROUP1 SPLIT1
- M32: BM32 GROUP1 SPLIT1
- M64: BM64 GROUP1 SPLIT1

## S3 measurements already observed

### Tuned start3, port 18013

First/cold measure:
- C1 `66.90866653149924`
- C4 `186.60153918601006`
- long `58.18529057634683`
- correctness hash `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- correctness TTFT `5.081645346s`

Hot repeat:
- C1 `67.74903556188706`
- C4 `186.49254976456643`
- long `62.67511482162822`
- correctness stock-exact
- correctness TTFT `0.053394466s`

Additional canonical hybrid measure same process:
- C1 `61.755682610028146`
- C4 `189.767407023313`
- long `62.120630097987394`
- correctness stock-exact

Canonical S3 gate summary:
`docs/evidence/r9700_phase3_tuned_s3_gate_20260912.json`

### Independent tuned start4, port 18014

Launch:
- VRAM before launch ~59,994,112 bytes
- stock systemd inactive
- runtime patch SHA exact `3935...5630d`
- S3 config SHA exact `8b6344...d3e6`

First measure:
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

All four C4 request hashes match the canonical stock hashes.

Interpretation: S3 materially improves the small-M/long-context balance while retaining a large C4 win. First request after a fresh process repeatedly incurs a 4-5s cold/lazy artifact; hot TTFT returns to ~50ms.

## LIVE STATE AT CHECKPOINT — START5

A fresh independent start5 was launched and must be inspected before any relaunch:
- label `hybrid_start5`
- container `hyperloom-r9700-p3-hybrid-s5-p18015`
- port `18015`
- container id `e45e8e1301ba19029feb9faea78ac69b547b18983addf09e3daaedb8c18aa2b5`
- launch time `2026-09-12T03:04:16Z`
- clean VRAM before launch ~59,994,112 bytes
- stock systemd inactive
- config SHA `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- runtime patch SHA `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- model load: 16.05 GiB in ~128.45 s
- `torch.compile`: ~133.75 s
- server log explicitly confirmed it selected the R9700 INT4 config file above
- last observed state before this handoff: startup/graph capture still progressing; no qzeros/layout/kernel crash observed

Do not launch start6 or another candidate before checking whether start5 is already HTTP-ready.

## Diagnostic traps discovered

1. `scripts/r9700_phase3_wait_isolated.py` can output `container: still_running` when `docker inspect` itself fails because the named container does not exist. Verify inspect rc/port/GPU ownership explicitly before trusting that string.
2. systemd service state alone is not enough to identify GPU ownership. A manually launched vLLM candidate can occupy ~28 GiB with stock systemd inactive.
3. Never launch candidate tests on non-clean VRAM.

## Exact resume sequence

1. Inspect start5 / port 18015. Keep it if alive; do not relaunch.
2. Wait for `/v1/models` HTTP 200.
3. Run `scripts/r9700_phase3_hybrid_case_measure.py` immediately for first/cold measure.
4. Run the same measure again hot.
5. Preserve start5 launch + both raw measurement JSONs into this active Phase3 branch.
6. Aggregate tuned S3 independent starts (start3/start4/start5), clearly separating first/cold from hot repeat where relevant.
7. Compute C1/C4/long medians and ranges against:
   - stock+queue1 C4 `158.489996`
   - healthy-fast stock `162.102053`
   - historical fast stock `165.699`
   - stock C1 ~63-65
   - stock long ~58.87
8. Promotion requires exact correctness, canonical C4 output hashes, independent process reproducibility, and no material cross-regime regression that invalidates the claim.
9. If start5 contradicts start3/start4 materially, run one more clean independent start6 before verdict. If it agrees, three S3 starts are sufficient for this gate.
10. Remove candidate after evidence capture.
11. Restore `inneros-vllm-canary-rocm10.service`; verify stock `/v1/models` HTTP 200.
12. Update `docs/R9700_PROJECT_CONTINUITY.md`, `FINAL_STATUS.md`, `docs/DEVELOPMENT_LEDGER.md` and README/public claims only after the gate is closed.
13. Commit/push raw evidence + aggregate + final verdict. Do not leave useful evidence only in dirty runtime worktrees.

## Current truth boundary

**PHASE3 S3 TUNED CANDIDATE KEEP FOR FINAL GATE / PROMOTION PENDING START5 + AGGREGATE + STOCK RESTORE**

Safe claim now: Phase3 #43389 repacked INT4 already reproduced ~20.8% C4 gain untuned across three fresh starts, and the R9700-specific S3 config materially improves small-M and long-context balance while retaining stock-exact correctness/hash behavior in start3/start4. Do not call S3 production-promoted until start5, aggregate and stock-restore gates are complete.
