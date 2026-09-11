# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-11 (America/Guayaquil)

This is the canonical restart file for a fresh ChatGPT/Codex session. Read this first, then `FINAL_STATUS.md`, `docs/evidence/r9700_v7_final_gate_summary_20260911.json`, `docs/DEVELOPMENT_LEDGER.md`, and `docs/COMMUNITY_FEEDBACK_LEDGER.md`.

## 1. Canonical identity

- repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- final presentation branch: `chatgpt/r9700-final-presentation-20260910`
- preservation/base branch: `chatgpt/r9700-rocm10-upstream-refresh-20260910`
- base preservation commit: `c4e4aae90f5582d76b7881ba932778bd94922611`
- measured-entrypoint fix commit: `97225ac5e7a314091f2231860d69d8e0ef63c8e6`
- parent ops task: `ops_9c4c37cfbb7e`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- normal endpoint: `http://127.0.0.1:8000/v1`
- stock service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- versions observed after final restore: Python 3.14.7; torch `2.12.0+rocm10.0.0`; HIP `7.15.26333`; vLLM `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`.

Always verify remote HEAD before editing. Do not reset or clean a research/runtime worktree merely because it is dirty.

## 2. Final truth boundary

**KERNEL KEEP / FULL-MODEL INTEGRATION NOT PROMOTED**

What is real:

- an experimental HyperLoom RDNA4 path physically executes on the R9700 / `gfx1201`;
- the packed-INT4 small-M W1 Triton kernel is a validated microkernel improvement;
- full Qwen3-Coder 30B AWQ boots with `R9700HybridWNA16Experts`;
- final v7 inference observed all 48 MoE layers on the custom route plus stock fallback;
- stock serving was restored after testing.

What is not claimed:

- no official AMD/upstream R9700 support;
- no “first port in the world” claim;
- no 1.477x full-model claim;
- no E2E speedup claim for v7;
- no promotion from a pathological slow stock spawn;
- no universal exact-output-parity claim for final v7.

## 3. Final clean v7 gate

Candidate source: `scripts/r9700_wna16_hybrid_patch_v7_clean.py`

SHA-256: `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`

Serving path:

- `scripts/r9700_v7_server_entry.py`
- `scripts/r9700_v7_full_model_launcher.py`
- `scripts/r9700_v7_full_model_collect.py`
- `scripts/r9700_v7_remove_candidate.py`

Final test conditions:

- stock systemd unit stopped through authorized host ops;
- stock container absent;
- VRAM used before candidate launch about 60 MB;
- stock `ROCM_ATTN` attention;
- `GPU_MAX_HW_QUEUES=1` stability control;
- patch/entrypoint bind-mounted read-only into a disposable ROCm10 container.

The first explicit-entrypoint attempt failed because `runpy` was re-entered under Python multiprocessing `spawn`. It was a harness/bootstrap failure, not a kernel failure. The fix was committed as `97225ac5e...` and the failure is preserved at `docs/evidence/r9700_v7_full_model_bootstrap_failure_20260911T0511Z.json`.

Corrected v7 then loaded the complete model and vLLM logged `Using R9700HybridWNA16Experts`.

Final path evidence:

- `48` x `custom_small_w1_stock_w2`
- `48` x `stock_full_fallback`

Dedicated streaming TTFT: about `64.9 ms`.

Candidate C4:

- `149.891101 tok/s`
- `150.842143 tok/s`
- median `150.366622 tok/s`

Fair stock+queue1 control:

- current clean control `157.489588 tok/s`
- prior clean factorial median `158.489996 tok/s`

Therefore final v7 remains roughly 4.5-5.1% below the selected stable stock baseline and is not promoted.

Correctness boundary:

- all four C4 request hashes matched stock exactly;
- dedicated v7 correctness hash: `e4810fc5ef6597d28440cc7e32d384180088ca95c385c7ddcf12c1439690f5f9`;
- dedicated stock+queue1 hash: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`;
- universal exact deterministic equivalence is therefore not closed for final v7.

Canonical machine summary: `docs/evidence/r9700_v7_final_gate_summary_20260911.json`.

## 4. Stable serving baseline

The clean serving factorial is closed. Read `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md` before quoting serving results.

Selected promotion control:

`stock attention + GPU_MAX_HW_QUEUES=1`

- C4 clean runs: `158.567959 / 158.412033 tok/s`
- median: `158.489996 tok/s`
- spread: about `0.098%`

Unified Attention results:

- default queues median: `138.764971 tok/s`
- queue1 median: `156.753550 tok/s`

Unified Attention does not beat stock+queue1 and is excluded from the final candidate.

Default-queue stock remains startup-bimodal, with healthy-fast observations above 160 tok/s and pathological slow observations around 72 tok/s. Never use a single slow stock start as a promotion baseline.

## 5. W1 kernel proof

Live Qwen MoE contract:

- `TritonWNA16Experts`
- FP16 activations
- 128 experts, top-k 8
- packed INT4 W4A16, group size 128
- W1 `uint8 [128,1536,1024]`
- W2 `uint8 [128,2048,384]`

Real-weight custom W1 versus stock Triton WNA16:

- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- `63/63` wins per shape across three campaigns
- cosine effectively 1.0

Evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`.

This is a W1 microkernel result only.

## 6. Recovered tuner — history gap closed

The tuner files once thought lost were recovered and preserved in Git.

Canonical note: `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md`

Raw tuner: `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`

Winner:

- BM16 / BN64 / BK32 / GROUP_M1
- warps4 / stages2 / waves4 / split-K1
- isolated M1..16 median speedup `1.209256x`
- minimum tested speedup `1.083843x`
- verdict: `MICROBENCH KEEP`

Recovered live override smokes remain `REJECT / NOT PROMOTED`: one did not observe the override; the second observed it but failed correctness/process-state promotion controls.

## 7. Historical evidence that remains valid

Phase 1 independent-process serving/concurrency result:

- baseline `19.893 / 19.942 / 19.987 tok/s`
- candidate `36.004 / 36.078 / 36.194 tok/s`
- paired median `+80.99%`

Interpretation: serving/concurrency, not kernel speedup.

Earlier full-model stock-layout smoke:

- 48 custom observations
- 48 fallback observations
- deterministic response hash matched stock for that smoke
- rollback passed

Evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`.

## 8. Rejected/invalid paths to preserve

Do not delete or reinterpret these as successes:

- AITER/FlyDSL sorting: HSA memory fault
- activation group pre-sum: correct but slower
- early FP16/BF16 mirror: dtype mismatch
- custom algebraic W2: slower than stock
- mmap/SIGUSR1/SIGUSR2 same-process gate: invalid under captured graphs
- recovered live tuned-config override: not promotable
- Unified Attention: valid test, but not better than stock+queue1
- final v7 full model: functional integration, but C4 below stock and strict exact-output parity incomplete

## 9. Runtime/worktree preservation

Historical AMD research worktree, keep intact unless explicitly archiving:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Final AMD runtime project:

- project id: `hyperloom-r9700-final-v7-20260910`
- worktree: `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

That runtime worktree contains additional untracked raw final-run JSON/probe artifacts. Do not `git clean` it. Essential promotion evidence is summarized in the committed final-gate JSON.

Clean presentation worktree/branch:

- branch: `chatgpt/r9700-final-presentation-20260910`
- use this branch for presentation/review work.

## 10. Current runtime after closure

Stock ROCm10 serving is restored and verified:

- systemd service active
- active container `inneros-vllm-canary-rocm10`
- `/v1/models` HTTP 200
- Qwen3-Coder 30B AWQ loaded
- R9700 / `gfx1201` recognized
- no experimental candidate/test container left running

## 11. Fresh-chat restart checklist

1. Read this file first.
2. Read `FINAL_STATUS.md` and `docs/evidence/r9700_v7_final_gate_summary_20260911.json`.
3. Verify remote HEAD of `chatgpt/r9700-final-presentation-20260910` before editing.
4. Treat Phase 2 as closed unless the user explicitly starts a new optimization phase.
5. Do not rerun old benchmarks merely to seek a better headline.
6. Never clean/reset either AMD research/runtime worktree before inventorying untracked evidence.
7. If preparing a presentation, use only the presentation-safe claims in `FINAL_STATUS.md`.
8. If future engineering resumes, start from clean v7 and the stable stock+queue1 control, and require exact correctness plus independently reproducible C4 parity or better before promotion.
