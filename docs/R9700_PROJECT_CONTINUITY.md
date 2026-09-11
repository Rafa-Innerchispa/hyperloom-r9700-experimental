# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-11 (America/Guayaquil)

This is the canonical restart file for a fresh ChatGPT/Codex session. Read this first, then consult `FINAL_STATUS.md`, `docs/DEVELOPMENT_LEDGER.md`, `docs/COMMUNITY_FEEDBACK_LEDGER.md`, and the result documents referenced below. Do not reset or clean the AMD worktree before reconciling its untracked research artifacts.

## 1. Canonical identity

- repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- active research branch: `chatgpt/r9700-rocm10-upstream-refresh-20260910`
- ops task: `ops_9c4c37cfbb7e`
- Codex preservation child task: `ops_cab302169eca`
- correlation id: `hyperloom-r9700-rocm10-refresh-20260910`
- AMD node: `ralfiia-amd`
- AMD project id: `hyperloom-r9700-amd-live-verify`
- AMD worktree: `/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- normal stock endpoint: `http://127.0.0.1:8000/v1`
- stock service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- exact image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- observed versions: Python 3.14.7; torch `2.12.0+rocm10.0.0`; HIP `7.15.26333`; vLLM `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`; Triton `3.8.0+git4cff872c.rocm10.0.0`; amd-aiter `0.1.20.post1`.

Always verify the current remote branch SHA before editing. The branch has advanced repeatedly during the ROCm10 refresh; never use a stale SHA from an old chat as the checkout target without checking remote HEAD.

## 2. Current truth boundary

**`KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED`**

Allowed claims:

- HyperLoom's experimental packed-INT4 W1 path physically runs on the Radeon AI PRO R9700 / `gfx1201`.
- The custom small-M W1 kernel repeatedly beats the stock Triton WNA16 W1 kernel in the validated real-Qwen-weight microkernel region.
- The full Qwen 30B hybrid has booted and exercised the custom W1 path across all 48 MoE layers with deterministic correctness and stock fallback.
- A clean serving factorial now identifies `stock attention + GPU_MAX_HW_QUEUES=1` as the fastest stable tested serving baseline.

Forbidden claims:

- do not say full Qwen is `1.47682x` faster;
- do not call Phase-1 serving/concurrency gains GPU-kernel gains;
- do not claim official AMD RDNA4 support;
- do not claim this is the first R9700/RDNA4 port;
- do not count the mmap/SIGUSR1/SIGUSR2 same-process result;
- do not claim Unified Attention fixes the R9700 process-start bimodality;
- do not promote a candidate by comparing it only to a pathological slow stock spawn.

## 3. Canonical proven evidence

### Phase-1 independent serving starts

- stock: `19.893 / 19.942 / 19.987 tok/s`
- candidate: `36.004 / 36.078 / 36.194 tok/s`
- paired median gain: `+80.99%`
- evidence: `docs/evidence/r9700_independent_process_final_20260908.json`
- SHA256: `dda9128a5ea17728e3eae59488d37952665b30c77e54cb440c225971fdfcf94f`

Interpretation: serving/concurrency only.

### Live Qwen AWQ/WNA16 contract

- backend: `TritonWNA16Experts`
- activation dtype: FP16
- experts: 128
- top-k: 8
- W1 packed uint8: `[128,1536,1024]`
- W2 packed uint8: `[128,2048,384]`
- quantization: `int4_w4a16`, group size 128
- evidence: `docs/evidence/r9700_live_moe_dtype_probe_20260909T024321Z.json`

### Custom W1 vs stock Triton WNA16

Three real-Qwen-weight child-process campaigns, 21 alternating paired HIP-event rounds per M:

- M1: `1.74518x`
- M2: `1.49210x`
- M4: `1.47445x`
- M8: `1.44613x`
- M16: `1.46273x`
- median across tested small-M region: `1.47682x`
- wins: `63/63` per tested shape across campaigns
- correctness cosine: effectively 1.0
- evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`
- SHA256: `6f1faf903dc62d31b2ef62b60beeb1394f68c9f778366ebd77b38a187a782a5e`

Interpretation: W1 microkernel only.

### Full-model functional integration

The full Qwen3-Coder 30B AWQ server booted with `R9700HybridWNA16Experts` and a deterministic request observed:

- 48 `custom_small_w1_stock_w2`
- 48 `stock_full_fallback`
- candidate response hash matched stock
- temporary bootstrap hook removed and stock restored
- evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`
- SHA256: `98565adc398ecf91c34e4a6f94e8ef7847c3e36b6123b791410878e66091206d`

Interpretation: load/routing/correctness/fallback/rollback proven, not E2E acceleration.

### Historical full-model E2E result

- healthy/stable stock C4 region: roughly `159-162 tok/s`
- integrated hybrid candidate: roughly `151-153 tok/s`
- v3 alignment reuse reduced a prior roughly 6-7% regression to roughly 5%, but did not reach parity

Therefore the full hybrid was not promoted.

## 4. Recovered gfx1201 WNA16 tuner

Canonical note: `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md`

Raw evidence: `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`

SHA256: `f56942b75f6621ae22078a8173ce3f6ea01c8980d07f279d8456c3d2aaba6d50`

Winner:

- `BLOCK_SIZE_M=16`
- `BLOCK_SIZE_N=64`
- `BLOCK_SIZE_K=32`
- `GROUP_SIZE_M=1`
- `num_warps=4`
- `num_stages=2`
- `waves_per_eu=4`
- `SPLIT_K=1`

Result:

- median isolated speedup M1..16: `1.2092561838928513x`
- minimum tested speedup: `1.083842659569329x`
- verdict: `MICROBENCH KEEP`

The historical live tuned-config smokes remain `REJECT / NOT PROMOTED` because path observation and deterministic correctness/process-state controls were insufficient.

## 5. Healthy stock reference from this refresh

After restoring the exact ROCm10 canary, direct API measurement produced:

- evidence generated on AMD: `docs/evidence/r9700_active_measure_active_20260911T015752Z.json`
- `/v1/models`: HTTP 200
- C1: `68.77085799282636 tok/s`
- C4: `162.10205336178268 tok/s`
- ~6012-token prompt: `62.282650281230964 tok/s`
- deterministic correctness SHA256: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- separate stock audit: `PTH_HITS=[]`

Historical stock/no-queue evidence also includes a fast C4 start at `165.699 tok/s`, alongside slow starts around 72 tok/s. Never substitute the low mode for the credible stock ceiling.

The generic local-model manager incorrectly reported `vllm_models_unavailable` during this refresh while the direct endpoint returned HTTP 200. For this campaign, project probes/direct endpoint evidence are authoritative.

## 6. 2026-09-11 clean serving factorial — CLOSED

Canonical result: `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md`

Machine summary: `docs/evidence/r9700_factorial_clean_summary_20260911.json`

Methodological correction: the earlier candidate launcher used plain `docker stop` against a stock container managed by an auto-restarting systemd unit. Logs proved stock restart attempts could overlap candidate execution. Therefore the preliminary queue1 run and the older Unified-Attention candidate campaign are preserved as historical evidence but excluded from clean causal/promotion statistics.

The clean rerun stopped `inneros-vllm-canary-rocm10.service` through authorized host ops, verified the stock container absent and used VRAM below 5 GiB, then launched a fresh candidate process with exact identity/environment capture.

### Clean cell results

`stock attention + GPU_MAX_HW_QUEUES=1`:

- C4: `158.567959 / 158.412033 tok/s`
- median C4: `158.489996 tok/s`
- C4 range/median: `0.098%`
- median C1: `63.892816 tok/s`
- median ~6K: `58.868927 tok/s`
- dedicated correctness: PASS / PASS

`Unified Attention + default queues`:

- C4: `136.900788 / 140.629155 tok/s`
- median C4: `138.764971 tok/s`
- C4 range/median: `2.687%`
- dedicated correctness: PASS / PASS
- one long-context completion was abnormally short/different; do not use its arithmetic long-context median as a performance claim

`Unified Attention + GPU_MAX_HW_QUEUES=1`:

- C4: `157.035577 / 156.471523 tok/s`
- median C4: `156.753550 tok/s`
- C4 range/median: `0.360%`
- median ~6K: `55.999820 tok/s`
- dedicated correctness: PASS / PASS

Computed comparisons:

- stock+queue1 vs same-day healthy-fast stock 162.102: `-2.2283%`
- Unified/default vs stock+queue1: `-12.4456%`
- Unified/queue1 vs stock+queue1: `-1.0956%`
- Unified/queue1 vs healthy-fast stock: `-3.2995%`

### Factorial conclusion

`GPU_MAX_HW_QUEUES=1` is the useful stability control in the tested environment. It strongly reduces observed process-start throughput bimodality, at a roughly 2.23% cost relative to the same-day healthy-fast stock observation.

Unified Attention does not improve the stable queue1 baseline and is not selected for the final HyperLoom gate.

**Final stable serving baseline selected:**

`stock attention + GPU_MAX_HW_QUEUES=1`

Reference C4 median: `158.489996 tok/s`.

The full-model promotion comparison must still include the healthy-fast stock ceiling `162.102 tok/s` and historical `165.699 tok/s`, not only the stability-limited queue1 baseline.

## 7. AMD candidate source state — DO NOT CONFUSE v6 WITH v7

The audited AMD worktree `scripts/r9700_wna16_hybrid_patch.py` is currently:

- patch name: `r9700_autoawq_stock_layout_hybrid_v6`
- SHA256: `0cf11f9fc86e33cde9aa8e6e386e38b9aad2ba09fc643cdb75e57f31703d26ee`
- still contains `mmap`, `SIGUSR1/SIGUSR2`, `runtime_gate_stock`, and the invalid runtime-gate machinery
- also contains the useful v3 alignment-reuse change

Do **not** use this v6 file as the final full-model candidate.

AMD also contains `scripts/r9700_make_hybrid_v7_clean.py`, intended to materialize a clean candidate without the invalid runtime gates. Codex task `ops_cab302169eca` is preserving/reconciling this source and the fresh raw evidence into Git. Do not reset/bootstrap/clean the AMD worktree before that task's output is inspected.

The canonical Git `scripts/r9700_wna16_hybrid_patch.py` may still represent an older v2 proof implementation until the clean candidate is explicitly committed. Never describe v3-v7 as shipped/versioned code without checking the actual file and commit.

## 8. Invalid/rejected paths worth preserving

- AITER/FlyDSL sorting: isolated HSA memory fault
- activation group pre-sum: correct but slower
- initial FP16/BF16 mirror: dtype mismatch, superseded by measured live FP16
- custom W2 algebraic path: slower than stock/reference
- same-process mmap/SIGUSR1/SIGUSR2 switching: invalid under captured graphs; signals reached EngineCore but telemetry stayed `runtime_gate_stock`
- historical tuned-config live override: not promotable
- old Unified-Attention campaign: historical only for clean causal/promotion use because stock systemd could auto-restart after plain Docker stop
- preliminary 2026-09-11 stock+queue1 run: excluded for the same systemd contamination reason

## 9. Immediate final gate

Do this next, in order:

1. inspect Codex task `ops_cab302169eca` and preserve the exact clean v7 source plus reproducible evidence in Git;
2. audit the committed v7 source to prove the mmap/signal/runtime-gate code is gone while the proven W1 path, stock fallback, and alignment reuse remain;
3. keep the stock systemd unit stopped during candidate measurement so no second model process can auto-restart;
4. launch full Qwen3-Coder 30B AWQ with **stock attention + `GPU_MAX_HW_QUEUES=1` + clean v7** using the exact ROCm10 image;
5. capture C1, C4, ~6K, TTFT/E2E, endpoint health, deterministic correctness, candidate path evidence, runtime identity, and raw JSON;
6. if the first v7 run is competitive, repeat in a second independent process start;
7. compare candidate against both:
   - stable queue1 baseline: `158.489996 tok/s` C4 median;
   - credible fast-stock observations: `162.102053` current and `165.699` historical;
8. promote only if the clean candidate repeatedly matches or beats the credible stock baseline without correctness, stability, or rollback regressions;
9. otherwise close Phase 2 honestly as a proven W1 kernel contribution plus stable serving configuration, with full hybrid integration unpromoted;
10. remove candidate, restore exact ROCm10 stock systemd service, verify direct `/v1/models`, run deterministic health/correctness and confirm `PTH_HITS=[]`;
11. update `FINAL_STATUS.md`, README, development/community ledgers as appropriate, commit, push, verify remote SHA, report final evidence status, and release the repo lock.

Do not reopen already-closed microbenchmarks just to manufacture a better headline number.

## 10. Restart checklist for a fresh chat

1. Read this file.
2. Verify remote HEAD of `chatgpt/r9700-rocm10-upstream-refresh-20260910` before editing.
3. Read `FINAL_STATUS.md` and the tail of `docs/DEVELOPMENT_LEDGER.md`.
4. Read `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md` before quoting serving results.
5. Check parent task `ops_9c4c37cfbb7e` and child preservation task `ops_cab302169eca`.
6. Acquire/renew the repo lock before writes.
7. Inspect the AMD worktree before any bootstrap/reset/clean.
8. Never trust the generic model-manager health result alone; confirm the direct endpoint/project probe.
9. Do not run v6 as the final candidate.
10. Resume at the clean-v7 full-model gate described in section 9.
