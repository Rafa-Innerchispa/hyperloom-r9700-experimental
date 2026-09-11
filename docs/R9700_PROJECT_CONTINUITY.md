# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-10 (America/Guayaquil)

This file is the restart point for a fresh ChatGPT/Codex session. Read it together with `FINAL_STATUS.md`, `docs/DEVELOPMENT_LEDGER.md`, `docs/COMMUNITY_FEEDBACK_LEDGER.md`, `docs/R9700_UNIFIED_ATTENTION_REFRESH_RESULT_20260910.md`, and `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md` before changing code or restarting the AMD runtime.

## Canonical identity

- Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Active research branch: `chatgpt/r9700-rocm10-upstream-refresh-20260910`
- Continuity checkpoint parent pushed at 2026-09-11 01:51 UTC: `939f982be8f276e7c703e641ca7aa1ccabbd6061`. Always verify current remote HEAD before editing because later evidence commits may advance it.
- Ops task: `ops_9c4c37cfbb7e`
- Correlation id: `hyperloom-r9700-rocm10-refresh-20260910`
- AMD node: `ralfiia-amd`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GB
- Runtime: ROCm 10 + vLLM
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Normal stock endpoint: `http://127.0.0.1:8000/v1`
- Normal stock container/service: `inneros-vllm-canary-rocm10`
- Exact ROCm 10 image observed: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Observed runtime versions: Python 3.14.7; torch `2.12.0+rocm10.0.0`; HIP `7.15.26333`; vLLM `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`; Triton `3.8.0+git4cff872c.rocm10.0.0`; amd-aiter `0.1.20.post1`.

## Truth boundary

Current technical verdict:

`KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED`

Do not weaken this gate for a better-looking benchmark claim. In particular:

- `1.47682x` is a routed W1 microkernel median against stock Triton WNA16 in the validated small-M region. It is **not** a full-Qwen speedup.
- Phase-1 serving/concurrency gains are separate from GPU-kernel gains.
- The process-local mmap/signal runtime gate was invalid under captured graphs and must not be counted.
- A slow/bimodal stock spawn is not a legitimate denominator for promoting the hybrid backend.
- Public status remains experimental; no claim of official AMD support and no claim of being the first RDNA4 port.

## What is proven

### Serving / process isolation

The original serving gate was rerun with independent process starts:

- stock: `19.893 / 19.942 / 19.987 tok/s`
- candidate: `36.004 / 36.078 / 36.194 tok/s`
- paired median gain: `+80.99%`
- evidence: `docs/evidence/r9700_independent_process_final_20260908.json`
- SHA256: `dda9128a5ea17728e3eae59488d37952665b30c77e54cb440c225971fdfcf94f`

This is serving/concurrency evidence only.

### Live Qwen WNA16 contract

Observed live backend contract on R9700:

- backend: `TritonWNA16Experts`
- activation: FP16
- experts: 128
- top-k: 8
- W1 packed uint8: `[128,1536,1024]`
- W2 packed uint8: `[128,2048,384]`
- quantization: `int4_w4a16`, group size 128

### Custom W1 kernel

The custom RDNA4 path keeps packed INT4 and precomputes `correction = zero_point * scale` once, removing qzero unpack/multiply from the hot W1 path. W2 remains stock WNA16.

Real-Qwen-weight stock-vs-custom campaign, three child-process campaigns, 21 alternating HIP-event rounds per M:

- M1: `1.74518x`
- M2: `1.49210x`
- M4: `1.47445x`
- M8: `1.44613x`
- M16: `1.46273x`
- median: `1.47682x`
- wins: `63/63` per tested shape aggregated across campaigns
- correctness cosine: effectively 1.0
- aggregate evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`
- SHA256: `6f1faf903dc62d31b2ef62b60beeb1394f68c9f778366ebd77b38a187a782a5e`

### Full-model functional integration

`R9700HybridWNA16Experts` booted the full Qwen 30B AWQ model. A deterministic request observed all 48 MoE layers and matched the stock response hash:

- 48 observations: `custom_small_w1_stock_w2`
- 48 observations: `stock_full_fallback`
- evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`
- SHA256: `98565adc398ecf91c34e4a6f94e8ef7847c3e36b6123b791410878e66091206d`

This proves loading/routing/correctness/fallback/rollback, not E2E acceleration.

## E2E promotion result so far

Stable C4 is the relevant full-model gate because C1 and some C4 stock starts have shown strong bimodality.

Historical stable region:

- stock C4: roughly `159-162 tok/s`
- hybrid C4: roughly `151-153 tok/s`

Thus the full hybrid was about 5% slower E2E and was not promoted. A later v3 routing/alignment reuse experiment reduced an earlier roughly 6-7% regression to roughly 5%, still insufficient. v3-v6 were not cleanly preserved in the canonical Git branch and must not be represented as shipped code.

## ROCm 10 refresh: Unified Attention plus single queue

The controlled three-start campaign recorded in `docs/R9700_UNIFIED_ATTENTION_REFRESH_RESULT_20260910.md` validated the **combined** setting:

- `VLLM_ROCM_USE_AITER=1`
- RDNA4 AITER Unified Attention selection
- `GPU_MAX_HW_QUEUES=1`

Canonical C4 throughput from that campaign:

- start 1: `158.017 tok/s`
- start 2: `157.296 tok/s`
- start 3: `157.469 tok/s`
- median: `157.469 tok/s`
- range/median: about `0.46%`
- deterministic response hash matched stock: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

The paired stock/no-queue starts were `71.940 / 71.776 / 165.699 tok/s` at C4 and reproduced the process-start bimodality.

Important: the candidate campaign changed two causal knobs at once. It validates the combination, but does **not** prove that Unified Attention alone fixes the bimodality. The required factorial gate must isolate Unified Attention from `GPU_MAX_HW_QUEUES=1`.

## Recovered bounded WNA16 tuner

Canonical recovery source: `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md`.

Raw source evidence: `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`.

Recovered SHA256 recorded by the canonical recovery note:

`f56942b75f6621ae22078a8173ce3f6ea01c8980d07f279d8456c3d2aaba6d50`

Winner configuration:

- `BLOCK_SIZE_M=16`
- `BLOCK_SIZE_N=64`
- `BLOCK_SIZE_K=32`
- `GROUP_SIZE_M=1`
- `num_warps=4`
- `num_stages=2`
- `waves_per_eu=4`
- `SPLIT_K=1`

Isolated-kernel result:

- median speedup over M=1,2,4,8,16: `1.2092561838928513x`
- minimum tested speedup: `1.083842659569329x`

Verdict: `MICROBENCH KEEP`. The two historical live override smokes remain `REJECT / NOT PROMOTED` because one did not observe the tuned path and the other had a deterministic hash mismatch while process state changed across restore.

## 2026-09-10/11 exact healthy-stock reference

After restoring the exact ROCm 10 canary, first startup appeared unavailable while weights/graphs compiled. Direct container logs proved this was not a hang:

- checkpoint weights: 15.66 GiB
- weight load: `123.11 s`
- `torch.compile`: `135.62 s`
- graph capture: `17 s`
- server HTTP started at `2026-09-11T01:56:59Z`
- vLLM warning: no device-specific MoE config file was found for `E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16`; stock therefore used the default MoE config.

A real API measurement immediately after readiness produced:

- evidence currently generated on AMD worktree: `docs/evidence/r9700_active_measure_active_20260911T015752Z.json`
- `/v1/models`: HTTP 200 with the expected Qwen model
- C1 decode: `68.77085799282636 tok/s`
- C4 aggregate: `162.10205336178268 tok/s`
- ~6012-token prompt decode: `62.282650281230964 tok/s`
- correctness SHA256: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

This is a healthy fast stock spawn and a valid reference point, not yet a multi-start distribution by itself.

A separate recovery audit of the current stock container reported `PTH_HITS=[]`, so no temporary HyperLoom `.pth` bootstrap hook was present in the restored stock runtime.

The generic `local_model_benchmark`/runtime manager incorrectly reported `vllm_models_unavailable` even while the direct endpoint returned HTTP 200. Treat that as an observability bug in the model manager; use the project probe/direct endpoint evidence for this campaign.

## AMD-only artifacts that must be preserved before any reset/bootstrap

As of this checkpoint, the AMD project worktree contains fresh untracked research helpers/artifacts not yet all present in the canonical Git tree. **Do not reset/bootstrap/clean that AMD worktree until they are reconciled.** Important names observed include:

- `scripts/r9700_make_hybrid_v7_clean.py`
- `scripts/r9700_active_measure.py`
- `scripts/r9700_candidate_wait.py`
- `scripts/r9700_container_recipe_probe.py`
- `scripts/r9700_current_capability_probe.py`
- `scripts/r9700_export_recovered_tuner.py`
- `scripts/r9700_force_stock_restore.py`
- `scripts/r9700_hybrid_patch_audit.py`
- `scripts/r9700_live_container_diag.py`
- `scripts/r9700_pilot_log_summary.py`
- `scripts/r9700_recovery_inventory.py`
- `scripts/r9700_refresh_audit.py`
- `scripts/r9700_restore_stable.py`
- `scripts/r9700_same_process_ab.py`
- `scripts/r9700_same_process_c4_measure.py`
- temporary helpers documenting v3-v6 work, including `_tmp_reuse_alignment_v3.py`, `_tmp_add_same_process_gate_v4.py`, `_tmp_add_signal_gate_v5.py`, and `_tmp_late_signal_install_v6.py`

The preserved canonical `scripts/r9700_wna16_hybrid_patch.py` is still the functional v2 implementation. Do not call v3-v7 'in Git' until a clean source file is actually committed.

A Codex repair task (`ops_cab302169eca`) was dispatched specifically to preserve the clean v7 source, the reproducible raw evidence, and provenance without resetting the AMD worktree.

## Invalid / rejected paths to avoid repeating

- AITER/FlyDSL sorting: isolated HSA memory fault.
- Activation group pre-sum: correct but slower.
- Initial FP16/BF16 mirror: dtype mismatch; superseded by actual FP16 measurement.
- Custom W2 algebraic path: slower; reject.
- Same-process mmap/SIGUSR1/SIGUSR2 runtime switching: signals reached EngineCore but captured execution stayed stock; invalid A/B.
- Historical tuned-config live override: not promotable because of path-observation/correctness/process-state problems.
- Any benchmark that compares a healthy candidate against an anomalously slow stock spawn without independent-process replication: reject.

## Immediate continuation gate

Run a process-isolated 2x2 factorial with identical model, benchmark shape, context, request count, launch arguments, and clean restarts:

1. stock attention + default queue setting
2. stock attention + `GPU_MAX_HW_QUEUES=1`
3. Unified Attention + default queue setting
4. Unified Attention + `GPU_MAX_HW_QUEUES=1`

The existing campaign already gives controlled observations for cells 1 and 4, but the missing cells 2 and 3 must be measured before attributing causality. Prefer multiple independent starts per missing cell. Capture throughput, TTFT, TPOT, E2E, response hash, launch env, GPU/runtime identity, and raw JSON evidence.

Then select the fastest **stable and correct** baseline. Only after that:

1. persist/inspect the clean v7 candidate source;
2. boot a separate full-model process with v7 over the winning serving baseline;
3. run C1, stable C4, and long-context (~6K) probes with correctness;
4. compare against the healthy stock distribution, not its pathological low mode;
5. restore the exact ROCm 10 stock server and verify direct API health plus absence of temporary hooks;
6. update `FINAL_STATUS.md` and the ledgers;
7. commit, push, and verify the remote SHA.

Promotion rule: change the full-model verdict only if the clean candidate repeatedly matches or beats the healthy stock baseline without correctness, stability, or rollback regressions.

## Restart checklist for another chat

1. Read this file.
2. Read `FINAL_STATUS.md`.
3. Read the tail of `docs/DEVELOPMENT_LEDGER.md` and `docs/COMMUNITY_FEEDBACK_LEDGER.md`.
4. Read both 2026-09-10 recovery/Unified Attention detailed notes before quoting their numbers.
5. Verify branch/SHA before editing.
6. Check task `ops_9c4c37cfbb7e`, Codex child task `ops_cab302169eca`, and acquire/renew the repo lock.
7. Inspect the AMD project worktree for the untracked artifacts listed above **before** any bootstrap/reset.
8. Check the exact ROCm10 canary and direct `/v1/models`; do not trust the current generic model-manager health result by itself.
9. Continue from the missing 2x2 cells and then the clean-v7 full-model gate. Do not rerun already-closed microbenchmarks simply to manufacture a nicer number.
