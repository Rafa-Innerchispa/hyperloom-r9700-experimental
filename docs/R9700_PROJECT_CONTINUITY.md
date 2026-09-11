# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-10 (America/Guayaquil)

This file is the restart point for a fresh ChatGPT/Codex session. Read it together with `FINAL_STATUS.md`, `docs/DEVELOPMENT_LEDGER.md`, and `docs/COMMUNITY_FEEDBACK_LEDGER.md` before changing code or restarting the AMD runtime.

## Canonical identity

- Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Active research branch: `chatgpt/r9700-rocm10-upstream-refresh-20260910`
- Remote SHA at this checkpoint: `fbababa9c5e5ff8e8982d50225658d4e5bc6324a`
- Ops task: `ops_9c4c37cfbb7e`
- Correlation id: `hyperloom-r9700-rocm10-refresh-20260910`
- AMD node: `ralfiia-amd`
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GB
- Runtime: ROCm 10 + vLLM
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Normal stock endpoint: `http://127.0.0.1:8000/v1`
- Normal stock container: `inneros-vllm-canary-rocm10`

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

The custom RDNA4 path keeps packed INT4 and precomputes `zero_point * scale` once, removing qzero unpack/multiply from the hot W1 path. W2 remains stock WNA16.

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

Thus the full hybrid was about 5% slower E2E and was not promoted. A later v3 routing/alignment reuse experiment reduced an earlier 6-7% regression to roughly 5%, still insufficient. v3-v6 were not cleanly preserved in the canonical Git branch and must not be represented as shipped code.

## ROCm 10 refresh findings

A three-start process-isolated campaign validated the **combined** setting:

- `VLLM_ROCM_USE_AITER=1`
- `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`
- `GPU_MAX_HW_QUEUES=1`

C4 throughput:

- start 1: `156.7611 tok/s`
- start 2: `157.4829 tok/s`
- start 3: `158.2656 tok/s`
- median: `157.4829 tok/s`
- IQR: `0.7522 tok/s` (~0.478%)
- deterministic response hash matched stock

Important: this campaign changed two causal knobs at once. It validates the combination, but does **not** prove that Unified Attention alone fixes the bimodality. The next gate must isolate the knobs.

## Recovered bounded WNA16 tuner

The raw tuner payload previously thought lost was recovered on the AMD project worktree and its SHA256 matched the prior inventory:

`87403d22bcf735feee14d99079687552c0527225910fae8b8495140cfa201893`

Winner:

`BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, num_warps=8, num_stages=1, waves_per_eu=1`

Objective candidate/stock ratio: `1.2082048235204528`.

This tuner result is kernel/config evidence only. Its old live C3/C4 runs are contextual and are not full-model promotion evidence because process isolation/capacity equivalence was not sufficient.

## AMD-only artifacts that must be preserved before any reset/bootstrap

As of this checkpoint, the AMD project worktree contains fresh untracked research helpers/artifacts not present in the canonical Git tree. **Do not reset/bootstrap/clean that AMD worktree until they are reconciled.** Important names observed include:

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

A fresh session should use the AMD Peer Ops/project registry to inspect these files before altering the AMD worktree.

## Invalid / rejected paths to avoid repeating

- AITER/FlyDSL sorting: isolated HSA memory fault.
- Activation group pre-sum: correct but slower.
- Initial FP16/BF16 mirror: dtype mismatch; superseded by actual FP16 measurement.
- Custom W2 algebraic path: slower; reject.
- Same-process mmap/SIGUSR1/SIGUSR2 runtime switching: signals reached EngineCore but captured execution stayed stock; invalid A/B.
- Any benchmark that compares a healthy candidate against an anomalously slow stock spawn without independent-process replication: reject.

## Immediate continuation gate

Run a process-isolated 2x2 factorial with identical model, benchmark shape, context, request count, launch arguments, and clean restarts:

1. stock attention + default queue setting
2. stock attention + `GPU_MAX_HW_QUEUES=1`
3. Unified Attention + default queue setting
4. Unified Attention + `GPU_MAX_HW_QUEUES=1`

Prefer multiple independent starts per cell. Capture throughput, TTFT, TPOT, E2E, response hash, launch env, GPU/runtime identity, and raw JSON evidence.

Then select the fastest **stable and correct** baseline. Only after that:

1. persist/inspect the clean v7 candidate source;
2. boot a separate full-model process with v7 over the winning serving baseline;
3. run C1, stable C4, and long-context (~6K) probes with correctness;
4. compare against the healthy stock distribution, not its pathological low mode;
5. restore the stock server and verify health;
6. update `FINAL_STATUS.md` and the ledgers;
7. commit, push, and verify the remote SHA.

Promotion rule: change the full-model verdict only if the clean candidate repeatedly matches or beats the healthy stock baseline without correctness, stability, or rollback regressions.

## Restart checklist for another chat

1. Read this file.
2. Read `FINAL_STATUS.md`.
3. Read the tail of `docs/DEVELOPMENT_LEDGER.md` and `docs/COMMUNITY_FEEDBACK_LEDGER.md`.
4. Verify branch/SHA before editing.
5. Check task `ops_9c4c37cfbb7e` and acquire/renew the repo lock.
6. Inspect the AMD project worktree for the untracked artifacts listed above **before** any bootstrap/reset.
7. Check `local_model_runtime_status(node="amd", backend="vllm")` and preserve/restore the stock endpoint.
8. Continue from the 2x2 gate. Do not rerun already-closed microbenchmarks simply to manufacture a nicer number.
