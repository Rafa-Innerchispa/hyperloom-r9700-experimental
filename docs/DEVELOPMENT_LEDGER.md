

## Precomputed correction kernel — PROVEN in paired BF16 routed W1

Evidence: `docs/evidence/r9700_wna16_precomputed_correction_tuning_20260908T221006Z.json`

Evidence SHA-256: `131fc642f7ef56f2d2073688c688b86a4f593161f57189218d897632d2a1616d`

The W1 candidate precomputes `correction = zero_point × scale` at weight-conversion/load time and keeps packed INT4 expert weights. This removes qzero unpack and the `zero_point × scale` multiply from the request hot path.

A fixed-config stability campaign then ran 21 alternating paired HIP-event measurements per M with no retuning between rounds:

- Evidence: `docs/evidence/r9700_wna16_correction_stability_20260908T221137Z.json`
- SHA-256: `11520e3839e44a7329be3697e8b97a13ed800aae18e52cbe3e2a21d9cb2419d6`
- Fixed config: `BM=16`, `BN=128`, `BK=32`, `num_warps=4`, `num_stages=1`, `waves_per_eu=4`.
- M1: `1.10727x` paired median speedup, `17/21` wins, numeric PASS.
- M2: `1.10040x`, `15/21` wins.
- M4: `1.13268x`, `21/21` wins.
- M8: `1.12323x`, `21/21` wins.
- M16: `1.09991x`, `20/21` wins.
- Cosine remained about `0.999995`; max absolute error `0.0078125`.

This is the first stable small-M W1 result that clears BF16 pre-dequantized reference by roughly ten percent under the paired synthetic routed harness. It does **not** yet prove an end-to-end serving gain.

## Hybrid vLLM Experts contract — PROVEN synthetic smoke

Evidence: `docs/evidence/r9700_wna16_hybrid_contract_smoke_20260908T222744Z.json`

Evidence SHA-256: `d98b119203cc816388619799de779013df0e8298526d10c5c1cd2e325a0e078a`

`R9700HybridWNA16Experts` and the AutoAWQ patch module successfully instantiated through the real vLLM modular MoE contract. The smoke test exercised:

- M1 via `custom_small_w1` packed/correction path.
- M20 via `generic_wna16_w1_fallback`.
- Real `FusedMoEKernel` construction using the monkeypatched Experts class.
- Finite outputs for both paths.
- Correction tensor shape `[8,1536,16]`, dtype FP16.

Truth boundary: this is a class/contract smoke inside the real ROCm/vLLM container. The production server was not patched or restarted and no serving throughput claim is made from this test.

## Float16 runtime-mirroring probe — FAIL preserved; dtype boundary unresolved

Evidence: `docs/evidence/r9700_wna16_correction_fp16_runtime_20260908T222146Z.json`

Evidence SHA-256: `b12622d0e4be44f067511b76222fd0593f52ab13ab8e633fd44e3de818888c1a`

A direct synthetic probe attempted to mirror the server launch flag `--dtype float16` by invoking the BF16 emulation reference with FP16 activations. Triton rejected `tl.dot(fp16, bf16)` with a compile-time dtype mismatch. This probe did not patch or restart the service and did not touch model weights.

Important interpretation: the failure does **not** invalidate the custom kernel or the hybrid contract smoke. It proves the direct benchmark did not yet reproduce the exact activation dtype/layout path used by the healthy live server. `Int4EmulationTritonExperts` keeps dequantized BF16 weights and `moe_kernel_quantize_input(..., quant_dtype=None)` returns activations unchanged, so the actual dtype entering the live MoE path must be measured in an isolated model process before enabling the hybrid backend.

## Immediate integration gate

Do not patch the resident serving process yet. The next gate is an isolated vLLM model process with instrumentation that records the real MoE `hidden_states.dtype` and verifies the hybrid backend on actual loaded Qwen3-Coder AWQ weights. Only after that passes should the independent process campaign be repeated for end-to-end throughput/TTFT.


## Real Qwen3-Coder AWQ W1 weights — PROVEN on physical R9700

Evidence: `docs/evidence/r9700_wna16_real_weight_layer_probe_20260909T021040Z.json`

Evidence SHA-256: `5f0f0a92e07a75677d6fcf3028e7f71fe24a370b854de53f0a893aa42059616b`

The precomputed-correction WNA16 kernel was revalidated using actual layer-0 Qwen3-Coder-30B-A3B-Instruct-AWQ checkpoint weights rather than randomly generated AWQ tensors. Eight real experts were loaded from `model-00001-of-00006.safetensors` while the resident vLLM service remained untouched.

Observed real-weight source shapes:

- combined W1 packed qweight: `[8, 2048, 192]`
- qzeros: `[8, 16, 192]`
- scales: `[8, 16, 1536]`, FP16
- AWQ group size: 128

Paired real-weight results versus the BF16 pre-dequantized routed W1 reference:

- M1: `1.10176x`, 14/21 wins, cosine `0.99999738`, max abs `0.001953125`.
- M2: `1.09598x`, 15/21 wins, cosine `0.99999744`, max abs `0.001953125`.
- M4: `1.12649x`, 20/21 wins, cosine `0.99999750`, max abs `0.001953125`.
- M8: `1.12302x`, 21/21 wins, cosine `0.99999750`, max abs `0.001953125`.
- M16: `1.12451x`, 21/21 wins, cosine `0.99999738`, max abs `0.00390625`.
- Median of paired medians across M1..16: `1.12302x`.
- Numeric gate: PASS for every tested M.

GPU probe allocation was bounded to about 789 MiB and left roughly 4.76 GB free during the measurement, avoiding a second full 30B model load.

Truth boundary: this closes the real-W1-weight microkernel gate, not the full-model serving gate. The benchmark uses actual checkpoint expert weights with synthetic routed activations. The resident vLLM process was not patched or restarted. Actual live MoE activation dtype/layout and end-to-end hybrid serving remain open.

## Revised immediate integration gate

The next safest step is to validate the hybrid Experts path against real checkpoint W1/W2 layouts in a bounded single-layer harness, then instrument the real vLLM MoE activation dtype/layout in a reversible single-model campaign before any production-like backend switch. Do not claim an end-to-end kernel serving gain until independent process A/B evidence clears that gate.


## Hybrid W1/W2 with real Qwen3-Coder AWQ weights — PROVEN bounded smoke

Evidence: `docs/evidence/r9700_wna16_hybrid_real_weight_smoke_20260909T022529Z.json`

Evidence SHA-256: `4e99a5778063db4f6f442afaf8bbb6ab33856322cd0ba8178b8bdf09b51b5b0e`

A bounded layer-0 harness loaded eight real Qwen3-Coder AWQ experts and exercised the full experimental hybrid path: packed/correction W1 -> SILU activation -> BF16 W2 fallback -> router-weighted MoE sum. Outputs were compared against a dequantized FP32 reference. The resident vLLM service stayed live and was not patched or restarted.

All tested cases passed for both activation dtypes:

- BF16 M1/M8/M16 custom-small-W1: cosine `0.9999958..0.9999961`, relative L2 about `0.00281..0.00287`.
- BF16 M20 generic packed-W1 fallback: cosine `0.9999963`, relative L2 `0.00272`.
- FP16 M1/M8/M16 custom-small-W1: cosine `0.99999845..0.99999851`, relative L2 about `0.00178..0.00182`.
- FP16 M20 generic packed-W1 fallback: cosine `0.99999839`, relative L2 `0.00178`.
- Every output was finite and every numeric gate passed.

This resolves the earlier synthetic `fp16 x bf16` concern for the hybrid design itself: W1 handles FP16/BF16, while W2 is intentionally cast to the validated BF16 fallback boundary. It does not yet prove the dtype emitted by the actual live Qwen MoE call site, nor an end-to-end serving speedup.

## Next gate after real-weight hybrid PASS

Instrument the actual model call path in a reversible single-model campaign to record live MoE `hidden_states.dtype`, shapes and routing sizes. Then run baseline vs hybrid serving under identical process-spawn controls and capture throughput, TTFT, E2E latency, GPU clock/runtime state and correctness evidence. Only then promote the backend beyond experimental status.


## Live Qwen MoE runtime path — PROVEN reversible instrumentation

Evidence: `docs/evidence/r9700_live_moe_dtype_probe_20260909T024321Z.json`

Evidence SHA-256: `1e781ba7f706885f2d96acf900ce5e36ed176418d999ff351f8865ccfa5f322e`

A reversible instrumentation campaign measured the actual MoE path used by the resident Qwen3-Coder-30B-A3B-Instruct-AWQ vLLM process. The probe temporarily instrumented the Triton Experts call site, issued one real OpenAI-compatible chat request, then restored the exact source bytes and restarted the original service.

Observed across 48 MoE calls:

- Experts implementation: `TritonWNA16Experts`.
- activation dtype: `torch.float16`.
- quant config: `int4_w4a16`.
- W1 dtype/shape: `torch.uint8`, `[128,1536,1024]`.
- W2 dtype/shape: `torch.uint8`, `[128,2048,384]`.
- top-k: 8, 128 global experts, SILU.
- real request: HTTP 200, 19 prompt + 21 completion tokens.
- exact runtime source restore SHA: `8d9c02d01440d09cba9b709aa7af18f47262acc7d159db483cf7f905505f6bee`, verified after restore.
- restored model endpoint health: PASS.

This supersedes the stale assumption that the current ROCm 10 / vLLM build is using BF16 Int4 emulation for the live Qwen MoE path. The current resident runtime is already using packed Triton WNA16 for both W1 and W2.

## Architecture consequence

The experimental backend should no longer dequantize W2 to BF16 by default. The preferred design is now stock-layout preserving: keep vLLM's official AutoAWQ conversion and packed W2 WNA16 path unchanged, replace only the small-M W1 execution when the custom correction kernel proves a repeatable advantage over the current stock WNA16 W1 kernel, and use stock WNA16 for all other cases.

The next performance gate is therefore custom W1 versus the current stock packed WNA16 W1 using actual checkpoint weights, FP16 activations and paired measurements. BF16 pre-dequantized W1 remains a useful correctness/reference baseline but is no longer the promotion baseline.


## Custom W1 vs current stock Triton WNA16 — PROMOTION GATE PASS

Per-run evidence:

- `docs/evidence/r9700_wna16_real_weight_vs_stock_20260909T025547Z.json` — SHA `362a03abb2588ca6e9b179ca7ed52321d7594aef62b0c41456d8847c0e5ddd55`
- `docs/evidence/r9700_wna16_real_weight_vs_stock_20260909T025706Z.json` — SHA `b3ff493cfe600d1a5b1d897168d1a983eaaf79e6c77c57726af0c54d66c4e23d`
- `docs/evidence/r9700_wna16_real_weight_vs_stock_20260909T025733Z.json` — SHA `84f5e926e883bb089fbf5542a339dd5cef0ab436068f49cd88826bacdfaf2f9e`

Aggregate evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`

Aggregate SHA-256: `6f1faf903dc62d31b2ef62b60beeb1394f68c9f778366ebd77b38a187a782a5e`

The comparison uses actual Qwen3-Coder layer-0 AutoAWQ W1 checkpoint weights, FP16 activations matching the measured live MoE dtype, identical N-first packed INT4/scales/zero-point layout, identical routing, and the current vLLM Triton WNA16 W1 kernel as the promotion baseline.

Across three consecutive child-process campaigns, each with 21 alternating paired HIP-event rounds per M:

- M1 median across campaigns: `1.74518x` vs stock; minimum campaign `1.73523x`; wins `63/63`.
- M2: `1.49210x`; wins `63/63`.
- M4: `1.47445x`; wins `63/63`.
- M8: `1.44613x`; wins `63/63`.
- M16: `1.46273x`; wins `63/63`.
- Median campaign-level speedup across the tested small-M region: `1.47682x`.
- All custom and stock outputs passed the numerical reference gates; custom-vs-stock cosine remained effectively 1.0.

This is materially stronger than the earlier BF16-reference comparison because the baseline is now the actual stock packed Triton WNA16 kernel used by the current runtime.

Truth boundary: these are repeated real-weight W1 microkernel campaigns, not independent full vLLM process starts and not end-to-end serving proof. The resident server was not patched or restarted for these measurements.

## Promotion architecture after stock gate

Proceed with a stock-layout-preserving backend: custom correction kernel only for W1 when `num_tokens <= 16`; stock Triton WNA16 W1 for larger M; stock packed Triton WNA16 W2 for all cases. Preserve vLLM's official AutoAWQ weight conversion, scales, qzeros, activation, routing and fallback behavior. End-to-end promotion still requires real model-server A/B evidence.


## Stock-layout hybrid real-weight smoke — PASS

Evidence: `docs/evidence/r9700_wna16_stock_layout_real_weight_smoke_20260909T030714Z.json`

Evidence SHA-256: `7ec25d34cdec2ef94888037a070cd789cd9e43774d20e5657e23b83c56697003`

The v2 hybrid preserves packed WNA16 W1/W2 layouts. On actual layer-0 Qwen3-Coder AWQ weights and FP16 routed activations:

- M1: `custom_small_w1_stock_w2`, cosine `0.99999917`, relative L2 `0.001259`.
- M8: `custom_small_w1_stock_w2`, cosine `0.99999905`, relative L2 `0.001242`.
- M20: `stock_full_fallback`, cosine `0.99999928`, relative L2 `0.001203`.
- W1 remains packed uint8 `[8,1536,1024]` with real scales/qzeros.
- W2 remains packed uint8 `[8,2048,384]` with real scales/qzeros.

This validates the intended architecture: only the proven small-M W1 kernel is substituted; stock WNA16 remains responsible for W2 and all unproven shapes.

## Full-model candidate boot — PASS with automatic stock restore

Evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`

Evidence SHA-256: `98565adc398ecf91c34e4a6f94e8ef7847c3e36b6123b791410878e66091206d`

The actual 30B Qwen vLLM server was restarted once with an ephemeral Python `.pth` bootstrap that installs the process-local v2 patch before model loading. No vLLM source file was overwritten.

Results:

- candidate model load health: PASS, ready in ~76.0 s;
- deterministic candidate response hash matched the stock-before response hash;
- path evidence: `48` observations of `stock_full_fallback` and `48` observations of `custom_small_w1_stock_w2`;
- the custom path was therefore exercised by all 48 MoE layers during the real request/decode path;
- bootstrap hook was externally neutralized before restore;
- stock server restore health: PASS, ready in ~74.0 s;
- temporary `.pth` removed: PASS.

This closes the full-model load/correctness smoke gate. It does not yet prove end-to-end performance improvement. The next gate is an identical-workload stock-vs-hybrid campaign across multiple independent process starts with TTFT, output throughput, E2E latency and GPU telemetry.


## 2026-09-10 — Final Phase 2 reconciliation

Final experimental verdict: **KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED**.

The committed evidence through the full-model candidate smoke remains valid:

- live backend/dtype discovery: `TritonWNA16Experts`, FP16, 128 experts, top-k 8;
- three real-weight custom-W1-vs-stock campaigns: custom small-M W1 won all 63 paired measurements per tested shape, with median speedup across M1..16 about `1.47682x`;
- full Qwen3-Coder 30B AWQ candidate boot: PASS;
- all 48 MoE layers observed `custom_small_w1_stock_w2` and stock fallback;
- deterministic candidate response hash matched stock;
- temporary bootstrap hook was removed and stock restored.

Post-checkpoint E2E work established a different result at the full serving level. Concurrency-1 measurements are strongly affected by the Radeon AI PRO R9700 bimodal process/start state and are not used as naive speedup evidence. Concurrency 4 was more stable: stock was observed around `159-162 tok/s`; the integrated hybrid candidate around `151-153 tok/s`. Reusing routing/alignment in the v3 integration reduced the earlier roughly 6-7% deficit to roughly 5%, but did not reach parity.

The mmap/SIGUSR1/SIGUSR2 same-process experiment is excluded from performance evidence. Although signals reached EngineCore, path telemetry remained `runtime_gate_stock`, so the custom path did not execute during the apparent `+0.79%` comparison. Graph capture/replay is the working explanation; no same-process speedup claim is allowed.

Later bounded-tuner/tuned-config files were referenced in the uncommitted experimental work but were never committed and are no longer present in the audited worktrees. Their metrics are not reconstructed. This means gfx1201-specific tuning remains an investigated lead, not a closed E2E promotion proof.

Preserved failed/partial results remain part of the engineering record: AITER/FlyDSL sorting HSA fault; activation group pre-sum correct but slower; early FP16/BF16 mismatch superseded by measured FP16 live dtype; custom W2 slower than stock/reference; invalid runtime-gated same-process A/B.

The final truth boundary is therefore explicit: the custom packed-INT4 W1 Triton kernel is worth keeping and is physically proven on `gfx1201`; the hybrid full-model integration is functionally proven but is not the default serving backend because the valid stable E2E comparison did not beat stock.


## 2026-09-10 — ROCm 10 RDNA4 Unified Attention refresh — THREE-START GATE PASS

Evidence: `docs/evidence/r9700_unified_attention_three_start_aggregate_20260910.json`

Detailed result: `docs/R9700_UNIFIED_ATTENTION_REFRESH_RESULT_20260910.md`

This campaign re-evaluated Vector.sys' R9700 bimodality warning against the newer vLLM RDNA4 AITER work without replacing ROCm 10. The resident image remains ROCm 10 / HIP 7.15.26333 / PyTorch 2.12.0+rocm10.0.0 / vLLM 0.27.1.dev5 / amd-aiter 0.1.20.post1. A minimal process-local overlay ports only the RDNA4 attention backend selection needed to exercise `ROCM_AITER_UNIFIED_ATTN`; RMSNorm remains native and MoE remains stock `TritonWNA16Experts`.

Three independent stock starts reproduced the reported process-start bimodality:

- C4 aggregate: `71.940 / 71.776 / 165.699 tok/s`.
- C1 decode: `22.322 / 22.289 / 69.343 tok/s`.
- ~6012-token prompt decode: `21.944 / 21.910 / 62.553 tok/s`.
- Stock C4 range/median: `130.56%`.

Three independent Unified Attention starts were tightly clustered:

- C4 aggregate: `158.017 / 157.296 / 157.469 tok/s`.
- C1 decode: `63.095 / 62.994 / 62.030 tok/s`.
- ~6012-token prompt decode: `56.288 / 56.264 / 56.088 tok/s`.
- Candidate C4 median: `157.469 tok/s`.
- Candidate C4 range/median: `0.46%`.
- Candidate C1 range/median: `1.69%`.
- Candidate long-context range/median: `0.36%`.

All candidate and stock deterministic correctness checks matched SHA-256 `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`.

Runtime log proof confirms the candidate selected `ROCM_AITER_UNIFIED_ATTN` and retained stock `TritonWNA16Experts` for the Qwen AWQ MoE path.

Interpretation boundary: candidate median C4 is about `+118.89%` versus the three-start stock median because two stock starts landed in the low-performance regime. However, the best stock start reached `165.699 tok/s`, about `4.97%` above candidate median. Therefore this is **not** claimed as a peak-throughput win over best stock. The valid result is that Unified Attention produced three consistently fast starts and is **KEEP_FOR_NEXT_GATE** as the new stable baseline candidate.

Next gate: investigate the remaining stock spawn-state bimodality and then evaluate HyperLoom's proven packed-INT4 small-M W1 substitution on top of the stable/fast RDNA4 attention baseline. Promotion requires the combined path to match or beat the best credible baseline without sacrificing correctness or reproducibility.


## 2026-09-10 — Recovered gfx1201 WNA16 tuner reconciliation

Recovered source evidence:

- `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json` — SHA-256 `f56942b75f6621ae22078a8173ce3f6ea01c8980d07f279d8456c3d2aaba6d50`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json` — SHA-256 `5d1891dbd8c9bf4a414542c68ee3e702f0c18e2c900eb5cbeaf92cba988239b9`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json` — SHA-256 `54e6a79cae81e21e39c34912008f66fb35f16e8e263faae84d4ef663eb428f8f`

Detailed interpretation: `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md`.

The bounded tuner found a gfx1201-specific WNA16 winner (`BM=16, BN=64, BK=32, GROUP_M=1, warps=4, stages=2, waves_per_eu=4, SPLIT_K=1`) with `1.20926x` median isolated speedup across M1..16 and minimum tested speedup `1.08384x`. This candidate is retained as **MICROBENCH KEEP**.

The live override is **REJECT / NOT PROMOTED**. In the first smoke `tuned_seen=false`; in the second smoke the tuned config was observed for M1/2/4/8/16 but the deterministic output hash did not match the pre-candidate baseline and the stock process changed from slow to fast regime across restore. No serving-level gain is claimed from these smokes.

Future use must integrate the configuration through a correctness-safe path and benchmark it against the stable/fast RDNA4 attention baseline established by the 2026-09-10 three-start Unified Attention campaign.


### Correction: Unified three-start candidate also set `GPU_MAX_HW_QUEUES=1`

A later source audit of the exact launcher found that the three `157-158 tok/s` candidate runs combined two factors: `ROCM_AITER_UNIFIED_ATTN` and `GPU_MAX_HW_QUEUES=1`. The measurements and hashes remain valid, but the earlier attribution to Unified Attention alone was too strong. The aggregate evidence schema has been revised to `hyperloom.r9700.unified_attention_plus_queue1_three_start.v2`, and the verdict is now **KEEP_FOR_FACTORIAL_GATE** pending separation of Unified-without-queue1 and stock-with-queue1.

No public or hackathon claim should state that Unified Attention alone eliminates R9700 process-start bimodality until that factorial gate is complete.

## 2026-09-11 — Clean serving factorial and systemd isolation correction

Canonical human result: `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md`

Machine summary: `docs/evidence/r9700_factorial_clean_summary_20260911.json`

A methodology audit found that the earlier candidate harness used plain `docker stop` against `inneros-vllm-canary-rocm10`, while that container is managed by an auto-restarting user systemd unit. Service logs proved restart attempts could overlap candidate execution. The preliminary 2026-09-11 stock+queue1 run and the older Unified-Attention candidate campaign are therefore preserved but excluded from clean causal/promotion statistics.

The factorial was rerun with `inneros-vllm-canary-rocm10.service` stopped through authorized host ops before each candidate sequence, with the stock container absent and VRAM below 5 GiB before fresh-process launch. All valid cells used the exact ROCm10 image/model/args and dedicated deterministic correctness probe.

Clean results, two independent process starts per cell:

- stock attention + `GPU_MAX_HW_QUEUES=1`: C4 `158.567959 / 158.412033 tok/s`; median `158.489996`; range/median `0.098%`; median long `58.868927`; correctness PASS/PASS.
- Unified Attention + default queues: C4 `136.900788 / 140.629155`; median `138.764971`; range/median `2.687%`; correctness PASS/PASS. One long-context completion was abnormally short/different, so no long-context performance claim is made from this cell.
- Unified Attention + `GPU_MAX_HW_QUEUES=1`: C4 `157.035577 / 156.471523`; median `156.753550`; range/median `0.360%`; median long `55.999820`; correctness PASS/PASS.

Computed C4 deltas:

- stock+queue1 vs same-day healthy-fast stock `162.102053`: `-2.2283%`;
- Unified/default vs stock+queue1: `-12.4456%`;
- Unified/queue1 vs stock+queue1: `-1.0956%`;
- Unified/queue1 vs healthy-fast stock: `-3.2995%`.

Conclusion: `GPU_MAX_HW_QUEUES=1` is the useful stability control in the tested R9700/ROCm10/vLLM environment. It strongly reduces observed process-start bimodality while sacrificing roughly 2.23% versus the same-day healthy-fast stock observation. Unified Attention does not improve the stable queue1 baseline and is not selected for the final HyperLoom gate.

Selected stable baseline for the final candidate: **stock attention + `GPU_MAX_HW_QUEUES=1`**, C4 median `158.489996 tok/s`. Promotion must still compare against the credible healthy-fast stock ceiling (`162.102053` same-day, `165.699` historical), not only the stability-limited denominator.

An audit of the AMD worktree after the factorial found `scripts/r9700_wna16_hybrid_patch.py` still at `r9700_autoawq_stock_layout_hybrid_v6`, SHA256 `0cf11f9fc86e33cde9aa8e6e386e38b9aad2ba09fc643cdb75e57f31703d26ee`. It retains the useful alignment-reuse change but also the invalid mmap/SIGUSR1/SIGUSR2 runtime-gate machinery. It must not be used as the final candidate. The next gate is to preserve/audit `v7_clean`, then run the full Qwen 30B candidate over stock attention + queue1 in clean process isolation.

## 2026-09-11 — Clean v7 full-model final gate

- Presentation branch: `chatgpt/r9700-final-presentation-20260910`.
- Base preservation commit: `c4e4aae90f5582d76b7881ba932778bd94922611`.
- Multiprocessing-safe explicit v7 entrypoint committed at `97225ac5e7a314091f2231860d69d8e0ef63c8e6`.
- Candidate patch: `scripts/r9700_wna16_hybrid_patch_v7_clean.py`, SHA-256 `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`.
- First explicit-entrypoint attempt failed because Python `spawn` re-entered the `runpy` API-server launch path. Classified as harness/bootstrap failure, preserved in `docs/evidence/r9700_v7_full_model_bootstrap_failure_20260911T0511Z.json`, then fixed without changing the v7 kernel source.
- Corrected candidate launched with stock systemd stopped, stock container absent, ~60 MB VRAM used before launch, stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, and read-only patch/entrypoint mounts.
- Full Qwen3-Coder 30B AWQ loaded and vLLM logged `Using R9700HybridWNA16Experts`.
- Real inference path evidence: `48` x `custom_small_w1_stock_w2`, `48` x `stock_full_fallback`.
- Candidate C4: `149.891101` and `150.842143 tok/s`; median `150.366622 tok/s`.
- Fresh fair stock+queue1 control: `157.489588 tok/s`; prior clean factorial median `158.489996 tok/s`.
- Candidate therefore remained about 4.5-5.1% below the selected stable stock baseline.
- All four C4 response hashes matched stock exactly. Dedicated strict correctness hash did not: v7 `e4810fc5...f5f9` vs stock `7931ecfb...5ce2`. Universal exact-output parity is not claimed.
- C1 remains excluded from promotion because R9700 process-start throughput is bimodal.
- Candidate and factorial test containers were removed. `inneros-vllm-canary-rocm10.service` was restored and direct `/v1/models` returned HTTP 200.
- Final verdict: **KERNEL KEEP / FULL-MODEL INTEGRATION NOT PROMOTED**.
- Canonical summary: `docs/evidence/r9700_v7_final_gate_summary_20260911.json`.
- Recovered tuner artifacts are present in Git; the previous evidence-gap statement is obsolete. Bounded tuner remains `MICROBENCH KEEP`; live override remains `REJECT / NOT PROMOTED`.


## 2026-09-11 — Phase 3 RDNA INT4 MoE repack — C4 FULL-MODEL KEEP

A fresh upstream review after the clean v7 rejection identified vLLM PR #43389, performance commit `f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3`, as a directly relevant ROCm RDNA INT4/W4A16 MoE optimization for the same Qwen3-30B-A3B AWQ family and gfx1201 class.

The five-file runtime patch applies cleanly to the deployed ROCm10 vLLM source commit `f46a9dfe2c5f57bebbd29556cbbb25eabd874226`. Runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`. The installed stable source was not modified; the patch is built into an isolated read-only overlay.

Pre-serving gates passed:

- patched modules compile;
- synthetic weight repack and zero-point conversion are bit-exact;
- actual Qwen layer-0 gate/down AWQ weights both repack exactly;
- real qzeros/scales match exactly;
- sampled dequantization delta is `0.0` for gate and down projections.

Three independent full Qwen3-Coder 30B candidate starts, with stock systemd stopped and clean VRAM before launch, produced canonical C4 results `191.450567 / 189.645424 / 191.426697 tok/s`. Median: `191.426697 tok/s`; range/median: `0.943%`.

This is `+20.7816%` versus the clean stable stock+queue1 median `158.489996 tok/s`, `+18.0902%` versus same-day healthy-fast stock `162.102053`, and `+15.5268%` versus the historical fast-stock observation `165.699`.

Comparable C4 response hashes matched stock. One cold/lazy strict-correctness request in start 2 temporarily differed and had abnormally high TTFT; an immediate hot repeat returned the canonical stock hash. One later same-process post-start-3 measurement fell to `145.119938 tok/s` and changed one C4 output hash. Temperature (~57 C), stock-service competition, and an obvious competing VRAM consumer were not observed. That degraded observation is preserved but excluded from the independent-start aggregate.

Small-M/long-context remain open: C1 median ~`60.367 tok/s`; ~6K-context median ~`55.199 tok/s`, both below the clean stock+queue1 baselines. Current verdict: **C4 FULL-MODEL KEEP / SMALL-M + LONG-CONTEXT OPTIMIZATION OPEN / PRODUCTION PROMOTION PENDING STABILITY GATE**.

The running vLLM still reports no tuned MoE config for `E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`. Current upstream code search has no R9700 INT4 W4A16 config match, so generating/tuning this configuration is the next HyperLoom target rather than duplicate upstream work.

Canonical Phase 3 summary: `docs/R9700_PHASE3_RESULT_20260911.md` and `docs/evidence/r9700_phase3_three_start_aggregate_20260911.json`.
