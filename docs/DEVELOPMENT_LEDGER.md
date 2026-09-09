

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
