

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
