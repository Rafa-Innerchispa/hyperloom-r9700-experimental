# HyperLoom Radeon AI PRO R9700 / RDNA4 — Final Experimental Status

Last reconciled: 2026-09-11 (America/Guayaquil)
Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
Presentation branch: `chatgpt/r9700-final-presentation-20260910`
Hardware: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
Workload: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
Runtime: ROCm 10 + vLLM + Triton

## Final verdict

**KERNEL KEEP / FULL-MODEL INTEGRATION NOT PROMOTED**

The project has a real experimental HyperLoom path running on the physical Radeon AI PRO R9700. The packed-INT4 small-M W1 Triton kernel is validated against stock vLLM Triton WNA16 with real Qwen3-Coder AWQ weights and measured FP16 activations. The complete 30B model also boots with `R9700HybridWNA16Experts`, reaches the custom W1 path in all 48 MoE layers, preserves stock fallback, serves requests, and restores stock cleanly after testing.

The full hybrid is intentionally **not** promoted as the default serving backend. In the final clean v7 gate, stable concurrency-4 throughput remained about 4.5-5.1% below the selected stock+queue1 baseline, and one strict deterministic single-request output probe did not match stock exactly. The kernel contribution is real; the full-model speedup claim is not.

## Final clean v7 gate — 2026-09-11

Canonical summary: `docs/evidence/r9700_v7_final_gate_summary_20260911.json`

Clean candidate:

- source: `scripts/r9700_wna16_hybrid_patch_v7_clean.py`
- patch SHA-256: `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`
- explicit serving entrypoint: `scripts/r9700_v7_server_entry.py`
- final entrypoint fix commit before the measured run: `97225ac5e7a314091f2231860d69d8e0ef63c8e6`
- exact image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- attention backend: stock `ROCM_ATTN`
- stability control: `GPU_MAX_HW_QUEUES=1`
- stock systemd unit stopped before launch
- stock container absent before launch
- VRAM used before candidate launch: about 60 MB
- patch and entrypoint mounted read-only

The first explicit-entrypoint attempt failed before serving because Python multiprocessing `spawn` re-entered the `runpy` API-server launch path. This was a harness/bootstrap bug, not a kernel failure. It is preserved at `docs/evidence/r9700_v7_full_model_bootstrap_failure_20260911T0511Z.json`. The entrypoint was corrected so the patch installs on import in child processes while the API server only starts under `if __name__ == "__main__"`.

The corrected clean launch then reached the real model path. vLLM reported:

- `Qwen3MoeForCausalLM`
- FP16 casting as expected
- `TRITON` WNA16 MoE backend
- stock `ROCM_ATTN`
- `Using R9700HybridWNA16Experts`
- all 48 MoE layers loaded

Observed full-model routing during real inference:

- `48` x `custom_small_w1_stock_w2`
- `48` x `stock_full_fallback`

Streaming TTFT on the dedicated probe: about `64.9 ms`.

### Final C4 comparison

Clean v7 candidate, two measurements from the same validated candidate process:

- `149.891101 tok/s`
- `150.842143 tok/s`
- median: `150.366622 tok/s`

Fair stock control with stock attention + `GPU_MAX_HW_QUEUES=1`, launched in a separate clean process:

- current control: `157.489588 tok/s`
- prior clean factorial median: `158.489996 tok/s`

Therefore v7 remained approximately:

- `4.5%` below the current fair stock+queue1 control using the candidate median
- `5.1%` below the prior clean stock+queue1 factorial median

This is consistent with the earlier Phase-2 full-model result: the integration penalty was reduced from the earlier 6-7% region, but parity was not reached.

### Correctness boundary from the final gate

The four C4 requests produced **exactly the same output hashes** under v7 and stock+queue1:

- `891b5901...2e68`
- `f3164647...6c5c`
- `fbe90e7b...a407`
- `fbe90e7b...a407`

However, the dedicated deterministic single-request probe differed:

- v7: `e4810fc5ef6597d28440cc7e32d384180088ca95c385c7ddcf12c1439690f5f9`
- stock+queue1: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

The final v7 candidate therefore does not have universal exact-output parity under the current strict gate. Earlier full-model smoke evidence did match stock on its deterministic request, so the integration is demonstrably functional, but the final promotion criterion remains unmet.

C1 is not used for promotion because R9700 serving showed strong process-start bimodality. Long-context measurements are retained as diagnostic evidence but do not override the stable C4 decision.

## Stock restoration after the final gate

After the v7 and fair stock+queue1 tests:

- experimental candidate container removed
- factorial test container removed
- `inneros-vllm-canary-rocm10.service` restored
- active container: `inneros-vllm-canary-rocm10`
- `/v1/models`: HTTP 200
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Python: `3.14.7`
- PyTorch: `2.12.0+rocm10.0.0`
- HIP: `7.15.26333`
- vLLM: `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`
- GPU identified as Radeon AI PRO R9700 / `gfx1201`

No final candidate was installed into the stock image or site-packages. The final v7 launch used read-only bind mounts into a disposable test container.

## Phase 1 — independent serving/concurrency result

Three independent vLLM process starts reproduced the Phase-1 serving/concurrency decision:

- baseline: `19.893 / 19.942 / 19.987 tok/s`
- candidate: `36.004 / 36.078 / 36.194 tok/s`
- paired median gain: `+80.99%`
- baseline inter-process spread: about `0.47%`
- verdict: `KEEP / KEEP / KEEP`

Evidence: `docs/evidence/r9700_independent_process_final_20260908.json`
SHA-256: `dda9128a5ea17728e3eae59488d37952665b30c77e54cb440c225971fdfcf94f`

This is a **serving/concurrency result**, not a GPU-kernel speedup.

## Phase 2 — live Qwen WNA16 contract

Measured live contract:

- backend: `TritonWNA16Experts`
- activation dtype: FP16
- experts: `128`
- top-k: `8`
- quantization: packed INT4 W4A16, group size `128`
- W1: `uint8 [128,1536,1024]`
- W2: `uint8 [128,2048,384]`

Evidence: `docs/evidence/r9700_live_moe_dtype_probe_20260909T024321Z.json`.

## Custom W1 versus stock Triton WNA16

Three real-weight child-process campaigns, 21 alternating paired HIP-event rounds per M:

- M1: `1.74518x`
- M2: `1.49210x`
- M4: `1.47445x`
- M8: `1.44613x`
- M16: `1.46273x`
- median M1..16: `1.47682x`
- wins: `63/63` per shape across campaigns
- custom-vs-stock cosine: effectively 1.0

Evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`
SHA-256: `6f1faf903dc62d31b2ef62b60beeb1394f68c9f778366ebd77b38a187a782a5e`

This proves a **routed W1 microkernel improvement only**. It must never be described as a 1.477x speedup for full Qwen.

## Earlier full-model functional proof

The earlier stock-layout hybrid smoke remains valid historical proof that the complete Qwen3-Coder 30B model can load and route through the experimental backend with fallback and rollback:

- 48 custom-path observations
- 48 stock-fallback observations
- deterministic request hash matched stock in that smoke
- temporary hook removed
- stock restored

Evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`
SHA-256: `98565adc398ecf91c34e4a6f94e8ef7847c3e36b6123b791410878e66091206d`

## Recovered gfx1201 tuner — evidence gap resolved

The previously uncommitted WNA16 tuner artifacts were recovered and are now preserved in Git. The earlier statement that these exact artifacts were unavailable is obsolete.

Canonical interpretation: `docs/R9700_RECOVERED_WNA16_TUNER_RESULT_20260910.md`

Raw bounded tuner: `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`
Recovered SHA-256: `f56942b75f6621ae22078a8173ce3f6ea01c8980d07f279d8456c3d2aaba6d50`

Recovered winner:

- `BLOCK_SIZE_M=16`
- `BLOCK_SIZE_N=64`
- `BLOCK_SIZE_K=32`
- `GROUP_SIZE_M=1`
- `num_warps=4`
- `num_stages=2`
- `waves_per_eu=4`
- `SPLIT_K=1`

Isolated result:

- median M1..16 speedup: `1.209256x`
- minimum tested speedup: `1.083843x`
- verdict: **MICROBENCH KEEP**

The two recovered live override smokes remain **REJECT / NOT PROMOTED**. One did not observe the tuned path; the other observed it but failed correctness/process-state promotion controls. Recovery closes the history gap, not the full-model performance gap.

## Clean serving factorial

The clean factorial campaign selected stock attention + `GPU_MAX_HW_QUEUES=1` as the stable promotion control:

- stock+queue1 median C4: `158.489996 tok/s`
- Unified Attention + default queues: `138.764971 tok/s`
- Unified Attention + queue1: `156.753550 tok/s`

Unified Attention did not improve the stable stock+queue1 baseline and is excluded from the final v7 gate.

Evidence: `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md` and `docs/evidence/r9700_factorial_clean_summary_20260911.json`.

## Rejected or invalid paths preserved

- AITER/FlyDSL sorting: isolated HSA memory fault
- activation group pre-sum: numerically correct but slower
- early FP16/BF16 mirror: dtype mismatch, superseded by measured FP16 contract
- custom algebraic W2: slower than stock/reference
- same-process mmap/SIGUSR1/SIGUSR2 gate: invalid because captured execution stayed on `runtime_gate_stock`
- naive default-queue stock comparisons: unsuitable for promotion because of R9700 startup bimodality
- Unified Attention: validly tested, but not better than stock+queue1
- recovered live WNA16 override: not promotable
- final v7 full-model hybrid: functional but not promoted because stable C4 is below stock and strict exact-output parity is incomplete

## Presentation-safe claims

You may say:

- An experimental HyperLoom RDNA4 path runs on a physical Radeon AI PRO R9700 / `gfx1201`.
- The project uses a real Qwen3-Coder 30B AWQ workload on ROCm 10 + vLLM.
- The live MoE backend and datatype were measured rather than assumed.
- A real packed-INT4 Triton W1 kernel executes on `gfx1201` and beats stock WNA16 W1 in the validated small-M microbenchmark region.
- The complete model boots with `R9700HybridWNA16Experts`; all 48 MoE layers reached the custom route with stock fallback available.
- The final clean v7 experiment was benchmarked against a controlled stock+queue1 baseline and was not promoted because E2E C4 remained about 5% slower.
- Negative results, unstable startup modes, failed approaches, correctness boundaries, and rollback are preserved as evidence rather than hidden.

Do **not** say:

- official AMD or upstream HyperLoom support for R9700
- first R9700/RDNA4 HyperLoom port in the world
- full Qwen is 1.477x faster
- the final hybrid improves end-to-end throughput
- the slow 72 tok/s stock startup proves a >100% candidate speedup
- exact deterministic parity is universally proven for final v7

## Closure

The R9700 Phase-2 investigation is technically closed as an **experimental, evidence-bounded result**. The W1 kernel remains `KEEP`. The clean v7 full-model integration is retained as research/prototype code but is **not promoted** into the stock serving path. Stock ROCm10 serving has been restored and verified healthy.

A future optimization phase should start from the preserved evidence, not rerun closed experiments. Promotion requires a new independently reproducible full-model candidate that matches stock correctness and repeatedly reaches or exceeds the credible stock C4 baseline.
