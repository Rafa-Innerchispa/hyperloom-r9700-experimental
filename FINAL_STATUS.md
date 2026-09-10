# HyperLoom Radeon AI PRO R9700 / RDNA4 Final Experimental Status

Date: 2026-09-10
Correlation: `hyperloom-r9700-finalize-20260910`
Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
Hardware: AMD Radeon AI PRO R9700 (`gfx1201`, 32 GiB)
Workload: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` + vLLM + ROCm 10

## Final verdict

**KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED**

HyperLoom has a real experimental RDNA4/R9700 path that executed on the physical `gfx1201` GPU. The custom packed-INT4 W1 Triton kernel is validated against the stock vLLM Triton WNA16 W1 kernel with real Qwen3-Coder AWQ weights and real FP16 activation dtype. The full 30B model also loaded and served a deterministic request with the hybrid backend, exercising the custom path across all 48 MoE layers while preserving stock fallback and rollback.

The integration is **not** promoted as the default serving backend because the most stable end-to-end concurrency-4 comparison remained approximately 5% slower than stock after the valid v3 alignment-overhead reduction. Therefore no full-model speedup claim is allowed.

## What is proven

### Phase 1: autonomous serving/concurrency decision

Three independent vLLM process starts reproduced the selected serving/concurrency improvement:

- baseline: `19.893 / 19.942 / 19.987 output tok/s`
- candidate: `36.004 / 36.078 / 36.194 output tok/s`
- paired median gain: `+80.99%`
- baseline inter-process spread: about `0.47%`
- verdict: `KEEP / KEEP / KEEP`

Evidence: `docs/evidence/r9700_independent_process_final_20260908.json`
SHA-256: `dda9128a5ea17728e3eae59488d37952665b30c77e54cb440c225971fdfcf94f`

This is a **serving/concurrency result**, not a GPU-kernel speedup.

### Phase 2: live WNA16 discovery

The live Qwen MoE path was measured as:

- Experts backend: `TritonWNA16Experts`
- activation dtype: `torch.float16`
- experts: `128`
- top-k: `8`
- quantization: `int4_w4a16`, group size `128`
- W1 packed dtype/shape: `uint8 [128,1536,1024]`
- W2 packed dtype/shape: `uint8 [128,2048,384]`

Evidence: `docs/evidence/r9700_live_moe_dtype_probe_20260909T024321Z.json`.

### Custom W1 versus stock Triton WNA16

Three consecutive real-weight child-process campaigns, 21 alternating paired HIP-event measurements per M:

- M1: about `1.745x`
- M2: about `1.492x`
- M4: about `1.474x`
- M8: about `1.446x`
- M16: about `1.463x`
- median across tested small-M region: about `1.477x`
- wins: `63/63` per tested shape across the three campaigns
- custom-vs-stock cosine: effectively `~1.0`

Aggregate evidence: `docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`.

This proves the routed W1 microkernel result only. It does **not** mean the complete Qwen model is 1.477x faster.

### Full-model hybrid integration

`R9700HybridWNA16Experts` loaded the complete Qwen3-Coder 30B AWQ model. A real deterministic request exercised:

- `48` observations of `custom_small_w1_stock_w2`
- `48` observations of `stock_full_fallback`
- candidate deterministic response hash matching stock
- automatic removal of the temporary bootstrap hook
- healthy stock restoration after the candidate run

Evidence: `docs/evidence/r9700_stock_layout_live_candidate_smoke_20260909T030903Z.json`.

This proves full-model boot, basic correctness, all-48-layer custom-path reachability, fallback, and rollback.

## End-to-end performance verdict

Radeon AI PRO R9700 showed a strong bimodal process/startup state at concurrency 1, with observed decode rates ranging roughly from the low 20s to 70+ tok/s under apparently equivalent conditions. Concurrency 1 is therefore not used as a naive proof of speedup.

Concurrency 4 was substantially more stable:

- stock states observed: approximately `159-162 tok/s`
- hybrid candidate after the valid alignment-reuse improvement: approximately `151-153 tok/s`
- resulting integrated regression: approximately `5%`

The v3 integration reduced duplicate `moe_align_block_size` work and improved the earlier approximately 6-7% regression, but did not recover parity with stock.

A same-process runtime-gate experiment using mmap and then `SIGUSR1`/`SIGUSR2` is explicitly **invalid for performance claims**. Signals reached the EngineCore process but path evidence remained `runtime_gate_stock`; the custom path was not activated. The apparent approximately `+0.79%` result is therefore excluded. The behavior is consistent with decode graph capture/replay preventing a Python runtime switch from changing the already-captured path.

## Tuning-config experiment record and evidence gap

Later uncommitted work referenced these artifacts:

- `docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json`
- `docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json`
- `scripts/r9700_vllm_wna16_bounded_tuner.py`
- `scripts/r9700_vllm_tuned_config_override.py`
- `scripts/r9700_tuned_config_live_smoke.py`

These files were never committed and are no longer present in any of the checked Phase-2/live worktrees. Their exact numerical result is therefore **not reconstructed or claimed**. The historical record shows this work was intended to test gfx1201-specific Triton/vLLM tuning after community feedback about missing RDNA4 configs, but without the raw JSON it cannot close the E2E promotion gate.

This evidence gap does not invalidate the committed microkernel or full-model smoke evidence. It prevents us from claiming that later tuning recovered the approximately 5% integrated deficit.

## Failed/partial experiments preserved as engineering evidence

- AITER/FlyDSL sorting path: isolated HSA memory fault; not promoted.
- activation group pre-sum: numerically correct but slower; rejected.
- early FP16/BF16 mirror: exposed a dtype mismatch; superseded after measuring the real FP16 live activation contract.
- custom algebraic W2: slower than stock/reference; rejected. Stock W2 remains the correct boundary.
- same-process runtime switching: invalid as an A/B performance proof because custom execution was not observed.

## Community feedback closure

The methodology incorporates the material recommendations received from the ROCm/AMD community and Vector.sys:

- independent process starts were adopted after the R9700 bimodal/spawn-lottery concern;
- concurrency/serving gains are kept separate from kernel gains;
- live backend and dtype were measured instead of inferred;
- Unified Attention was probed but not falsely claimed as validated on this workload;
- AITER/FlyDSL was investigated and its failing path preserved rather than hidden;
- gfx1201-specific tuning was investigated, but the uncommitted tuner artifacts are not treated as evidence now that they are unavailable;
- all public claims remain explicitly experimental and do not imply official AMD support.

## Runtime baseline reconfirmed on 2026-09-10

The AMD node was observed running:

- container: `inneros-vllm-canary-rocm10`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- launch: `--max-model-len 8192 --gpu-memory-utilization 0.82 --dtype float16`
- PyTorch: `2.12.0+rocm10.0.0`
- HIP: `7.15.26333`
- vLLM: `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`
- GPU: AMD Radeon AI PRO R9700, `gfx1201`

Final post-documentation health and hook absence must be checked once more after commit/push; that check is operational evidence, not a benchmark result.

## Allowed claims

- Experimental HyperLoom path on Radeon AI PRO R9700 / `gfx1201`.
- Real Qwen3-Coder 30B AWQ workload on ROCm 10 + vLLM.
- Three independent process starts validated the Phase-1 serving/concurrency decision.
- A real packed-INT4 Triton W1 kernel runs on physical `gfx1201`.
- With real Qwen weights and measured FP16 activation dtype, the custom small-M W1 kernel beat stock Triton WNA16 W1 in the tested microbenchmark region.
- The complete model booted with the experimental hybrid backend and all 48 MoE layers exercised the custom path.
- Stock fallback and rollback were validated.

## Forbidden claims

- Official AMD or upstream HyperLoom support for Radeon AI PRO R9700.
- “First port in the world.”
- `1.477x` speedup of the complete Qwen model.
- Any end-to-end speedup from the hybrid kernel integration.
- Any performance conclusion from the invalid same-process runtime-gate A/B.

## Closure definition

The engineering investigation is considered complete as an **experimental technical result** when this status, the development ledger and community-feedback ledger are committed/pushed and stock serving is reconfirmed healthy. The optimized W1 kernel remains a `KEEP`; the full-model hybrid backend remains research code and is not promoted into the default serving path until a future independent-process E2E campaign reaches stock parity or better with preserved raw evidence.
