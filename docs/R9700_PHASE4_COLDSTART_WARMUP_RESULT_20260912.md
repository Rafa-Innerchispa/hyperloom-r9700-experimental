# R9700 Phase 4 — Cold-start directed warmup result

Captured: 2026-09-12

## Result

A fresh stock-equivalent ROCm10/vLLM process with `--jit-monitor-verbose` reproduced the cold first-request penalty and exposed the exact Triton specialization missing from startup warmup.

Untreated fresh first request:

- TTFT: **6.0748 s**
- output hash: exact canonical stock hash
- unexpected inference-time JIT: **yes**
- kernel: `vllm.v1.attention.ops.prefix_prefill._fwd_kernel`

The relevant specialization included:

- `BLOCK_DMODEL=128`
- `BLOCK_DMODEL_PADDED=128`
- `BLOCK_M=128`
- `BLOCK_N=64`
- `BLOCK_SIZE=32`
- `CAUSAL=True`
- `KV_FROM_CACHE=False`
- `SKIP_DECODE=True`
- `PHYSICAL_BLOCK_SIZE=16`
- `num_queries_per_kv=8`
- `num_unroll_cache=4`
- `num_unroll_request=1`
- Q/K/V FP16
- Triton warps=4, stages=1

This confirms that HTTP readiness did not imply all user-facing Triton specializations were compiled.

## Directed warmup A/B

A second completely fresh process was launched from clean VRAM. Before the first externally visible request, the exact deterministic correctness request was issued as an internal readiness warmup.

Internal warmup:

- TTFT: **5.4139 s**
- canonical output hash: **PASS**
- expected `_fwd_kernel` JIT warning: **observed**

First user request immediately afterward:

- TTFT: **0.0663 s**
- elapsed: **1.2601 s**
- canonical output hash: **PASS**
- unexpected inference-time JIT warning: **none**

Gate:

- correctness: PASS
- no user-visible inference JIT: PASS
- first-user TTFT < 500 ms: PASS
- overall: **PASS**

Relative to the untreated fresh request, first-user TTFT was reduced by approximately **98.91%**, roughly a **91.6x** reduction in first-token latency.

## Interpretation

The ~5–6 second delay is not an intrinsic steady-state cost of the model, the R9700, or the Phase 3 S3 INT4 repack. It is an uncovered ROCm prefix-prefill Triton specialization that can be consumed deliberately during readiness warmup.

The lowest-risk operational mitigation is therefore initially a readiness-stage internal warmup request rather than an invasive vLLM source modification. A service should not be advertised as user-ready until that warmup completes successfully.

## Remaining gate before Phase 4 closure

1. Apply the readiness warmup to the selected Phase 3 S3 candidate.
2. Verify fresh-process correctness and absence of user-visible JIT.
3. Re-measure C1/C4/~6K context to ensure S3 performance is not materially regressed.
4. Repeat fresh starts for reproducibility.
5. Package the warmup as an explicit service/readiness helper with failure behavior and rollback.
6. Restore stock ROCm10 operational default and verify HTTP 200 after test campaigns.
7. Update `main`, continuity and final Phase 4 verdict.

Machine summary: `docs/evidence/r9700_phase4_coldstart_warmup_ab_summary_20260912.json`.
