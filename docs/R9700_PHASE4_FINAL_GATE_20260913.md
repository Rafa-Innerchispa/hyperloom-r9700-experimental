# R9700 Phase 4 final gate — 2026-09-13 UTC

## Verdict

**PASS — S3 + readiness warmup is reproducible across three independent fresh processes on the tested workload.**

This closes the Phase 4 cold-start/parity investigation. It does **not** claim universal performance uplift, production support by AMD, or superiority for every request shape.

## Stack

- GPU: AMD Radeon AI PRO R9700 32 GiB, gfx1201
- Runtime image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- vLLM patch SHA256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 MoE config SHA256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- Queue control: `GPU_MAX_HW_QUEUES=1`
- Attention: stock `ROCM_ATTN`

## Cold-start root cause and mitigation

HTTP readiness alone was insufficient. After `/v1/models` returned 200, the first real inference could still JIT-compile:

`vllm.v1.attention.ops.prefix_prefill._fwd_kernel`

The captured specialization includes `BLOCK_DMODEL=128`, `BLOCK_M=128`, `BLOCK_N=64`, FP16, causal attention, `KV_FROM_CACHE=False`, `SKIP_DECODE=True`, and 8 query heads per KV head.

A deterministic readiness-stage inference now absorbs that compile before user traffic. The readiness request itself intentionally pays the JIT cost and verifies the canonical output hash. The service should only be considered user-ready after this warmup passes.

## Three fresh-process results

Controlled stock queue1 C4 baseline: **158.489996 tok/s**.

| Fresh process | Warmup JIT TTFT | First user TTFT | C1 tok/s | C4 tok/s | Fresh ~6K long tok/s | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Start 1 | 4.905 s | 50.9 ms | 69.96 | 190.60 | 62.00 / 61.97 | PASS |
| Start 2 | 4.720 s | 53.6 ms | 69.68 | 194.08 | 61.96 | PASS |
| Start 3 | 4.624 s | 52.9 ms | 68.15 | 189.98 | 61.57 / 60.79 | PASS |

Aggregate using the **first post-warmup C4 measurement from each independent process**:

- C4 median: **190.5969 tok/s**
- C4 mean: **191.5536 tok/s**
- C4 range: **189.9816–194.0823 tok/s**
- Median C4 gain vs controlled stock: **+20.258%**
- C1 median: **69.6790 tok/s**
- First-user TTFT median: **52.94 ms**
- Readiness JIT TTFT median: **4.720 s**
- Fresh-prefix long-context median by process: **61.955 tok/s**
- Canonical correctness hash: stock exact in every fresh process
- Canonical C4 hashes: stock exact in every fresh process
- New JIT during first user inference after readiness warmup: **0 lines in every fresh process**

## Important anomaly handled

Start 1 produced one transient long-context observation near 47 tok/s. It was not treated as a success result. Repeated long-context plus two fresh-prefix controls returned ~62 tok/s. Start 2 and Start 3 independently reproduced ~61–62 tok/s. The final long-context claim is therefore based on fresh-prefix controls rather than exact-prompt cache reuse.

## Operational outcome

The reproducible deployment pattern for this experiment is now:

1. launch the S3 candidate from a clean GPU state;
2. wait for HTTP 200;
3. run `scripts/r9700_readiness_warmup.py` or equivalent deterministic readiness inference;
4. require the canonical correctness hash;
5. only then expose the process to user traffic.

This avoids changing vLLM internals solely to eliminate the first-request JIT penalty and keeps the mitigation reversible.

## Evidence

Canonical machine-readable aggregate:

`docs/evidence/r9700_phase4_three_start_aggregate_20260913.json`

Key runtime evidence was produced by:

- `scripts/r9700_phase4_s3_verbose_launcher.py`
- `scripts/r9700_phase4_s3_warmup_probe.py`
- `scripts/r9700_phase4_s3_measure.py`
- `scripts/r9700_phase4_s3_long_fresh_probe.py`
- `scripts/r9700_readiness_warmup.py`

## Next action

Phase 4 experimentation is complete. The experimental S3 container must be removed and the stable `inneros-vllm-canary-rocm10.service` restored and verified with `/v1/models` HTTP 200. Promotion of the S3 recipe into a persistent service should be a separate controlled change, not silently folded into this benchmark run.
