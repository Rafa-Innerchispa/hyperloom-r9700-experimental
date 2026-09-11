# HyperLoom on Radeon AI PRO R9700 — Presentation Brief

## One-line story

We adapted and validated an experimental HyperLoom optimization path on a physical AMD Radeon AI PRO R9700 (`gfx1201`) using ROCm 10, vLLM and a real Qwen3-Coder 30B AWQ workload, then followed the evidence all the way from a faster packed-INT4 W1 microkernel to a full-model integration that we deliberately did **not** promote when end-to-end throughput remained below stock.

## Why this project matters

The work is not a synthetic “hello GPU” demo. It exercises a real 30B MoE coding model, discovers the live quantized execution contract, builds a hardware-specific Triton optimization, integrates it into vLLM’s WNA16 MoE path, validates routing across all 48 MoE layers, measures correctness and throughput, preserves failed experiments, and restores a clean stock runtime after every promotion test.

Architecture:

`HyperLoom → R9700/gfx1201 → ROCm 10 → vLLM → Qwen3-Coder 30B AWQ → MoE → AutoAWQ/WNA16 → packed INT4 → Triton RDNA4 → custom W1 + stock W2/fallback → evidence gate`

## What we proved

### 1. The real live workload contract

The Qwen MoE path on the R9700 was measured as:

- FP16 activations
- `TritonWNA16Experts`
- 128 experts, top-k 8
- packed INT4 W4A16, group size 128
- W1 packed `uint8 [128,1536,1024]`
- W2 packed `uint8 [128,2048,384]`

This prevented us from optimizing an assumed or stale datatype/backend path.

### 2. A real RDNA4 W1 kernel win

With real Qwen weights and real FP16 activations, the custom packed-INT4 small-M W1 kernel beat stock Triton WNA16 W1 across the tested M1..16 region:

- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- `63/63` paired wins per tested shape across three campaigns
- numerical cosine effectively 1.0

This is the strongest kernel result and is a **microkernel claim only**.

### 3. Full Qwen 30B integration is real

The clean `v7` candidate booted the complete model with `R9700HybridWNA16Experts` and real inference observed:

- 48 `custom_small_w1_stock_w2` routes
- 48 `stock_full_fallback` routes
- stock `ROCM_ATTN`
- `GPU_MAX_HW_QUEUES=1`
- dedicated streaming TTFT about `64.9 ms`

The custom path is therefore not a disconnected benchmark kernel; it is reachable inside the real full-model serving path.

### 4. The promotion gate rejected the full hybrid

Clean v7 C4:

- `149.891 tok/s`
- `150.842 tok/s`
- median `150.367 tok/s`

Fair clean stock+queue1 control:

- current `157.490 tok/s`
- prior clean factorial median `158.490 tok/s`

The integrated candidate remained about 4.5-5.1% below the selected stable stock baseline.

All four C4 output hashes matched stock exactly, but one dedicated strict deterministic single-request probe did not. Therefore the full hybrid is retained as experimental research code rather than promoted into the stock serving path.

## The engineering lesson

The project produced a genuine local kernel acceleration, but the complete system exposed additional routing, scheduling and integration overhead. HyperLoom’s value here is not merely “it found a faster kernel.” The more meaningful result is that the optimization workflow carried the experiment through measurement, integration, falsification and rollback without turning a microbenchmark win into a misleading product claim.

That distinction is especially important on the R9700 because serving showed strong process-start throughput bimodality. A pathological stock start around 72 tok/s could make the candidate appear more than twice as fast. We explicitly rejected that comparison and used a stable `GPU_MAX_HW_QUEUES=1` stock control instead.

## Additional useful findings

- Phase-1 independent process work reproduced a `+80.99%` paired median serving/concurrency decision. This is kept separate from kernel claims.
- A recovered gfx1201 WNA16 bounded tuner found a `1.209256x` isolated median improvement with a hardware-specific configuration; live override smokes were not promotable.
- Unified Attention was tested and did not beat the stable stock+queue1 baseline.
- AITER/FlyDSL, activation pre-sum, custom W2 and same-process signal switching produced failures or regressions and are preserved as negative evidence.

## Demo/evidence sequence

For a technical presentation, show the evidence in this order:

1. physical GPU/runtime identity: Radeon AI PRO R9700 / `gfx1201`, ROCm 10, vLLM, Qwen3-Coder 30B AWQ;
2. live WNA16/FP16/paked-INT4 contract discovery;
3. W1 microkernel comparison and `1.47682x` median M1..16 result;
4. full-model log line `Using R9700HybridWNA16Experts`;
5. path counts `48 custom + 48 fallback`;
6. clean C4 stock vs v7 comparison;
7. correctness boundary and explicit non-promotion;
8. stock runtime restored, HTTP 200.

## Presentation-safe headline

**Experimental HyperLoom RDNA4 optimization on Radeon AI PRO R9700: a validated ~1.48x small-M W1 microkernel improvement integrated into a real Qwen3-Coder 30B serving path, with full-system promotion withheld after controlled E2E validation.**

## Do not claim

- official AMD or upstream HyperLoom R9700 support
- “first in the world”
- 1.477x acceleration of the full 30B model
- an E2E speedup for the final v7 hybrid
- that the 72 tok/s slow stock mode is the proper baseline
- universal exact deterministic parity for final v7

Canonical sources: `FINAL_STATUS.md`, `docs/R9700_PROJECT_CONTINUITY.md`, and `docs/evidence/r9700_v7_final_gate_summary_20260911.json`.
