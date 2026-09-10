# R9700 ROCm 10 / RDNA4 Unified Attention refresh result — 2026-09-10

## Why this campaign exists

Vector.sys correctly warned that single-start measurements on Radeon AI PRO R9700 can be confounded by process-start bimodality. Since that feedback, upstream vLLM added explicit RDNA4/gfx120x AITER handling and AITER now lists Radeon AI PRO R9700/gfx1201 as experimentally supported. Our installed ROCm 10 image predates the relevant vLLM RDNA4 gate, so this campaign ports only the minimal backend-selection behavior required to exercise AITER Unified Attention without enabling unsupported AITER RMSNorm/CK paths.

## Controlled stack

- Physical AMD Radeon AI PRO R9700 / gfx1201 / 32 GiB
- ROCm 10 / HIP 7.15.26333
- PyTorch 2.12.0+rocm10.0.0
- vLLM 0.27.1.dev5+gf46a9dfe2.d20260827.rocm100
- amd-aiter 0.1.20.post1
- QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
- max model length 8192
- gpu-memory-utilization 0.82
- dtype float16

The candidate changes only attention backend selection. RMSNorm remains native. Qwen MoE remains stock `TritonWNA16Experts`. This is intentionally not yet the HyperLoom custom W1 integration.

## Three independent process starts

### Stock

C4 aggregate throughput:

- 71.940 tok/s
- 71.776 tok/s
- 165.699 tok/s

C1 decode:

- 22.322 tok/s
- 22.289 tok/s
- 69.343 tok/s

~6012-token prompt / 64-token decode:

- 21.944 tok/s
- 21.910 tok/s
- 62.553 tok/s

This reproduces the severe process-start bimodality that motivated Vector's warning.

### RDNA4 AITER Unified Attention candidate

C4 aggregate throughput:

- 158.017 tok/s
- 157.296 tok/s
- 157.469 tok/s

C1 decode:

- 63.095 tok/s
- 62.994 tok/s
- 62.030 tok/s

~6012-token prompt / 64-token decode:

- 56.288 tok/s
- 56.264 tok/s
- 56.088 tok/s

Candidate C4 median: **157.469 tok/s**.
Candidate C4 range/median: **0.46%**.
Stock C4 median: **71.940 tok/s**.
Stock C4 range/median: **130.56%**.

The deterministic correctness request produced the same SHA-256 on stock and all candidate runs:

`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

Runtime logs explicitly reported:

- `Setting kv cache block size to 64 for ROCM_AITER_UNIFIED_ATTN backend`
- `Using TritonWNA16Experts`

## Interpretation

Unified Attention is **not** being claimed as faster than the best stock spawn. The best stock C4 run reached 165.699 tok/s, about 5% above the candidate median. Instead, the important result is that all three candidate starts landed tightly around 157–158 tok/s while stock reproduced low/fast process-start modes.

Therefore the correct Phase verdict is:

**RDNA4 AITER Unified Attention: KEEP_FOR_NEXT_GATE**

It appears to be a strong candidate for eliminating or bypassing the practically harmful low-performance regime, but three starts are not enough to claim that the underlying ROCm bimodal bug is solved universally.

## What changes for HyperLoom

This result raises the baseline HyperLoom must beat. We should no longer optimize the custom packed-INT4 W1 against a randomly slow stock process and call the difference a product win.

Next comparison must use:

1. stable/fast RDNA4 attention baseline;
2. the real Qwen AWQ WNA16 path;
3. HyperLoom custom W1 + stock W2 on top of the same attention baseline;
4. independent process starts and identical prompts;
5. C4 as the primary serving gate, plus C1, long-context, TTFT, correctness, clocks/power when available;
6. promotion only if the combined candidate reaches or beats the best credible baseline without sacrificing correctness or reproducibility.

This is a stronger hackathon story than a one-off port: HyperLoom is being tested as an adaptive optimization system against a moving RDNA4 upstream baseline, with obsolete workarounds rejected and upstream improvements incorporated when they win.


## 2026-09-10 correction — candidate launcher included `GPU_MAX_HW_QUEUES=1`

A source-level audit of the exact candidate launcher found that all three candidate process starts enabled both `ROCM_AITER_UNIFIED_ATTN` **and** `GPU_MAX_HW_QUEUES=1`. The throughput/correctness numbers above remain valid measurements of that combined configuration, but attributing the stability improvement to Unified Attention alone would be incorrect.

The aggregate JSON was revised to schema `hyperloom.r9700.unified_attention_plus_queue1_three_start.v2`. The current verdict is therefore **KEEP_FOR_FACTORIAL_GATE**, not an isolated Unified-Attention promotion. The required separation is:

1. Unified Attention with `GPU_MAX_HW_QUEUES` unset;
2. stock attention with `GPU_MAX_HW_QUEUES=1`;
3. compare each against the existing stock/no-queue and Unified+queue1 results using independent process starts.

This correction preserves the original measurements while tightening their claim boundary before any hackathon/public use.
