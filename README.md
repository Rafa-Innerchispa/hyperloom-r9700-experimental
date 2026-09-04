# Hyperloom R9700 Experimental

Experimental AMD Lab Program Challenge 1 port for running Hyperloom on the AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 64 CUs, 32 GiB-class VRAM).

## Status

**LIVE EXPERIMENTAL PASS** for the architecture-neutral Hyperloom benchmark path.

On 2026-09-04 the patched Hyperloom build executed on the physical R9700 and successfully produced both baseline and candidate `benchmark_report.json` artifacts using Hyperloom's `bypass` backend + InferenceX against the existing ROCm 10 vLLM server.

This is **not official AMD Hyperloom support** and does not yet include a Magpie R9700 runner, TraceLens RDNA4 profiling or RDNA4-specific kernel optimization.

## Experimental upstream patch

Base: `AMD-AGI/Hyperloom@9ae79d6a8c9fec7ed041735e70fb19ef39850813`

The minimal identity port is:

```python
"r9700": ("gfx1201", 64)
```

plus:

```python
"gfx1201": "r9700"
```

Reproducible patch:

`patches/hyperloom-r9700-gfx1201.patch`

## Live validation

### Preflight

- physical GPU: AMD Radeon AI PRO R9700
- `rocm-smi`: `gfx1201`
- Hyperloom autodetect: `r9700`
- dispatch identity: `gfx1201`, 64 CU
- backend: `bypass`
- result: **PASS**

### Baseline

Workload: ISL=32, OSL=32, concurrency=1.

- completed: 10/10
- output throughput: **21.7953 tok/s**
- total token throughput: **43.5905 tok/s**
- mean TTFT: **68.47 ms**
- mean E2E: **1467.81 ms**

### Candidate

Same model/GPU/server, concurrency=2.

- completed: 20/20
- output throughput: **36.5927 tok/s**
- total token throughput: **73.1854 tok/s**
- mean TTFT: **121.06 ms**
- mean E2E: **1746.44 ms**

Aggregate output throughput changed by **+67.89%**. This is a concurrency/workload tuning result: latency also increased. It is not presented as a universal 67.89% GPU speedup.

Raw evidence:

`evidence/live-r9700-results-20260904.json`

## Why the bypass backend first

Hyperloom already ships `HYPERLOOM_BENCHMARK_BACKEND=bypass`. It runs serving-framework benchmarks directly in Python and produces the report contract consumed by Hyperloom without requiring a board-specific Magpie shell runner. That makes it the safest first route for an RDNA4 GPU that upstream Hyperloom does not yet recognize.

## Tests

The upstream-style development worktree passed **11 focused tests**, covering the R9700 identity, parser acceptance, product-name autodetection and `gfx1201` fallback detection.

## Truth boundary / Phase 2

Still intentionally not claimed:

- official AMD support
- Magpie `vllm_r9700.sh` / `sglang_r9700.sh`
- TraceLens profiling on RDNA4
- RDNA4-specific kernel optimization
- portability of `gfx942/gfx950` compiled artifacts
- full autonomous Think → Decide → Implement Hyperloom optimization session

Those are the next engineering layer after proving the architecture-neutral execution path.

## Challenge project

Judge-facing integration, Builder submission, demo and full evidence map:

`Rafa-Innerchispa/amd-ralfiia-hybrid-ops-copilot`
