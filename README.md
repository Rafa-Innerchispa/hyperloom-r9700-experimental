# Hyperloom R9700 Experimental

Experimental AMD Lab Program Challenge 1 port that explores running Hyperloom on the AMD Radeon AI PRO R9700 (`gfx1201`, RDNA 4, 64 CUs, 32 GB VRAM).

## Goal

Hyperloom currently exposes official runner identities for AMD Instinct families. This project adds an **experimental** R9700 identity and validates the architecture-neutral optimization path without claiming official AMD support.

The first target is deliberately narrow:

1. Accept and auto-detect `r9700` / `gfx1201`.
2. Use Hyperloom's own `bypass` benchmark backend with vLLM on ROCm 10.
3. Run a real `baseline -> candidate -> compare` loop on the local R9700.
4. Keep Instinct/CDNA-specific Magpie, TraceLens and kernel paths gated until they are reimplemented or revalidated for RDNA4.

## Verified development state

- Upstream base: `AMD-AGI/Hyperloom` commit `9ae79d6a8c9fec7ed041735e70fb19ef39850813`.
- Experimental identity: `r9700 -> (gfx1201, 64 CUs)`.
- CLI/parser accepts `--gpu-type r9700` in the patched working tree.
- Product-name and `gfx1201` auto-detection covered by tests.
- 11 focused upstream tests pass in the development worktree.
- Live execution on the physical R9700 remains the next acceptance gate.

## Why the bypass backend first

Hyperloom already ships `HYPERLOOM_BENCHMARK_BACKEND=bypass`. It runs serving-framework benchmarks directly in Python and writes the same report contract consumed by the optimizer, without requiring a Magpie board-specific benchmark script. That makes it the safest first path for an RDNA4 board that does not yet have an official Magpie runner.

## Truth boundary

This repository does **not** claim official Hyperloom support for Radeon AI PRO R9700. A successful experiment means the patched Hyperloom orchestration + bypass benchmark path runs on the R9700 and produces reproducible evidence. Instinct-specific kernels must not be reused as if `gfx942/gfx950` artifacts were compatible with `gfx1201`.

## Challenge project

The judge-facing integration, evidence map, Builder submission and demo UI live in:

`Rafa-Innerchispa/amd-ralfiia-hybrid-ops-copilot`
