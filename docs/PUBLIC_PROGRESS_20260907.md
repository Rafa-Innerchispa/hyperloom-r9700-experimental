# HyperLoom on Radeon AI PRO R9700: Experimental Compatibility Progress

Date: 2026-09-07

## What is working

We have a reproducible experimental path running on a real AMD Radeon AI PRO R9700 (`gfx1201`) workstation GPU using ROCm 10, vLLM, and `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.

The current branch is `codex/hyperloom-r9700-master-20260907` and the verified runtime checkpoint is based on commit `149bd5f69edd327e3712839812b00eaec1a8680a`.

Verified runtime evidence includes:

- AMD Radeon AI PRO R9700 / `gfx1201`
- ROCm 10 container runtime
- local OpenAI-compatible vLLM endpoint
- Qwen3-Coder 30B AWQ served locally
- bounded agent E2E evidence with deterministic KEEP/REJECT gating
- reproducible runtime manifest and SHA-256 evidence
- focused tests on Windows and AMD
- AMD compile checks and clean diff validation

## Why this matters

HyperLoom upstream still has open RDNA4 support work. That means people with newer RDNA4 cards can encounter a gap between owning capable hardware and having a documented, reproducible path for running and evaluating HyperLoom-style optimization workflows on it.

This project is useful because it turns one successful workstation setup into an auditable compatibility path: exact hardware, runtime, model, launch configuration, evidence files, tests, and explicit claim boundaries.

The value is not that only one person can make an R9700 run. The value is reducing the amount of trial-and-error required for the next person and producing evidence that can be compared with upstream work.

## What we are not claiming

This is **not** official AMD-AGI HyperLoom support for the R9700.

We are **not** claiming a GEAK, Arbor, Triton, or kernel-level optimization win from the serving/concurrency results.

We are **not** claiming a specific AWQ kernel backend until the runtime logs/profiler expose enough evidence to identify it.

The Radeon AI PRO R9700 is not being treated as identical to an RX 9070 XT simply because both are RDNA4-class hardware.

## Current upstream context

At the time of this checkpoint:

- AMD-AGI/Hyperloom issue #1033 for RDNA4 support remains open.
- AMD-AGI/Hyperloom pull request #1032 remains open and unmerged.

Our work should therefore be read as an experimental compatibility and validation effort that can inform future upstream support, not as a replacement for it.

## Next measurement

The final closure step is a bounded five-spawn benchmark using `scripts/r9700_multispawn_harness.py --execute` against the already running local vLLM service, without restarting vLLM.

That run will capture per-spawn evidence and aggregate metrics including output tokens/sec, total tokens/sec, mean E2E latency, p95 latency, token counts, failures, and the harness audit verdict.

Until that benchmark is committed, the project status remains **experimental / partially closed**, even though the local R9700 execution path itself is already working.

## Reproducibility first

The goal of this repository is not to turn a workstation success into a marketing claim. It is to leave enough evidence that another engineer with comparable RDNA4 hardware can understand what worked, what was changed, what remains unknown, and where upstream support still differs from this experimental path.
