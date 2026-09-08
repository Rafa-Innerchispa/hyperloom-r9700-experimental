# HyperLoom on Radeon AI PRO R9700: Experimental Compatibility Progress

Updated: 2026-09-08

## What is working

We have a reproducible experimental path running on a real AMD Radeon AI PRO R9700 (`gfx1201`) workstation GPU using ROCm 10, vLLM, and `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.

The current branch is `codex/hyperloom-r9700-master-20260907`.

Verified runtime evidence now includes:

- AMD Radeon AI PRO R9700 / `gfx1201`
- ROCm 10 container runtime
- local OpenAI-compatible vLLM endpoint
- Qwen3-Coder 30B A3B AWQ served locally
- bounded agent E2E evidence with deterministic KEEP/REJECT gating
- completed five-spawn benchmark with per-spawn evidence
- runtime-bound AWQ backend identification for both linear and MoE paths
- reproducible runtime manifest v2 and SHA-256 evidence
- focused local tests and compile checks

## Five-spawn benchmark result

The previously planned bounded five-spawn benchmark has now been executed against the resident local vLLM service without restarting it.

Each spawn measured the same baseline/candidate policy and recorded its own E2E artifact. Across the five spawns:

- all `5/5` spawns completed;
- each arm completed `18/18` valid requests with `0` failures;
- candidate concurrency was `2`;
- aggregate baseline median output throughput was approximately `19.76 output tok/s`;
- aggregate candidate median output throughput was approximately `35.93 output tok/s`;
- median throughput gain was approximately `80.9%`;
- the harness audit passed;
- gate outcomes were `4 KEEP` and `1 REJECT`.

The single rejection is useful evidence that the gate is active rather than decorative. The candidate still completed successfully, but its p95 latency reached approximately `1.307x` baseline, above the allowed `1.25x` limit.

These results are classified as **serving/concurrency scaling**. They are not being presented as a GEAK, Arbor, Triton, or kernel-level optimization win.

## AWQ backend is no longer unknown

A model being labeled AWQ does not imply one universal AWQ kernel. The current Qwen3 MoE model uses different execution paths for dense linear layers and routed expert layers.

We added `scripts/r9700_awq_backend_probe.py`, a permanent observer that binds its result to the running vLLM container and model, inspects the installed dispatch logic, and replays vLLM's backend-selection logic without loading a second copy of model weights.

Current measured result on the R9700:

### Dense / linear AWQ

- `VLLM_USE_TRITON_AWQ=false`
- backend classification: `VLLM_CUSTOM_OP__C_AWQ`
- kernel entry: `torch.ops._C.awq_gemm`

### Qwen3 MoE AWQ

The live configuration is AWQ uint4 / group size 128 with 128 experts, top-8 routing, hidden size 2048, MoE intermediate size 768, and a single-GPU TP/DP/EP topology of 1/1/1.

vLLM's WNA16 backend oracle selects:

- backend: `EMULATION`
- implementation: `vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe.Int4EmulationTritonExperts`

This is valuable because it turns an unknown into a concrete RDNA4 optimization target. Future work can now test whether changes move the MoE path from emulation to a more specialized backend while preserving correctness and latency gates.

## Why this matters

HyperLoom upstream still has open RDNA4 support work. There is a real gap between owning capable RDNA4 hardware and having a documented, auditable way to run, measure, and improve the stack on that hardware.

This project turns one successful workstation setup into a reproducible compatibility and optimization lab: exact hardware, runtime, model, launch configuration, backend selection, evidence files, tests, deterministic gates, and explicit claim boundaries.

The value is not merely that the R9700 can serve a model. The value is that we can now identify where the stack falls back, measure changes against a pinned baseline, reject regressions automatically, and provide evidence useful to future upstream enablement.

## What we are not claiming

This is **not** official AMD-AGI HyperLoom support for the R9700.

We are **not** claiming the serving/concurrency gain is a kernel-level optimization.

We are **not** claiming that every AWQ workload on RDNA4 uses the same backend.

We are **not** treating the Radeon AI PRO R9700 as identical to an RX 9070 XT simply because both are RDNA4-class hardware.

## Current upstream context

At this checkpoint:

- AMD-AGI/Hyperloom issue #1033 for RDNA4 support remains open.
- AMD-AGI/Hyperloom pull request #1032 remains open and unmerged.

Our work should therefore be read as an experimental compatibility, observability, and validation effort that can inform future upstream support, not as a replacement for it.

## Reproducibility tools

Key tools now include:

- `scripts/r9700_multispawn_harness.py`
- `scripts/r9700_upstream_agent_e2e.py`
- `scripts/r9700_awq_backend_probe.py`
- `scripts/r9700_runtime_manifest.py`
- `scripts/r9700_evidence_audit.py`
- `docs/AWQ_BACKEND_OBSERVABILITY.md`

The AWQ observer supports strict mode: if it cannot prove the backend, it fails instead of replacing missing evidence with a guessed label.

## Next optimization target

The strongest next target is no longer “make it run.” It already runs. The next target is to determine why the Qwen3 AWQ MoE path on `gfx1201` falls through to `EMULATION`, identify the closest specialized WNA16 backend that can be enabled safely, and benchmark any change against the pinned five-spawn baseline.

That is a much more useful AMD challenge story: not another generic application layer, but an evidence-driven RDNA4 compatibility and backend-enablement project on real workstation hardware.
