# ROCm Hyperloom

[![Tests](https://github.com/AMD-AGI/Hyperloom/actions/workflows/tests-coverage.yml/badge.svg)](https://github.com/AMD-AGI/Hyperloom/actions/workflows/tests-coverage.yml)
[![Lint](https://github.com/AMD-AGI/Hyperloom/actions/workflows/lint.yml/badge.svg)](https://github.com/AMD-AGI/Hyperloom/actions/workflows/lint.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

ROCm™ Hyperloom is an autonomous agentic system designed to optimize end-to-end inference workloads
(targeting both host code and GPU kernels) on AMD GPUs. Using advanced AI agents and profiling tools,
Hyperloom analyzes your workload, identifies performance bottlenecks, implements targeted optimizations,
and validates the performance and correctness of the optimizations without requiring manual intervention.
 
The system operates through a sophisticated multi-stage pipeline. First TraceLens, the profiling brain of
the workload understanding stage, consumes traces collected by Magpie (which in turn relies on IntelliKit
for some low-level GPU profiling tools), captures bottlenecks, and derives the roofline targets that seed
the optimization search tree.

Next, Hyperloom employs a self-evolving code optimization engine following an iterative agentic loop (Think
→ Decide → Implement → Benchmark). Arbor intelligently explores the optimization space using a Dynamic
Specialist Agent and Knowledge Base. In parallel to Arbor, GEAK, a multi-agent GPU performance optimizer,
optimizes hot kernels. Once optimizations are identified and validated, Hyperloom prepares
the optimized code and generates a report with all proposed changes and expected performance improvements.
This end-to-end automation enables developers to achieve significant performance improvements while
maintaining code quality and reducing the manual effort traditionally required for GPU optimization.

<p align="center"><img width="600" alt="Hyperloom architecture" src="docs/images/Hyperloom_architecture.png" /></p>

Hyperloom combines:

- Trace analysis, identifying bottleneck kernels and bridge planning through
  [TraceLens](https://github.com/AMD-AGI/TraceLens) Agent (backend support
   from [Magpie](https://github.com/AMD-AGI/Magpie) and
   [IntelliKit](https://github.com/AMDResearch/intellikit))
- Kernel optimization through the
  [GEAK](https://github.com/AMD-AGI/GEAK) backend.
- Agentic search space exploration through
  [Arbor](https://arxiv.org/abs/2606.12563), a tree-based cognition layer
  with dynamic agents, long-horizon campaigns, and self-evolving optimization
  guided by a curated knowledge base of hardware learnings, pitfalls, and
  prior campaign artifacts.

## Supported Features

| Feature | Options |
|------|-------|
| Workload | Inference serving |
| Platform | MI300X, MI325X, MI355X |
| Framework | SGLang, vLLM |
| Kernel Language | HIP, Triton, FlyDSL |
| LLM Backend | Claude |

## Get Started

| Goal | Guide |
|------|-------|
| Set up Hyperloom and run a demo | [Quickstart](examples/README.md) |
| Launch and monitor an optimization | [Run an optimization](docs/how-to/optimize.md) |
| Understand the algorithm | [Optimization loop](docs/conceptual/optimization-loop.md) |

## Documentation

| Topic | Link |
|-------|------|
| ROCm Docs | [Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html) |
| Authentication and credentials | [Authentication & credentials](docs/reference/authentication.md) |
| Environment variables | [Environment variables](docs/reference/environment-variables.md) |
| Components | [Components](docs/components/index.md) |
| Compatibility | [Compatibility matrix](docs/compatibility.rst) |
| Troubleshooting | [Troubleshooting](docs/reference/troubleshooting.md) |
| Operations | [Operations & self-hosting](docs/reference/operations.md) |
| Session output schema | [`session_breakdown.json`](docs/reference/session-breakdown.md) |

## File Issues and Feedback

If you encounter any problem or bugs while running Hyperloom, feel free to open an
[issue](https://github.com/AMD-AGI/Hyperloom/issues/new/choose), or provide us with
feedback on how to improve Hyperloom by completing the
[beta survey](https://www.feedback.amd.com/se/5A1E27D2004A9E15).

---

## Developer Entry Points

- Runtime package: `src/hyperloom/`
- Main agent instructions: [`src/hyperloom/inference_optimizer/SKILL.md`](src/hyperloom/inference_optimizer/SKILL.md)
- CLI entry point: `python -m hyperloom.inference_optimizer.cli optimize`
- Operator tools: `python -m hyperloom.inference_optimizer.tools.*`
- Compute-partition sweep: `python3 scripts/partition_mode_sweep.py` — sets each
  AMD partition mode (`SPX`/`DPX`/`QPX`/`CPX`) on one card in turn, runs the same
  benchmark on every partition that mode creates, sums the throughput and restores
  the entry mode. Answers which shape a workload wants before a session commits to
  one; `optimize` itself only ever reads the mode. Needs privilege for the set, so
  it is a script rather than part of the loop.
- Platform tuning audit: `python3 scripts/platform_audit.py` — checks the host CPU
  tuning that silently changes benchmark results. Judges Core Performance Boost and
  the cpufreq governor against [AMD's BIOS & Workload Tuning Guide for EPYC 9004][58011];
  records determinism, SMT and NPS without a verdict, because chapter 5 varies those
  by workload or the OS layer can only infer them. Reads `/sys`, `/proc` and — as
  root — the HWCR MSR; no credentials, nothing written. Exit `0` on target, `1` a
  knob is wrong, `2` unresolved, which CI should treat as missing coverage rather
  than as a failure. The BIOS-only knobs are not reachable this way; see below.
- BIOS audit over the BMC: `sudo python3 scripts/platform_audit_bmc.py --bmc-user <ro>`
  — covers the three knobs the OS cannot see (High Performance profile, APBDIS, DF
  C-states), targeted per [58011][58011] §4.2.1, §4.4.3 and §4.4.4. Without
  `--bmc-user` it refuses to run unless `--allow-account-creation` is passed, because
  that path **mints a temporary ADMINISTRATOR account on the BMC**; exit `3` means
  such an account was left enabled or could not be confirmed revoked, and should page
  someone. The script's docstring has the account lifecycle and the rest of the exit
  codes.
- Documentation source: `docs/`

[58011]: https://docs.amd.com/v/u/en-US/58011-epyc-9004-tg-bios-and-workload

For contribution workflow, testing, and linting, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Licensing

Hyperloom is released under the **MIT License**. The full license text
is in [`LICENSE`](LICENSE).

You may use Hyperloom commercially, modify it, and distribute it under
the terms of the MIT license, provided the copyright notice and the
permission notice are retained in all copies or substantial portions of
the software.

Third-party tools and agents (Cursor, Visual Studio, and Claude Code)
that Hyperloom invokes are governed by their own separate license terms
and are NOT covered by the MIT license above — see the "Third-Party
Tools and Agents" section in [`LICENSE`](LICENSE). You are responsible
for reviewing and complying with each tool's individual license.

A few files distributed *inside* Hyperloom are also third-party — reference
kernels and a Triton oracle carried in forge's knowledge base and examples.
They keep their own licences; [`THIRD_PARTY.md`](THIRD_PARTY.md) lists them and
`REUSE.toml` carries the machine-readable form.

For security-relevant issues, see [`SECURITY.md`](SECURITY.md). For
contribution conventions, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Experimental Radeon AI PRO R9700 / gfx1201 status

This fork contains experimental RDNA4 work validated on a physical AMD Radeon AI PRO R9700 (`gfx1201`) with ROCm 10, vLLM, and `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.

The current evidence supports these claims:

- three independent process starts reproduced the autonomous serving/concurrency decision (`+80.99%` paired median); this is **not** a kernel-speedup claim;
- the live Qwen MoE path was measured as FP16 `TritonWNA16Experts` with packed INT4 W4A16 weights;
- a custom packed-INT4 small-M W1 Triton kernel beat the stock vLLM Triton WNA16 W1 kernel across the tested real-weight microbenchmark region (about `1.477x` median across M1..16; `63/63` paired wins per shape across three campaigns);
- the complete 30B model booted with the clean v7 `R9700HybridWNA16Experts` backend and real inference observed `48` custom-path plus `48` stock-fallback routes;
- the clean final gate used stock `ROCM_ATTN` plus `GPU_MAX_HW_QUEUES=1` as the stable comparison control;
- clean v7 C4 measured `149.891` and `150.842 tok/s` versus a current fair stock+queue1 control of `157.490 tok/s` and a prior clean stock+queue1 median of `158.490 tok/s`;
- all four C4 response hashes matched stock exactly, while one dedicated strict deterministic probe did not, so universal exact-output parity is not claimed;
- stock ROCm10 serving was restored after the experiment and `/v1/models` returned HTTP 200.

Final experimental verdict: **KERNEL KEEP / FULL-MODEL INTEGRATION NOT PROMOTED**.

See [`FINAL_STATUS.md`](FINAL_STATUS.md), [`docs/R9700_PROJECT_CONTINUITY.md`](docs/R9700_PROJECT_CONTINUITY.md), [`docs/evidence/r9700_v7_final_gate_summary_20260911.json`](docs/evidence/r9700_v7_final_gate_summary_20260911.json), [`docs/DEVELOPMENT_LEDGER.md`](docs/DEVELOPMENT_LEDGER.md), and [`docs/COMMUNITY_FEEDBACK_LEDGER.md`](docs/COMMUNITY_FEEDBACK_LEDGER.md) for the evidence boundaries and preserved negative results.

This work does not claim official AMD support, upstream HyperLoom support for the R9700, a “first port,” or a full-model `1.477x` speedup.
