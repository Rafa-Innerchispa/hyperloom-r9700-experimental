# Final Status

## Current Checkpoint

Correlation: `hyperloom-awq-observer-20260908`

Canonical target branch: `codex/hyperloom-r9700-master-20260907`

Proof branch: `chatgpt/hyperloom-r9700-awq-proof-20260908`

Status: **CLOSED FOR THIS EXPERIMENTAL CHECKPOINT.** The bounded live R9700 benchmark completed, the runtime AWQ backend observer is implemented, and the measured backend paths are now captured as reproducible evidence. This remains experimental RDNA4/R9700 enablement work, not a claim of official upstream HyperLoom support.

## What Was Completed

- Live bounded multi-spawn benchmark: **5/5 spawns completed**.
- Each arm: **18/18 requests passed**, zero request failures.
- Candidate concurrency: **2**.
- Median baseline output throughput: **19.764 output tok/s**.
- Median candidate output throughput: **35.935 output tok/s**.
- Median throughput gain: **~80.9%**.
- Candidate output throughput range: **28.55-36.31 tok/s**.
- Candidate total throughput range: **86.28-109.74 tok/s**.
- Candidate p95 E2E range: **926.97-1189.64 ms**.
- Gate decisions: **4 KEEP / 1 REJECT**.
- The rejected spawn was functionally successful but exceeded the p95 gate at approximately **1.307x baseline**, so the rejection is expected and correct.
- Harness audit: `audit_ok: true`.

## AWQ Backend Evidence

### Dense / Linear AWQ

Measured live configuration:

- `VLLM_USE_TRITON_AWQ=false`
- Backend classification: `VLLM_CUSTOM_OP__C_AWQ`
- Primary GEMM entry: `torch.ops._C.awq_gemm`
- Large-token dequantization path can use `torch.ops._C.awq_dequantize`

This classification comes from introspecting the installed live vLLM implementation and runtime environment. It is not inferred merely from the model being AWQ-quantized.

### MoE AWQ

Model architecture and routing facts observed:

- Architecture: `Qwen3MoeForCausalLM`
- Hidden size: `2048`
- MoE intermediate size: `768`
- Local experts: `128`
- Experts per token: `8`
- `norm_topk_prob=true`

The vLLM WNA16 MoE selector chose:

- Backend: `EMULATION`
- Experts implementation: `vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe.Int4EmulationTritonExperts`

This is now a concrete optimization target: determine why earlier specialized WNA16 backends are not eligible on `gfx1201` / Radeon AI PRO R9700, then test a safe specialized path if feasible.

## Evidence Files

- `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260908T035235773281Z.json`
- `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260908T035310633550Z.json`
- `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260908T035355104232Z.json`
- `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260908T035434219313Z.json`
- `docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260908T035519562264Z.json`
- `docs/evidence/r9700_multispawn_plan.json`
- `docs/evidence/r9700_awq_backend_probe_20260908T040657Z.json`
- `docs/evidence/r9700_vllm_rocm10_launch_manifest_20260908T040845Z.json`

Observer probe SHA-256:

- `d654988e2c02f1cebe125e118b5700437b3d0689416ded120c1db73bb290822c`

Runtime manifest SHA-256:

- `c674caea87dc5efe4281c620d30a12eeebbe9662cfddd57de5d0d886e6c9c513`

## Runtime Observed On AMD

- Host: `ralfiia-amd`
- Endpoint: `http://127.0.0.1:8000/v1`
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- GPU: `AMD Radeon AI PRO R9700`, `gfx1201`
- Docker image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- In-container Python: `3.14.7`
- In-container torch: `2.12.0+rocm10.0.0`
- In-container `torch.version.hip`: `7.15.26333`
- In-container vLLM: `0.27.1.dev5+gf46a9dfe2.d20260827`

## Validation Performed

- New observer unit tests: **6/6 passed** using standard-library `unittest`.
- `compileall` for `scripts/r9700_awq_backend_probe.py`: PASS.
- `compileall` for `scripts/r9700_runtime_manifest.py`: PASS.
- Live AWQ backend observer: `evidence_status=proved`.
- Live runtime manifest: AWQ evidence `proved`.
- Multi-spawn harness audit: `audit_ok: true`.
- Cached Git diff check before commit: PASS.

A full historical pytest suite is **not** claimed for this checkpoint. A focused pytest attempt encountered the repository-level conftest dependency on `httpx`; the new observer tests were therefore also validated directly with `unittest` instead of pretending a broader suite passed. Humanity survives another truthful test report.

## Claims Allowed

- Local OpenAI-compatible vLLM on AMD node `.5` serves `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` on Radeon AI PRO R9700 / `gfx1201`.
- The bounded five-spawn benchmark completed successfully and shows reproducible serving/concurrency scaling under the recorded gate.
- The measured dense AWQ path is the vLLM custom `_C` AWQ path with `VLLM_USE_TRITON_AWQ=false`.
- The measured WNA16 MoE selector chose `EMULATION` / `Int4EmulationTritonExperts` for the reconstructed live configuration.
- The current concurrency result is a serving/runtime result, not proof of a custom kernel optimization.

## Claims Forbidden

- Official AMD-AGI/HyperLoom support for Radeon AI PRO R9700.
- A GEAK, Arbor, Marlin, Machete, or other specialized kernel win unless separately measured and evidenced.
- That AWQ inherently implies Triton, Marlin, Machete, or any other specific backend.
- RX 9070 XT and Radeon AI PRO R9700 being the same product.
- That the MoE `EMULATION` selection is a failure; it is a measured fallback and an optimization target.

## Next Engineering Targets

1. Trace the WNA16 eligibility checks that lead `gfx1201` / R9700 to `EMULATION` rather than a more specialized backend.
2. If technically safe, enable one specialized MoE backend behind a narrow experimental path and benchmark it against this pinned evidence baseline.
3. Package the reproducible observer, benchmark evidence, and truth boundaries as the core technical story for the AMD AI Academy Challenge.
