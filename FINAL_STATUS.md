# Final Status

## Current Checkpoint

Correlation: `hyperloom-r9700-master-amd-challenge1-20260907`

Branch: `codex/hyperloom-r9700-master-20260907`

Status: PARTIAL checkpoint. Phase 0 truth boundary, Phase 1 manifest tooling/docs, and Phase 2 harness/tests are implemented and tested. Live multi-spawn execution remains blocked until an explicit benchmark window because it can saturate the resident R9700/vLLM service.

## Evidence Added

- Runtime manifest: `docs/evidence/r9700_vllm_rocm10_launch_manifest_20260907T1525Z.json`
- Runtime manifest SHA-256: `db7e94c7d48a80bf862b6c4fe27ea68cf55906d7bf3d0eadd8eaab4330d2fa97`
- AMD-generated multi-spawn plan: `docs/evidence/r9700_multispawn_plan_20260907_amd.json`
- Windows focused unit tests: `5 passed`
- AMD focused unit tests: `10 passed`
- AMD compile check: `python3 -m compileall scripts/r9700_runtime_manifest.py scripts/r9700_multispawn_harness.py`
- `git diff --check`: PASS

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

## Claims Allowed

- Local OpenAI-compatible vLLM on AMD node `.5` serves `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.
- The target GPU is AMD Radeon AI PRO R9700 workstation GPU, `gfx1201`, when `rocm-smi` reports that identity.
- Saved E2E evidence demonstrates bounded agent candidate selection and deterministic KEEP/REJECT gating.
- The earlier throughput improvement is serving/concurrency scaling.

## Claims Forbidden

- Official AMD-AGI/Hyperloom support for R9700.
- Kernel optimization or GEAK/Arbor win from the concurrency result.
- RX 9070 XT and Radeon AI PRO R9700 being the same product.
- `VLLM_USE_TRITON_AWQ=1` being required on this stack without runtime proof.

## Next 3 Actions

1. Schedule a benchmark window for `scripts/r9700_multispawn_harness.py --execute`.
2. Capture AWQ backend evidence from vLLM logs/profiler before claiming a specific AWQ kernel.
3. Have Rafael/ChatGPT review upstream and hackathon wording before any public post or PR.
