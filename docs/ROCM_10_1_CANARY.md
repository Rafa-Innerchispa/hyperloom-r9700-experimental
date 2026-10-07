# ROCm 10.1 side-by-side canary on Radeon AI PRO R9700

The stock ROCm 10 runtime remains the operational baseline.

Two ROCm 10.1 candidates are kept separate:

1. **PyTorch compatibility canary**
   - `rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0`
   - used for lightweight framework/GPU smoke tests.

2. **vLLM benchmark canary**
   - `rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0`
   - AMD's documented ROCm 10.1 + vLLM 0.29 pairing.

## Coexistence rule

Both container images can exist on the host at the same time. The PyTorch smoke canary can be launched while stock is online if it does not allocate meaningful VRAM.

The full 30B vLLM comparison must use an exclusive GPU benchmark window. The current stock model can consume nearly all R9700 VRAM, so running two equivalent 30B servers concurrently would either fail or produce useless performance comparisons.

## Promotion gate

ROCm 10.1 does not replace stock until it passes:

- exact correctness/output gates;
- C1/C4 throughput;
- TTFT and cold/warm behavior;
- VRAM/OOM behavior;
- repeated process starts;
- sustained soak;
- clean stock restore.

Use `python3 scripts/rocm10_1_canary.py pytorch` or `vllm` to print the candidate plan. The script is planning-only and does not mutate the host.
