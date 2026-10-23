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

A separate candidate port does not create additional VRAM. The stock service may be stopped only after its image, configuration, endpoint health, and rollback command have been recorded.

## Host safety gate

Before preparing the exclusive 30B window, run:

```bash
python3 scripts/rocm10_1_host_gate.py \
  --output docs/evidence/rocm10_1_host_gate.json
```

The gate is read-only and fails closed when disk, memory, active swap I/O, image identity, stock health, cleanup state, or the candidate port are unsafe. It never pulls an image and never starts or stops a container.

The current minimum thresholds are:

- at least 70 GiB free on `/`;
- at most 85% root usage;
- at least 6 GiB available memory;
- no active storage cleanup;
- exact ROCm 10.1 image already present;
- stock container and `127.0.0.1:8000` healthy;
- candidate port `18029` free;
- no active swap thrashing above 1 MiB/s.

See `docs/ROCM_10_1_STORAGE_MEMORY_GATE.md` for the exact sequence, cleanup boundaries, AMD-only device flags, and rollback requirements.

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
