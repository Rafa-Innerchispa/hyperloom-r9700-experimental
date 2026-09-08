# R9700 AWQ Backend Observability

## Purpose

A model being labeled `AWQ` does not identify a single execution kernel. On the current Radeon AI PRO R9700 / ROCm 10 / vLLM stack, dense linear layers and Mixture-of-Experts layers take different backend paths. This repository therefore records those paths separately and fails closed when it cannot prove them.

## Live measured result

Runtime target:

- GPU: AMD Radeon AI PRO R9700 (`gfx1201`)
- vLLM platform: `vllm.platforms.rocm.RocmPlatform`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- architecture: `Qwen3MoeForCausalLM`
- quantization: AWQ uint4, group size 128, zero point enabled
- topology used by the live server: single GPU, TP=1, DP=1, EP=1

### Dense / linear AWQ path

The installed vLLM reports `VLLM_USE_TRITON_AWQ=false`.

For the live build, `AutoAWQLinearMethod` calls `vllm._custom_ops.awq_gemm` for the small-token path. The installed implementation resolves that call to:

`torch.ops._C.awq_gemm`

The observer therefore classifies the current dense/linear backend as:

`VLLM_CUSTOM_OP__C_AWQ`

For larger token counts, vLLM may dequantize through `torch.ops._C.awq_dequantize` and then perform the matrix multiplication. The observer records the installed dispatch source so the classification is tied to the actual vLLM build rather than inferred from the model name.

### MoE AWQ path

Qwen3-Coder 30B A3B is a Mixture-of-Experts model. vLLM uses its WNA16 backend oracle for those routed expert layers.

Replaying the same oracle with the live model configuration produced:

- 128 local experts
- top-8 experts per token
- hidden size 2048
- MoE intermediate size 768
- AWQ uint4 / group size 128
- TP=1 / DP=1 / EP=1
- routing method `RenormalizeNaive`
- requested MoE backend `auto`

The selected backend is:

`EMULATION`

The selected implementation class is:

`vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe.Int4EmulationTritonExperts`

This is useful evidence, not a failure of the experiment. It identifies an actual optimization target: the current RDNA4/R9700 stack reaches the model successfully but does not select a more specialized WNA16 MoE backend ahead of the emulation path.

## Permanent probe

Run:

```bash
python3 scripts/r9700_awq_backend_probe.py --strict
```

The probe:

1. discovers the running local vLLM/ROCm container;
2. binds evidence to its launch command and local `/v1/models` response;
3. inspects the installed vLLM dispatch functions;
4. reconstructs the live model's MoE configuration without loading a second copy of the model weights;
5. invokes vLLM's own backend-selection logic;
6. records relevant runtime log signals;
7. writes a hashed JSON evidence artifact under `docs/evidence/`.

`--strict` exits non-zero unless the backend evidence is proved. Missing evidence is never converted into a guessed backend.

## Runtime manifest integration

`scripts/r9700_runtime_manifest.py` schema v2 embeds the backend observer result. A manifest can therefore state the exact launch/runtime configuration and the proved AWQ dispatch paths in one artifact.

Run:

```bash
python3 scripts/r9700_runtime_manifest.py
```

## Safety and truth boundary

The observer does not restart vLLM and does not load a second copy of the model weights. It uses local read-only inspection plus a short-lived Python process inside the existing container to replay backend selection.

These results describe the measured InnerChispa experimental R9700 compatibility path. They are not a claim of official AMD-AGI/HyperLoom R9700 support, and the serving/concurrency benchmark is not being presented as a GEAK, Arbor, Triton, or kernel-level optimization win.

## Why this matters for the AMD challenge

The result turns an earlier unknown into a measurable engineering target. The project can now compare future vLLM/ROCm/HyperLoom changes against a known baseline and answer two separate questions:

- Did dense AWQ dispatch change or improve?
- Did the Qwen3 MoE path graduate from `EMULATION` to a more specialized backend on RDNA4?

That makes backend enablement and optimization reproducible instead of anecdotal.
