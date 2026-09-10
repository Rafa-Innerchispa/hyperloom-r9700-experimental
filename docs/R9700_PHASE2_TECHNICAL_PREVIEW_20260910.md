# R9700 Phase 2 Technical Preview — Final Experimental Result

Date: 2026-09-10

## Scope

This technical preview documents the experimental HyperLoom path exercised on an AMD Radeon AI PRO R9700 (`gfx1201`) with ROCm 10, vLLM and `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.

It is not an announcement of official AMD or upstream HyperLoom support.

## Architecture actually exercised

`HyperLoom -> R9700/gfx1201 -> vLLM -> Qwen3-Coder 30B AWQ -> MoE -> AutoAWQ -> TritonWNA16Experts -> packed INT4 -> custom small-M W1 Triton kernel -> stock WNA16 W2/fallback`

The live model path was measured rather than inferred. The runtime exposed FP16 activations, 128 experts, top-k 8, group-size-128 INT4 W4A16, packed uint8 W1/W2 tensors, and `TritonWNA16Experts`.

## Kernel result

The custom W1 path precomputes the zero-point/scale correction at load/conversion time while preserving packed INT4 weights. Against the stock vLLM Triton WNA16 W1 kernel, three real-weight campaigns produced approximately:

| Routed M | Median speedup vs stock WNA16 W1 |
|---:|---:|
| 1 | 1.745x |
| 2 | 1.492x |
| 4 | 1.474x |
| 8 | 1.446x |
| 16 | 1.463x |

Across the tested region the median was approximately `1.477x`; the candidate won all `63/63` paired measurements per tested shape across the three campaigns. Numerical agreement remained effectively cosine 1.0.

This result applies to the measured routed W1 microkernel, not to total model throughput.

## Full-model integration result

`R9700HybridWNA16Experts` booted the complete model. During a real deterministic request, all 48 MoE layers exercised the custom-small-W1/stock-W2 path and also the stock fallback path. The candidate response hash matched the stock response. The ephemeral bootstrap hook was removed and the stock server was restored.

Therefore the following gates are PASS:

- complete model load;
- basic deterministic correctness;
- custom-path reachability through all 48 MoE layers;
- stock fallback;
- rollback to the unmodified serving configuration.

## End-to-end result

The R9700 exhibited a strong bimodal process/startup state at concurrency 1, so C1 is not used as a simplistic performance proof. Concurrency 4 was more stable.

Observed stable-region values were approximately:

- stock: `159-162 tok/s` aggregate output throughput;
- hybrid: `151-153 tok/s` after the valid alignment-reuse improvement.

The v3 integration removed one source of duplicate MoE routing/alignment work and reduced the full-model regression from approximately 6-7% to approximately 5%. It still did not reach stock parity.

Final Phase-2 promotion verdict:

**KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED**

## Experiments deliberately excluded from claims

A same-process mmap/signal runtime switch was attempted to avoid spawn-state variation. Signals reached EngineCore, but execution telemetry stayed on `runtime_gate_stock`; the custom path was not observed. The apparent approximately `+0.79%` difference from that run is invalid and excluded.

A later gfx1201-specific bounded-tuning/config-override experiment was also created. Its files were never committed and are no longer present in the audited worktrees, so its exact metrics are not reconstructed. It cannot be used to promote the backend.

## Negative results retained

- AITER/FlyDSL sorting: HSA memory fault in the tested path.
- activation group pre-sum: correct but slower.
- initial FP16/BF16 mirror: dtype mismatch, later resolved by measuring the real live FP16 contract.
- custom W2 algebraic path: slower than stock/reference.
- dynamic same-process backend switch: invalid A/B because graph replay did not execute custom code.

Negative results are part of the technical preview because they define the path that was actually validated and prevent future work from repeating dead ends without a new hypothesis.

## What remains upstreamable/useful

The useful engineering artifact is narrow and concrete: a gfx1201-tested packed-INT4 small-M W1 approach, real-weight benchmark harnesses, live backend/dtype observability, full-model hybrid/fallback integration proof, and a methodology hardened against R9700 process-state variance.

The full hybrid serving backend should remain opt-in research code until a future campaign demonstrates independent-process E2E parity or improvement with raw evidence preserved in Git.
