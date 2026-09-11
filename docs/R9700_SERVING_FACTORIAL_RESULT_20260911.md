# R9700 Serving Factorial Result — 2026-09-11

## Purpose

This experiment separates two settings that had been changed together in the earlier ROCm 10 refresh campaign on the physical Radeon AI PRO R9700 / `gfx1201`:

1. RDNA4 AITER Unified Attention (`ROCM_AITER_UNIFIED_ATTN`)
2. `GPU_MAX_HW_QUEUES=1`

The goal is to identify the fastest **stable and correct** serving baseline before the final full-model HyperLoom W1 integration gate.

## Methodology correction

The first version of the serving harness stopped the stock Docker container directly. The stock runtime is actually managed by `inneros-vllm-canary-rocm10.service`, which can automatically restart the container. Systemd logs proved that restart attempts could overlap a candidate load or measurement.

Therefore:

- the preliminary `stock + queue1` measurement is preserved but excluded from clean statistics;
- the older Unified Attention three-start candidate campaign is preserved as historical evidence but is no longer used as clean causal or promotion evidence;
- all results below were rerun with the stock **systemd unit stopped through the authorized host-ops plane**;
- each candidate launch verified that the stock container was not running and that used VRAM had returned below 5 GiB before starting a fresh process;
- each launch recorded its exact image, command line, and relevant environment variables;
- all valid measurements used the same Qwen3-Coder 30B AWQ model and the same ROCm 10 image.

This matters because a benchmark against an auto-restarting second model process is not a benchmark. It is an accidental stress test wearing a lab coat.

## Runtime under test

- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- dtype: FP16
- max model length: 8192
- GPU memory utilization: 0.82
- stock MoE backend: `TritonWNA16Experts`
- dedicated correctness reference SHA-256: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

## Clean results

| Serving cell | Clean C4 runs (tok/s) | C4 median | C4 range / median | C1 median | ~6K median | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stock attention + `GPU_MAX_HW_QUEUES=1` | 158.567959 / 158.412033 | **158.489996** | **0.098%** | 63.892816 | 58.868927 | PASS / PASS |
| Unified Attention + default queues | 136.900788 / 140.629155 | **138.764971** | 2.687% | 66.869576 | not promoted* | PASS / PASS |
| Unified Attention + `GPU_MAX_HW_QUEUES=1` | 157.035577 / 156.471523 | **156.753550** | 0.360% | 62.660559 | 55.999820 | PASS / PASS |

\* The Unified-Attention/default-queue long-context runs were not stable as an output-length benchmark: one run produced an abnormally short/different long completion while the dedicated deterministic correctness probe still matched the canonical hash. The arithmetic median of the long decode rates is intentionally not used as a product/performance claim.

Healthy fast stock reference, observed immediately before this factorial: **162.102053 tok/s C4**. Historical stock/no-queue starts also included a fast observation of **165.699 tok/s**, alongside low-mode starts near 72 tok/s.

## Comparisons

Computed directly by `scripts/r9700_factorial_summary.py` from only the six clean JSON measurements:

- `stock + queue1` versus healthy-fast stock reference: **-2.2283% C4**
- Unified/default versus `stock + queue1`: **-12.4456% C4**
- Unified/queue1 versus `stock + queue1`: **-1.0956% C4**
- Unified/queue1 versus healthy-fast stock reference: **-3.2995% C4**

## Interpretation

### `GPU_MAX_HW_QUEUES=1` is the useful stability control

Two clean independent starts with stock attention and `GPU_MAX_HW_QUEUES=1` landed at 158.568 and 158.412 tok/s C4, only about 0.10% apart. This is dramatically more repeatable than the previously observed stock/no-queue process-start distribution, which alternated between roughly 72 tok/s and 160+ tok/s.

The evidence supports this bounded statement:

> On this R9700/ROCm10/vLLM/Qwen AWQ configuration, `GPU_MAX_HW_QUEUES=1` strongly reduces the observed process-start throughput bimodality and provides a stable C4 operating point around 158.49 tok/s.

It does **not** prove that queue1 increases peak throughput. The current healthy-fast stock observation remains about 2.23% faster.

### Unified Attention does not improve the stable baseline

Unified Attention without queue1 produced 136.90 and 140.63 tok/s C4. Adding queue1 recovered stability and much of the throughput, but the resulting 156.75 tok/s median was still about 1.10% below stock attention + queue1.

Therefore Unified Attention is not selected for the final HyperLoom serving gate. This supersedes the earlier tentative interpretation that Unified Attention itself was responsible for the stable 157-158 tok/s campaign. That earlier campaign changed both Unified Attention and `GPU_MAX_HW_QUEUES=1` and was additionally vulnerable to stock systemd auto-restart contamination.

## Baseline selected for the final HyperLoom gate

Primary stable serving baseline:

`stock attention + GPU_MAX_HW_QUEUES=1`

C4 reference distribution from clean runs:

`158.567959 / 158.412033 tok/s`, median `158.489996 tok/s`.

The final HyperLoom candidate must also be compared against the credible healthy-fast stock ceiling, currently `162.102053 tok/s` in the direct same-day probe and `165.699 tok/s` in the earlier independent stock campaign. A candidate is not promoted merely because it beats a deliberately stability-limited denominator.

## Raw clean evidence

Stock + queue1:

- `docs/evidence/r9700_factorial_stock_queue1_clean1_20260911T022250Z.json`
- `docs/evidence/r9700_factorial_stock_queue1_clean2_20260911T023018Z.json`

Unified Attention + default queues:

- `docs/evidence/r9700_factorial_unified_defaultq_clean1_20260911T023713Z.json`
- `docs/evidence/r9700_factorial_unified_defaultq_clean2_20260911T024424Z.json`

Unified Attention + queue1:

- `docs/evidence/r9700_factorial_unified_queue1_clean1_20260911T025257Z.json`
- `docs/evidence/r9700_factorial_unified_queue1_clean2_20260911T030032Z.json`

Machine-readable aggregate:

- `docs/evidence/r9700_factorial_clean_summary_20260911.json`

Reproducible harness:

- `scripts/r9700_factorial_cell_launcher.py`
- `scripts/r9700_factorial_stock_queue1.py`
- `scripts/r9700_factorial_unified_defaultq.py`
- `scripts/r9700_factorial_unified_queue1.py`
- `scripts/r9700_factorial_summary.py`

## Next promotion gate

Do not run the current AMD `v6` hybrid as the final candidate. The audited AMD worktree copy still contains the invalid mmap/signal runtime-gate machinery.

The next gate is:

1. preserve and inspect `v7_clean`, which removes the invalid runtime gates while retaining the proven W1 path and alignment-reuse improvement;
2. version the exact candidate source and smoke evidence before restarting a model process;
3. run the full Qwen 30B candidate with stock attention + `GPU_MAX_HW_QUEUES=1` in a clean process;
4. measure C1, C4, ~6K context, TTFT/E2E and deterministic correctness;
5. repeat in a second independent process if the first candidate is competitive;
6. compare against both the 158.49 tok/s stable baseline and the 162.10-165.70 tok/s healthy-fast stock observations;
7. promote only if the clean full-model candidate repeatedly matches or beats the credible stock baseline without correctness, stability, or rollback regressions.

Until that gate passes, the project verdict remains:

**`KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED`**
