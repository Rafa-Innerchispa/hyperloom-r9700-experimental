# HyperLoom Radeon AI PRO R9700 / RDNA4 — Final Experimental Status

Last reconciled: 2026-09-11 22:48 America/Guayaquil / 2026-09-12 03:48 UTC
Repository: `Rafa-Innerchispa/hyperloom-r9700-experimental`
Active engineering branch: `chatgpt/r9700-phase3-int4-repack-20260911`
Hardware: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
Workload: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
Runtime: ROCm 10 + vLLM + Triton

## Final verdict

**PHASE 3 S3 FULL-MODEL EXPERIMENTAL PROMOTION: PASS FOR BATCHED C4 / CONCURRENT SERVING**

**Stock ROCm10 remains the operational default until an explicit deployment decision is made.**

Phase 3 closes the full-model performance gap that Phase 2 did not close. The selected candidate combines the relevant vLLM #43389 RDNA INT4/W4A16 MoE repack/interleave path with an R9700-specific tuned `int4_w4a16` MoE configuration, stock `ROCM_ATTN`, and `GPU_MAX_HW_QUEUES=1`.

Across three independent fresh candidate processes, the conservative first-measurement C4 median is `188.598166 tok/s`. This is:

- about `+19.0%` versus the established stable stock+queue1 baseline (`158.489996 tok/s`), and
- about `+13.9%` versus a freshly restored healthy-hot stock observation from the same closure campaign (`165.577595 tok/s`).

The hot-repeat C4 median is `192.460522 tok/s`, about `+16.2%` versus the freshly restored healthy-hot stock observation.

This is a real full-model serving result for the tested C4 workload. It is **not** a universal acceleration claim: C1 is near parity with healthy stock and long-context decode is within roughly `1.7-3.1%` of the freshly restored healthy-hot stock observation. The selected candidate is therefore promoted as the best experimental R9700 configuration for batched/concurrent C4 serving, not installed as the operational default and not described as faster in every workload regime.

Canonical final gate: `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`.

## Phase 3 implementation

Phase 3 backports the relevant RDNA INT4/W4A16 MoE repack/interleave work into the stable ROCm10 image without editing the installed runtime in place.

Runtime patch SHA-256:

`3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`

Candidate controls:

- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- attention: stock `ROCM_ATTN`
- stability control: `GPU_MAX_HW_QUEUES=1`
- fresh independent process for each canonical start
- stock systemd unit inactive during candidate runs
- clean VRAM before launch
- patched vLLM files mounted read-only into disposable containers

Real Qwen AutoAWQ validation before serving passed for repacked weights, qzeros, scales and sampled dequantization on the tested layer. The Phase 3 candidate uses the real full model, not a synthetic-only approximation.

## R9700-specific S3 MoE config

Canonical config:

`docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`

Observed config SHA-256:

`8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`

Mounted at the vLLM config name:

`E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Selected shape configuration:

- M1: `BM16 BN64 BK32 GROUP1 SPLIT1`, warps4, stages2, waves4
- M2/M4/M8/M16: `BM16 GROUP1 SPLIT1`
- M32: `BM32 GROUP1 SPLIT1`
- M64: `BM64 GROUP1 SPLIT1`

This tuning is the main difference between the untuned Phase 3 C4-focused candidate and the selected S3 configuration that recovers a much better small-M / long-context balance.

## Three independent S3 process starts

Canonical aggregate:

`docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`

### First measurement from each fresh process

C1:

- median: `69.496803 tok/s`
- min: `66.908667`
- max: `70.214798`

C4:

- median: `188.598166 tok/s`
- min: `186.601539`
- max: `193.662339`
- range / median: about `3.74%`

~6K-context decode:

- median: `61.801768 tok/s`
- min: `58.185291`
- max: `63.134073`

All canonical candidate correctness requests returned the stock hash:

`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

The four canonical C4 outputs also matched the stock hashes exactly on all selected S3 starts.

### Hot repeat from each process

C1 median: `67.749036 tok/s`

C4:

- median: `192.460522 tok/s`
- min: `186.492550`
- max: `193.948262`
- range / median: about `3.87%`

~6K-context decode median: `62.675115 tok/s`

The best single C4 observation, `193.948262 tok/s`, is retained as evidence but is **not** the headline result. The conservative headline is the independent-start first-measurement median of `188.598166 tok/s`.

## Fresh stock restore control

After the candidate campaign:

- the Phase 3 candidate container was removed;
- R9700 VRAM returned to roughly 60 MB before stock restart;
- `inneros-vllm-canary-rocm10.service` was restored;
- `/v1/models` returned HTTP 200;
- the normal Qwen3-Coder model loaded successfully;
- stock correctness hash remained canonical.

Fresh restored-stock first measurement:

- C1 `68.964437 tok/s`
- C4 `162.909696 tok/s`
- long-context `63.161537 tok/s`

Fresh restored-stock hot measurement:

- C1 `69.455893 tok/s`
- C4 `165.577595 tok/s`
- long-context `63.752833 tok/s`

Against that stronger same-session hot stock control, the selected S3 medians are approximately:

First-measure S3 median:

- C1: `+0.06%`
- C4: `+13.90%`
- long-context: `-3.06%`

Hot-repeat S3 median:

- C1: `-2.46%`
- C4: `+16.24%`
- long-context: `-1.69%`

This is why the final claim is a strong batched/concurrent C4 improvement with near-parity in the other measured regimes, not universal acceleration.

## Cold first-request behavior

Every fresh S3 process showed a reproducible first-request latency artifact on the dedicated correctness request:

- first-request correctness TTFT: roughly `4.61-5.08 s`
- subsequent hot correctness TTFT: roughly `49-54 ms`

The candidate still produced the canonical correctness hash and healthy C1/C4/long-context measurements. The behavior is therefore treated as a cold/lazy compile/cache penalty that must be disclosed and should be addressed before any latency-sensitive production deployment.

The freshly restored stock runtime also showed a similar cold first-request effect before normalizing on its hot measurement, reinforcing that this behavior is not evidence of a candidate-only steady-state correctness regression.

## Long-soak stability

Canonical evidence:

`docs/evidence/r9700_phase3_long_soak_stability_20260912.json`

A prior same-process Phase 3 observation once fell to about `145.12 tok/s`. It is excluded from the canonical independent-start campaign.

A later roughly 10-hour soak did **not** reproduce the 145 tok/s extreme event:

- late single C4: `188.870452 tok/s`
- eight-round C4 median: `178.180645 tok/s`
- minimum: `175.520296`
- maximum: `189.760035`
- deterministic C4 hash vector retained

Telemetry did not support thermal throttling, power collapse or sclk collapse as the cause of the earlier anomalous observation. The long-soak evidence supports persistence of the Phase 3 C4 advantage while also documenting remaining intra-process variability honestly.

## Untuned Phase 3 milestone

Before S3 tuning, three independent fresh Phase 3 starts produced:

- `191.450567 tok/s`
- `189.645424 tok/s`
- `191.426697 tok/s`

Median C4: `191.426697 tok/s`, roughly `+20.8%` versus the stable stock+queue1 control.

However, untuned small-M and long-context performance were weaker:

- C1 median about `60.367 tok/s`
- ~6K-context median about `55.199 tok/s`

The S3 tune was therefore selected not because it maximizes a single C4 number, but because it provides the strongest measured cross-regime full-model configuration while preserving a large C4 advantage.

## Phase 2 historical result remains valid

Phase 2 is not rewritten as a success merely because Phase 3 later found a better path.

The clean Phase 2 v7 hybrid:

- booted the full Qwen3-Coder 30B model;
- reached the custom W1 path in all 48 MoE layers;
- preserved stock fallback;
- served requests;
- but produced a C4 median around `150.366622 tok/s` against a stable stock+queue1 baseline around `158.489996 tok/s`.

Therefore Phase 2 v7 was correctly **NOT PROMOTED**.

That negative result motivated the fresh upstream RDNA search that found the relevant INT4 MoE repack path used by Phase 3.

## Phase 2 W1 microkernel proof

The real-weight custom W1 versus stock Triton WNA16 result remains valid:

- M1: `1.74518x`
- M2: `1.49210x`
- M4: `1.47445x`
- M8: `1.44613x`
- M16: `1.46273x`
- median M1..16: `1.47682x`
- `63/63` wins per tested shape aggregate
- cosine effectively 1.0

Evidence:

`docs/evidence/r9700_wna16_stock_gate_aggregate_20260909T025832Z.json`

This is a **W1 microkernel result only**. It must never be described as a `1.477x` full-Qwen speedup.

## Stable serving baseline and rejected alternatives

The clean serving factorial previously selected stock attention + `GPU_MAX_HW_QUEUES=1` as the stable promotion control:

- stock+queue1 C4 median: `158.489996 tok/s`
- Unified Attention + default queues: `138.764971 tok/s`
- Unified Attention + queue1: `156.753550 tok/s`

Unified Attention did not beat stock+queue1 and remains excluded from the selected Phase 3 configuration.

Other negative/invalid paths remain preserved rather than rewritten:

- AITER/FlyDSL sorting: isolated HSA memory fault
- activation group pre-sum: correct but slower
- early FP16/BF16 mirror: dtype mismatch
- custom algebraic W2: slower than stock/reference
- mmap/SIGUSR1/SIGUSR2 same-process gate: invalid under captured graphs
- recovered live WNA16 override: not promotable
- Phase 2 clean v7 full-model hybrid: functional but slower than stock
- pathological slow stock process starts: never valid promotion baselines

## Presentation-safe claims

You may say:

- An experimental HyperLoom/RDNA4 path runs on a physical Radeon AI PRO R9700 / `gfx1201` with a real Qwen3-Coder 30B AWQ workload.
- Phase 2 validated a real packed-INT4 small-M W1 Triton microkernel improvement but rejected its slower full-model integration.
- Phase 3 backported a directly relevant RDNA INT4/W4A16 MoE repack path, validated it against real Qwen AutoAWQ weights, and reproduced a full-model C4 improvement across independent fresh processes.
- The selected R9700 S3 configuration achieved a conservative independent-start C4 median of about `188.6 tok/s`, about `+19%` versus the stable stock+queue1 baseline and about `+13.9%` versus a freshly restored healthy-hot stock measurement from the same closure campaign.
- Hot S3 C4 median was about `192.5 tok/s`.
- C1 remained near parity and long-context remained within a few percent of healthy stock under the tested workload.
- Correctness and canonical C4 output hashes matched stock in the selected S3 gate.
- The project preserves negative results, unstable modes, cold-start limitations and rollback evidence rather than hiding them.

Do **not** say:

- official AMD support or official upstream HyperLoom support for R9700
- first R9700/RDNA4 HyperLoom port in the world
- full Qwen is `1.477x` faster
- every workload is 19-21% faster
- `193.95 tok/s` is the universal reproducible result
- the candidate is already installed as the production/default service
- cold-start latency is solved

## Closure

The Phase 3 S3 engineering gate is closed as an **experimental full-model promotion for batched/concurrent C4 serving**.

The selected configuration is preserved in Git with machine-readable evidence, independent-process replication, correctness hashes, long-soak context and a fresh stock restore control. The stock ROCm10 service remains active and healthy as the operational default.

Any future deployment step should be treated as a separate operational decision. The highest-value remaining engineering work is reducing the cold first-request penalty and further tightening C1/long-context parity without sacrificing the independently reproduced C4 advantage.
