# HyperLoom R9700 — Recovery + ROCm 10 / upstream refresh

Date: 2026-09-10
Status: active engineering record
Scope: Radeon AI PRO R9700 / gfx1201, ROCm 10, vLLM, Qwen3-Coder-30B-A3B-Instruct-AWQ, experimental HyperLoom integration.

## Why this document exists

This file is a durable record created after reconstructing project state across GitHub, InnerOS Project Runtime Registry, the physical AMD node, the technical ledgers, Discord/community feedback, and recovered uncommitted evidence. It exists specifically to prevent context loss. Do not delete or rewrite historical failures into successes.

## Current project thesis

The work is NOT obsolete.

The project has two independent layers:

1. HyperLoom system/agent layer: autonomous search, bounded experiments, KEEP/REJECT decisions, reproducible evidence, rollback, and serving/concurrency optimization.
2. Radeon RDNA4 backend layer: kernels, vLLM/AITER integration, tuned configs, attention, MoE/WNA16 behavior, and architecture-specific optimization on gfx1201.

Upstream improvements in vLLM/AITER reduce the amount of manual compatibility patching that HyperLoom must carry. They do not replace HyperLoom. They improve the baseline that HyperLoom must optimize against.

For the hackathon, this is a stronger thesis: HyperLoom should demonstrate that it can observe a changing upstream stack, identify which previously local patches are now redundant, adopt the new baseline, retain only optimizations that still win, and produce evidence-backed KEEP/REJECT decisions.

## What was already proven before this refresh

### Serving/concurrency campaign

The original single-run result (21.80 -> 36.59 tok/s, +67.89%) was superseded after community feedback about the R9700 bimodal process-spawn issue.

Three independent vLLM starts produced:

- baseline: 19.893 / 19.942 / 19.987 tok/s
- candidate: 36.004 / 36.078 / 36.194 tok/s
- paired median gain: +80.99%
- baseline spread: approximately 0.47%
- verdict: KEEP / KEEP / KEEP

This remains a serving/concurrency result, NOT a kernel speedup claim.

### Real WNA16 / packed INT4 work

The real Qwen AutoAWQ MoE path was identified as TritonWNA16Experts with FP16 activations, 128 experts, top-k 8, packed uint8 INT4 weights and group size 128.

A custom RDNA4 W1 path keeps INT4 packed and precomputes correction = zero_point * scale at weight load time, removing zero-point unpack/multiply from the hot path.

Against stock vLLM WNA16 with real checkpoint weights, the custom W1 previously showed approximately:

- M=1: ~1.745x
- M=2: ~1.49x
- M=4: ~1.47x
- M=8: ~1.45x
- M=16: ~1.46x
- global median: ~1.477x
- candidate wins: 63/63 measurements per tested shape aggregated across three campaigns
- correctness cosine: approximately 0.999999

This is a W1 microkernel/routed-W1 result only. It is not a whole-model speedup.

### Full model integration

R9700HybridWNA16Experts booted the real Qwen3-Coder 30B AWQ model. All 48 MoE layers were observed using the experimental custom-small-W1 + stock-W2 route with a stock fallback path available. Deterministic candidate output matched stock in the validated smoke test. Temporary hook restoration was verified in the successful integration test.

### End-to-end result before upstream refresh

The stable C4 comparison was approximately:

- stock: ~159-162 tok/s
- integrated hybrid candidate: ~151-153 tok/s

The integrated backend was therefore approximately 5% slower E2E even though isolated W1 was faster. The previous honest verdict was:

KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED

This negative E2E result must remain visible.

## Recovered uncommitted work on 2026-09-10

The physical AMD Project Runtime worktree still contained post-checkpoint files that had not been committed. They were recovered after a previous reconciliation made the canonical checkout appear clean.

Recovered evidence includes:

- docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json
- docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json
- docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json
- multiple r9700_stock_layout_e2e_single_* JSON files
- multiple r9700_stock_layout_e2e_v3_single_* JSON files
- r9700_same_process_ab_* JSON files
- r9700_same_process_c4_measure_20260909T043617Z.json
- r9700_signal_gate_probe_* JSON files
- r9700_force_stock_restore_* JSON files
- r9700_wna16_stock_layout_real_weight_smoke_20260909T040134Z.json

Recovered scripts include:

- scripts/r9700_vllm_wna16_bounded_tuner.py
- scripts/r9700_vllm_tuned_config_override.py
- scripts/r9700_tuned_config_live_smoke.py
- scripts/r9700_stock_layout_e2e_single.py
- scripts/r9700_stock_layout_e2e_aggregate.py
- scripts/r9700_stock_layout_e2e_v3_pair.py
- scripts/r9700_same_process_ab.py
- scripts/r9700_same_process_c4_measure.py
- scripts/r9700_signal_gate_probe.py
- scripts/r9700_force_stock_restore.py

Temporary runtime-gate artifacts also existed, including r9700_sameproc_*.pth and an empty sameproc directory. These are evidence of the failed same-process switching experiment and must not be reinstalled as runtime hooks. They should be archived/quarantined, not silently deleted.

## What the recovered tuner actually showed

The bounded WNA16 tuner found a candidate config:

- BLOCK_SIZE_M=16
- BLOCK_SIZE_N=64
- BLOCK_SIZE_K=32
- GROUP_SIZE_M=1
- num_warps=4
- num_stages=2
- waves_per_eu=4
- SPLIT_K=1

Its median isolated speedup across M=1,2,4,8,16 was ~1.209x with minimum ~1.084x for the bounded comparison captured in that tuner file.

The first live tuned-config smoke did not observe the tuned path (`tuned_seen=false`) and failed its overall gate despite restoring a healthy stock service.

The second live smoke did observe the tuned config for M=1,2,4,8,16 (`tuned_seen=true`, `only_small_m=true`) but failed its overall correctness/hash gate. Therefore the config is an optimization lead, not a promoted production config.

The bimodal process behavior is visible in these recovered files: equivalent stock requests appear around ~22 tok/s in one process state and ~69-70 tok/s in another. This reinforces the need for independent process starts and GPU runtime-state capture.

## Current physical runtime before refresh

Verified on the real AMD node on 2026-09-10:

- GPU: AMD Radeon AI PRO R9700, gfx1201, 32 GB
- active container: inneros-vllm-canary-rocm10
- image: rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
- model: QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
- max model length: 8192
- gpu memory utilization: 0.82
- dtype: float16
- torch: 2.12.0+rocm10.0.0
- HIP: 7.15.26333
- vLLM: 0.27.1.dev5+gf46a9dfe2.d20260827.rocm100
- Triton: 3.8.0+git4cff872c.rocm10.0.0
- AITER: amd-aiter 0.1.20.post1
- active VLLM_ROCM_* env overrides: none
- active HyperLoom/r9700/sameproc .pth hooks found in site-packages: none

The stable serving container is therefore currently clean of the failed same-process hook experiments.

## What changed upstream

### Vector.sys recommendation that is still valid

The warning about R9700 run-to-run/process-spawn variability remains methodologically important. Single-run comparisons must not be used for final claims. Independent process starts and telemetry remain required.

### Vector.sys recommendation that partially moved upstream

Vector recommended testing AITER Unified Attention and noted that gfx1201 support was gated/unfinished. That recommendation was correct at the time.

Since then, vLLM PR #43615, `[ROCm] Enable AITER and FP8 inference on GFX120x`, was merged on 2026-08-03. It separates RDNA4 AITER Triton support from MI3xx-only CK paths, enables gfx120x support where appropriate, prefers ROCM_AITER_UNIFIED_ATTN on gfx12 when AITER is available, and defaults AITER RMSNorm off on gfx12 because that kernel is not safe there.

vLLM v0.28.0 is the current stable release as of this audit and contains an explicit RDNA4 helper path (`is_aiter_found_and_supported_on_rdna4`).

AITER also published v0.1.21.post2 on 2026-09-09. The project README now documents Radeon AI PRO R9700/gfx1201 as an experimental support target rather than an entirely unsupported architecture.

Important nuance: not every AITER operator is automatically safe or first-class on gfx1201. The open AITER issue #3294 still documents incomplete general support, and community reports still show model/operator-specific limitations. Unified Attention support and generalized AITER MoE/CK/FlyDSL support are not the same thing.

### Meaning for HyperLoom

We should NOT keep a local patch whose only purpose is to bypass a gate that upstream now handles correctly. That would be technical debt masquerading as innovation.

Instead:

1. refresh the baseline to a current RDNA4-capable vLLM/AITER stack while keeping ROCm 10;
2. test Unified Attention on the real Qwen AWQ workload;
3. keep unsafe/non-RDNA4 AITER subpaths disabled unless validated independently;
4. tune the actual INT4 WNA16/MoE shapes used by Qwen on the R9700;
5. rerun HyperLoom custom W1 against this stronger stock baseline;
6. let HyperLoom KEEP only optimizations that still improve the refreshed baseline;
7. run E2E C4 + long-context campaigns with multiple process starts, TTFT/TPOT/throughput, SCLK/GFX state and power;
8. restore stock and verify health after every campaign.

## Hackathon framing after upstream changes

The project becomes more compelling, not less:

Old framing:

"We patched HyperLoom/vLLM so it runs on an unsupported R9700 path."

Stronger framing:

"HyperLoom is an autonomous optimization system running on physical RDNA4 that can continuously reconcile upstream changes, discover which local workarounds are obsolete, generate/test architecture-specific candidates, reject regressions, preserve reproducible evidence, and retain only measurable wins on real Qwen workloads."

That is closer to the actual HyperLoom/agentic-optimization thesis and much more defensible in a hackathon review.

## Updated completion gates

Do not declare the R9700 work fully complete until all of these are closed:

1. recovered uncommitted evidence is permanently archived/versioned;
2. refreshed vLLM/AITER RDNA4 baseline is tested on ROCm 10;
3. AITER Unified Attention is tested on the real Qwen AWQ workload, including long context;
4. unsafe AITER paths remain disabled unless independently proven;
5. gfx1201 WNA16/MoE tuned config is regenerated/validated against current upstream behavior;
6. custom HyperLoom W1 is rebenchmarked against the refreshed stock baseline;
7. at least three independent process starts per final arm are used;
8. C4 and long-context E2E metrics include TTFT/TPOT/throughput plus GPU clock/power state;
9. correctness/fallback/rollback remain PASS;
10. stable stock serving is restored and healthy;
11. ledgers/README/FINAL_STATUS contain only evidence-backed claims;
12. final commit is pushed and remote SHA verified.

## Primary upstream/community references

- vLLM PR #43615: https://github.com/vllm-project/vllm/pull/43615
- vLLM PR #46192 historical Unified Attention proposal: https://github.com/vllm-project/vllm/pull/46192
- vLLM releases: https://github.com/vllm-project/vllm/releases
- ROCm AITER gfx1201 tracking issue #3294: https://github.com/ROCm/aiter/issues/3294
- AITER releases: https://github.com/ROCm/aiter/releases
- AITER repo: https://github.com/ROCm/aiter
- ROCm/vLLM optimization guidance supplied by Vector: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html
- Original community report supplied by Vector: https://www.reddit.com/r/ROCm/comments/1uaedpw/2_radeon_ai_pro_r9700_rdna4gfx1201_on_vllm_0221/
- R9700 serving reference supplied in the AITER issue discussion: https://github.com/andysalerno/r9700-serving

## Preservation rule

Every new experiment must produce a timestamped evidence file and update DEVELOPMENT_LEDGER.md in the same work session. Failed experiments are evidence and must be retained with FAIL/PARTIAL status. Temporary runtime hooks must be quarantined after capture and must never become an invisible dependency of the stable server.

## Fresh upstream verification added after the initial recovery record

### ROCm 10 target status

AMD ROCm 10.0.0 now lists the Radeon AI PRO R9700/R9700S/R9600D family as RDNA4 `gfx1201` in the official ROCm 10 device tables and compatibility material. This means the base GPU/runtime target itself is an official ROCm 10 target. It does NOT mean HyperLoom or every vLLM/AITER kernel is officially supported on R9700.

### AITER current support boundary

Current AITER README explicitly lists `AMD Radeon AI PRO R9700 | gfx1201 (RDNA4) | Experimental`. The RDNA footnote states that Triton and most FlyDSL kernels run, as do most HIP kernels including normalization, RoPE, quantization, activation and some GEMM/attention, while most CK and ASM kernels remain CDNA-only.

Therefore the correct policy is capability-by-capability validation, not a global `AITER=works` or `AITER=does-not-work` label.

### Current vLLM release boundary

vLLM v0.28.0 is the latest stable release found during this audit. Its published ROCm artifact is built for ROCm 7.2.2, so it is NOT a blind drop-in replacement for the existing ROCm 10 canary. We should port/cherry-pick the relevant RDNA4 support into the ROCm 10 environment or build an equivalent current vLLM/AITER stack against ROCm 10, then validate it on the physical R9700.

### Current AITER release boundary

AITER v0.1.21.post2 was published on 2026-09-09. The physical ROCm 10 container currently has `amd-aiter 0.1.20.post1`. Because published binary wheels target ROCm 7.x families, the ROCm 10 path should use a compatible ROCm 10 package if AMD supplies one in the canary channel or build the current AITER source against the ROCm 10 environment rather than downgrading the node.

### R9700 queue/spawn issue

ROCm issue #6347 remains open and describes process-start bimodality and later degradation on R9700. Independent community work also reports `GPU_MAX_HW_QUEUES=1` as a promising single-GPU stabilization workaround in related RDNA4 queue/firmware scenarios. This is NOT yet accepted as a universal fix in our project. It is now an explicit bounded experiment: compare default queue behavior versus `GPU_MAX_HW_QUEUES=1` across multiple independent process starts with identical model/config and GPU telemetry.

### Consequence for the hackathon

The experimental contribution is no longer 'make ROCm see an R9700'. ROCm 10 already does that officially. The contribution is the agentic optimization and validation system above a rapidly changing RDNA4 software stack: detect capability changes, retire obsolete bypasses, tune missing gfx1201 shapes, benchmark real workloads, preserve negative evidence, and retain only optimizations that survive comparison with the newest safe upstream baseline.

### Live capability probe of the existing ROCm 10 container

A read-only probe against the actually running `inneros-vllm-canary-rocm10` container confirmed:

- R9700 detection: PASS
- ROCm platform detection: PASS
- `on_gfx12x()`: true
- AITER package present: true (`amd-aiter 0.1.20.post1`)
- generic `is_aiter_found_and_supported()`: false
- newer `is_aiter_found_and_supported_on_rdna4` helper: absent
- newer `on_rdna4` platform helper: absent in this build
- no active `VLLM_ROCM_*` override was present

This proves that the current limitation is specifically the older vLLM capability/gating layer, not failure to detect the R9700 and not absence of AITER. The refreshed upstream path should replace this obsolete gating logic rather than stacking another permanent local bypass on top of it.

Evidence: `docs/evidence/r9700_current_capability_probe_20260910.json`.
