# HyperLoom R9700 Community Feedback Ledger

Canonical external-feedback register for the experimental Radeon AI PRO R9700 (`gfx1201`) HyperLoom work.

Last reconciled: 2026-09-08
Canonical engineering branch at reconciliation: `codex/hyperloom-r9700-master-20260907`

## Purpose

External feedback must not live only in Discord threads, screenshots, or ChatGPT summaries. Every meaningful recommendation is tracked here with a stable ID, source, technical interpretation, implementation status, evidence, and next action.

Duplicate reposts are deduplicated by author + technical content + referenced links. A repost in another Discord channel is recorded as another occurrence of the same feedback item, not as a new recommendation.

Status vocabulary:

- `VALIDATED`: recommendation addressed and supported by reproducible evidence.
- `IMPLEMENTED_PENDING_RUN`: tooling/code exists, but the decisive live run is still pending.
- `PARTIAL`: some requested evidence exists, but one or more requested measurements remain open.
- `INVESTIGATED`: recommendation was technically evaluated and bounded; no production change was made.
- `DEFERRED`: deliberately not pursued now, with a documented reason.
- `PENDING`: not yet closed.

## Recovered people / sources

### ROCm AI Assistant (Discord app)

Recovered occurrence: 2026-09-04.

Main feedback:

- HyperLoom official targets are Instinct/CDNA, not Radeon AI PRO R9700.
- `gfx1201` is supported by the broader ROCm/vLLM Radeon stack, but that does not imply official HyperLoom support.
- Do not assume the observed throughput gain is a GEAK/Arbor kernel win.
- Document which HyperLoom component actually fired.
- Document the attention backend, whether AITER is active, the vLLM version, and relevant server details.
- Be aware that HyperLoom knowledge/tuning guidance is calibrated primarily for Instinct/CDNA.
- A Radeon vLLM warmup regression was mentioned for some older vLLM releases; current runtime version must be recorded before applying that advice.
- Multi-GPU/RCCL behavior on Radeon PCIe systems should not be assumed to match Instinct systems.

### Guo Hongwei `[HACK]`

Recovered occurrence: 2026-09-07/08.

Main feedback:

- The original `21.80 -> 36.59 tok/s` comparison changed concurrency from 1 to 2, so it should be described as aggregate throughput scaling, not a port/kernel optimization.
- Publish vLLM version, quantization/precision, launch parameters, and repeated runs.
- Report throughput, TTFT, and end-to-end latency together.
- Cross-reference upstream HyperLoom RDNA4 work:
  - https://github.com/AMD-AGI/Hyperloom/issues/1033
  - https://github.com/AMD-AGI/Hyperloom/pull/1032
- Preserve the distinction between GPU detection/roofline enablement and true E2E RDNA4 support.

### Vector.sys `[AMD]`

Recovered occurrences:

- detailed comment in the `Experimental Hyperloom support...` thread, 2026-09-07/08;
- repost/reference of the same detailed comment in another channel after the project was posted across multiple server channels.

These occurrences are treated as **one canonical feedback set**, not two different reviewers.

Main feedback:

- ROCm/ROCm#6347 may confound single-process R9700 measurements because vLLM decode can land in a fast/slow process-spawn state.
- Repeat baseline/candidate across independent vLLM process spawns, not only repeated requests inside one process.
- Study the 4x-R9700 report that uses repeated runs to filter spawn variability.
- Evaluate AITER Unified Attention / the gfx1201 AITER enablement path as a bounded Phase-2 target.
- Study missing/pre-tuned Triton configs for gfx1201, but verify that the exact tuning mechanism applies to the model/quantization in use.
- Do not prioritize FP8 KV-cache as an optimization target without evidence on this exact stack.
- Study:
  - https://github.com/ROCm/ROCm/issues/6347
  - https://forum.level1techs.com/t/vllm-0-23-x-rocm-7-14-upgrade-on-4x-radeon-ai-pro-r9700-50-regression-on-one-model-unaffected-on-another-but-a-free-9-via-kernel-tuning/253460
  - https://www.reddit.com/r/ROCm/comments/1uaedpw/2_radeon_ai_pro_r9700_rdna4gfx1201_on_vllm_0221/
  - https://discuss.vllm.ai/t/rdna4-fp8-support/2610
  - https://github.com/ROCm/aiter/issues/3294
  - https://github.com/bluefalcon13/vllm-rocm
  - https://www.reddit.com/r/ROCm/comments/1u3slct/gfx1201_enablement_rebuilding_aiter/
  - https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html

## Recommendation reconciliation

| ID | Recommendation | Source | Status | Current conclusion / evidence | Next action |
|---|---|---|---|---|---|
| FDBK-001 | Do not call concurrency scaling a kernel/port optimization. | Guo, Vector, ROCm AI Assistant | VALIDATED | Current gain is classified as autonomous serving/concurrency optimization by Qwen, not GEAK/Arbor. | Preserve this claim boundary in README/submission. |
| FDBK-002 | Repeat baseline/candidate, preserve request-level evidence, independently audit the gate. | Guo, Vector | VALIDATED | Harness runs 3 rounds per arm, 6 requests/round, with request-level evidence and independent audit. Five completed harness executions produced 5/5 successful runs and zero request failures. | Keep evidence immutable. |
| FDBK-003 | Eliminate ROCm #6347 process-spawn lottery by using independent vLLM process starts. | Vector | IMPLEMENTED_PENDING_RUN | `scripts/r9700_independent_process_harness.py` and tests exist. Same-process five-run evidence is explicitly not treated as proof against #6347. | Execute bounded independent-process benchmark and publish result. |
| FDBK-004 | Publish vLLM version, quantization, model and launch parameters. | Guo, ROCm AI Assistant | VALIDATED | Runtime manifest/probes capture ROCm/vLLM/PyTorch versions, model `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`, AWQ config, max model length and launch command. | Surface compact version in public README. |
| FDBK-005 | Publish throughput + TTFT + E2E latency. | Guo | PARTIAL | Throughput, mean E2E and p95 E2E are captured. TTFT is not yet part of canonical published benchmark evidence. | Add TTFT to decisive independent-process run or explicitly state why unavailable. |
| FDBK-006 | Cross-reference upstream HyperLoom RDNA4 issue/PR and respect integration boundary. | Guo | VALIDATED | #1033 and PR #1032 were reviewed. Project does not claim official upstream support or merged RDNA4 support. | Re-check upstream state immediately before final publication. |
| FDBK-007 | Identify which HyperLoom component actually produced the current gain. | ROCm AI Assistant, Guo | VALIDATED | Current ~80% median gain is a Qwen-selected serving/concurrency decision, not a proven GEAK/Arbor kernel optimization. | Phase 2 may target a real kernel/backend change. |
| FDBK-008 | Identify attention backend and whether AITER is active. | ROCm AI Assistant, Vector | VALIDATED | `scripts/r9700_attention_backend_probe.py` proved `ROCM_ATTN` selected; `TRITON_ATTN` is valid alternative; AITER is installed but current vLLM reports it unsupported for this R9700 path. Evidence: `docs/evidence/r9700_attention_backend_probe_20260908T142410Z.json`, SHA-256 `2d31cfdae1ff1752e86c2cbdc0ef36f0e6515301cb81c39a7ae140cd5d680bf8`. | Keep probe in final public branch. |
| FDBK-009 | Check Radeon vLLM warmup regression and version range. | ROCm AI Assistant | INVESTIGATED | Current runtime is vLLM 0.27.x, newer than the range mentioned in feedback. No workaround is justified from that comment alone. | Verify exact upstream source before citing publicly. |
| FDBK-010 | Evaluate AITER Unified Attention / gfx1201 enablement. | Vector | INVESTIGATED | Current image contains AITER, but installed vLLM support gate rejects this Radeon path and image AITER arch list is not `gfx1201`. Simply setting a flag would not be a clean proof. | Treat as Phase-2 experimental branch, not final baseline mutation. |
| FDBK-011 | Generate/use gfx1201 Triton tuning configs for possible ~7-9% gain. | Vector | INVESTIGATED | Referenced community gain used FP8 model/shapes. Current model is AWQ INT4 and dense AWQ uses vLLM `_C.awq_gemm`; direct transfer is not justified. | Focus first on observed WNA16 MoE emulation fallback and tune only matching shapes/backends. |
| FDBK-012 | Avoid chasing FP8 KV-cache without evidence. | Vector | DEFERRED | Current project uses AWQ and has higher-value unresolved RDNA4 backend work. FP8 KV is outside finalization critical path. | Revisit only in separate benchmark matrix. |
| FDBK-013 | Investigate actual MoE backend instead of inferring from `AWQ`. | Vector/AITER follow-up | VALIDATED | Runtime probe proves dense AWQ path `VLLM_CUSTOM_OP__C_AWQ` / `torch.ops._C.awq_gemm`, while Qwen3 MoE WNA16 selects `EMULATION` with `Int4EmulationTritonExperts`. | Investigate why specialized gfx1201 MoE backend is ineligible; do not call fallback a kernel win. |
| FDBK-014 | Compare MoE fallback with active upstream gfx1201 AITER limitations. | Follow-up research | INVESTIGATED | AITER gfx1201 tracking/issues are relevant to observed fallback, but equivalence/root cause is not proven. | Cross-reference exact shapes/config before causal claim. |
| FDBK-015 | Make public repo show current evidence, not only 2026-09-04 numbers. | Internal publication audit prompted by community review | PENDING | Default branch `main` still presents older `21.80 -> 36.59 (+67.89%)` checkpoint. Canonical engineering work lives on separate history. | Publish clean final branch based on `main`, then merge without force-rewriting history. |

## Known measured checkpoint before independent-process validation

These numbers are valid as same-resident-vLLM-process serving/concurrency evidence, not as proof against ROCm #6347:

- completed harness executions: 5/5
- requests per arm per execution: 18
- request failures: 0
- candidate concurrency selected by Qwen: 2
- median baseline output throughput: ~19.764 tok/s
- median candidate output throughput: ~35.935 tok/s
- median gain: ~80.9%
- gate decisions: 4 KEEP, 1 REJECT

The rejected run is useful evidence: it completed successfully but exceeded the allowed p95 latency ratio, so the gate rejected it rather than blindly choosing the faster aggregate-throughput result.

## Feedback ingestion rule

From this checkpoint forward:

1. Every external technical comment gets a stable `FDBK-###` ID or is linked as a duplicate occurrence of an existing ID.
2. Preserve source identity/handle, date/channel when known, and original links.
3. Convert advice into a testable engineering statement.
4. Track status and evidence; never mark addressed because it was merely discussed.
5. If feedback is intentionally not followed, record the technical reason.
6. Before public release, all `PENDING`, `PARTIAL`, and `IMPLEMENTED_PENDING_RUN` items must be reviewed explicitly.
7. Discord reposts do not create duplicate engineering tasks.

## Recovery limitation

As of 2026-09-08, the connected InnerOS Discord bot does not index the external Discord server/thread where all community discussions occurred. Historical recovery therefore depends on messages preserved in chats or explicitly supplied by the user. The currently recovered distinct external sources are ROCm AI Assistant, Guo Hongwei, and Vector.sys. This is a known ingestion gap, not evidence that no other commenter exists.


## 2026-09-10 final reconciliation

The feedback ledger is now reconciled against the later Phase-2 evidence and final E2E truth boundary.

- **FDBK-007 / bimodal R9700 process state:** CLOSED. The methodology switched to independent process starts for the Phase-1 serving/concurrency result. Concurrency-1 E2E kernel comparisons are not used naively because the R9700 can enter materially different performance states.
- **FDBK-008 / Unified Attention:** INVESTIGATED / NOT VALIDATED FOR THIS WORKLOAD. The probe observed the platform selecting the Triton attention path and an explicit Unified-Attention attempt did not establish a supported end-to-end path. No UA performance claim is made.
- **FDBK-009 / gfx1201 AITER status:** UPDATED. AITER now publicly describes Radeon AI PRO R9700 / gfx1201 support as experimental. Our targeted AITER/FlyDSL sorting experiment encountered an isolated HSA memory fault, so it is not promoted into this workload path.
- **FDBK-010 / AITER quant/sorting:** ATTEMPTED / PARTIAL-FAIL. The failing experiment is preserved as evidence rather than silently retried or presented as success.
- **FDBK-014 / missing gfx1201 tuning configs:** INVESTIGATED BUT NOT CLOSED. A bounded tuner and live config-override experiment were created after the checkpoint, but their raw JSON/scripts were never committed and are no longer present in the audited worktrees. Their numerical outcome is therefore not reconstructed and cannot support promotion.
- **Kernel-versus-serving claim separation:** CLOSED AS POLICY. The `+80.99%` Phase-1 result is explicitly a serving/concurrency result. The approximately `1.477x` Phase-2 result is explicitly a small-M routed W1 microkernel result versus stock Triton WNA16. Neither is represented as an end-to-end full-model kernel speedup.
- **Full-model promotion gate:** NOT PASSED. The hybrid backend boot/correctness/fallback/rollback gate passed, but the more stable concurrency-4 E2E comparison remained approximately 5% below stock after the valid alignment-reuse improvement. Final verdict: `KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED`.

This ledger therefore closes the recommendation-review cycle without manufacturing a favorable benchmark. Future work may revisit gfx1201 tuning or graph-recapture-safe integration, but it starts from this recorded verdict rather than repeating already closed experiments.
