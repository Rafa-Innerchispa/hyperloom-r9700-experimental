# HyperLoom R9700 Project Continuity

Last reconciled: 2026-09-12 (America/Guayaquil)

This is the canonical restart file for a fresh ChatGPT/Codex session.

## 0. CURRENT ACTIVE STATE — READ THIS FIRST

**PHASE 4 IS ACTIVE: COLD-START / PARITY OPTIMIZATION.**

Do not restart Phase 2 or Phase 3 experiments. Phase 3 S3 is closed and preserved.

Canonical active handoff:

`docs/R9700_PHASE4_ACTIVE_HANDOFF_20260912.md`

Canonical Phase 4 root-cause note:

`docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`

Current active branch:

`chatgpt/r9700-phase4-coldstart-parity-20260912`

Current Phase 4 task:

`ops_d0c3d4eebd67`

Phase 3 closure SHA:

`321d5d0cff9020e70ba11cc8929e09b80dc3d655`

Phase 4 root-cause commit:

`7c8627ae6c132f83d2b487cc066406e053dd7e88`

Phase 4 active-handoff commit:

`9f1093612b90eeaec164c60c0d177ce52127e44f`

### Exact next gate

1. Start a fresh isolated stock-equivalent process with `--jit-monitor-verbose`.
2. Reproduce the first real inference request that currently incurs ~4.6-5.1 s TTFT.
3. Capture verbose Triton specialization details for the unexpected `prefix_prefill::_fwd_kernel` JIT.
4. Design the minimal reversible startup warmup/precompile for that specialization.
5. Fresh-process A/B until the inference-time JIT warning disappears and first-user TTFT moves toward the hot ~50 ms class.
6. Preserve stock-exact correctness.
7. Apply the mitigation to S3 and confirm Phase 3 C4/C1/long-context behavior is not materially regressed.
8. Restore stock ROCm10 operational default and verify `/v1/models` HTTP 200.
9. Commit raw evidence, negative results and final verdict before closing Phase 4.

### Phase 4 root cause already proven

The ~5 s first-request spike is not S3-specific. It also occurs on freshly restored stock ROCm10 after HTTP readiness. vLLM emits an unexpected Triton JIT warning for `_fwd_kernel` during that first request. The decoder path strongly identifies:

`ROCM_ATTN -> chunked_prefill_paged_decode -> prefix_prefill.context_attention_fwd -> @triton.jit _fwd_kernel`

Do not claim this is solved until a fresh-process A/B removes the inference-time JIT warning and normalizes first-user TTFT.

---

## 1. PHASE 3 S3 CLOSED RESULT — PRESERVE, DO NOT REDO

**PHASE 3 S3 FULL-MODEL EXPERIMENTAL PROMOTION: PASS FOR BATCHED C4 / CONCURRENT SERVING.**

**Stock ROCm10 remains the operational default and was restored healthy at Phase 3 closure.**

Read these Phase 3 artifacts when historical detail is required:

1. `FINAL_STATUS.md`
2. `docs/R9700_PHASE3_S3_CLOSURE_20260912.md`
3. `docs/evidence/r9700_phase3_s3_final_gate_summary_20260912.json`
4. `docs/evidence/r9700_phase3_s3_three_start_aggregate_20260912.json`
5. `docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`
6. `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`
7. `docs/DEVELOPMENT_LEDGER.md`

## 2. Canonical identity

- repo: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Phase 3 branch: `chatgpt/r9700-phase3-int4-repack-20260911`
- Phase 4 branch: `chatgpt/r9700-phase4-coldstart-parity-20260912`
- Phase 3 task: `ops_d07579f84116` (closed)
- Phase 4 task: `ops_d0c3d4eebd67` (active)
- GPU: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201` / 32 GiB
- model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- stock endpoint: `http://127.0.0.1:8000/v1`
- stock service/container: `inneros-vllm-canary-rocm10.service` / `inneros-vllm-canary-rocm10`
- image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Phase 3 runtime patch SHA-256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA-256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- serving controls: stock `ROCM_ATTN`, `GPU_MAX_HW_QUEUES=1`, clean fresh process

Important preservation commits from the Phase 3 closure sequence:

- long-soak evidence: `db41aaf4d5da26340c1afb520101b5577f9b5cbc`
- S3 config: `f11328fe5d58ca6997b423b584c51cfa2a3a5907`
- S3 initial gate: `58c13fb29ffb4693bdf168d47a399d654e81115d`
- detailed active handoff: `4c96b44b955626d4608dc6eb2a6c99051dd553d1`
- pre-closure continuity checkpoint: `222c47501e3f91271efa1b0c8f92bb3145695b2b`
- S3 three-start aggregate: `8459866f9961f0bae7bedd120c99dce8ce9a797b`
- start5 launch evidence: `9b2f7690aec94ecdc99c06a255fbd82a58e25be6`
- start5 first measure: `bfe8a2a11b4d49d6553c985b78c732f6b9344865`
- start5 hot measure: `59705bef2f2914ad3b27ce31a9bc2d1cf0fcd841`
- final S3 machine summary: `2d22dd72f0a938d033cb845fe8c35093f4b54f7c`
- final status closure: `5521740f0a58891265c7d47c4b99590a8307e1d5`
- canonical Phase 3 closure: `321d5d0cff9020e70ba11cc8929e09b80dc3d655`

Always verify current remote HEAD before editing further.

## 3. Final Phase 3 S3 result

The selected candidate combines the relevant vLLM #43389 RDNA INT4/W4A16 MoE repack/interleave path with the R9700-specific S3 MoE config.

Three independent fresh S3 processes passed correctness and reproducibility gates.

### Conservative first-measurement aggregate

- C1 median: `69.496803 tok/s`
- C4 median: `188.598166 tok/s`
- C4 min/max: `186.601539 / 193.662339`
- C4 range/median: about `3.74%`
- ~6K-context median: `61.801768 tok/s`

### Hot-repeat aggregate

- C1 median: `67.749036 tok/s`
- C4 median: `192.460522 tok/s`
- C4 min/max: `186.492550 / 193.948262`
- C4 range/median: about `3.87%`
- ~6K-context median: `62.675115 tok/s`

Combined first+hot C4 median: `190.529344 tok/s`.

Correctness:

- canonical stock hash: `7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`
- selected S3 correctness requests returned that stock hash
- all four canonical C4 response hashes matched stock in the selected S3 gate

## 4. Fresh stock restore control

After S3 evidence capture, the candidate was removed and R9700 VRAM returned to about 60 MB before stock restart.

The stock service was restored successfully:

- `inneros-vllm-canary-rocm10.service`: active/running
- `/v1/models`: HTTP 200
- model loaded successfully
- correctness hash remained canonical

Fresh restored-stock first measurement:

- C1 `68.964437`
- C4 `162.909696`
- long `63.161537`

Fresh restored-stock hot measurement:

- C1 `69.455893`
- C4 `165.577595`
- long `63.752833`

Against that strong same-session healthy-hot stock control:

- conservative S3 first C4 gain: about `+13.90%`
- S3 hot C4 gain: about `+16.24%`
- C1 near parity
- long-context within a few percent

Against the stable stock+queue1 C4 median `158.489996`:

- S3 conservative first C4 gain: about `+19.0%`
- S3 hot C4 gain: about `+21.4%`

## 5. Cold first-request limitation that led to Phase 4

Fresh S3 processes repeatedly showed:

- first dedicated correctness TTFT: roughly `4.61-5.08 s`
- hot correctness TTFT: roughly `49-54 ms`

Freshly restored stock also showed the same class of cold first-request delay. Phase 4 then proved the delay is associated with an unexpected post-readiness Triton JIT compile in the ROCm prefix-prefill path.

## 6. Long-soak evidence

A same-process untuned Phase 3 observation once dropped to about `145.12 tok/s`; it is not used in the canonical independent-start result.

A later roughly 10-hour soak did not reproduce the extreme 145 event:

- late C4: `188.870452`
- 8-round median: `178.180645`
- min/max: `175.520296 / 189.760035`
- deterministic C4 hash vector remained stable

Telemetry did not support thermal throttle, power collapse or sclk collapse as the explanation for the earlier anomaly.

Evidence: `docs/evidence/r9700_phase3_long_soak_stability_20260912.json`.

## 7. S3 config

Canonical config:

`docs/evidence/r9700_phase3_tuned_moe_config_s3_20260912.json`

Mounted as:

`E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Selected config:

- M1: BM16 BN64 BK32 GROUP1 SPLIT1 warps4 stages2 waves4
- M2/M4/M8/M16: BM16 GROUP1 SPLIT1
- M32: BM32 GROUP1 SPLIT1
- M64: BM64 GROUP1 SPLIT1

## 8. Phase 2 remains a valid negative result

Phase 2 custom W1 proof is real:

- M1 `1.74518x`
- M2 `1.49210x`
- M4 `1.47445x`
- M8 `1.44613x`
- M16 `1.46273x`
- median `1.47682x`
- 63/63 wins per tested shape aggregate
- cosine effectively 1

This is a W1 microkernel result only.

The clean Phase 2 v7 full-model hybrid remained roughly 4.5-5.1% below stable stock C4 and was correctly **not promoted**. That negative result remains preserved.

## 9. Worktree preservation

Phase 3 worktree:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-phase3-int4-repack-20260911`

AMD runtime/raw-evidence worktree:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910`

Historical AMD research worktree:

`/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__hyperloom-r9700-amd-live-verify`

Do not `git clean`, hard reset, or delete these before inventory/archive.

Known diagnostic trap: `scripts/r9700_phase3_wait_isolated.py` can misleadingly print `container: still_running` if `docker inspect` fails because the named container does not exist. Verify inspect rc/listener/GPU ownership instead.

## 10. Safe claim boundary

Safe:

- Phase 3 selected experimental full-model S3 candidate reproduced about `188.6 tok/s` conservative C4 median across three fresh processes;
- about `+19%` versus stable stock+queue1 and about `+13.9%` versus freshly restored healthy-hot stock;
- hot S3 median about `192.5 tok/s`;
- C1 near parity and long-context within a few percent of healthy stock;
- correctness and canonical C4 hashes matched stock;
- Phase 4 root cause identifies the cold first-request spike as an unexpected ROCm prefix-prefill Triton JIT seen on stock too.

Do not claim:

- official AMD/upstream R9700 support
- first port in the world
- `1.47682x` full-model acceleration
- universal 19-21% acceleration
- best single `193.948` observation as the universal result
- candidate is already deployed as operational default
- cold first-request latency is solved

## 11. Restart rule

A new ChatGPT/Codex session must:

1. read `docs/R9700_PHASE4_ACTIVE_HANDOFF_20260912.md` first;
2. read `docs/R9700_PHASE4_COLDSTART_ROOT_CAUSE_20260912.md`;
3. verify remote HEAD of `chatgpt/r9700-phase4-coldstart-parity-20260912`;
4. inspect active ops task `ops_d0c3d4eebd67`;
5. verify runtime state before any mutation;
6. continue from the exact Phase 4 next gate, not from earlier benchmarks.

Do not repeat closed Phase 2/3 experiments merely to reconstruct context.