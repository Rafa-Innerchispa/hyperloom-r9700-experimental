# HyperLoom R9700 canonical handoff — 2026-09-15

This file is the canonical cross-chat handoff for the current HyperLoom R9700 work. It exists in GitHub `main` so the project does not depend on ChatGPT MCP/task continuity being available in every chat.

## Repository

- Repo: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- Canonical branch: `main`
- Current canonical main after PR #15: `0e366e170a62e33c3adc464fd98cf41bdf738d3b`

## Hardware / workload target

- GPU: AMD Radeon AI PRO R9700
- Architecture: `gfx1201` / RDNA4
- VRAM: 32 GiB
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- Goal: validated experimental vLLM source stack for R9700/gfx1201, preserving existing S3 and stock fallback runtimes until a separate manual promotion decision.

## Previously validated runtime baseline

The pre-existing validated S3 runtime remains the known serving baseline and MUST NOT be modified merely to test the new vLLM 0.29 lane.

- Validated S3 image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Validated S3 runtime patch SHA256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- Preserve `ROCM_ATTN`
- Preserve `GPU_MAX_HW_QUEUES=1`
- Preserve deterministic warmup/correctness contract
- Stock remains boot/default fallback
- S3 remains manual-only
- Phase 5 CLOSED/PASS
- Phase 6 CLOSED/PASS
- Real automatic fallback to stock was previously demonstrated under controlled S3 failure

Do not replay Phases 2–6 unless a new verifiable regression exists.

## Known performance evidence from the older validated lane

Phase 4 fresh-process result:

- Candidate C4 median: `190.5969 tok/s`
- Controlled stock + queue1 median: `158.489996 tok/s`
- Observed improvement in that controlled experiment: `+20.26%`
- Correctness hashes matched exactly

This result belongs to the older validated lane. Do NOT present it as proof that the new vLLM 0.29 stack has the same performance.

## vLLM 0.29 source candidate

Exact source identity:

- vLLM tag: `v0.29.0`
- vLLM commit: `98dff2a81d747d1dba01a47f939f48c3526d4206`
- Final R9700 candidate patch SHA256: `372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9`
- Patch file: `docs/evidence/vllm_v0_29_0_r9700_awq_triton_candidate.patch`

The patch changes exactly five intended vLLM source paths and preserves fail-closed behavior.

## PR #13 — source candidate

- PR: `#13`
- Purpose: version the isolated five-file R9700 vLLM source candidate
- Final head before merge: `22ab69655821d249b2a51d00c36d4adb7f886c39`
- Merge SHA: `b9c0d9499e5aece2aab5f3ef963eda82a4ed9c63`
- No live runtime mutation

Local validation before merge included focused tests, complete `scripts/tests`, `compileall`, and `git diff --check`.

## PR #14 — exact clean-source apply gate

- PR: `#14`
- Title: `ci: gate R9700 patch on clean vLLM v0.29.0 source`
- Head SHA: `44c37d7406d83117f9ef90a0005b0484bf19b7c4`
- Merge SHA: `784eafcde78dfe54507cf2b3183dc3e02869f8c9`
- Final patch SHA256: `372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9`

Hosted clean-source result on exact vLLM 0.29.0:

- `apply_check=PASS`
- `apply=PASS`
- exactly five intended paths changed
- `diff_check=PASS`
- `py_compile=PASS`
- direct AutoAWQ -> Triton rejection preserved
- classic non-interleaved WNA16 path preserved
- new interleaved path present
- `runtime_mutation=false`
- `gpu_runtime_tested=false`
- `automatic_promotion_allowed=false`

The hosted gate caught context mismatches in `fused_moe.py`. Those hunks were reanchored to the exact vLLM 0.29 blobs without broadening scope.

Local validation around this phase reached:

- focused candidate + apply-gate tests: `21 passed`
- complete `scripts/tests`: `387 passed`
- `compileall`: PASS
- `git diff --check`: PASS

## PR #15 — physical GPU runtime gate preparation

- PR: `#15`
- Title: `R9700: gate isolated ROCm 10 vLLM 0.29 GPU runtime proof`
- Final head SHA: `5996f370129f3f1105aada51b5a39170552a3528`
- Merge SHA: `0e366e170a62e33c3adc464fd98cf41bdf738d3b`
- Changed files: 7 new files, no deletions
- This PR prepared the runtime proof lane but DID NOT itself prove physical GPU runtime success.

Exact candidate build identity:

- ROCm generation: `10.0`
- Python: `3.13`
- Torch: `2.13.0`
- Base tag: `rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.13.0`
- Base digest: `sha256:c820e27bba8090875760d10b92e52aae790c776a937fa00c4357289dbc0addec`
- ROCm Triton repo: `https://github.com/ROCm/triton.git`
- ROCm Triton validated pin for the matching vLLM 0.29 ROCm Docker recipe: `f0b55c0`
- Rust: `1.95`
- Build arch: `PYTORCH_ROCM_ARCH=gfx1201`
- Isolated candidate port: `18029`
- Isolated candidate container: `hyperloom-r9700-vllm029-gfx1201-candidate`

Files added by PR #15 include:

- `docker/Dockerfile.r9700-vllm029-rocm10`
- `scripts/r9700_vllm029_gpu_runtime_plan.py`
- `scripts/r9700_vllm029_gpu_preflight.py`
- tests for Dockerfile contract, runtime plan, and preflight
- `docs/R9700_VLLM029_GPU_RUNTIME_GATE_20260915.md`

Before merge, all material hosted gates were green:

- Tests + Coverage: PASS
- Lint: PASS
- CodeQL: PASS
- Docs: PASS
- Packaging: PASS
- REUSE: PASS
- Secret Scan: PASS
- Forge E2E: PASS

The repo's general `CI E2E (single-GPU smoke run)` is not evidence of the physical R9700 target because the workflow uses a self-hosted `Hyperloom-e2e-ci` runner and secret-configured external GPU/model metadata without proving `R9700/gfx1201` identity. Do not use that workflow as the physical hardware proof.

## Exact definition of “the new stack works on R9700”

Do NOT make the final claim yet.

The project may say the new vLLM 0.29 stack works on the physical R9700 only after all of these pass in the isolated lane:

1. isolated ROCm 10 source build PASS
2. import/ABI PASS, including pinned ROCm Triton availability
3. exact `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` load PASS on physical `gfx1201`
4. intended interleaved WNA16 path proved, not fallback
5. canonical deterministic correctness hash PASS
6. existing S3/stock runtime remains untouched

Performance measurement comes after correctness/path proof. A performance number is not required to establish correctness, but no speed claim should be made before measurement.

## Required next execution sequence

From canonical `main` at `0e366e170a62e33c3adc464fd98cf41bdf738d3b`:

1. Run the read-only preflight from `scripts/r9700_vllm029_gpu_preflight.py` on the physical R9700 host.
2. Fail closed if R9700/gfx1201 identity, `/dev/kfd`, `/dev/dri`, patch identity, model path, candidate port, or preserved-runtime boundaries are not proved.
3. Build `docker/Dockerfile.r9700-vllm029-rocm10` in isolation.
4. Verify build/import identity: Python 3.13, Torch 2.13.0, HIP present, pinned ROCm Triton present, vLLM 0.29.0, ROCm platform.
5. Launch only the isolated candidate on port `18029`, never ports `8000`, `18018`, or historical Phase 3 `18011`.
6. Load the exact AWQ model on physical `gfx1201`.
7. Use/reuse backend/path probes to prove the intended interleaved WNA16 path rather than a fallback path.
8. Run deterministic canonical warmup/correctness and require exact expected hash.
9. Only after correctness + path proof, measure bounded performance against controlled stock and validated S3.
10. Promotion remains a separate manual decision. Do not alter boot/default fallback automatically.

## Upstream research / current conclusions

Recent upstream review around 2026-09-15 found:

- ROCm 10 officially lists Radeon AI PRO R9700/R9700S as RDNA4 `gfx1201`. This is GPU support, not official support for this exact HyperLoom patch.
- vLLM 0.29.0 is the stable release used for this lane.
- HyperLoom upstream moved its default to vLLM 0.29.0 around this work window, reducing version drift versus upstream.
- vLLM still has a direct rejection for `AutoAWQConfig -> Triton` in its WNA16 oracle path, which is why the local candidate remains necessary.
- Upstream PR `vllm-project/vllm#43389` was the important related ROCm AWQ/GPTQ INT4 repack work during this investigation and was not treated as merged/replacement evidence at the time of the source gate.
- Newer AITER/gfx1201 activity does not justify switching the validated lane to AITER. Keep `ROCM_ATTN` unless new isolated evidence proves a reason to change it.

Do not claim official AMD support for HyperLoom, universal speedup, or “world first”. Position the work as experimental / bleeding-edge.

## Git/worktree cleanup already performed

Cleanup principle: do not delete work that contains uncommitted evidence.

Already done during this session:

- 11 clean historical worktrees from already-closed/merged phases were moved out of the active area to reversible quarantine.
- PR #6 and PR #7 were closed as `superseded`; their branch history was preserved.
- Phase 3 worktree was intentionally preserved because it contained unversioned evidence/handoff material.
- `rocm10-upstream-refresh` was intentionally preserved because it had substantial uncommitted/unversioned material.
- Git worktree metadata pruning was not forced manually when the safe prune operation was unavailable. Do not edit `.git/worktrees` by hand just to make the list prettier.

The cleanup work may need a later safe `git worktree prune` from a trusted local shell after confirming quarantined paths and dirty trees.

## MCP / coordination status

Historical coordination/task identifiers seen during this effort:

- source candidate task: `ops_4685195982f6`
- clean source apply task: `ops_5d02fcbc356e`

These IDs may now be stale because the source apply and runtime-preparation work has already merged into GitHub.

Important continuity rule going forward:

- GitHub `main` is the canonical source of truth for code, handoffs, hashes, PRs and technical state.
- MCP/task coordination is useful as a live orchestration layer, but must NOT be the only place where critical project state exists.
- If ChatGPT MCP access is missing in another chat, start from this file and GitHub `main`; do not guess task state.
- When MCP connectivity is restored, synchronize task state from this GitHub handoff rather than overwriting GitHub truth with stale MCP state.

## Copy/paste starter for a new ChatGPT chat

Use this prompt in a new chat:

> CONTINÚA HYPERLOOM R9700 desde el estado canónico actual. No repitas Phases 2–6 ni trabajo ya cerrado. Primero lee en GitHub `Rafa-Innerchispa/hyperloom-r9700-experimental` branch `main` el archivo `docs/HYPERLOOM_R9700_HANDOFF_2026-09-15.md`. Main canónico después de PR #15: `0e366e170a62e33c3adc464fd98cf41bdf738d3b`. PR #14 y #15 están MERGED. El patch vLLM 0.29 final es SHA256 `372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9`. La siguiente fase NO es más documentación: ejecutar el gate físico aislado en la AMD Radeon AI PRO R9700/gfx1201 con ROCm 10 + Python 3.13 + Torch 2.13.0 + ROCm Triton `f0b55c0` + vLLM 0.29.0, usando puerto 18029 y sin tocar S3/stock. Debes probar build, import/ABI, carga exacta de `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`, camino WNA16 interleaved real y hash canónico. Sólo después medir rendimiento. `ROCM_ATTN` y `GPU_MAX_HW_QUEUES=1` se preservan salvo evidencia nueva aislada. Si MCP no está disponible, usa GitHub como fuente canónica y continúa sin inventar estado MCP.

## Truth boundary

At this checkpoint:

- source candidate: DONE
- exact clean-source apply: DONE / PASS
- runtime-proof tooling/build recipe: DONE / merged
- physical R9700 vLLM 0.29 runtime execution: NOT YET PASS
- production promotion: NOT DONE and not authorized automatically

That is the exact project state to resume from.