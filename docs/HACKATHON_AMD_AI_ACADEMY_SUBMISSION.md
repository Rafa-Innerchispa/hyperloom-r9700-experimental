# AMD AI Academy Challenge — Submission Package

Checked: 2026-09-13

Target event: **Lablab x AMD AI Academy Challenge** (September 1 – December 1, 2026)

Repository: `https://github.com/Rafa-Innerchispa/hyperloom-r9700-experimental`

Participant model: **individual participation**. The repository is published under the InnerChispa ecosystem, but the challenge submission must be presented as the work of the individual participant, not as a team entry.

> This document is the canonical hackathon-facing narrative for HyperLoom R9700. It does not redefine the technical evidence in the repository.

## 1. Recommended project title

**HyperLoom R9700: Production-Safe Local AI on AMD RDNA4**

Alternative short title if the submission UI has a tighter limit:

**HyperLoom R9700**

## 2. One-line pitch

An experimental, evidence-driven path for running, validating and safely promoting local Qwen3-Coder inference on an AMD Radeon AI PRO R9700 (`gfx1201`) using ROCm, vLLM, AutoAWQ, WNA16/Triton and automatic fallback to a known-good stock backend.

## 3. Short description

HyperLoom R9700 explores how an AMD Radeon AI PRO R9700 can serve a real 30B-class MoE coding workload locally while preserving reproducibility, correctness and operational safety. The project combines ROCm 10, vLLM, Qwen3-Coder-30B-A3B-Instruct-AWQ, AutoAWQ, packed INT4 WNA16 and Triton with an experimental R9700 path, then adds a production-style promotion controller with exact hash preflight, canonical-output readiness checks, a transient health guard and automatic stock fallback.

The result is not a claim of official AMD or upstream HyperLoom support. It is a reproducible experimental integration, measured on real R9700 hardware and designed to fail closed instead of silently serving an unverified backend.

## 4. Long description

### The problem

Local AI builders increasingly have capable workstation GPUs, but getting a large quantized MoE model to run is only the first step. A useful local inference system also needs answers to harder questions:

- Is the exact runtime and optimization bundle reproducible?
- Does an optimized path preserve model correctness?
- Are throughput gains real measurements rather than a benchmark artifact?
- Can an experimental backend be promoted without allowing two runtimes to fight for one GPU?
- If the experimental path disappears, can the system recover automatically to a known-good backend?

The Radeon AI PRO R9700 (`gfx1201`, RDNA4) is particularly interesting for this experiment because the project is intentionally operating outside a claim of official HyperLoom support. That makes evidence, bounded claims and reversibility more important, not less.

### The solution

HyperLoom R9700 builds an experimental end-to-end path around the real local workload:

`HyperLoom -> Radeon AI PRO R9700 -> ROCm 10 -> vLLM -> Qwen3-Coder -> MoE -> AutoAWQ -> WNA16 -> packed INT4 -> Triton RDNA4 -> validated backend -> guarded production promotion`

The work was split into two distinct concerns:

1. **Evidence and optimization.** Establish a reproducible R9700 path, preserve exact configuration/runtime hashes, validate model output against stock behavior, and measure concurrency with an exclusive benchmark lease so overlapping runs cannot fake a result.
2. **Operational promotion.** Treat the experimental backend as something that must earn the right to serve traffic. Promotion is transactional, exact-hash preflight is fail-closed, readiness requires canonical model identity and output, only one backend may own the physical GPU, and a transient guard can automatically restore stock service after loss of the promoted backend.

### Why this matters

A benchmark screenshot is easy. A local AI system that can explain exactly what ran, prove that the result remained correct, survive a failed backend and return to a known-good model is much closer to something people can trust.

The project therefore treats **reasoning/optimization and execution/operations as separate trust boundaries**. Experimental optimization may move quickly; promotion to a serving role is deliberately conservative.

## 5. AMD technology used

- **AMD Radeon AI PRO R9700**, 32 GB VRAM
- **RDNA4 / `gfx1201`**
- **ROCm 10 runtime/container stack**
- PyTorch ROCm build
- vLLM ROCm runtime
- Triton AMD execution path
- AutoAWQ / packed INT4 WNA16 MoE path

Observed final promoted runtime during Phase 6 acceptance included:

- Python 3.14.7
- torch `2.12.0+rocm10.0.0`
- HIP runtime `7.15.26333`
- vLLM `0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`
- model `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- execution reaching `TritonWNA16Experts`

## 6. Model and workload

Primary workload:

`QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`

The project deliberately uses a real coding model and its actual MoE/AWQ path rather than a tiny synthetic model chosen only because it is convenient to demo.

## 7. Architecture

```text
                        +---------------------------+
                        |  HyperLoom / KernelForge  |
                        |  experiment + evidence    |
                        +-------------+-------------+
                                      |
                                      v
+----------------+     +--------------+--------------+
| Phase 5 exact  | --> | R9700 experimental overlay |
| hashes/config  |     | ROCm + vLLM + AWQ/WNA16    |
+----------------+     +--------------+--------------+
                                      |
                                      v
                            +---------+---------+
                            | Radeon AI PRO     |
                            | R9700 / gfx1201   |
                            +---------+---------+
                                      |
                         measured + correctness
                                      |
                                      v
                    +-----------------+-----------------+
                    | Phase 6 promotion controller      |
                    | preflight -> cutover -> readiness |
                    +---------+---------------+----------+
                              |               |
                     PASS     |               | FAIL/loss
                              v               v
                    +---------+----+   +------+---------+
                    | S3 optimized |   | stock backend  |
                    | backend      |   | known-good     |
                    +------+-------+   +----------------+
                           |
                           v
                    transient guard
                           |
                           +---- automatic fallback ---->
```

A single physical R9700 cannot honestly provide simultaneous blue/green GPU ownership. The design therefore uses a transactional `stop -> clean -> verify route/GPU ownership -> start -> validate` cutover instead of pretending both full model servers can coexist.

## 8. Validated evidence

### Phase 5 — optimization/evidence closure

Phase 5 is closed and its evidence is immutable unless a new verified regression requires a targeted check.

Key measured results on the validated workload:

| Measurement | Result |
| --- | ---: |
| Clean exclusive-lock soak | 5/5 PASS |
| Concurrency 1 median aggregate output throughput | 68.34 tok/s |
| Concurrency 4 median aggregate output throughput | 191.92 tok/s |
| Fresh ~6K workload | ~62 tok/s |
| Correctness | stock-exact canonical hashes |

**Important interpretation:** the C1 vs C4 numbers are aggregate throughput under different concurrency. They are **not** a claim that the GPU itself became 2.8x faster per request. Concurrency can increase aggregate throughput while changing latency, so throughput and latency must be evaluated together.

The benchmark harness also found and fixed a concurrency-measurement bug by enforcing an exclusive benchmark lease. That correction is part of the evidence, not an inconvenient footnote to hide.

### Phase 6 — production-style operational closure

Phase 6 did not rerun performance benchmarks. It validated whether the experimental backend could be promoted and recovered safely.

Validated sequence:

`stock -> controlled S3 promotion -> strict canonical readiness -> health guard -> induced S3 loss -> automatic stock fallback -> exact-model recovery -> final S3 re-promotion -> strict canonical readiness -> guard PASS`

Focused Phase 6 validation:

- controller tests: 11/11 PASS
- readiness tests: 4/4 PASS
- combined focused suite: 15/15 PASS
- Python compile checks: PASS
- exact Phase 5 bundle/config hash preflight: PASS
- automatic fallback to stock: PASS
- exact model recovery: PASS
- final S3 re-promotion: PASS
- no orphan S3 container: PASS
- no double GPU ownership: PASS

Final live S3 state during acceptance served the exact model on `127.0.0.1:8000`, used approximately 28.47 GB VRAM and passed the post-promotion guard.

## 9. Interesting engineering findings

### TCP TIME_WAIT is not a serving backend

An early port-availability check used `bind()` and could mistake TCP `TIME_WAIT` for an active route owner. The control plane was changed to probe for a real listener using `connect_ex` semantics.

### A failed service start can leave real resources behind

A failed `ExecStartPost` could terminate the systemd-side `docker run` client while leaving the named container alive. Exact-name cleanup was added so failed promotion cannot silently retain the GPU or port.

### Cold output is not an excuse to weaken correctness

One first cold S3 sample differed from the canonical stock hash. Stock independently reproduced the canonical output, so the gate was not relaxed. The final readiness policy may consume the cold transient but requires **two consecutive canonical outputs** within a bounded number of attempts before promotion succeeds.

### Local-first still needs production discipline

Running locally removes a cloud dependency; it does not remove the need for health checks, evidence, rollback and fail-closed behavior.

## 10. What makes the project different

- It uses a **real R9700 + large quantized MoE coding model**, not a toy inference demo.
- It records exact runtime and overlay hashes so an experimental result can be tied to an exact software state.
- It distinguishes **aggregate concurrency throughput** from raw GPU speed, avoiding a common benchmark overclaim.
- It makes correctness a promotion gate rather than a retrospective check.
- It demonstrates a complete automatic fallback transaction on one physical GPU.
- It preserves a safe boot policy: stock remains the default; experimental S3 remains manual-only.
- It is explicitly experimental and does not present community/RDNA work as official AMD or official HyperLoom support.

## 11. Demo flow

Recommended video length: approximately 2–3 minutes.

### Scene 1 — the hardware and thesis

Show the Radeon AI PRO R9700 and the repository.

Narration idea:

> We wanted to answer a harder question than “can this model run?” Can an experimental RDNA4 optimization path produce reproducible evidence and be promoted like a real service without sacrificing correctness or recovery?

### Scene 2 — exact workload

Show the runtime/model identity:

- R9700 / `gfx1201`
- ROCm 10
- vLLM
- Qwen3-Coder-30B-A3B-Instruct-AWQ
- AutoAWQ/WNA16/Triton path

### Scene 3 — evidence

Show the Phase 5 closure table and make the concurrency caveat explicit:

- C1 median: 68.34 tok/s
- C4 median: 191.92 tok/s aggregate
- 5/5 clean soak
- stock-exact correctness

### Scene 4 — promotion safety

Show the Phase 6 state/controller and explain:

- exact-hash preflight
- only one GPU owner
- exact model readiness
- two consecutive canonical outputs
- transient guard

### Scene 5 — automatic fallback proof

Use recorded evidence rather than inducing another live failure for the camera.

Show:

- guard trigger `service_not_active`
- stock service automatically restored
- exact model healthy again
- no orphan S3 container
- final successful S3 re-promotion

### Scene 6 — close

> HyperLoom R9700 is not a claim of official support. It is an experiment in making local AI optimization measurable, reversible and safe enough to trust.

## 12. Suggested screenshots / submission media

1. **Hero image:** R9700 + `HyperLoom R9700` title + local AI pipeline.
2. **Architecture image:** experiment/evidence -> R9700 runtime -> Phase 6 promotion/fallback.
3. **Evidence image:** Phase 5 C1/C4/soak/correctness table with the aggregate-throughput caveat visible.
4. **Operational image:** stock/S3 state machine and fail-closed promotion flow.
5. **Fallback image:** recorded guard event -> stock recovery -> final S3 re-promotion.
6. **GitHub image:** public repository and final Phase 6 documentation.

Avoid screenshots containing private hostnames, credentials, internal IPs beyond the intentionally local `127.0.0.1` contract, secrets, tokens or unrelated InnerOS coordination data.

## 13. Suggested technology tags

Use only tags available in the challenge UI. Preferred order where available:

1. AMD ROCm
2. AMD Developer Cloud / AMD AI ecosystem, only if actually used in a challenge activity or workload
3. Python
4. PyTorch
5. vLLM
6. Triton
7. LLM inference
8. Local AI
9. AI infrastructure
10. Open source

Do not tag a technology merely to collect visibility.

## 14. Experimental limitations and claim boundaries

The submission should state these plainly:

- This is an **experimental R9700/RDNA4 integration**.
- It is not official AMD support.
- It is not official upstream HyperLoom support for the R9700.
- No “world first” or “first R9700 port” claim is made.
- Phase 5 performance numbers are specific to the tested hardware, software, model, configuration and workload.
- C1/C4 scaling is aggregate throughput scaling, not a universal per-request GPU acceleration claim.
- Phase 6 proves the tested control-plane/fallback path, not general high-availability guarantees for arbitrary systems.
- Stock remains the boot/default path; experimental S3 promotion remains deliberate/manual.

## 15. AI Academy learning and community angle

The AI Academy Challenge is a three-month growth/community program rather than a single weekend build. HyperLoom R9700 should therefore be presented as a continuing learning project with public evidence.

Useful challenge activities derived from this project:

- document what changed between stock and experimental R9700 paths;
- publish a tutorial on reproducible ROCm/vLLM R9700 evidence without overstating benchmark results;
- publish the Phase 6 promotion/fallback design as an operational lesson for local AI;
- share the TCP `TIME_WAIT` and orphan-container findings as practical debugging notes;
- help other AMD community members distinguish aggregate concurrency throughput from raw GPU speed;
- answer Discord/community questions where our evidence is actually relevant;
- continue AMD AI Academy learning modules and connect lessons back to measured R9700 experiments;
- use referrals/community promotion only authentically, not as spam.

## 16. Submission strategy as of 2026-09-13

The official live challenge dashboard currently has submissions open and remains early in the program. Publish a clean project entry early enough to establish the project narrative, then improve it with evidence, tutorial material and demo media as the Academy progresses.

Do **not** wait until the final weeks to explain the project for the first time.

At the same time, avoid pretending the project is “finished forever.” The strongest Academy story is a chronological evidence trail: hypothesis -> experiment -> bugs found -> measurements corrected -> operational safeguards -> community learning.

## 17. Copy-ready “What I built” answer

I built an experimental R9700/RDNA4 local-AI inference path around Qwen3-Coder-30B-A3B-Instruct-AWQ using ROCm, vLLM, AutoAWQ, WNA16 and Triton, then added an evidence and promotion layer so the optimized backend has to prove exact configuration, model identity and canonical output before it can serve traffic.

On the validated Phase 5 workload, clean measurements produced a 68.34 tok/s median at concurrency 1 and 191.92 tok/s aggregate median at concurrency 4, with a 5/5 exclusive-lock soak and stock-exact correctness hashes. Those figures are concurrency throughput measurements, not a claim that the GPU became 2.8x faster per request.

I then built Phase 6 as a production-style safety layer for one physical R9700: transactional cutover, exact-hash preflight, strict readiness, single-GPU ownership, transient health guard and automatic fallback to a known-good stock service. A controlled S3 service loss triggered automatic stock recovery with the exact model, after which S3 was successfully re-promoted and guarded again.

The project is deliberately framed as experimental community work. I do not claim official AMD support, official HyperLoom R9700 support, universal speedups or a world-first port.

## 18. Copy-ready “What I learned” answer

The biggest lesson was that optimizing local AI is not just a kernel or tokens-per-second problem. Measurement itself can lie if concurrent benchmarks overlap, a port can look occupied because of TCP state even when no server owns it, a failed service transition can leave a container holding the GPU, and a cold first inference can differ from a steady-state canonical output.

Those failures changed the project. I added an exclusive benchmark lease, real listener probes, exact-name container cleanup, immutable configuration hashes and a readiness policy that requires two consecutive canonical outputs instead of relaxing correctness when a cold sample disagrees.

The AMD R9700 experiment therefore became as much about trustworthy execution as acceleration: preserve the exact evidence, make claims narrowly, and make the experimental path reversible.

## 19. Copy-ready “Why AMD?” answer

The Radeon AI PRO R9700 gives this project a useful constraint: a single 32 GB RDNA4 workstation GPU powerful enough for a serious local coding model, but different enough from the better-established accelerator paths that assumptions have to be tested rather than inherited.

ROCm, PyTorch, vLLM and Triton make it possible to investigate the full local inference stack instead of treating the GPU as a black box. The result is useful even when an experiment fails, because the project records which layer failed, which configuration produced the result and how the system recovered.

## 20. Reuse for other events

The October **AMD Developer Hackathon: ACT III** is a separate event. This repository and its evidence may later provide a technical foundation for an ACT III project, but the current submission must remain specifically scoped to the **Lablab x AMD AI Academy Challenge**. Do not combine event rules, dates, team rules or submission narratives.

## 21. Canonical evidence references

Start with:

- `FINAL_STATUS.md`
- `docs/R9700_PHASE5_CANARY_CLOSURE_20260913.md`
- `docs/R9700_PHASE6_LIVE_ACCEPTANCE_20260913.md`
- `docs/R9700_PHASE6_PROMOTION_CONTROL_20260913.md`

Canonical post-Phase-6 `main` at package creation:

`801d48b50f61ddae2aa9051eefcca8c0bb02d100`

Any future metric added to the submission should point back to repository evidence. If a number cannot be traced to a run, artifact or documented observation, it does not belong in the pitch.