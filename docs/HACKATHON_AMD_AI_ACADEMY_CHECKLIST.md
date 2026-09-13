# AMD AI Academy Challenge — Execution Checklist

Checked: 2026-09-13

Target: **Lablab x AMD AI Academy Challenge**, September 1 – December 1, 2026.

This checklist is intentionally separate from runtime engineering. Phase 5 and Phase 6 are CLOSED/PASS; do not rerun benchmarks, induce another failure, restart the production path or alter boot policy merely to produce hackathon media.

## A. Submission identity

- [x] Public repository exists: `Rafa-Innerchispa/hyperloom-r9700-experimental`
- [x] Experimental claim boundary documented
- [x] Phase 5 evidence documented
- [x] Phase 6 automatic-fallback evidence documented
- [x] Submission narrative drafted in `docs/HACKATHON_AMD_AI_ACADEMY_SUBMISSION.md`
- [ ] Create/update the Lablab project entry as the **individual participant**
- [ ] Use project title: `HyperLoom R9700: Production-Safe Local AI on AMD RDNA4`
- [ ] Add one-line pitch from the canonical submission package
- [ ] Add repository URL
- [ ] Select only technologies actually used
- [ ] Add public demo/video when ready

## B. Submission quality gate

Before publishing or materially updating the Lablab entry:

- [ ] No claim of official AMD R9700 HyperLoom support
- [ ] No claim of official upstream HyperLoom R9700 support
- [ ] No “first”, “world first” or similar novelty claim
- [ ] C1/C4 results described as **aggregate concurrency throughput**, not raw GPU speedup
- [ ] Every performance number traces to Phase 5 evidence
- [ ] Phase 6 described as control-plane/fallback validation, not a new performance run
- [ ] No private hostnames, credentials, tokens or internal coordination screenshots
- [ ] No claim of simultaneous blue/green GPU serving on one physical R9700
- [ ] Stock boot/default fallback is represented accurately
- [ ] S3 remains described as deliberate/manual promotion

## C. Hero visual

Create a clean 16:9 project cover showing:

- `HyperLoom R9700`
- Radeon AI PRO R9700 / RDNA4
- ROCm + vLLM + Qwen3-Coder
- a compact “measure -> validate -> promote -> guard -> fallback” visual
- experimental wording visible but not dominant

Avoid dense terminal screenshots as the hero image. Humans continue to insist that seeing things is easier than reading logs, so give them one useful picture.

## D. Architecture visual

Create one diagram with two clearly separated layers:

### Evidence / optimization

`HyperLoom -> ROCm -> vLLM -> Qwen3-Coder AWQ/MoE -> WNA16/Triton -> R9700`

### Operational trust

`exact hashes -> stock validation -> transactional promotion -> canonical readiness -> transient guard -> automatic stock fallback`

The diagram should visually communicate the project thesis: **optimization may be experimental; promotion is conservative**.

## E. Evidence visual

Use a small table or chart with:

- clean exclusive-lock soak: `5/5 PASS`
- C1 median aggregate output throughput: `68.34 tok/s`
- C4 median aggregate output throughput: `191.92 tok/s`
- fresh ~6K: `~62 tok/s`
- correctness: stock-exact canonical hashes

Place this sentence directly on or next to the visual:

> C1/C4 are aggregate concurrency-throughput measurements, not a claim of equivalent per-request GPU acceleration.

## F. Demo video, 2–3 minutes

### 0:00–0:20 — thesis

- [ ] Show R9700 / repository
- [ ] State problem: running is not enough; experimental local AI needs evidence and recovery

### 0:20–0:50 — stack

- [ ] R9700 / gfx1201
- [ ] ROCm 10
- [ ] vLLM
- [ ] Qwen3-Coder-30B-A3B-Instruct-AWQ
- [ ] AutoAWQ / WNA16 / Triton path

### 0:50–1:20 — Phase 5 evidence

- [ ] show 5/5 soak
- [ ] show C1 68.34 tok/s
- [ ] show C4 191.92 tok/s aggregate
- [ ] explicitly explain concurrency caveat
- [ ] show stock-exact correctness

### 1:20–2:10 — Phase 6 safety

- [ ] exact-hash preflight
- [ ] one physical GPU / one owner
- [ ] canonical readiness
- [ ] two consecutive canonical outputs
- [ ] guard
- [ ] recorded automatic stock fallback evidence
- [ ] recorded final S3 re-promotion evidence

**Do not induce another failure for the video.** Use the already-recorded accepted evidence.

### 2:10–2:40 — lessons / close

- [ ] mention TIME_WAIT listener bug
- [ ] mention orphan-container bug
- [ ] mention cold correctness finding
- [ ] close on measurable + reversible local AI

## G. Public technical content / XP plan

The Academy is a multi-month learning/community program. Build a public evidence trail rather than uploading one project on the last day.

### Tutorial 1

**Working title:** `Benchmarking Local LLM Throughput on AMD Without Lying to Yourself`

Cover:

- aggregate throughput vs per-request speed
- concurrency and latency
- overlapping-run contamination
- exclusive benchmark lease
- reproducibility

### Tutorial 2

**Working title:** `Fail-Closed Local AI: Safely Promoting an Experimental vLLM Backend on One GPU`

Cover:

- one-GPU transactional cutover
- route ownership
- exact hashes
- canonical readiness
- health guard
- automatic stock fallback

### Tutorial 3

**Working title:** `Three Bugs a Benchmark Screenshot Will Never Show You`

Cover:

- TCP TIME_WAIT vs real listener ownership
- orphan container after failed service transition
- cold first output vs canonical steady state

### Community actions

- [ ] Complete relevant AMD AI Academy learning modules
- [ ] Relate each learning module to a real R9700 experiment or note
- [ ] Share bounded technical findings in AMD/Lablab Discord
- [ ] Answer questions only where evidence is available
- [ ] Link repository evidence instead of arguing from memory
- [ ] Publish tutorials progressively during the challenge
- [ ] Keep community/referral actions authentic and non-spammy

## H. Suggested project-update cadence

Rather than treating December 1 as the first time anyone hears about the project:

1. **Now:** publish/refresh the project entry with current canonical story.
2. **Next evidence update:** architecture image + Phase 5 evidence graphic.
3. **Next learning update:** first benchmark/reproducibility tutorial.
4. **Next operational update:** Phase 6 safety/fallback tutorial.
5. **Demo update:** 2–3 minute video using recorded evidence.
6. **Final challenge period:** adapt to the actual final challenge requirements once Lablab unlocks/publishes them.

Do not invent final-challenge requirements before they are officially published.

## I. Public README follow-up

The repository README is still heavily upstream-oriented. A later docs-only pass should add a concise R9700 experimental section near the top linking to:

- `FINAL_STATUS.md`
- Phase 5 closure
- Phase 6 live acceptance
- AMD AI Academy submission package

That README change should preserve upstream attribution and make it impossible to mistake this experimental fork for official upstream R9700 support.

## J. Separate future event

**AMD Developer Hackathon: ACT III** is a separate October event. Reuse of this technical foundation is allowed as a later planning decision, but:

- do not mix ACT III dates into the AI Academy submission;
- do not import ACT III team rules into this individual challenge;
- do not describe current AI Academy work as an ACT III submission;
- create a separate event-facing document when ACT III work actually starts.

## K. Completion definition

The hackathon-facing package is considered ready for a strong early public entry when all of these are true:

- [ ] Lablab project entry published/updated
- [x] canonical narrative exists
- [x] technical evidence is traceable
- [ ] hero image uploaded
- [ ] architecture/evidence image uploaded
- [ ] public demo video linked
- [ ] at least one public learning/tutorial artifact linked
- [ ] participant identity follows individual-entry rules
- [ ] all claims pass the quality gate above

This checklist can evolve with newly published official challenge requirements, but technical evidence already closed in Phase 5/6 must not be rewritten to chase leaderboard wording.