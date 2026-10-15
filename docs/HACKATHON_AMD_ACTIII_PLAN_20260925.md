# AMD Developer Hackathon ACT III — Preparation Plan

Checked: 2026-09-25

## Status

Participant approval is already confirmed by email dated 2026-09-08. Do **not** re-register unless Lablab/AMD explicitly asks for a new action.

Public schedule currently states:

- Online build: **October 12–17, 2026**
- On-site phase: **October 17–18, 2026**
- Submission deadline: **October 18, 2026 15:00 UTC**
- Tracks: **TBA**

## Core rule

ACT III is not the AMD AI Academy submission.

The Academy documents the ongoing R9700 learning/porting journey. ACT III should turn those lessons into a product-like tool that another AMD developer could use.

## Working concept

### AMD Runtime Doctor / Portability Agent

An evidence-driven agent for AMD AI infrastructure that can:

- identify GPU model / architecture and ROCm capability
- fingerprint Python, PyTorch, HIP, Triton and vLLM
- inspect model quantization and backend selection
- detect fallback vs intended optimized path
- prepare an isolated candidate runtime
- run deterministic correctness gates
- compare candidate vs known-good baseline
- emit a reproducible evidence bundle
- recommend bounded remediation
- keep production runtime untouched until an explicit promotion gate

## Why HyperLoom R9700 is useful here

The R9700 work gives ACT III a real failure corpus rather than invented toy errors:

- unsupported/new architecture assumptions
- vLLM version drift
- WNA16/AWQ backend selection
- packed INT4 interleave
- container/image failures
- exact runtime hashing
- cold-start correctness
- one-GPU ownership
- fallback and recovery

The hackathon product should generalize those lessons.

## What must remain separate

Do not copy the Academy pitch as the ACT III pitch.

Do not claim:

- official AMD R9700 HyperLoom support
- official upstream HyperLoom R9700 support
- world-first status
- universal performance improvements

Do not choose a final track before AMD/Lablab publishes the ACT III tracks.

## Preparation before October 12

- [ ] Track ACT III page for tracks, rules, judging criteria and required technologies
- [ ] Confirm AMD Developer Program account remains connected
- [ ] Claim/verify AMD Developer Cloud credit if still available
- [ ] Create a dedicated ACT III branch or repo area
- [ ] Define a small product contract independent of HyperLoom-specific filenames
- [ ] Convert existing R9700 probes into reusable hardware/runtime checks
- [ ] Create one deliberately broken test environment for demo-safe diagnosis
- [ ] Design a simple judge-facing UI/report
- [ ] Prepare a cloud-capable execution path where feasible
- [ ] Keep physical R9700 as a high-value validation target
- [ ] Create separate ACT III submission copy only after tracks are announced

## Candidate judge demo

1. "Here is an AMD runtime that appears healthy."
2. Agent fingerprints it.
3. Agent discovers the model is on fallback / wrong backend / incompatible stack.
4. Agent explains the exact incompatibility and evidence.
5. Agent builds or selects an isolated candidate.
6. Deterministic correctness gate passes.
7. Intended AMD execution path is proven.
8. Report shows what changed and what remains experimental.

This tells a product story rather than a week-long debugging diary, while remaining grounded in real AMD infrastructure work.
