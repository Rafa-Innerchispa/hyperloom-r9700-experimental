# AMD Programs Status — 2026-09-25

This document separates the two active AMD/Lablab opportunities so the project does not mix rules, dates, claims, or submission narratives.

## 1. Lablab x AMD AI Academy Challenge

**Presented by:** AMD × NativelyAI × lablab.ai  
**Dates:** September 1 – December 1, 2026  
**Format:** 3-month AMD-powered developer growth and community program  
**Prize pool:** USD 5,000  
**Participation:** individual only  
**Current public structure:** five Mini-Challenges plus XP from broader Quest activity such as Discord/community/referrals/hackathon participation. Mini-Challenge grading is performed by AMD.

### Enrollment status

The participant was approved by email on **2026-09-04** for the **AMD Lab Program**. The approval explicitly states that this is not a one-off hackathon but a three-month quest of AMD-powered challenges.

Challenge 1 in that approval is:

> **Build Your First AI App on AMD**

with two paths:

- **Starter** — guided path
- **Builder** — original project

The same approval requires an AMD Developer account connected to lablab.ai for XP tracking.

### Does HyperLoom R9700 fit?

**Yes, technically it is a strong fit for the Builder path.**

HyperLoom R9700 is an original AMD infrastructure/AI project using a physical Radeon AI PRO R9700 (RDNA4 / gfx1201), ROCm, PyTorch, vLLM, Triton and a real 30B-class MoE coding model. The project is not merely running a demo model: it is documenting compatibility gaps, kernel/backend behavior, correctness, reproducibility, promotion/fallback safety and the practical work required to make a modern local AI stack behave reliably on AMD hardware.

The Academy framing should therefore be:

**hypothesis -> build -> incompatibility discovered -> investigation -> patch/port -> physical evidence -> correctness -> performance -> operational safety -> public learning**

That framing matches the program's stated "Learn. Build. Share. Grow." structure better than treating the work as a single benchmark screenshot.

### Current technical lane

The original evidence remains valid for the exact historical software stack that produced it. Do not rewrite old results.

Current active experimental lane as of 2026-09-25:

- physical AMD Radeon AI PRO R9700 / gfx1201
- ROCm 10
- Python 3.13
- PyTorch 2.13
- ROCm Triton pin f0b55c0
- vLLM 0.30.0
- Qwen3-Coder-30B-A3B-Instruct-AWQ
- experimental RDNA WNA16 / packed INT4 interleave path
- fail-closed runtime proof before any performance claim

The vLLM 0.29 candidate is retained as historical development evidence, not the final target.

### Submission validity boundary

The engineering work is aligned with Challenge 1 Builder, but technical relevance alone is not the same thing as a formally valid submission.

Before claiming Challenge 1 completion, the project must still be packaged in the exact currently exposed Lablab submission flow. At minimum the existing Academy checklist expects:

- individual participant identity
- public repository
- clean project description
- AMD technologies actually used
- demo/media when required by the form
- bounded claims linked to evidence
- no claim of official AMD or official upstream HyperLoom R9700 support

If a Mini-Challenge publishes a more specific container/interface/output contract, that challenge-specific contract overrides generic project packaging.

## 2. AMD Developer Hackathon: ACT III

This is a **separate event**. Do not reuse the Academy submission copy verbatim.

**Enrollment:** approved by email on **2026-09-08**. No additional signup is currently needed.  
**Online build:** October 12–17, 2026  
**On-site phase:** October 17–18, 2026  
**Submission deadline:** October 18, 2026 at 15:00 UTC  
**Registration closes:** kickoff, October 12 at 15:00 UTC  
**Tracks:** currently TBA on the public event page

The event is positioned around pushing AI on real infrastructure and uses AMD Developer Program access.

### ACT III project strategy

Reuse the technical foundation, not the Academy story.

Working direction:

**AMD Runtime Doctor / Portability Agent**

A developer-facing agent/tool that can inspect an AMD AI runtime, identify GPU/ROCm/vLLM/model compatibility, run safe correctness and backend-path probes, detect silent fallback, produce reproducible evidence, and recommend or apply bounded fixes in an isolated lane.

HyperLoom R9700 becomes one proving case for that product rather than the entire product.

Potential demo flow:

1. Detect AMD GPU and architecture.
2. Inspect ROCm / PyTorch / Triton / vLLM compatibility.
3. Load or inspect an actual model configuration.
4. Detect unsupported or fallback execution paths.
5. Prepare an isolated candidate path.
6. Run correctness and backend-path gates.
7. Compare against a known-good baseline.
8. Produce a signed/evidence-oriented report.
9. Never promote automatically without an explicit safe gate.

Do not lock the final ACT III story until tracks and judging criteria are published.

## 3. Separation rule

| Program | Story |
| --- | --- |
| AMD AI Academy | Continuous learning and evidence trail from making serious local AI work on AMD R9700 |
| AMD ACT III | New product/tool built from those lessons for validating and improving AMD AI infrastructure |

The repository may share core code/evidence, but event-facing docs, screenshots, demos, dates, requirements and submission claims must remain separate.
