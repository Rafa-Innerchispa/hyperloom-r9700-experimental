# AMD 2026 Program Map — Canonical Repository Strategy

Updated: 2026-10-06

This document is the canonical map for the AMD workstreams under the Rafa-Innerchispa GitHub account.

## Non-negotiable structure

There are TWO separate AMD programs:

1. **Lablab x AMD AI Academy Challenge** — three-month program with 5 Mini Challenges.
2. **AMD Developer Hackathon: ACT III** — separate October hackathon.

They must not share submission narratives, deadlines, repos, or evidence claims.

---

## A. AMD AI Academy — parent R&D line

### Research / optimization hub

**Canonical repo:** `Rafa-Innerchispa/hyperloom-r9700-experimental`

Role:
- HyperLoom / ROCm / R9700 research and optimization
- reproducible performance evidence
- correctness gates
- promotion/fallback experiments
- reusable AMD capabilities that may feed Mini Challenges or InnerOS

This is **not** the default repo for every Mini Challenge.

Promotion rule:

```
probe -> measured evidence -> architectural decision -> reusable capability
```

Child repos named `hyperloom-r9700-*` are bounded R&D probes, not standalone hackathon products.

---

## B. Mini Challenge repository map

### Mini Challenge 1

**Dedicated canonical repo:** NOT FOUND as of 2026-10-06.

Existing evidence:
- `hyperloom-r9700-experimental` contains an AMD AI Academy submission package and Academy execution checklist.
- No GitHub repository or commit was found explicitly labelled Mini Challenge 1 / MC1.

Status: **needs classification before creating a canonical MC1 repo**.

Rule:
- Do not silently rename HyperLoom into MC1 unless the official MC1 brief proves that is correct.
- Once the official MC1 scope is identified, create a dedicated repo and preserve provenance back to HyperLoom where applicable.

Planned naming:
`Rafa-Innerchispa/amd-academy-mc1-<official-theme>`

---

### Mini Challenge 2 — OCR

**Canonical repo:** `Rafa-Innerchispa/infralens-ocr-amd`

Public/product name:
**ChispaVision / InfraLens OCR**

Purpose:
- AMD ROCm OCR
- difficult road signs / plates
- Asset Passport product extension
- local-first physical infrastructure workflow

Status:
- technical artifact completed
- final 512-token artifact validated on physical Radeon AI PRO R9700
- public immutable artifact preserved
- submission copy exists in `docs/SUBMISSION.md`

Important:
- `Rafa-Innerchispa/infralens-ocr` is an older duplicate/legacy repo.
- Do not use it as the canonical MC2 source going forward.
- The AMD local checkout may be repaired/re-hydrated, but the final challenge artifact must not be mutated.

Canonical final image:
`us-central1-docker.pkg.dev/innerops-agentic-platform/amd-academy-public/chispavision-mc2:final-512`

Digest:
`sha256:dbfcaf89fc2d47823406c1ceeefef464a98a7e4328226511ab1625c3d721868f`

---

### Mini Challenge 3 — RAG

**Dedicated canonical repo:** NOT CREATED as of 2026-10-06.

Official challenge direction verified from live submissions:
- Retrieval-Augmented Generation
- mixed-filetype corpus
- exact citations / grounded answers
- AMD ROCm container/runtime
- tight query-time budget
- robust handling of PDFs, documents, spreadsheets, code/logs and images

Planned canonical repo:
`Rafa-Innerchispa/amd-academy-mc3-rag`

Architecture direction:
```
documents/images
    -> parsers + ChispaVision OCR where needed
    -> deterministic corpus/index
    -> retrieval
    -> grounded generation
    -> exact citations / provenance
    -> fail-closed unanswerable behavior
    -> AMD runtime metrics
    -> HyperLoom optimization/evidence lane
```

This repo must reuse verified ideas from MC2 and HyperLoom without turning into a copy of either repo.

Immediate priority: **build and validate MC3 now.**

---

### Mini Challenge 4

Official theme not yet preserved in this repository map.

Planned repo naming:
`Rafa-Innerchispa/amd-academy-mc4-<official-theme>`

Do not invent implementation before the official brief is captured.

---

### Mini Challenge 5

Official theme not yet preserved in this repository map.

Planned repo naming:
`Rafa-Innerchispa/amd-academy-mc5-<official-theme>`

Do not invent implementation before the official brief is captured.

---

## C. AMD Developer Hackathon — ACT III

**Canonical repo:** `Rafa-Innerchispa/inneros-amd-act-iii`

This is NOT an Academy Mini Challenge.

Project direction:
- sovereign/local-first agentic compute fabric
- AMD local ROCm + AMD cloud routing
- explicit policy/capability routing
- execution evidence
- cost/privacy/latency reasoning
- replay and verification
- multi-agent execution

ACT III should consume proven capabilities from HyperLoom and the Mini Challenges, but the final submission must remain original to the ACT III build window and respect the pre-hackathon baseline.

---

## D. Repository rules for every Mini Challenge

Each Mini Challenge should have exactly one canonical submission repo.

Required minimum structure:

```
README.md
Dockerfile
src/
tests/
scripts/
docs/
  SUBMISSION.md
  ARCHITECTURE.md
  EVIDENCE.md
```

Every repo must record:
- official challenge brief
- exact grader contract
- AMD hardware/runtime used
- reproducible tests
- measured evidence
- known limitations
- submission copy
- immutable final artifact reference when applicable
- lineage to HyperLoom/InnerOS without claiming pre-existing work as newly built

---

## E. Current priority order

1. Preserve/finalize MC2 submission state. Do not reopen the frozen artifact.
2. Create and build the dedicated MC3 RAG repo.
3. Identify the exact official MC1 brief and split it into a dedicated repo if needed.
4. Capture MC4/MC5 briefs the moment AMD publishes them.
5. Continue HyperLoom as the AMD optimization/evidence backbone.
6. Keep ACT III separate and prepare it as the flagship integrated demo.

## Operating principle

**Academy:** learn, measure and accumulate reusable AMD capabilities through the Mini Challenges.

**HyperLoom:** optimize and prove the AMD execution layer.

**ACT III:** combine the strongest proven capabilities into one surprising, auditable flagship system.

Do not collapse these three levels into one repository again.
