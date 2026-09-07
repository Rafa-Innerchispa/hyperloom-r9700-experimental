# AMD DevQuest -> Radeon AI PRO R9700 -> HyperLoom: Provenance Record

This document records the developer journey behind the R9700 HyperLoom experiment while keeping a strict boundary between externally documented facts, private evidence, and user-attested events.

## Evidence levels

- `DOCUMENTED_EXTERNAL`: supported by a message, email, credential, repository artifact, or other evidence retained outside this repository.
- `DOCUMENTED_REPO`: supported by versioned artifacts in this repository.
- `PRIVATE_EVIDENCE`: evidence exists or was received privately but is not published here.
- `USER_ATTESTED`: reported by the developer but not yet backed by an artifact available for independent review.
- `NOT_CLAIMED`: intentionally excluded from public claims until stronger evidence exists.

## Timeline

### 2026-07-23 - AMD Advancing AI 2026, San Francisco

Status: `DOCUMENTED_EXTERNAL`

AMD event staff notified Rafael Lopez that he was in the **Top 10 of the DevQuest Challenge** and instructed him to report to the DevZone by 4 PM.

A separate SMS received at the event conveyed the same operational instruction: the participant was in the Top 10 leaderboard and should check in at the DevZone with Lindsey Brown by 4 PM.

Public-safe claim:

> Top 10 participant in the AMD DevQuest Challenge at Advancing AI 2026 in San Francisco.

Do not publish a more precise finishing position unless independently documented. The developer recalls finishing 10th, but the retained AMD message currently establishes only `Top 10`.

### Physical Radeon AI PRO R9700 handoff at Advancing AI 2026

Status: `USER_ATTESTED` + `PRIVATE_EVIDENCE_PENDING`

The developer reports that an AMD Radeon AI PRO R9700 was handed over physically at the event after the DevQuest result and that a paper acknowledging receipt was signed onsite.

No copy or photo of the signed handoff document is currently retained in this repository.

Allowed wording:

> The Radeon AI PRO R9700 used in this project was received physically during AMD Advancing AI 2026 following the developer's Top 10 DevQuest participation. The developer reports signing an onsite receipt; a copy is not currently available for public verification.

Stronger wording such as `AMD awarded this R9700 specifically for finishing 10th` should remain `NOT_CLAIMED` until the handoff document, prize record, photograph, or equivalent AMD evidence is recovered.

### 2026-08-04 - AMD ROCm Certified Associate

Status: `DOCUMENTED_EXTERNAL`

The developer received the AMD ROCm Certified Associate credential via AMD/Credly. This credential is relevant context for subsequent ROCm, vLLM and R9700 engineering work.

Public-safe claim:

> AMD ROCm Certified Associate.

### 2026-09-04 onward - HyperLoom R9700 experimental work

Status: `DOCUMENTED_REPO`

This repository documents an experimental HyperLoom path for the AMD Radeon AI PRO R9700 (`gfx1201`). The current repository truth boundary remains explicit:

- physical R9700 execution: experimentally validated for the architecture-neutral/bypass path;
- ROCm 10 + vLLM serving: real local runtime;
- concurrency scaling: measured and reported as serving/workload tuning, not as a kernel win;
- official HyperLoom R9700 support: **not claimed**;
- RDNA4-specific GEAK/Arbor kernel optimization: **not yet claimed**.

See the repository README and evidence artifacts for the technical boundary.

## Narrative that is safe to use

A concise public version is:

> The project grew out of AMD's developer ecosystem. After placing in the Top 10 of the DevQuest Challenge at Advancing AI 2026 in San Francisco, the developer received and began working with a Radeon AI PRO R9700, later earned the AMD ROCm Certified Associate credential, and used that hardware to build a local ROCm 10 + vLLM development stack. That work evolved into this experimental HyperLoom/gfx1201 effort, where community feedback is now being converted into stricter reproducibility, multi-spawn validation and potential upstream contributions.

This wording intentionally avoids claiming official AMD sponsorship of HyperLoom or official R9700 HyperLoom support.

## Claims matrix

| Claim | Status | Public use |
|---|---|---|
| Top 10 AMD DevQuest at Advancing AI 2026 | DOCUMENTED_EXTERNAL | Yes |
| Exact final place was 10th | USER_ATTESTED | Prefer `Top 10` until documented |
| R9700 physically received at Advancing AI 2026 | USER_ATTESTED / PRIVATE_EVIDENCE_PENDING | Yes, with attribution and caveat |
| Signed onsite receipt for R9700 handoff | USER_ATTESTED | Mention only as developer report until recovered |
| AMD ROCm Certified Associate | DOCUMENTED_EXTERNAL | Yes |
| Physical R9700 used for this project | DOCUMENTED_REPO + local evidence | Yes |
| Official AMD HyperLoom support for R9700 | NOT_CLAIMED | No |
| Current throughput gain is a kernel optimization | NOT_CLAIMED | No |
| Current throughput gain is serving/concurrency scaling | DOCUMENTED_REPO | Yes |

## Evidence recovery backlog

If available later, preserve without publishing sensitive data:

1. photo or scan of the signed R9700 handoff paper;
2. AMD prize/hardware allocation message tied to DevQuest;
3. event badge/photo showing DevZone handoff context;
4. screenshot/export of the SMS, with phone numbers and unrelated personal data redacted;
5. public Credly credential URL or badge metadata suitable for a developer profile.

Any recovered artifact should be hashed and referenced from an evidence manifest rather than embedding private contact details in the public repository.
