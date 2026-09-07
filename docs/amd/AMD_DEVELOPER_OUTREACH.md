# AMD Developer Outreach Draft

## Route A: AMD AI Developer Program Featured Project

Pitch:

InnerChispa is testing local agentic inference workflows on an AMD Radeon AI PRO R9700 using ROCm 10, vLLM, and a local Qwen coding model. The project focuses on reproducibility, bounded autonomous changes, deterministic evidence, and an upstream-friendly path for RDNA4/R9700 validation.

Evidence to include:

- Launch manifest hash from `scripts/r9700_runtime_manifest.py`.
- Saved E2E evidence paths under `docs/evidence/`.
- Clear limitation that this is experimental, not official support.

## Route B: Instinct Evaluation Program

Ask:

Developer Cloud MI300X hours or evaluation access to compare the same Hyperloom evidence harness on officially supported Instinct hardware against the R9700 experimental path.

Why:

The R9700 path can validate local workstation feasibility. Instinct access would validate whether the same harness produces comparable evidence on supported production-class targets.

## Hardware/Compute Requests

1. Developer Cloud MI300X hours.
2. Instinct MI300X or MI325X evaluation access.
3. Radeon Test Drive or future workstation GPU access where eligible.

No application should be submitted automatically. Human review is required.

