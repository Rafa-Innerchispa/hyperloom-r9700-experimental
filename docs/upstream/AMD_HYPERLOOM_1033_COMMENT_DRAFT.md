# Draft Comment for AMD-AGI/Hyperloom#1033

Do not post without human review.

We have been testing an experimental local Hyperloom/KernelForge path on an AMD Radeon AI PRO R9700 workstation GPU (`gfx1201`) with ROCm 10, vLLM, and an OpenAI-compatible local Qwen coding model.

What we can contribute safely:

- A runtime manifest collector that records GPU identity, Docker image, vLLM and torch/ROCm versions, model id, context length, and sanitized launch command.
- A small benchmark harness that separates serving/concurrency scaling from kernel optimization.
- Fail-closed evidence auditing for request-level benchmark samples.
- Documentation warning that `gfx1201` alone is not enough to claim a specific product identity.

What we should not claim:

- Official R9700 support.
- RX 9070 XT and Radeon AI PRO R9700 being the same product.
- Kernel-level GEAK/Arbor gains from serving concurrency measurements.

Current upstream context checked on 2026-09-07:

- Issue #1033 is open for RDNA4 GPU support.
- PR #1032 is open, large, and merge-conflicted/dirty according to the GitHub API, so a smaller focused contribution may be easier to review.

