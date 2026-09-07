# Project Story

InnerChispa started this AMD path as a practical "first AI app on AMD" challenge: make local agentic software run on the hardware Rafael actually has, not on a perfect cloud lab. The experiment evolved into a reproducibility and upstream-readiness track for Hyperloom on an AMD Radeon AI PRO R9700 workstation GPU.

The current result is intentionally narrow and honest. We have a local ROCm 10/vLLM/Qwen stack, a bounded local-openai agent backend, repeated measurements, and a deterministic gate that accepts or rejects a candidate. The measured `concurrency=1 -> 2` improvement is a serving/concurrency result, not a kernel optimization claim.

Rafael has stated that the R9700 arrived from the AMD DevQuest/Advancing AI San Francisco path. Until repository or email evidence is attached to this repo, that provenance remains USER-ATTESTED and must be worded that way in public material.

