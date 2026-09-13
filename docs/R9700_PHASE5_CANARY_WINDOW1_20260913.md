# R9700 Phase 5 — Controlled Canary Window 1

Date: 2026-09-13 UTC / 2026-09-12 America/Guayaquil

## Verdict

**PASS — reversible canary packaging, readiness warmup, full serving gate, and rollback were all validated.**

This is not a silent production-default promotion. Stock ROCm10 was restored after the controlled window.

## Canary identity

- Service: `inneros-vllm-hyperloom-s3-canary.service`
- Container: `inneros-vllm-hyperloom-s3-canary`
- Port: `18018`
- Runtime patch SHA256: `3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d`
- S3 config SHA256: `8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6`
- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`
- GPU: AMD Radeon AI PRO R9700 / gfx1201

## Safety gates

The service stayed fail-closed during preparation. Preflight requires stock inactive, port 18018 free, VRAM clean, exact overlay/config hashes, and a valid bundle manifest.

`ExecStartPost` runs the deterministic readiness warmup. In this window the warmup returned the canonical correctness hash and paid the expected late prefix-prefill Triton JIT before systemd marked the canary active.

Readiness warmup TTFT: `5.215961 s`.

Canonical correctness hash:
`7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2`

## Full canary measurement

- correctness: PASS
- hot correctness TTFT: `51.9998 ms`
- C1: `68.8094 tok/s`
- C4: `191.4875 tok/s`
- canonical four C4 hashes: PASS
- fresh-prefix ~6K y: `61.8344 tok/s`
- fresh-prefix ~6K z: `62.1267 tok/s`

All configured gates passed:
- C1 >= 60
- C4 >= 180
- exact correctness hash
- canonical C4 hashes
- both fresh-long >= 55
- hot correctness TTFT < 500 ms

Compared with the controlled stock+queue1 C4 baseline `158.489996 tok/s`, this canary observation is about `+20.5%` on the tested C4 workload. Do not generalize this number to all workloads.

## Rollback / restore

The first canary window deliberately ended in rollback rather than promotion.

Validated sequence:
1. stop `inneros-vllm-hyperloom-s3-canary.service`;
2. R9700 VRAM returned to the clean idle class (`59,994,112` bytes);
3. start `inneros-vllm-canary-rocm10.service`;
4. stock container `inneros-vllm-canary-rocm10` became active;
5. `/v1/models` returned HTTP 200;
6. stock VRAM returned to ~28.61 GB loaded-model class.

Stock again emitted the same first-inference `_fwd_kernel` JIT class after HTTP readiness, reinforcing the Phase 4 conclusion that the cold-JIT issue is not S3-specific.

## Engineering conclusion

Phase 5 has crossed an important boundary: S3 + deterministic readiness is no longer only a benchmark recipe. It has been packaged as a separate systemd canary, started through explicit safety gates, measured end-to-end, and rolled back cleanly to stock.

The next decision must be explicit: either run additional controlled canary windows/soak or promote the canary recipe as the operational default with a defined rollback policy. Do not silently change default routing merely because this first canary window passed.
