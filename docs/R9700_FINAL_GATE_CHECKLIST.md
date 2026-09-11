# R9700 final gate checklist

## Completed before the final full-model run
- [x] `v7_clean` recovered and preserved in remote Git.
- [x] `v7_clean` SHA256: `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`.
- [x] Real-weight M1/M8 custom + M20 stock fallback correctness smoke PASS.
- [x] Clean serving factorial completed; stock attention + `GPU_MAX_HW_QUEUES=1` selected.
- [x] Unified Attention excluded from the final candidate.
- [x] Historical same-process/systemd-overlap evidence demoted from promotion evidence.
- [x] Explicit server entrypoint replaces executable `.pth` bootstrap.

## Final promotion gate
- [ ] Fresh-process full Qwen v7 candidate boots on physical R9700/gfx1201.
- [ ] `/v1/models` HTTP 200.
- [ ] Canonical deterministic correctness hash matches.
- [ ] Full-model evidence observes `custom_small_w1_stock_w2` and `stock_full_fallback`.
- [ ] C1/C4/~6K throughput and TTFT captured.
- [ ] Repeat independent candidate start if performance is close enough to promotion threshold.
- [ ] Candidate removed and stock ROCm10 service restored healthy.
- [ ] `FINAL_STATUS.md`, continuity, ledgers and presentation claims updated.
- [ ] Final evidence committed/pushed and remote SHA verified.

Stable C4 comparison target: **158.490 tok/s** under stock attention + `GPU_MAX_HW_QUEUES=1`. The historical healthy-fast stock C4 reference is **162.102 tok/s**, but default-queue startup is bimodal.

Do not convert the ~1.477x W1 microkernel result into a full-model claim.
