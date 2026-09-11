# R9700 continuity addendum — final presentation gate

Read this after `docs/R9700_PROJECT_CONTINUITY.md`.

## Durable checkpoint
- Base preservation commit: `c4e4aae90f5582d76b7881ba932778bd94922611`.
- Clean candidate: `scripts/r9700_wna16_hybrid_patch_v7_clean.py`.
- v7 SHA256: `eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269`.
- Real-weight v7 smoke PASS: `docs/evidence/r9700_wna16_stock_layout_real_weight_smoke_20260911T031621Z.json`, evidence SHA `2fdadeca5e00f452670d5f0743997fdf90662696cdadfca621d9d7482d113719`.
- Smoke paths: M1 custom, M8 custom, M20 stock fallback; all correctness gates PASS.

## Clean serving factorial
- stock attention + `GPU_MAX_HW_QUEUES=1`: C4 runs `158.568`, `158.412`; median `158.490 tok/s`, ~0.098% spread.
- Unified Attention + default queues: median C4 `138.765 tok/s`.
- Unified Attention + queue1: median C4 `156.754 tok/s`.
- healthy-fast stock reference without queue1: `162.102 tok/s`, but default-queue startup is historically bimodal.

Decision: final v7 gate uses **stock attention + `GPU_MAX_HW_QUEUES=1`**. Unified Attention is excluded.

## Methodology correction
Older campaigns that stopped only the Docker stock container while the systemd unit could auto-restart are diagnostic/history only, not clean promotion evidence. The final campaign must stop `inneros-vllm-canary-rocm10.service`, verify low VRAM, start a fresh candidate process, capture evidence, remove candidate, restore stock, and verify stock HTTP health.

## Final execution path
Use `scripts/r9700_v7_server_entry.py` + `scripts/r9700_v7_full_model_launcher.py` + `scripts/r9700_v7_full_model_collect.py` + `scripts/r9700_v7_remove_candidate.py`. Do not use signal/mmap gates or executable `.pth`/sitecustomize bootstrap. The legacy `.pth` file is intentionally inert.

Current truth until final evidence is committed: **KERNEL KEEP / FULL-MODEL INTEGRATION NOT YET PROMOTED**. Never present the ~1.477x W1 microkernel speedup as full-model acceleration.
