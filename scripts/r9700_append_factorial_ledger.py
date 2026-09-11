#!/usr/bin/env python3
"""Idempotently append the clean 2026-09-11 serving factorial to the development ledger."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "DEVELOPMENT_LEDGER.md"
MARKER = "## 2026-09-11 — Clean serving factorial and systemd isolation correction"
SECTION = r'''

## 2026-09-11 — Clean serving factorial and systemd isolation correction

Canonical human result: `docs/R9700_SERVING_FACTORIAL_RESULT_20260911.md`

Machine summary: `docs/evidence/r9700_factorial_clean_summary_20260911.json`

A methodology audit found that the earlier candidate harness used plain `docker stop` against `inneros-vllm-canary-rocm10`, while that container is managed by an auto-restarting user systemd unit. Service logs proved restart attempts could overlap candidate execution. The preliminary 2026-09-11 stock+queue1 run and the older Unified-Attention candidate campaign are therefore preserved but excluded from clean causal/promotion statistics.

The factorial was rerun with `inneros-vllm-canary-rocm10.service` stopped through authorized host ops before each candidate sequence, with the stock container absent and VRAM below 5 GiB before fresh-process launch. All valid cells used the exact ROCm10 image/model/args and dedicated deterministic correctness probe.

Clean results, two independent process starts per cell:

- stock attention + `GPU_MAX_HW_QUEUES=1`: C4 `158.567959 / 158.412033 tok/s`; median `158.489996`; range/median `0.098%`; median long `58.868927`; correctness PASS/PASS.
- Unified Attention + default queues: C4 `136.900788 / 140.629155`; median `138.764971`; range/median `2.687%`; correctness PASS/PASS. One long-context completion was abnormally short/different, so no long-context performance claim is made from this cell.
- Unified Attention + `GPU_MAX_HW_QUEUES=1`: C4 `157.035577 / 156.471523`; median `156.753550`; range/median `0.360%`; median long `55.999820`; correctness PASS/PASS.

Computed C4 deltas:

- stock+queue1 vs same-day healthy-fast stock `162.102053`: `-2.2283%`;
- Unified/default vs stock+queue1: `-12.4456%`;
- Unified/queue1 vs stock+queue1: `-1.0956%`;
- Unified/queue1 vs healthy-fast stock: `-3.2995%`.

Conclusion: `GPU_MAX_HW_QUEUES=1` is the useful stability control in the tested R9700/ROCm10/vLLM environment. It strongly reduces observed process-start bimodality while sacrificing roughly 2.23% versus the same-day healthy-fast stock observation. Unified Attention does not improve the stable queue1 baseline and is not selected for the final HyperLoom gate.

Selected stable baseline for the final candidate: **stock attention + `GPU_MAX_HW_QUEUES=1`**, C4 median `158.489996 tok/s`. Promotion must still compare against the credible healthy-fast stock ceiling (`162.102053` same-day, `165.699` historical), not only the stability-limited denominator.

An audit of the AMD worktree after the factorial found `scripts/r9700_wna16_hybrid_patch.py` still at `r9700_autoawq_stock_layout_hybrid_v6`, SHA256 `0cf11f9fc86e33cde9aa8e6e386e38b9aad2ba09fc643cdb75e57f31703d26ee`. It retains the useful alignment-reuse change but also the invalid mmap/SIGUSR1/SIGUSR2 runtime-gate machinery. It must not be used as the final candidate. The next gate is to preserve/audit `v7_clean`, then run the full Qwen 30B candidate over stock attention + queue1 in clean process isolation.
'''

text = LEDGER.read_text(encoding="utf-8")
if MARKER not in text:
    text = text.rstrip() + SECTION
    print("appended")
else:
    print("already-present")
LEDGER.write_text(text.rstrip() + "\n", encoding="utf-8")
