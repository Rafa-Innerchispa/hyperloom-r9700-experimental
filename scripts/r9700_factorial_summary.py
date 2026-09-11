#!/usr/bin/env python3
"""Summarize the clean 2026-09-11 R9700 serving factorial evidence.

The summary is computed only from explicitly named clean evidence files. The
preliminary stock+queue1 measurement and the older Unified Attention campaign
are intentionally excluded because the stock systemd unit could auto-restart
while those candidates were running.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
OUT = EVIDENCE / "r9700_factorial_clean_summary_20260911.json"

CELLS = {
    "stock_queue1": [
        "r9700_factorial_stock_queue1_clean1_20260911T022250Z.json",
        "r9700_factorial_stock_queue1_clean2_20260911T023018Z.json",
    ],
    "unified_defaultq": [
        "r9700_factorial_unified_defaultq_clean1_20260911T023713Z.json",
        "r9700_factorial_unified_defaultq_clean2_20260911T024424Z.json",
    ],
    "unified_queue1": [
        "r9700_factorial_unified_queue1_clean1_20260911T025257Z.json",
        "r9700_factorial_unified_queue1_clean2_20260911T030032Z.json",
    ],
}
CANONICAL_HASH = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"
HEALTHY_FAST_STOCK_C4 = 162.10205336178268


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def pct_delta(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> int:
    summary: dict[str, object] = {
        "schema": "hyperloom.r9700.serving_factorial_summary.v1",
        "date": "2026-09-11",
        "truth_boundary": "clean process-isolated serving-factor analysis; not HyperLoom full-model promotion evidence",
        "excluded": [
            "preliminary stock+queue1 run while stock systemd auto-restart was not disabled",
            "older Unified Attention candidate campaign for causal/promotion use because it used plain docker stop against an auto-restarting systemd unit",
        ],
        "healthy_fast_stock_reference_c4_tok_s": HEALTHY_FAST_STOCK_C4,
        "cells": {},
    }

    for cell, names in CELLS.items():
        rows = []
        for name in names:
            path = EVIDENCE / name
            data = json.loads(path.read_text(encoding="utf-8"))
            correctness_hash = data["correctness"]["text_sha256"]
            rows.append(
                {
                    "file": f"docs/evidence/{name}",
                    "c1_tok_s": float(data["c1"]["decode_tok_s"]),
                    "c4_tok_s": float(data["c4"]["aggregate_tok_s"]),
                    "long_tok_s": float(data["long_context"]["decode_tok_s"]),
                    "correctness_hash": correctness_hash,
                    "correctness_match": correctness_hash == CANONICAL_HASH,
                    "health_ok": bool(data["health"]["ok"]),
                }
            )
        c4 = [row["c4_tok_s"] for row in rows]
        c1 = [row["c1_tok_s"] for row in rows]
        long_values = [row["long_tok_s"] for row in rows]
        med_c4 = median(c4)
        summary["cells"][cell] = {
            "runs": rows,
            "median_c1_tok_s": median(c1),
            "median_c4_tok_s": med_c4,
            "median_long_tok_s": median(long_values),
            "c4_range_tok_s": max(c4) - min(c4),
            "c4_range_pct_of_median": (max(c4) - min(c4)) / med_c4 * 100.0,
            "all_correctness_match": all(row["correctness_match"] for row in rows),
            "all_health_ok": all(row["health_ok"] for row in rows),
        }

    cells = summary["cells"]
    stock_q1 = cells["stock_queue1"]["median_c4_tok_s"]
    ua_default = cells["unified_defaultq"]["median_c4_tok_s"]
    ua_q1 = cells["unified_queue1"]["median_c4_tok_s"]
    summary["comparisons"] = {
        "unified_defaultq_vs_stock_queue1_c4_pct": pct_delta(ua_default, stock_q1),
        "unified_queue1_vs_stock_queue1_c4_pct": pct_delta(ua_q1, stock_q1),
        "stock_queue1_vs_healthy_fast_stock_c4_pct": pct_delta(stock_q1, HEALTHY_FAST_STOCK_C4),
        "unified_queue1_vs_healthy_fast_stock_c4_pct": pct_delta(ua_q1, HEALTHY_FAST_STOCK_C4),
    }
    summary["interpretation"] = {
        "fastest_stable_tested_candidate_baseline": "stock attention + GPU_MAX_HW_QUEUES=1",
        "queue1_effect": "strongly reduces observed process-start throughput bimodality in two clean independent runs, while sitting slightly below the healthy-fast stock ceiling",
        "unified_attention_effect": "does not improve the stable queue1 baseline in these clean runs; it reduces C4 throughput slightly when added to queue1 and is substantially slower without queue1",
        "promotion_rule": "run clean HyperLoom v7 full-model candidate over stock attention + GPU_MAX_HW_QUEUES=1 and also compare against the healthy-fast stock ceiling before promotion",
    }

    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
