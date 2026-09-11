#!/usr/bin/env python3
"""Run the existing real-weight WNA16 smoke against the clean v7 candidate."""
from __future__ import annotations

from pathlib import Path
import r9700_wna16_stock_layout_real_weight_smoke as base

ROOT = Path(__file__).resolve().parents[1]
base.PATCH = ROOT / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py"
base.SCHEMA = "hyperloom.r9700.wna16_v7_clean_real_weight_smoke.v1"

if __name__ == "__main__":
    raise SystemExit(base.main())
