#!/usr/bin/env python3
"""Emit the exact clean-v7 candidate source for cross-worktree preservation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py"
print(path.read_text(encoding="utf-8"), end="")
