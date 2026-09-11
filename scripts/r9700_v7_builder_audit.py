#!/usr/bin/env python3
"""Read-only audit/export of the AMD-only v7 builder and clean candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "scripts" / "r9700_make_hybrid_v7_clean.py",
    ROOT / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py",
    ROOT / "scripts" / "r9700_wna16_hybrid_v7_clean.py",
    ROOT / "scripts" / "r9700_wna16_hybrid_patch.py",
]
MARKERS = [
    "PATCH_NAME",
    "RUNTIME_GATE",
    "SIGNAL_GATE",
    "SIGUSR1",
    "SIGUSR2",
    "mmap",
    "runtime_gate_stock",
    "custom_small_w1_stock_w2",
    "stock_full_fallback",
    "moe_align_block_size",
    "reuse",
]

out: list[dict[str, object]] = []
for path in FILES:
    row: dict[str, object] = {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
    if path.exists():
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        row.update(
            {
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "lines": len(text.splitlines()),
                "markers": {marker: marker in text for marker in MARKERS},
            }
        )
        if path.name in {"r9700_make_hybrid_v7_clean.py", "r9700_wna16_hybrid_patch_v7_clean.py"}:
            row["source"] = text
    out.append(row)
print(json.dumps(out, indent=2, sort_keys=True))
