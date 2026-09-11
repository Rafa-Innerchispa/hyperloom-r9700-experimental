#!/usr/bin/env python3
"""Read-only Git state probe for the AMD live-verify worktree."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str]) -> dict[str, object]:
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}

print(json.dumps({
    "root": str(ROOT),
    "head": run(["git", "rev-parse", "HEAD"]),
    "branch": run(["git", "branch", "--show-current"]),
    "status": run(["git", "status", "--short", "--branch"]),
    "remotes": run(["git", "remote", "-v"]),
}, indent=2, sort_keys=True))
