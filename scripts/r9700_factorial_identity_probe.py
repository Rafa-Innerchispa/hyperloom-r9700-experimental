#!/usr/bin/env python3
"""Print sanitized identity for the active R9700 factorial test container."""
from __future__ import annotations

import json
import subprocess

NAME = "hyperloom-r9700-unified-pilot"
proc = subprocess.run(["docker", "inspect", NAME], capture_output=True, text=True, check=False)
if proc.returncode:
    print(json.dumps({"ok": False, "name": NAME, "error": proc.stderr[-3000:]}, indent=2))
    raise SystemExit(2)
row = json.loads(proc.stdout)[0]
env = row.get("Config", {}).get("Env", []) or []
relevant = sorted(
    item
    for item in env
    if item.startswith("GPU_MAX_HW_QUEUES=")
    or item.startswith("VLLM_ROCM_USE_AITER")
)
out = {
    "ok": True,
    "name": NAME,
    "image": row.get("Config", {}).get("Image"),
    "started_at": row.get("State", {}).get("StartedAt"),
    "pid": row.get("State", {}).get("Pid"),
    "running": row.get("State", {}).get("Running"),
    "relevant_env": relevant,
    "cmd": row.get("Config", {}).get("Cmd", []),
}
print(json.dumps(out, indent=2, sort_keys=True))
