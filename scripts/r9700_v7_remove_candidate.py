#!/usr/bin/env python3
"""Remove only the named clean-v7 full-model candidate container."""
from __future__ import annotations

import json
import subprocess

NAME = "hyperloom-r9700-v7-candidate"
p = subprocess.run(["docker", "rm", "-f", NAME], capture_output=True, text=True, check=False)
print(json.dumps({"name": NAME, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}, indent=2, sort_keys=True))
raise SystemExit(0 if p.returncode == 0 else 2)
