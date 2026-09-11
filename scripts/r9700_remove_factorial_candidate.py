#!/usr/bin/env python3
"""Remove only the named R9700 factorial test container."""
from __future__ import annotations

import json
import subprocess

NAME = "hyperloom-r9700-unified-pilot"
proc = subprocess.run(["docker", "rm", "-f", NAME], capture_output=True, text=True, check=False)
print(json.dumps({"name": NAME, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}, indent=2, sort_keys=True))
raise SystemExit(0 if proc.returncode == 0 else 2)
