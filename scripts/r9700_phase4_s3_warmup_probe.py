from __future__ import annotations

import pathlib
import subprocess
import sys

script = pathlib.Path(__file__).with_name("r9700_phase4_warmup_then_user_probe.py")
proc = subprocess.run(
    [
        sys.executable,
        str(script),
        "--port", "18017",
        "--container", "hyperloom-r9700-p4-s3-verbose-p18017",
        "--label", "s3",
    ],
    text=True,
    timeout=300,
)
raise SystemExit(proc.returncode)
