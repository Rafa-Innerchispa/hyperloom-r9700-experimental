#!/usr/bin/env python3
"""Prove Phase5 benchmark clients fail closed when an exclusive lease is held."""
from __future__ import annotations

import fcntl
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "var" / "r9700_phase5_benchmark.lock"
MEASURE = ROOT / "scripts" / "r9700_phase5_canary_measure.py"
SOAK = ROOT / "scripts" / "r9700_phase5_canary_soak.py"


def probe(argv: list[str]) -> dict:
    proc = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=30)
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
        "lock_rejected": proc.returncode == 4 and "benchmark_lock_busy" in proc.stdout,
    }


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        measure = probe([sys.executable, str(MEASURE)])
        soak = probe([sys.executable, str(SOAK), "--rounds", "1", "--interval-sec", "0"])
    out = {
        "schema": "hyperloom.r9700.phase5.benchmark_lock_selftest.v1",
        "measure": measure,
        "soak": soak,
        "pass": bool(measure["lock_rejected"] and soak["lock_rejected"]),
        "truth_boundary": "No inference traffic should be sent because both clients must fail before health/request execution.",
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
