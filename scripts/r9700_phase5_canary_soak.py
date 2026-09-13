from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEASURE = ROOT / "scripts" / "r9700_phase5_canary_measure.py"
EVIDENCE = ROOT / "docs" / "evidence"
LOCK_PATH = ROOT / "var" / "r9700_phase5_benchmark.lock"
LOCK_HELD_ENV = "R9700_PHASE5_BENCH_LOCK_HELD"


def acquire_soak_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({
            "schema": "hyperloom.r9700.phase5.canary_soak_lock.v1",
            "pass": False,
            "error": "benchmark_lock_busy",
            "lock_path": str(LOCK_PATH),
            "action": "abort_without_sending_inference_traffic",
        }, indent=2))
        raise SystemExit(4)
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "kind": "soak", "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    handle.flush()
    return handle


def run_round(index: int) -> dict:
    env = os.environ.copy()
    env[LOCK_HELD_ENV] = "1"
    proc = subprocess.run(
        [sys.executable, str(MEASURE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        env=env,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    evidence_path = pathlib.Path(lines[0]) if lines else None
    payload = None
    if evidence_path and evidence_path.exists():
        payload = json.loads(evidence_path.read_text())
    return {
        "round": index,
        "returncode": proc.returncode,
        "evidence_path": str(evidence_path) if evidence_path else None,
        "payload": payload,
        "stderr_tail": proc.stderr[-2000:],
    }


def metric(row: dict, path: tuple[str, ...]) -> float:
    cur = row["payload"]
    for part in path:
        cur = cur[part]
    return float(cur)


def stats(values: list[float]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "mean": statistics.fmean(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--interval-sec", type=float, default=15.0)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 24:
        raise SystemExit("--rounds must be between 1 and 24")

    lock_handle = acquire_soak_lock()
    captured = dt.datetime.now(dt.timezone.utc)
    rounds: list[dict] = []
    aborted = False
    abort_reason = None

    for index in range(1, args.rounds + 1):
        row = run_round(index)
        rounds.append(row)
        payload = row.get("payload") or {}
        if row["returncode"] != 0 or not payload.get("pass"):
            aborted = True
            abort_reason = f"round {index} failed measurement gate"
            break
        if index < args.rounds:
            time.sleep(args.interval_sec)

    passed_rows = [r for r in rounds if r.get("payload") and r["payload"].get("pass")]
    summary = {
        "schema": "hyperloom.r9700.phase5.canary_soak.v2",
        "captured_at_utc": captured.isoformat(),
        "requested_rounds": args.rounds,
        "completed_rounds": len(rounds),
        "passed_rounds": len(passed_rows),
        "interval_sec": args.interval_sec,
        "benchmark_lock": "exclusive",
        "aborted": aborted,
        "abort_reason": abort_reason,
        "rounds": rounds,
        "pass": (not aborted and len(passed_rows) == args.rounds),
    }

    if passed_rows:
        summary["aggregate"] = {
            "c1_tok_s": stats([metric(r, ("c1", "decode_tok_s")) for r in passed_rows]),
            "c4_tok_s": stats([metric(r, ("c4", "aggregate_tok_s")) for r in passed_rows]),
            "long_y_tok_s": stats([metric(r, ("long_fresh_y", "decode_tok_s")) for r in passed_rows]),
            "long_z_tok_s": stats([metric(r, ("long_fresh_z", "decode_tok_s")) for r in passed_rows]),
            "correctness_ttft_sec": stats([metric(r, ("correctness", "ttft_sec")) for r in passed_rows]),
        }

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"r9700_phase5_canary_soak_{captured.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(path)
    print(json.dumps({k: summary.get(k) for k in ("requested_rounds", "completed_rounds", "passed_rounds", "aborted", "abort_reason", "aggregate", "pass")}, indent=2, sort_keys=True))
    # Keep the lock handle alive until all evidence is written and the process exits.
    _ = lock_handle
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
