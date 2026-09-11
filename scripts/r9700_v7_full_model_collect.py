#!/usr/bin/env python3
"""Collect health, throughput, TTFT, correctness, and path evidence for v7."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "hyperloom-r9700-v7-candidate"
BASE = "http://127.0.0.1:8000/v1"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
CANONICAL_HASH = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"
REMOTE_PATHS = "/tmp/r9700_candidate_paths.jsonl"


def run(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def health(timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [x.get("id") for x in payload.get("data", []) if isinstance(x, dict)]
        return {"ok": response.status == 200 and MODEL in ids, "status": response.status, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def wait_health(timeout_sec: float = 480.0) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_sec:
        attempts += 1
        last = health()
        if last.get("ok"):
            return {"ok": True, "ready_sec": time.monotonic() - started, "attempts": attempts, "health": last}
        time.sleep(3)
    return {"ok": False, "ready_sec": time.monotonic() - started, "attempts": attempts, "health": last}


def stream_ttft_probe() -> dict[str, Any]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the integers 1 through 16 separated by single spaces and nothing else."}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_data: float | None = None
    lines = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                lines += 1
                if first_data is None:
                    first_data = time.perf_counter()
        ended = time.perf_counter()
        return {
            "ok": True,
            "ttft_sec": None if first_data is None else first_data - started,
            "e2e_sec": ended - started,
            "data_events": lines,
        }
    except Exception as exc:
        return {"ok": False, "elapsed_sec": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}


def parse_paths(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "docs" / "evidence" / f"r9700_v7_full_model_result_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "hyperloom.r9700.v7_full_model_result.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE,
        "model": MODEL,
        "canonical_correctness_hash": CANONICAL_HASH,
        "truth_boundary": "full-model candidate evidence; promotion still requires independent-start replication if competitive",
    }

    state = run(["docker", "inspect", "-f", "{{.State.Running}}", CANDIDATE], 20)
    payload["candidate_running"] = state.returncode == 0 and state.stdout.strip() == "true"
    if not payload["candidate_running"]:
        payload.update({"pass": False, "error": "candidate_container_not_running"})
    else:
        ready = wait_health()
        payload["health"] = ready
        if not ready.get("ok"):
            payload.update({"pass": False, "error": "candidate_endpoint_not_ready"})
        else:
            measure = run(["python3", str(ROOT / "scripts" / "r9700_active_measure.py")], 240)
            payload["measure_returncode"] = measure.returncode
            payload["measure_stdout_tail"] = measure.stdout[-6000:]
            payload["measure_stderr_tail"] = measure.stderr[-3000:]
            if measure.returncode == 0 and measure.stdout.splitlines():
                source = Path(measure.stdout.splitlines()[0].strip())
                if source.is_file():
                    dest = ROOT / "docs" / "evidence" / f"r9700_v7_full_model_measure_{stamp}.json"
                    shutil.copy2(source, dest)
                    payload["measurement_file"] = str(dest.relative_to(ROOT))
                    measurement = json.loads(dest.read_text(encoding="utf-8"))
                    payload["measurement"] = measurement
                    observed = ((measurement.get("correctness") or {}).get("text_sha256"))
                    payload["correctness_match"] = observed == CANONICAL_HASH
            payload["stream_probe"] = stream_ttft_probe()

            paths_dest = ROOT / "docs" / "evidence" / f"r9700_v7_full_model_paths_{stamp}.jsonl"
            cp = run(["docker", "cp", f"{CANDIDATE}:{REMOTE_PATHS}", str(paths_dest)], 40)
            payload["path_copy_returncode"] = cp.returncode
            if cp.returncode == 0 and paths_dest.is_file():
                rows = parse_paths(paths_dest.read_text(encoding="utf-8", errors="replace"))
                counts = Counter(str(row.get("path")) for row in rows if row.get("path") is not None)
                payload["path_file"] = str(paths_dest.relative_to(ROOT))
                payload["path_rows"] = len(rows)
                payload["path_counts"] = dict(sorted(counts.items()))
                payload["custom_path_seen"] = counts.get("custom_small_w1_stock_w2", 0) > 0
                payload["stock_fallback_seen"] = counts.get("stock_full_fallback", 0) > 0
            else:
                payload["path_copy_stderr"] = cp.stderr[-3000:]

            logs = run(["docker", "logs", "--tail", "240", CANDIDATE], 30)
            payload["logs_tail"] = (logs.stdout + "\n" + logs.stderr)[-30000:]
            payload["pass"] = bool(
                measure.returncode == 0
                and payload.get("correctness_match")
                and payload.get("custom_path_seen")
                and payload.get("stock_fallback_seen")
            )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(
        {
            "pass": payload.get("pass"),
            "health": payload.get("health"),
            "measurement_file": payload.get("measurement_file"),
            "correctness_match": payload.get("correctness_match"),
            "stream_probe": payload.get("stream_probe"),
            "path_counts": payload.get("path_counts"),
            "result_file": str(out_path.relative_to(ROOT)),
            "error": payload.get("error"),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if payload.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
