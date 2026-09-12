from __future__ import annotations

import json
import subprocess
import time
import urllib.request

PORT = 18016
NAME = "hyperloom-r9700-p4-stock-verbose-p18016"
start = time.monotonic()
last = ""
attempts = 0

while time.monotonic() - start < 600:
    attempts += 1
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=4) as resp:
            body = resp.read().decode()[:1000]
            if resp.status == 200:
                print(json.dumps({
                    "ok": True,
                    "port": PORT,
                    "status": resp.status,
                    "elapsed_sec": time.monotonic() - start,
                    "attempts": attempts,
                    "body": body,
                }, indent=2))
                raise SystemExit(0)
    except Exception as exc:
        last = f"{type(exc).__name__}: {exc}"

    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", NAME],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        print(json.dumps({
            "ok": False,
            "port": PORT,
            "container_exists": False,
            "inspect_stderr": proc.stderr[-1000:],
            "last": last,
            "elapsed_sec": time.monotonic() - start,
        }, indent=2))
        raise SystemExit(3)
    if not proc.stdout.strip().startswith("running"):
        print(json.dumps({
            "ok": False,
            "port": PORT,
            "container_exists": True,
            "container": proc.stdout.strip(),
            "last": last,
            "elapsed_sec": time.monotonic() - start,
        }, indent=2))
        raise SystemExit(4)
    time.sleep(5)

print(json.dumps({
    "ok": False,
    "port": PORT,
    "container_exists": True,
    "container": "running_but_not_ready",
    "last": last,
    "elapsed_sec": time.monotonic() - start,
}, indent=2))
raise SystemExit(5)
