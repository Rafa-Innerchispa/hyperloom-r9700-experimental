from __future__ import annotations
import datetime as dt
import json
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
NAME = "inneros-vllm-canary-rocm10"
SERVICE = "inneros-vllm-canary-rocm10.service"

def run(argv, timeout=30):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)

started = time.monotonic()
last_error = ""
models_body = ""
status = None
attempts = 0
while time.monotonic() - started < 480:
    attempts += 1
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as resp:
            status = resp.status
            models_body = resp.read().decode("utf-8", "replace")[:4000]
            if status == 200:
                break
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    inspect = run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", NAME])
    if inspect.returncode == 0 and not inspect.stdout.strip().startswith("running"):
        break
    time.sleep(5)

svc = run(["systemctl", "--user", "is-active", SERVICE])
inspect = run(["docker", "inspect", "-f", "{{.Name}} {{.State.Status}} {{.State.ExitCode}} {{.Config.Image}}", NAME])
vram = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
out = {
    "schema": "hyperloom.r9700.phase4.stock_restore.v1",
    "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "service": SERVICE,
    "service_state": svc.stdout.strip(),
    "container": inspect.stdout.strip(),
    "http_status": status,
    "models_body": models_body,
    "last_error": last_error,
    "attempts": attempts,
    "elapsed_sec": time.monotonic() - started,
    "vram": vram.stdout.strip(),
}
out["pass"] = out["service_state"] == "active" and status == 200 and "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ" in models_body
path = ROOT / "docs" / "evidence" / f"r9700_phase4_stock_restore_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(path)
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out["pass"] else 2)
