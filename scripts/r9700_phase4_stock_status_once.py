from __future__ import annotations
import json
import subprocess
import urllib.request

BASE = "http://127.0.0.1:8000"
NAME = "inneros-vllm-canary-rocm10"
SERVICE = "inneros-vllm-canary-rocm10.service"

def run(argv, timeout=30):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    return {"rc": p.returncode, "stdout": p.stdout[-12000:], "stderr": p.stderr[-4000:]}

health = {"ok": False}
try:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=8) as resp:
        body = resp.read().decode("utf-8", "replace")[:4000]
        health = {"ok": resp.status == 200, "status": resp.status, "body": body}
except Exception as exc:
    health = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

out = {
    "health": health,
    "service": run(["systemctl", "--user", "is-active", SERVICE]),
    "inspect": run(["docker", "inspect", "-f", "{{.Name}} {{.State.Status}} {{.State.ExitCode}} {{.Config.Image}}", NAME]),
    "logs": run(["docker", "logs", "--tail", "80", NAME]),
    "vram": run(["rocm-smi", "--showmeminfo", "vram", "--json"]),
}
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if health.get("ok") else 2)
