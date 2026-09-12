from __future__ import annotations

import json
import subprocess
import urllib.request

PORT = 18016
NAME = "hyperloom-r9700-p4-stock-verbose-p18016"


def run(argv: list[str], timeout: int = 20) -> dict:
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    return {"rc": proc.returncode, "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-4000:]}

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as resp:
        health = {"ok": resp.status == 200, "status": resp.status, "body": resp.read().decode()[:1000]}
except Exception as exc:
    health = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

out = {
    "health": health,
    "inspect": run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", NAME]),
    "logs": run(["docker", "logs", "--tail", "120", NAME]),
    "vram": run(["rocm-smi", "--showmeminfo", "vram", "--json"]),
}
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if health.get("ok") else 2)
