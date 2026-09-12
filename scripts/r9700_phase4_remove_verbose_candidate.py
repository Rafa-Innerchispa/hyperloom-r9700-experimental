from __future__ import annotations

import json
import subprocess

NAME = "hyperloom-r9700-p4-stock-verbose-p18016"

inspect = subprocess.run(
    ["docker", "inspect", "-f", "{{.Name}} {{.State.Status}}", NAME],
    text=True,
    capture_output=True,
    timeout=10,
)
out = {"name": NAME, "existed": inspect.returncode == 0, "inspect": inspect.stdout.strip()}
if inspect.returncode == 0:
    if not inspect.stdout.strip().startswith(f"/{NAME} "):
        raise RuntimeError(f"refusing unexpected container identity: {inspect.stdout.strip()!r}")
    rm = subprocess.run(["docker", "rm", "-f", NAME], text=True, capture_output=True, timeout=30)
    out.update({"remove_rc": rm.returncode, "stdout": rm.stdout.strip(), "stderr": rm.stderr[-1000:]})
    if rm.returncode != 0:
        print(json.dumps(out, indent=2, sort_keys=True))
        raise SystemExit(2)
else:
    out.update({"remove_rc": 0, "idempotent_missing": True})

vram = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True, capture_output=True, timeout=20)
out["vram"] = vram.stdout.strip()
print(json.dumps(out, indent=2, sort_keys=True))
