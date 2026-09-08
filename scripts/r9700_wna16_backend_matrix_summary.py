#!/usr/bin/env python3
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
files = sorted((root / "docs" / "evidence").glob("r9700_wna16_backend_matrix_*.json"))
if not files:
    raise SystemExit("no evidence")
p = files[-1]
data = json.loads(p.read_text())
live = data.get("live_probe", {})
print("evidence=", p.name)
print("status=", data.get("evidence_status"), "sha256=", data.get("probe_sha256"))
print("runtime=", json.dumps(live.get("runtime", {}), sort_keys=True))
print("model=", json.dumps(live.get("model", {}), sort_keys=True))
for name, row in (live.get("matrix") or {}).items():
    print(name, json.dumps(row, sort_keys=True))
print("selected=", json.dumps(live.get("selected", {}), sort_keys=True))
for k, src in (live.get("hard_gates") or {}).items():
    compact = " ".join(str(src).split())
    print("hard_gate", k, compact[:1200])
