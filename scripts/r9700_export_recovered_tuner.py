from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "docs/evidence/r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json",
    "docs/evidence/r9700_tuned_config_live_smoke_20260909T052123Z.json",
    "docs/evidence/r9700_tuned_config_live_smoke_20260909T052523Z.json",
]
for rel in FILES:
    p = ROOT / rel
    print(f"=====FILE:{rel}=====")
    if not p.exists():
        print("__MISSING__")
        continue
    text = p.read_text(encoding="utf-8")
    # Validate JSON before emitting verbatim.
    json.loads(text)
    print(text)
    print(f"=====END:{rel}=====")
