from pathlib import Path
import hashlib

p = Path(__file__).resolve().parents[1] / "scripts" / "r9700_wna16_hybrid_patch.py"
text = p.read_text(encoding="utf-8")
print("SHA256", hashlib.sha256(text.encode()).hexdigest())
print("LINES", len(text.splitlines()))
for i,line in enumerate(text.splitlines(),1):
    low=line.lower()
    if any(k in low for k in ("v3","v4","v5","v6","runtime_gate","signal","align","moe_align","fallback","small_w1","threshold","pth","evidence")):
        print(f"{i:04d}: {line}")
