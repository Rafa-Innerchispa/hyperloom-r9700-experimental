from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
src = root / "scripts" / "r9700_wna16_hybrid_patch.py"
dst = root / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py"
raw = src.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
expected = "0cf11f9fc86e33cde9aa8e6e386e38b9aad2ba09fc643cdb75e57f31703d26ee"
if sha != expected:
    raise SystemExit(f"source SHA mismatch: {sha} != {expected}")
lines = raw.decode("utf-8").splitlines()
remove = set([20,21])
remove.update(range(56,93))
remove.update(range(263,272))
remove.update(range(389,402))
out=[]
for n,line in enumerate(lines,1):
    if n in remove:
        continue
    if n == 53:
        line = 'PATCH_NAME = "r9700_autoawq_stock_layout_hybrid_v7_clean"'
    out.append(line)
text = "\n".join(out) + "\n"
for forbidden in ("RUNTIME_GATE", "SIGNAL_GATE", "SIGUSR", "_runtime_custom_enabled", "runtime_gate_stock", "import mmap", "import signal"):
    if forbidden in text:
        raise SystemExit(f"forbidden residue: {forbidden}")
if "moe_align_block_size" not in text or "custom_small_w1_stock_w2" not in text:
    raise SystemExit("expected v3 alignment/custom path missing")
dst.write_text(text, encoding="utf-8")
print({"source_sha256":sha,"dest_sha256":hashlib.sha256(text.encode()).hexdigest(),"dest":str(dst),"lines":len(out)})
