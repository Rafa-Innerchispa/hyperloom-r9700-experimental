from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
PORT = 18017
BASE = f"http://127.0.0.1:{PORT}"
CANONICAL_CORRECTNESS = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"
CANONICAL_C4 = [
    "891b5901302b3d6901fd310b055c7f3fa03ed1ea6a4cf1c1deebd0e5fc372e68",
    "f3164647b31f3f554d6d6f1be77d63f87736ffb85b89bc3e3b59fcdab5ac6c5c",
    "fbe90e7b9c033267a5017a845902599a1c0b6fead46d2a709ecad63ddb97a407",
    "fbe90e7b9c033267a5017a845902599a1c0b6fead46d2a709ecad63ddb97a407",
]


def req(prompt: str, max_tokens: int = 96) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic(); first = None; chunks: list[str] = []; usage: dict = {}
    with urllib.request.urlopen(request, timeout=240) as resp:
        status = resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices"):
                text = obj["choices"][0].get("text") or ""
                if text and first is None:
                    first = time.monotonic()
                chunks.append(text)
    end = time.monotonic(); rendered = "".join(chunks)
    ct = usage.get("completion_tokens") or max(1, len(rendered.split()))
    return {
        "ok": status == 200,
        "status": status,
        "elapsed_sec": end - t0,
        "ttft_sec": None if first is None else first - t0,
        "completion_tokens": ct,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_tok_s": None if first is None else ct / max(end - first, 1e-9),
        "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "text_len": len(rendered),
    }


def c4(prompt: str, max_tokens: int = 128) -> dict:
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(req, prompt + f"\nCase {i}:", max_tokens) for i in range(4)]
        rows = [f.result() for f in futures]
    wall = time.monotonic() - t0
    total = sum(row["completion_tokens"] for row in rows)
    hashes = [row["text_sha256"] for row in rows]
    return {
        "wall_sec": wall,
        "completion_tokens": total,
        "aggregate_tok_s": total / max(wall, 1e-9),
        "rows": rows,
        "hashes": hashes,
        "canonical_hashes_match": hashes == CANONICAL_C4,
    }


def health() -> dict:
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as resp:
            return {"ok": resp.status == 200, "status": resp.status, "body": resp.read().decode()[:1000]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


short = "Implement a Python LRU cache with get and put methods. Return only code."
long_prompt = (" x" * 6000) + "\nSummarize the repeated marker pattern in one sentence."
captured = dt.datetime.now(dt.timezone.utc)
out = {
    "schema": "hyperloom.r9700.phase4.s3_measure.v1",
    "captured_at_utc": captured.isoformat(),
    "port": PORT,
    "health": health(),
}
if out["health"]["ok"]:
    out["correctness"] = req("Return exactly a compact Python function add(a,b) that returns a+b.", 64)
    out["correctness"]["canonical_match"] = out["correctness"]["text_sha256"] == CANONICAL_CORRECTNESS
    out["c1"] = req(short, 128)
    out["c4"] = c4(short, 128)
    out["long_context"] = req(long_prompt, 64)
    out["pass"] = bool(out["correctness"]["canonical_match"] and out["c4"]["canonical_hashes_match"])
else:
    out["pass"] = False

ev = ROOT / "docs" / "evidence"; ev.mkdir(parents=True, exist_ok=True)
path = ev / f"r9700_phase4_s3_measure_{captured.strftime('%Y%m%dT%H%M%SZ')}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(path)
print(json.dumps({key: out.get(key) for key in ("correctness", "c1", "c4", "long_context", "pass")}, indent=2, sort_keys=True))
raise SystemExit(0 if out["pass"] else 2)
