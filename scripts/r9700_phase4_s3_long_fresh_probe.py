from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:18017"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"


def req(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
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
    end = time.monotonic(); rendered = "".join(chunks); ct = usage.get("completion_tokens") or 64
    return {
        "status": status,
        "ok": status == 200,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "elapsed_sec": end - t0,
        "ttft_sec": None if first is None else first - t0,
        "decode_tok_s": None if first is None else ct / max(end - first, 1e-9),
        "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }

captured = dt.datetime.now(dt.timezone.utc)
prompts = {
    "fresh_y": (" y" * 6000) + "\nSummarize the repeated marker pattern in one sentence.",
    "fresh_z": (" z" * 6000) + "\nSummarize the repeated marker pattern in one sentence.",
}
rows = {label: req(prompt) for label, prompt in prompts.items()}
out = {
    "schema": "hyperloom.r9700.phase4.s3_long_fresh.v1",
    "captured_at_utc": captured.isoformat(),
    "port": 18017,
    "prefix_cache_control": "distinct leading token repeated for each 6K prompt; no exact shared x/y/z long prefix",
    "rows": rows,
}
ev = ROOT / "docs" / "evidence"; ev.mkdir(parents=True, exist_ok=True)
path = ev / f"r9700_phase4_s3_long_fresh_{captured.strftime('%Y%m%dT%H%M%SZ')}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(path); print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if all(row["ok"] for row in rows.values()) else 2)
