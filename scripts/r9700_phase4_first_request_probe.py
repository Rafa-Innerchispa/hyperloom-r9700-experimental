from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 18016
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
NAME = "hyperloom-r9700-p4-stock-verbose-p18016"
PROMPT = "Return exactly a compact Python function add(a,b) that returns a+b."
CANONICAL_SHA256 = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"


def stream_request() -> dict:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    text: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=240) as resp:
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
                chunk = obj["choices"][0].get("text") or ""
                if chunk and first is None:
                    first = time.monotonic()
                text.append(chunk)
    end = time.monotonic()
    rendered = "".join(text)
    completion_tokens = usage.get("completion_tokens") or max(1, len(rendered.split()))
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    return {
        "ok": status == 200,
        "status": status,
        "elapsed_sec": end - t0,
        "ttft_sec": None if first is None else first - t0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_tok_s": None if first is None else completion_tokens / max(end - first, 1e-9),
        "text_sha256": digest,
        "canonical_sha256": CANONICAL_SHA256,
        "canonical_match": digest == CANONICAL_SHA256,
        "text_len": len(rendered),
    }


def health() -> dict:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as resp:
        return {"ok": resp.status == 200, "status": resp.status, "body": resp.read().decode()[:1000]}


captured = dt.datetime.now(dt.timezone.utc)
since = captured.isoformat()
result = {
    "schema": "hyperloom.r9700.phase4.first_request_verbose.v1",
    "captured_at_utc": captured.isoformat(),
    "container": NAME,
    "port": PORT,
    "health": health(),
}
result["first_request"] = stream_request()
time.sleep(1.0)
logs = subprocess.run(
    ["docker", "logs", "--since", since, NAME],
    text=True,
    capture_output=True,
    timeout=30,
)
all_logs = (logs.stdout or "") + "\n" + (logs.stderr or "")
keywords = (
    "Triton kernel JIT",
    "_fwd_kernel",
    "constexpr",
    "signature",
    "specialization",
    "cache key",
    "cache_key",
    "JIT monitor",
    "jit monitor",
)
interesting = [line for line in all_logs.splitlines() if any(key in line for key in keywords)]
result["docker_logs_rc"] = logs.returncode
result["jit_lines"] = interesting[-300:]
result["log_tail"] = all_logs.splitlines()[-300:]

out_dir = ROOT / "docs" / "evidence"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"r9700_phase4_first_request_verbose_{captured.strftime('%Y%m%dT%H%M%SZ')}.json"
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(out)
print(json.dumps({
    "health": result["health"],
    "first_request": result["first_request"],
    "jit_lines": result["jit_lines"],
}, indent=2, sort_keys=True))
raise SystemExit(0 if result["health"]["ok"] and result["first_request"]["ok"] else 1)
