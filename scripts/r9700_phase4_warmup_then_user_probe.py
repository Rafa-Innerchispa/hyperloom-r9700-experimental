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


def request_once() -> dict:
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
    chunks: list[str] = []
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
                text = obj["choices"][0].get("text") or ""
                if text and first is None:
                    first = time.monotonic()
                chunks.append(text)
    end = time.monotonic()
    rendered = "".join(chunks)
    ct = usage.get("completion_tokens") or max(1, len(rendered.split()))
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    return {
        "ok": status == 200,
        "status": status,
        "elapsed_sec": end - t0,
        "ttft_sec": None if first is None else first - t0,
        "completion_tokens": ct,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_tok_s": None if first is None else ct / max(end - first, 1e-9),
        "text_sha256": digest,
        "canonical_match": digest == CANONICAL_SHA256,
        "text_len": len(rendered),
    }


def docker_logs_since(since: str) -> list[str]:
    proc = subprocess.run(
        ["docker", "logs", "--since", since, NAME],
        text=True,
        capture_output=True,
        timeout=30,
    )
    return ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()


def jit_lines(lines: list[str]) -> list[str]:
    keys = ("Triton kernel JIT", "_fwd_kernel", "constexpr", "specialization", "cache key", "cache_key")
    return [line for line in lines if any(key in line for key in keys)][-200:]


def health() -> dict:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as resp:
        return {"ok": resp.status == 200, "status": resp.status}


captured = dt.datetime.now(dt.timezone.utc)
result = {
    "schema": "hyperloom.r9700.phase4.warmup_then_user.v1",
    "captured_at_utc": captured.isoformat(),
    "container": NAME,
    "port": PORT,
    "health": health(),
}

warmup_since = dt.datetime.now(dt.timezone.utc).isoformat()
result["warmup_request"] = request_once()
time.sleep(1.5)
warmup_logs = docker_logs_since(warmup_since)
result["warmup_jit_lines"] = jit_lines(warmup_logs)

# Give log buffers a clean separation before measuring the first externally visible request.
time.sleep(1.0)
user_since = dt.datetime.now(dt.timezone.utc).isoformat()
result["first_user_request"] = request_once()
time.sleep(1.5)
user_logs = docker_logs_since(user_since)
result["first_user_jit_lines"] = jit_lines(user_logs)

user = result["first_user_request"]
result["gate"] = {
    "canonical_match": bool(user.get("canonical_match")),
    "no_user_inference_jit": not result["first_user_jit_lines"],
    "user_ttft_under_500ms": user.get("ttft_sec") is not None and user["ttft_sec"] < 0.5,
}
result["pass"] = all(result["gate"].values())

out_dir = ROOT / "docs" / "evidence"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"r9700_phase4_warmup_then_user_{captured.strftime('%Y%m%dT%H%M%SZ')}.json"
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(out)
print(json.dumps({
    "warmup_request": result["warmup_request"],
    "warmup_jit_lines": result["warmup_jit_lines"],
    "first_user_request": result["first_user_request"],
    "first_user_jit_lines": result["first_user_jit_lines"],
    "gate": result["gate"],
    "pass": result["pass"],
}, indent=2, sort_keys=True))
raise SystemExit(0 if result["pass"] else 2)
