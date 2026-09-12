#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
DEFAULT_PROMPT = "Return exactly a compact Python function add(a,b) that returns a+b."
CANONICAL_SHA256 = "7931ecfbe6d2b41843001499ef498b96a4d7ddc101f77926bd867468271c5ce2"


def wait_models(base_url: str, timeout_sec: float) -> dict:
    start = time.monotonic()
    last_error = ""
    while time.monotonic() - start < timeout_sec:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return {"ok": True, "elapsed_sec": time.monotonic() - start, "status": resp.status}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    return {"ok": False, "elapsed_sec": time.monotonic() - start, "error": last_error}


def warmup(base_url: str, model: str) -> dict:
    payload = {
        "model": model,
        "prompt": DEFAULT_PROMPT,
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic(); first = None; chunks: list[str] = []; usage: dict = {}
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
    end = time.monotonic(); rendered = "".join(chunks)
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    ct = usage.get("completion_tokens") or max(1, len(rendered.split()))
    return {
        "ok": status == 200 and digest == CANONICAL_SHA256,
        "status": status,
        "elapsed_sec": end - t0,
        "ttft_sec": None if first is None else first - t0,
        "completion_tokens": ct,
        "decode_tok_s": None if first is None else ct / max(end - first, 1e-9),
        "text_sha256": digest,
        "canonical_sha256": CANONICAL_SHA256,
        "canonical_match": digest == CANONICAL_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume the known R9700/vLLM prefix-prefill cold JIT before exposing readiness.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    args = parser.parse_args()

    readiness = wait_models(args.base_url, args.ready_timeout)
    if not readiness["ok"]:
        print(json.dumps({"schema": "hyperloom.r9700.readiness_warmup.v1", "ready": readiness, "pass": False}, indent=2, sort_keys=True))
        return 2
    try:
        result = warmup(args.base_url, args.model)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(json.dumps({
            "schema": "hyperloom.r9700.readiness_warmup.v1",
            "ready": readiness,
            "warmup": {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            "pass": False,
        }, indent=2, sort_keys=True))
        return 3

    out = {"schema": "hyperloom.r9700.readiness_warmup.v1", "ready": readiness, "warmup": result, "pass": bool(result["ok"])}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
