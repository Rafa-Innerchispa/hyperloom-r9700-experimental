#!/usr/bin/env python3
"""Reversible live-model MoE dtype/layout probe for the resident R9700 vLLM.

Temporarily instruments Int4EmulationTritonExperts.apply inside the existing
container, restarts the same container, sends one bounded chat request, reads
only shape/dtype metadata written by the worker, then restores the exact source
bytes and restarts the original service. The restore runs in a finally block.

This probe does not enable the experimental hybrid backend and does not persist
site-package changes after completion.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.live_moe_dtype_probe.v3"
CONTAINER = "inneros-vllm-canary-rocm10"
BASE = "http://127.0.0.1:8000/v1"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
TARGET = "/opt/python/lib/python3.14/site-packages/vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
PROBE_FILE = "/tmp/r9700_live_moe_dtype_probe.jsonl"
MARKER = "R9700_LIVE_MOE_DTYPE_PROBE_V3_TRITON_APPLY"


def run(argv: list[str], timeout: float = 120.0, input_text: str | None = None) -> dict[str, Any]:
    try:
        p = subprocess.run(argv, input=input_text, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "argv": argv}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": type(exc).__name__ + ":" + str(exc), "argv": argv}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def health(timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        ids = [x.get("id") for x in payload.get("data", []) if isinstance(x, dict)]
        return {"ok": r.status == 200 and MODEL in ids, "status": r.status, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__ + ":" + str(exc)}


def wait_health(timeout_sec: float = 300.0) -> dict[str, Any]:
    started = time.perf_counter(); attempts = 0; last: dict[str, Any] = {}
    while time.perf_counter() - started < timeout_sec:
        attempts += 1; last = health()
        if last.get("ok"):
            return {"ok": True, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}
        time.sleep(2)
    return {"ok": False, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}


def write_container_text(path: str, text: str) -> dict[str, Any]:
    code = "import sys;from pathlib import Path;Path(sys.argv[1]).write_text(sys.stdin.read(),encoding='utf-8')"
    return run(["docker", "exec", "-i", CONTAINER, "python3", "-c", code, path], 60, text)


def chat_request() -> dict[str, Any]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Return exactly one short sentence explaining why deterministic benchmarks matter."}],
        "temperature": 0,
        "max_tokens": 48,
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode(); payload = json.loads(raw)
        return {"ok": r.status == 200, "status": r.status, "elapsed_sec": time.perf_counter() - started, "usage": payload.get("usage")}
    except Exception as exc:
        return {"ok": False, "elapsed_sec": time.perf_counter() - started, "error": type(exc).__name__ + ":" + str(exc)}


def telemetry() -> dict[str, Any]:
    r = run(["rocm-smi", "--showuse", "--showpower", "--showclocks"], 20)
    lines = [x.strip() for x in (r.get("stdout") or "").splitlines() if "GPU[0]" in x]
    return {"returncode": r.get("returncode"), "gpu0_lines": lines[:80]}


def instrument(original: str) -> str:
    needle = '''    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
'''
    count = original.count(needle)
    if count < 1 or count > 4:
        raise RuntimeError(f"instrumentation_anchor_count={count}")
    block = '''    ):
        # R9700_LIVE_MOE_DTYPE_PROBE_V3_TRITON_APPLY
        try:
            import json as _r9700_json, os as _r9700_os
            if not getattr(self, "_r9700_dtype_probe_written", False):
                self._r9700_dtype_probe_written = True
                _row = {
                    "pid": _r9700_os.getpid(),
                    "experts_class": self.__class__.__name__,
                    "hidden_states_dtype": str(hidden_states.dtype),
                    "hidden_states_shape": list(hidden_states.shape),
                    "w1_dtype": str(w1.dtype),
                    "w1_shape": list(w1.shape),
                    "w2_dtype": str(w2.dtype),
                    "w2_shape": list(w2.shape),
                    "topk_weights_dtype": str(topk_weights.dtype),
                    "topk_weights_shape": list(topk_weights.shape),
                    "topk_ids_dtype": str(topk_ids.dtype),
                    "topk_ids_shape": list(topk_ids.shape),
                    "activation": str(activation),
                    "global_num_experts": int(global_num_experts),
                    "apply_router_weight_on_input": bool(apply_router_weight_on_input),
                    "quant_config_name": self.quant_config.config_name(hidden_states.dtype),
                }
                with open("/tmp/r9700_live_moe_dtype_probe.jsonl", "a", encoding="utf-8") as _f:
                    _f.write(_r9700_json.dumps(_row, sort_keys=True) + "\\n")
        except Exception:
            pass
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
'''
    return original.replace(needle, block)

def main() -> int:
    captured = datetime.now(timezone.utc)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": captured.isoformat(),
        "container": CONTAINER,
        "target": TARGET,
        "marker": MARKER,
        "truth_boundary": {
            "temporary_site_package_instrumentation": True,
            "hybrid_backend_enabled": False,
            "service_restart_required": True,
            "exact_source_restore_required": True,
            "payload_content_recorded": False,
        },
    }
    original = ""; original_sha = ""; patched_sha = ""; restore: dict[str, Any] = {}; modified = False
    try:
        if not health().get("ok"):
            raise RuntimeError("pre_probe_service_not_healthy")
        read = run(["docker", "exec", CONTAINER, "cat", TARGET], 60)
        if read.get("returncode") != 0:
            raise RuntimeError("target_read_failed:" + str(read.get("stderr"))[-1000:])
        original = read["stdout"]; original_sha = sha(original); patched = instrument(original); patched_sha = sha(patched)
        out["original_sha256"] = original_sha; out["patched_sha256"] = patched_sha
        clear = run(["docker", "exec", CONTAINER, "python3", "-c", f"from pathlib import Path;Path('{PROBE_FILE}').unlink(missing_ok=True)"], 30)
        out["clear_probe"] = {"returncode": clear.get("returncode"), "stderr_tail": str(clear.get("stderr") or "")[-1000:]}
        wr = write_container_text(TARGET, patched)
        if wr.get("returncode") == 0:
            modified = True
        if wr.get("returncode") != 0:
            raise RuntimeError("patch_write_failed:" + str(wr.get("stderr"))[-1000:])
        verify = run(["docker", "exec", CONTAINER, "python3", "-m", "py_compile", TARGET], 60)
        if verify.get("returncode") != 0:
            raise RuntimeError("patched_source_compile_failed:" + str(verify.get("stderr"))[-2000:])
        restart = run(["docker", "restart", "--time", "30", CONTAINER], 90)
        out["instrumented_restart"] = {"returncode": restart.get("returncode"), "stderr_tail": str(restart.get("stderr") or "")[-1000:]}
        if restart.get("returncode") != 0:
            raise RuntimeError("instrumented_restart_failed")
        ready = wait_health(); out["instrumented_health"] = ready
        if not ready.get("ok"):
            raise RuntimeError("instrumented_service_not_healthy")
        out["telemetry_before_request"] = telemetry()
        out["request"] = chat_request()
        out["telemetry_after_request"] = telemetry()
        if not out["request"].get("ok"):
            raise RuntimeError("instrumented_request_failed")
        probe = run(["docker", "exec", CONTAINER, "cat", PROBE_FILE], 30)
        rows = []
        for line in (probe.get("stdout") or "").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict): rows.append(row)
        out["observations"] = rows
        out["observation_count"] = len(rows)
        if not rows:
            raise RuntimeError("no_moe_observations_captured")
        out["pass"] = True
    except Exception as exc:
        out["pass"] = False; out["error"] = type(exc).__name__ + ":" + str(exc)
    finally:
        if original and modified:
            wr = write_container_text(TARGET, original)
            restore["write_returncode"] = wr.get("returncode")
            rr = run(["docker", "restart", "--time", "30", CONTAINER], 90)
            restore["restart_returncode"] = rr.get("returncode")
            ready = wait_health(); restore["health"] = ready
            reread = run(["docker", "exec", CONTAINER, "cat", TARGET], 60)
            restored_text = reread.get("stdout") or ""
            restore["restored_sha256"] = sha(restored_text) if reread.get("returncode") == 0 else None
            restore["sha_matches_original"] = restore.get("restored_sha256") == original_sha
            restore["service_ok"] = bool(ready.get("ok"))
            if not restore["sha_matches_original"] or not restore["service_ok"]:
                out["pass"] = False
                out["restore_failure"] = True
        out["restore"] = restore
    clean = json.loads(json.dumps(out, sort_keys=True, allow_nan=False)); clean.pop("probe_sha256", None)
    out["probe_sha256"] = hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    stamp = captured.strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "docs" / "evidence" / f"r9700_live_moe_dtype_probe_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": out.get("pass"), "path": str(path.relative_to(ROOT)), "sha256": out["probe_sha256"], "observations": out.get("observations", []), "restore": out.get("restore", {}), "error": out.get("error")}, indent=2, sort_keys=True))
    return 0 if out.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
