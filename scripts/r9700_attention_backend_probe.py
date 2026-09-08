#!/usr/bin/env python3
"""Prove the attention backend selected by the live R9700 vLLM stack.

This probe is read-only with respect to the resident service. It binds evidence to
its live Docker launch, reconstructs the installed vLLM EngineConfig without
loading model weights, and asks vLLM's own ROCm attention selector for the exact
backend class. It also records whether the installed AITER package is considered
supported by this vLLM build on gfx1201.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.attention_backend_observability.v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_SECRET_RE = re.compile(r"(?i)(authorization|api[-_]?key|token|secret|password)(=|\s+)[^\s,]+|(sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+)")


def redact(text: str) -> str:
    return _SECRET_RE.sub(lambda m: (m.group(1) + (m.group(2) or "=") + "REDACTED") if m.group(1) else "REDACTED", str(text or ""))


def stable_json_sha256(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_command(argv: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"argv": argv, "returncode": proc.returncode, "stdout": redact(proc.stdout)[-60000:], "stderr": redact(proc.stderr)[-12000:]}
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "returncode": None, "error": type(exc).__name__}


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        try:
            value = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def last_json_object(text: str) -> dict[str, Any] | None:
    rows = parse_json_lines(text)
    return rows[-1] if rows else None


def cli_flag(argv: list[str], name: str, default: str = "") -> str:
    try:
        idx = argv.index(name)
    except ValueError:
        return default
    return argv[idx + 1] if idx + 1 < len(argv) else default


def choose_backend(valid: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in valid if isinstance(row, dict) and isinstance(row.get("priority"), int)]
    return min(rows, key=lambda row: row["priority"]) if rows else None


def discover_live_container(run: Callable[..., dict[str, Any]] = run_command) -> dict[str, Any]:
    result = run(["docker", "ps", "--format", "{{json .}}"])
    rows = parse_json_lines(str(result.get("stdout") or ""))
    candidates = []
    for row in rows:
        blob = " ".join(str(row.get(k, "")) for k in ("Image", "Names", "Command"))
        if re.search(r"\b(vllm|rocm)\b", blob, re.IGNORECASE):
            candidates.append(row)
    return {"selected": candidates[0] if candidates else {}, "candidates": candidates, "probe": result}


def inspect_container(container: str, run: Callable[..., dict[str, Any]] = run_command) -> dict[str, Any]:
    result = run(["docker", "inspect", container])
    if result.get("returncode") != 0:
        return {"available": False, "probe": result}
    try:
        row = json.loads(str(result.get("stdout") or "[]"))[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return {"available": False, "probe": result}
    cmd = [str(x) for x in row.get("Config", {}).get("Cmd", []) or []]
    env = [str(x) for x in row.get("Config", {}).get("Env", []) or []]
    return {
        "available": True,
        "image": row.get("Config", {}).get("Image"),
        "cmd": [redact(x) for x in cmd],
        "env_signals": [redact(x) for x in env if re.search(r"AITER|ROCM|VLLM|TRITON|ATTN|HIP", x, re.IGNORECASE)],
        "model_path": cli_flag(cmd, "--model"),
        "served_model_name": cli_flag(cmd, "--served-model-name"),
        "dtype": cli_flag(cmd, "--dtype", "auto"),
        "max_model_len": cli_flag(cmd, "--max-model-len", "8192"),
        "tensor_parallel_size": cli_flag(cmd, "--tensor-parallel-size", "1"),
        "data_parallel_size": cli_flag(cmd, "--data-parallel-size", "1"),
    }


def request_models(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer local"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": type(exc).__name__, "url": url}
    models = payload.get("data") if isinstance(payload, dict) else []
    ids = [m.get("id") for m in models if isinstance(m, dict)] if isinstance(models, list) else []
    return {"available": True, "url": url, "model_ids": ids}


def _inside_probe_code(*, model_path: str, dtype: str, max_model_len: int, tp_size: int, dp_size: int) -> str:
    model_json = json.dumps(model_path)
    dtype_json = json.dumps(dtype)
    return f'''import importlib.util, inspect, json, os, traceback\nout={{}}\ntry:\n import torch, vllm\n from vllm.engine.arg_utils import EngineArgs\n from vllm.platforms import current_platform\n from vllm.platforms.rocm import RocmPlatform\n from vllm.config import set_current_vllm_config\n from vllm.v1.attention.selector import AttentionSelectorConfig, get_attn_backend\n out["versions"]={{"vllm":getattr(vllm,"__version__",None),"torch":torch.__version__,"torch_hip":getattr(torch.version,"hip",None)}}\n out["platform"]={{"class":type(current_platform).__module__+"."+type(current_platform).__name__,"is_rocm":current_platform.is_rocm(),"capability":str(current_platform.get_device_capability()),"device_name":current_platform.get_device_name()}}\n cfg=EngineArgs(model={model_json},dtype={dtype_json},max_model_len={int(max_model_len)},trust_remote_code=True,tensor_parallel_size={int(tp_size)},data_parallel_size={int(dp_size)}).create_engine_config()\n h=cfg.model_config.hf_config\n head_size=getattr(h,"head_dim",None) or (getattr(h,"hidden_size")//getattr(h,"num_attention_heads"))\n num_heads=getattr(h,"num_attention_heads",None)\n cache_dtype=getattr(cfg.cache_config,"cache_dtype","auto") if cfg.cache_config else "auto"\n block_size=(cfg.cache_config.block_size if cfg.cache_config and getattr(cfg.cache_config,"user_specified_block_size",False) else None)\n kv=cfg.kv_transfer_config\n use_kv=bool(kv is not None and kv.is_kv_transfer_instance)\n selector=AttentionSelectorConfig(head_size=head_size,dtype=cfg.model_config.dtype,kv_cache_dtype=cache_dtype,block_size=block_size,use_mla=False,has_sink=False,use_sparse=False,use_mm_prefix=False,use_per_head_quant_scales=False,attn_type="decoder",has_sliding_window=False,use_non_causal=cfg.attention_config.use_non_causal,use_batch_invariant=False,use_kv_connector=use_kv,use_pcp=cfg.parallel_config.prefill_context_parallel_size>1)\n valid,invalid=RocmPlatform.get_valid_backends(device_capability=RocmPlatform.get_device_capability(),attn_selector_config=selector,num_heads=num_heads)\n out["selector"]={{"config":repr(selector),"valid":[{{"name":b.name,"priority":p,"path":b.get_path()}} for b,p in valid],"invalid":{{b.name:reasons for b,reasons in invalid.items()}}}}\n with set_current_vllm_config(cfg):\n  selected=get_attn_backend(head_size=head_size,dtype=cfg.model_config.dtype,kv_cache_dtype=cache_dtype,use_mla=False,has_sink=False,use_sparse=False,use_mm_prefix=False,use_per_head_quant_scales=False,attn_type="decoder",num_heads=num_heads,has_sliding_window=False)\n out["selected_backend"]={{"class":selected.__module__+"."+selected.__name__,"name":selected.get_name() if hasattr(selected,"get_name") else selected.__name__}}\n out["model_config"]={{"architecture":getattr(h,"architectures",None),"head_size":head_size,"num_attention_heads":num_heads,"num_key_value_heads":getattr(h,"num_key_value_heads",None),"dtype":str(cfg.model_config.dtype),"cache_dtype":cache_dtype,"block_size_for_selector":block_size,"max_model_len":cfg.model_config.max_model_len}}\n out["aiter"]={{"module_found":bool(importlib.util.find_spec("aiter")),"env":{{k:v for k,v in os.environ.items() if k.startswith("VLLM_ROCM_USE_AITER") or k in ("AITER_ROCM_ARCH","VLLM_USE_TRITON_AWQ")}}}}\n try:\n  import vllm._aiter_ops as ao\n  fn=getattr(ao,"is_aiter_found_and_supported",None)\n  out["aiter"]["vllm_supported"]=fn() if fn else None\n  out["aiter"]["support_function_source"]=inspect.getsource(fn) if fn else ""\n except Exception as exc:\n  out["aiter"]["probe_error"]=type(exc).__name__+":"+str(exc)\nexcept Exception as exc:\n out["error"]=type(exc).__name__+":"+str(exc)\n out["trace"]=traceback.format_exc()[-12000:]\nprint(json.dumps(out,default=str,sort_keys=True))'''


def collect_attention_evidence(*, base_url: str = DEFAULT_BASE_URL, run: Callable[..., dict[str, Any]] = run_command) -> dict[str, Any]:
    live = discover_live_container(run=run)
    selected = live.get("selected") or {}
    container = str(selected.get("Names") or "")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": {
            "claim": "runtime-bound installed-vLLM attention selector evidence; not official HyperLoom R9700 support",
            "service_restarted": False,
            "second_model_copy_loaded": False,
        },
        "container": {"name": container, "candidate_count": len(live.get("candidates") or [])},
        "endpoint": request_models(base_url),
    }
    if not container:
        result["evidence_status"] = "unproved_no_vllm_container"
        result["probe_sha256"] = stable_json_sha256(result)
        return result
    inspected = inspect_container(container, run=run)
    result["container"].update(inspected)
    model_path = str(inspected.get("model_path") or "")
    if not model_path:
        result["evidence_status"] = "unproved_model_path_missing"
        result["probe_sha256"] = stable_json_sha256(result)
        return result
    try:
        max_model_len = int(str(inspected.get("max_model_len") or "8192"))
        tp_size = int(str(inspected.get("tensor_parallel_size") or "1"))
        dp_size = int(str(inspected.get("data_parallel_size") or "1"))
    except ValueError:
        max_model_len, tp_size, dp_size = 8192, 1, 1
    code = _inside_probe_code(model_path=model_path,dtype=str(inspected.get("dtype") or "auto"),max_model_len=max_model_len,tp_size=tp_size,dp_size=dp_size)
    inner = run(["docker", "exec", container, "python3", "-c", code], timeout=120.0)
    parsed = last_json_object(str(inner.get("stdout") or "")) or {}
    result["vllm_selection"] = parsed
    result["selection_probe"] = {"returncode": inner.get("returncode"), "stderr_tail": str(inner.get("stderr") or "")[-4000:]}
    valid = ((parsed.get("selector") or {}).get("valid") if isinstance(parsed, dict) else None) or []
    predicted = choose_backend(valid)
    selected_backend = parsed.get("selected_backend") if isinstance(parsed, dict) else None
    result["selection_summary"] = {
        "selected_backend": selected_backend,
        "priority_prediction": predicted,
        "aiter_vllm_supported": ((parsed.get("aiter") or {}).get("vllm_supported") if isinstance(parsed, dict) else None),
    }
    runtime_bound = bool(result["endpoint"].get("available")) and bool(model_path)
    rocm = isinstance(parsed.get("platform"), dict) and parsed["platform"].get("is_rocm") is True
    selected_proved = isinstance(selected_backend, dict) and bool(selected_backend.get("class"))
    priority_consistent = predicted is not None and isinstance(selected_backend, dict) and (predicted.get("path") == selected_backend.get("class"))
    result["evidence_status"] = "proved" if runtime_bound and rocm and selected_proved and priority_consistent and inner.get("returncode") == 0 else "partial"
    result["probe_sha256"] = stable_json_sha256(result)
    return result


def write_evidence(payload: dict[str, Any], output: Path | None = None) -> Path:
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "docs" / "evidence" / f"r9700_attention_backend_probe_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    payload = collect_attention_evidence(base_url=args.base_url)
    output = write_evidence(payload, Path(args.output) if args.output else None)
    selected = payload.get("selection_summary", {}).get("selected_backend") or {}
    print(json.dumps({"ok": payload.get("evidence_status") == "proved", "evidence_status": payload.get("evidence_status"), "selected_backend": selected.get("name"), "selected_class": selected.get("class"), "aiter_vllm_supported": payload.get("selection_summary", {}).get("aiter_vllm_supported"), "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output), "probe_sha256": payload.get("probe_sha256")}, sort_keys=True))
    return 0 if (not args.strict or payload.get("evidence_status") == "proved") else 2


if __name__ == "__main__":
    raise SystemExit(main())
