#!/usr/bin/env python3
"""Prove which AWQ execution paths vLLM selects on the live R9700 ROCm server.

The probe is deliberately read-only with respect to the running service. It binds
its evidence to the live Docker container and OpenAI-compatible /models endpoint,
then replays vLLM's own quantization/backend selection logic in a short-lived
`docker exec` Python process. It does not load a second copy of model weights.
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
SCHEMA = "hyperloom.r9700.awq_backend_observability.v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

_SECRET_RE = re.compile(
    r"(?i)(authorization|api[-_]?key|token|secret|password)(=|\s+)[^\s,]+|"
    r"(crsr_[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+)"
)


def redact(text: str) -> str:
    return _SECRET_RE.sub(
        lambda m: (m.group(1) + (m.group(2) or "=") + "REDACTED") if m.group(1) else "REDACTED",
        str(text or ""),
    )


def stable_json_sha256(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_command(argv: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": redact(proc.stdout)[-60000:],
            "stderr": redact(proc.stderr)[-12000:],
        }
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


def classify_linear_backend(*, use_triton_awq: bool, awq_gemm_source: str) -> dict[str, Any]:
    src = str(awq_gemm_source or "")
    if use_triton_awq:
        return {
            "backend": "TRITON_AWQ",
            "kernel_entry": "vllm.model_executor.layers.quantization.awq_triton.awq_gemm_triton",
            "proved": "awq_gemm_triton" in src,
        }
    return {
        "backend": "VLLM_CUSTOM_OP__C_AWQ",
        "kernel_entry": "torch.ops._C.awq_gemm",
        "proved": "torch.ops._C.awq_gemm" in src,
    }


def discover_live_container(run: Callable[..., dict[str, Any]] = run_command) -> dict[str, Any]:
    result = run(["docker", "ps", "--format", "{{json .}}"])
    rows = parse_json_lines(str(result.get("stdout") or ""))
    candidates = []
    for row in rows:
        blob = " ".join(str(row.get(k, "")) for k in ("Image", "Names", "Command"))
        if re.search(r"\b(vllm|rocm)\b", blob, re.IGNORECASE):
            candidates.append(row)
    selected = candidates[0] if candidates else {}
    return {"selected": selected, "candidates": candidates, "probe": result}


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
        "env_signals": [
            redact(x)
            for x in env
            if re.search(r"AWQ|VLLM|TRITON|MARLIN|MACHETE|HIP|ROCM|AITER", x, re.IGNORECASE)
        ],
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


def _inside_probe_code(
    *, model_path: str, dtype: str, max_model_len: int, tp_size: int, dp_size: int
) -> str:
    # Values are JSON-encoded before interpolation so paths cannot turn into code.
    model_json = json.dumps(model_path)
    dtype_json = json.dumps(dtype)
    return f'''import inspect, json, os, traceback\nout={{}}\ntry:\n import torch, vllm\n from vllm import envs\n from vllm.platforms import current_platform\n out["versions"]={{"vllm":getattr(vllm,"__version__",None),"torch":torch.__version__,"torch_hip":getattr(torch.version,"hip",None)}}\n out["platform"]={{"class":type(current_platform).__module__+"."+type(current_platform).__name__,"is_rocm":current_platform.is_rocm(),"is_cuda":current_platform.is_cuda(),"capability":str(current_platform.get_device_capability()),"device_name":current_platform.get_device_name()}}\n import vllm._custom_ops as ops\n gemm_src=inspect.getsource(ops.awq_gemm)\n deq_src=inspect.getsource(ops.awq_dequantize)\n use_triton=bool(envs.VLLM_USE_TRITON_AWQ)\n out["linear_awq"]={{"use_triton_awq":use_triton,"awq_gemm_source":gemm_src,"awq_dequantize_source":deq_src,"backend":"TRITON_AWQ" if use_triton else "VLLM_CUSTOM_OP__C_AWQ","kernel_entry":"awq_gemm_triton" if use_triton else "torch.ops._C.awq_gemm"}}\n from vllm.engine.arg_utils import EngineArgs\n cfg=EngineArgs(model={model_json},dtype={dtype_json},max_model_len={int(max_model_len)},trust_remote_code=True,tensor_parallel_size={int(tp_size)},data_parallel_size={int(dp_size)}).create_engine_config()\n h=cfg.model_config.hf_config\n qc=cfg.quant_config\n out["model_config"]={{"architecture":getattr(h,"architectures",None),"quant_config_class":type(qc).__module__+"."+type(qc).__name__ if qc else None,"quant_name":qc.get_name() if qc and hasattr(qc,"get_name") else None,"quant_repr":repr(qc),"hidden_size":getattr(h,"hidden_size",None),"moe_intermediate_size":getattr(h,"moe_intermediate_size",None),"num_local_experts":getattr(h,"num_local_experts",None),"num_experts_per_tok":getattr(h,"num_experts_per_tok",None),"norm_topk_prob":getattr(h,"norm_topk_prob",None)}}\n if all(getattr(h,n,None) is not None for n in ("moe_intermediate_size","num_local_experts","num_experts_per_tok")):\n  from vllm.model_executor.layers.fused_moe.layer import make_parallel_config, create_fused_moe_router\n  from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig\n  from vllm.model_executor.layers.fused_moe.activation import MoEActivation\n  from vllm.model_executor.layers.quantization.auto_awq import AutoAWQMoEMethod\n  pc=make_parallel_config(tp_size={int(tp_size)},dp_size={int(dp_size)},pcp_size=1,is_sequence_parallel=False,parallel_config=cfg.parallel_config)\n  router=create_fused_moe_router(top_k=h.num_experts_per_tok,global_num_experts=h.num_local_experts,eplb_state=None,renormalize=bool(getattr(h,"norm_topk_prob",False)),use_grouped_topk=False,num_expert_group=None,topk_group=None,custom_routing_function=None,scoring_func="softmax",routed_scaling_factor=1.0,e_score_correction_bias=None,num_fused_shared_experts=0,shared_expert_weight=1.0)\n  mc=FusedMoEConfig(num_experts=h.num_local_experts,experts_per_token=h.num_experts_per_tok,hidden_dim=h.hidden_size,intermediate_size=h.moe_intermediate_size,num_local_experts=h.num_local_experts,num_logical_experts=h.num_local_experts,activation=MoEActivation.from_str("silu"),device=cfg.device_config.device,routing_method=router.routing_method_type,moe_parallel_config=pc,in_dtype=cfg.model_config.dtype,moe_backend=cfg.kernel_config.moe_backend,max_num_tokens=cfg.scheduler_config.max_num_batched_tokens,has_bias=False,is_lora_enabled=False,max_capture_size=cfg.compilation_config.max_cudagraph_capture_size)\n  method=AutoAWQMoEMethod(qc,mc)\n  out["moe_awq"]={{"applicable":True,"backend":getattr(method.wna16_moe_backend,"value",str(method.wna16_moe_backend)),"experts_cls":method.experts_cls.__module__+"."+method.experts_cls.__name__,"routing_method":getattr(router.routing_method_type,"name",str(router.routing_method_type)),"tp_size":pc.tp_size,"dp_size":pc.dp_size,"ep_size":pc.ep_size,"use_ep":pc.use_ep,"all2all_backend":pc.all2all_backend,"max_num_tokens":cfg.scheduler_config.max_num_batched_tokens}}\n else:\n  out["moe_awq"]={{"applicable":False}}\nexcept Exception as exc:\n out["error"]=type(exc).__name__+":"+str(exc)\n out["trace"]=traceback.format_exc()[-12000:]\nprint(json.dumps(out,default=str,sort_keys=True))'''


def collect_backend_evidence(
    *,
    base_url: str = DEFAULT_BASE_URL,
    run: Callable[..., dict[str, Any]] = run_command,
) -> dict[str, Any]:
    live = discover_live_container(run=run)
    selected = live.get("selected") or {}
    container = str(selected.get("Names") or "")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": {
            "claim": "runtime-bound vLLM backend selection evidence; not official upstream HyperLoom R9700 support",
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
    dtype = str(inspected.get("dtype") or "auto")

    code = _inside_probe_code(
        model_path=model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    inner = run(["docker", "exec", container, "python3", "-c", code], timeout=120.0)
    parsed = last_json_object(str(inner.get("stdout") or "")) or {}
    result["vllm_selection"] = parsed
    result["selection_probe"] = {
        "returncode": inner.get("returncode"),
        "stderr_tail": str(inner.get("stderr") or "")[-4000:],
    }

    linear = parsed.get("linear_awq") if isinstance(parsed, dict) else None
    linear_proof = False
    if isinstance(linear, dict):
        classified = classify_linear_backend(
            use_triton_awq=bool(linear.get("use_triton_awq")),
            awq_gemm_source=str(linear.get("awq_gemm_source") or ""),
        )
        linear.update(classified)
        linear_proof = bool(classified.get("proved"))

    moe = parsed.get("moe_awq") if isinstance(parsed, dict) else None
    moe_proof = isinstance(moe, dict) and (
        moe.get("applicable") is False or (bool(moe.get("backend")) and bool(moe.get("experts_cls")))
    )
    platform = parsed.get("platform") if isinstance(parsed, dict) else {}
    runtime_bound = bool(container) and bool(result["endpoint"].get("available")) and bool(model_path)
    rocm_proof = isinstance(platform, dict) and platform.get("is_rocm") is True

    logs = run(["docker", "logs", "--tail", "1200", container], timeout=30.0)
    signal_lines = []
    for line in (str(logs.get("stdout") or "") + "\n" + str(logs.get("stderr") or "")).splitlines():
        if re.search(r"awq|wna16|triton|marlin|machete|quantiz|kernel", line, re.IGNORECASE):
            signal_lines.append(redact(line)[-700:])
    result["runtime_log_signals"] = signal_lines[-120:]

    proved = runtime_bound and rocm_proof and linear_proof and moe_proof and not parsed.get("error")
    result["evidence_status"] = "proved" if proved else "unproved"
    result["proof_checks"] = {
        "runtime_bound": runtime_bound,
        "rocm_platform": rocm_proof,
        "linear_awq_backend": linear_proof,
        "moe_backend_or_not_applicable": moe_proof,
        "inside_probe_error": parsed.get("error") if isinstance(parsed, dict) else "invalid_probe_output",
    }
    result["probe_sha256"] = stable_json_sha256(result)
    return result


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "docs" / "evidence" / f"r9700_awq_backend_probe_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true", help="exit non-zero unless backend evidence is proved")
    args = parser.parse_args()

    result = collect_backend_evidence(base_url=args.base_url)
    output = Path(args.output) if args.output else default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    selection = result.get("vllm_selection") or {}
    linear = selection.get("linear_awq") or {}
    moe = selection.get("moe_awq") or {}
    print(json.dumps({
        "ok": result.get("evidence_status") == "proved",
        "evidence_status": result.get("evidence_status"),
        "linear_backend": linear.get("backend"),
        "moe_backend": moe.get("backend") if moe.get("applicable") is not False else "not_applicable",
        "moe_experts_cls": moe.get("experts_cls"),
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "probe_sha256": result.get("probe_sha256"),
    }, sort_keys=True))
    if args.strict and result.get("evidence_status") != "proved":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
