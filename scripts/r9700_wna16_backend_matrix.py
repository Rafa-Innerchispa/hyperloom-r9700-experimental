#!/usr/bin/env python3
"""Read-only WNA16 MoE backend eligibility probe for R9700/gfx1201.

This script inspects the live vLLM ROCm container and replays the installed
WNA16 backend oracle without loading a second model or restarting the service.
It records per-backend incompatibility reasons, kernel support decisions and a
small source snapshot proving the active hard gates in the installed runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from r9700_awq_backend_probe import (
    DEFAULT_BASE_URL,
    discover_live_container,
    inspect_container,
    last_json_object,
    request_models,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.wna16_backend_matrix.v1"


def stable_json_sha256(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_command(argv: list[str], timeout: float = 120.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-100000:],
            "stderr": proc.stderr[-20000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "returncode": None, "error": type(exc).__name__}


def inside_probe_code(model_path: str, dtype: str, max_model_len: int, tp_size: int, dp_size: int) -> str:
    model_json = json.dumps(model_path)
    dtype_json = json.dumps(dtype)
    return textwrap.dedent(
        f'''
        import inspect, json, traceback
        out={{}}
        try:
            import torch, vllm
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.platforms import current_platform
            from vllm.engine.arg_utils import EngineArgs
            from vllm.model_executor.layers.fused_moe.layer import make_parallel_config, create_fused_moe_router
            from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            from vllm.model_executor.layers.quantization.auto_awq import AutoAWQMoEMethod
            from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
                WNA16MoEBackend, _backend_incompatibility_reason, backend_to_kernel_cls
            )
            from vllm.model_executor.layers.quantization.utils.marlin_utils import check_moe_marlin_supports_config
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
            from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingExpertsBase

            cfg=EngineArgs(
                model={model_json}, dtype={dtype_json}, max_model_len={int(max_model_len)},
                trust_remote_code=True, tensor_parallel_size={int(tp_size)}, data_parallel_size={int(dp_size)}
            ).create_engine_config()
            h=cfg.model_config.hf_config
            qc=cfg.quant_config
            pc=make_parallel_config(
                tp_size={int(tp_size)}, dp_size={int(dp_size)}, pcp_size=1,
                is_sequence_parallel=False, parallel_config=cfg.parallel_config
            )
            router=create_fused_moe_router(
                top_k=h.num_experts_per_tok,
                global_num_experts=h.num_local_experts,
                eplb_state=None,
                renormalize=bool(getattr(h,"norm_topk_prob",False)),
                use_grouped_topk=False,
                num_expert_group=None,
                topk_group=None,
                custom_routing_function=None,
                scoring_func="softmax",
                routed_scaling_factor=1.0,
                e_score_correction_bias=None,
                num_fused_shared_experts=0,
                shared_expert_weight=1.0,
            )
            mc=FusedMoEConfig(
                num_experts=h.num_local_experts,
                experts_per_token=h.num_experts_per_tok,
                hidden_dim=h.hidden_size,
                intermediate_size=h.moe_intermediate_size,
                num_local_experts=h.num_local_experts,
                num_logical_experts=h.num_local_experts,
                activation=MoEActivation.from_str("silu"),
                device=cfg.device_config.device,
                routing_method=router.routing_method_type,
                moe_parallel_config=pc,
                in_dtype=cfg.model_config.dtype,
                moe_backend=cfg.kernel_config.moe_backend,
                max_num_tokens=cfg.scheduler_config.max_num_batched_tokens,
                has_bias=False,
                is_lora_enabled=False,
                max_capture_size=cfg.compilation_config.max_cudagraph_capture_size,
            )
            activation_format=(
                mk.FusedMoEActivationFormat.BatchedExperts
                if pc.use_batched_activation_format
                else mk.FusedMoEActivationFormat.Standard
            )
            out["runtime"]={{
                "vllm":getattr(vllm,"__version__",None),
                "torch":torch.__version__,
                "torch_hip":getattr(torch.version,"hip",None),
                "platform_class":type(current_platform).__module__+"."+type(current_platform).__name__,
                "is_rocm":current_platform.is_rocm(),
                "is_cuda_alike":current_platform.is_cuda_alike(),
                "capability":str(current_platform.get_device_capability()),
                "device_name":current_platform.get_device_name(),
            }}
            out["model"]={{
                "architecture":getattr(h,"architectures",None),
                "quant_config_class":type(qc).__module__+"."+type(qc).__name__ if qc else None,
                "quant_repr":repr(qc),
                "hidden_size":getattr(h,"hidden_size",None),
                "moe_intermediate_size":getattr(h,"moe_intermediate_size",None),
                "num_local_experts":getattr(h,"num_local_experts",None),
                "num_experts_per_tok":getattr(h,"num_experts_per_tok",None),
                "zero_point":getattr(qc,"zero_point",None),
                "group_size":getattr(qc,"group_size",None),
            }}
            backends=[
                WNA16MoEBackend.FLASHINFER_TRTLLM,
                WNA16MoEBackend.MARLIN,
                WNA16MoEBackend.BATCHED_MARLIN,
                WNA16MoEBackend.TRITON,
                WNA16MoEBackend.HUMMING,
                WNA16MoEBackend.EMULATION,
            ]
            matrix={{}}
            for backend in backends:
                row={{"backend":backend.value}}
                reason=_backend_incompatibility_reason(
                    backend=backend,
                    moe_config=mc,
                    quant_config=qc,
                    may_have_zp=bool(getattr(qc,"zero_point",False)),
                    may_have_bias=True,
                    allow_tile_padding=True,
                )
                row["oracle_incompatibility_reason"]=reason
                row["oracle_passed"]=reason is None
                kernels=[]
                if reason is None:
                    for cls in backend_to_kernel_cls(backend):
                        supported, k_reason=cls.is_supported_config(
                            cls, mc, kInt4Static, None, activation_format
                        )
                        kernels.append({{
                            "class":cls.__module__+"."+cls.__name__,
                            "supported":bool(supported),
                            "reason":k_reason,
                        }})
                row["kernel_checks"]=kernels
                matrix[backend.value]=row
            out["matrix"]=matrix
            selected=AutoAWQMoEMethod(qc,mc)
            out["selected"]={{
                "backend":getattr(selected.wna16_moe_backend,"value",str(selected.wna16_moe_backend)),
                "experts_cls":selected.experts_cls.__module__+"."+selected.experts_cls.__name__,
            }}
            out["hard_gates"]={{
                "marlin_check_source":inspect.getsource(check_moe_marlin_supports_config),
                "triton_device_source":inspect.getsource(TritonWNA16Experts._supports_current_device),
                "humming_device_source":inspect.getsource(HummingExpertsBase._supports_current_device),
                "oracle_source":inspect.getsource(_backend_incompatibility_reason),
            }}
            triton=matrix.get("TRITON",{{}})
            out["candidate_assessment"]={{
                "primary_target":"TRITON_AUTOAWQ_LAYOUT_ADAPTER",
                "why":triton.get("oracle_incompatibility_reason"),
                "safe_next_step":"offline/isolated AWQ layout conversion correctness test before any service restart",
                "concurrency_must_remain_fixed":2,
            }}
        except Exception as exc:
            out["error"]=type(exc).__name__+":"+str(exc)
            out["trace"]=traceback.format_exc()[-16000:]
        print(json.dumps(out,default=str,sort_keys=True))
        '''
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    live = discover_live_container(run=run_command)
    selected = live.get("selected") or {}
    container = str(selected.get("Names") or "")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": {
            "official_hyperloom_r9700_support": False,
            "service_restarted": False,
            "second_model_copy_loaded": False,
            "purpose": "backend eligibility and root-cause evidence only",
        },
        "endpoint": request_models(args.base_url),
        "container": {"name": container, "candidate_count": len(live.get("candidates") or [])},
    }
    if not container:
        result["evidence_status"] = "unproved_no_vllm_container"
    else:
        inspected = inspect_container(container, run=run_command)
        result["container"].update(inspected)
        model_path = str(inspected.get("model_path") or "")
        try:
            max_model_len = int(str(inspected.get("max_model_len") or "8192"))
            tp_size = int(str(inspected.get("tensor_parallel_size") or "1"))
            dp_size = int(str(inspected.get("data_parallel_size") or "1"))
        except ValueError:
            max_model_len, tp_size, dp_size = 8192, 1, 1
        if not model_path:
            result["evidence_status"] = "unproved_model_path_missing"
        else:
            code = inside_probe_code(
                model_path=model_path,
                dtype=str(inspected.get("dtype") or "auto"),
                max_model_len=max_model_len,
                tp_size=tp_size,
                dp_size=dp_size,
            )
            inner = run_command(["docker", "exec", container, "python3", "-c", code], timeout=180.0)
            parsed = last_json_object(str(inner.get("stdout") or "")) or {}
            result["live_probe"] = parsed
            result["selection_probe"] = {
                "returncode": inner.get("returncode"),
                "stderr_tail": str(inner.get("stderr") or "")[-6000:],
            }
            matrix = parsed.get("matrix") if isinstance(parsed, dict) else None
            selected_backend = (parsed.get("selected") or {}).get("backend") if isinstance(parsed, dict) else None
            result["evidence_status"] = (
                "proved" if isinstance(matrix, dict) and selected_backend else "partial"
            )

    result["probe_sha256"] = stable_json_sha256(result)
    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = ROOT / "docs" / "evidence" / f"r9700_wna16_backend_matrix_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence_status": result.get("evidence_status"),
        "output": str(out_path),
        "probe_sha256": result.get("probe_sha256"),
        "selected": ((result.get("live_probe") or {}).get("selected") or {}),
        "candidate_assessment": ((result.get("live_probe") or {}).get("candidate_assessment") or {}),
    }, sort_keys=True))
    if args.strict and result.get("evidence_status") != "proved":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
