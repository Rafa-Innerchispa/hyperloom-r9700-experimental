#!/usr/bin/env python3
"""Bounded real-checkpoint smoke for the R9700 hybrid WNA16 Experts path.

Loads eight layer-0 experts from the actual Qwen3-Coder AWQ checkpoint mounted by
the resident vLLM container, converts W1/W2 through the experimental hybrid
backend, and compares full MoE outputs against a dequantized FP32 reference.

The running vLLM server process is not patched or restarted. Only a child
interpreter in the existing container is used, and the test is bounded to one
expert subset so it fits beside the resident 30B model.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from r9700_awq_backend_probe import discover_live_container

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.wna16_hybrid_real_weight_smoke.v1"
PATCH = ROOT / "scripts" / "r9700_wna16_hybrid_patch.py"
MODEL = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"


def run(argv: list[str], timeout: float = 480.0) -> dict[str, Any]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "returncode": p.returncode,
        "stdout": p.stdout[-320000:],
        "stderr": p.stderr[-80000:],
    }


def digest(payload: dict[str, Any]) -> str:
    clean = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clean.pop("probe_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def child_code() -> str:
    return textwrap.dedent(
        rf'''
import hashlib,json,sys,traceback,torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0,'/tmp')
import r9700_wna16_hybrid_patch as hp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig,FusedMoEParallelConfig,RoutingMethodType
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import make_wna16_moe_quant_config,_unpack_and_dequant_int4_awq,make_wna16_moe_kernel,WNA16MoEBackend

MODEL=Path({MODEL!r});SHARD=MODEL/'model-00001-of-00006.safetensors';E=8;K=2048;I=768;N=I*2;TOPK=8

def thash(t):
    x=t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()

def load_real():
    gate_qw=[];gate_qz=[];gate_s=[];up_qw=[];up_qz=[];up_s=[];down_qw=[];down_qz=[];down_s=[]
    with safe_open(str(SHARD),framework='pt',device='cpu') as f:
        for e in range(E):
            p=f'model.layers.0.mlp.experts.{{e}}.'
            gate_qw.append(f.get_tensor(p+'gate_proj.qweight'));gate_qz.append(f.get_tensor(p+'gate_proj.qzeros'));gate_s.append(f.get_tensor(p+'gate_proj.scales'))
            up_qw.append(f.get_tensor(p+'up_proj.qweight'));up_qz.append(f.get_tensor(p+'up_proj.qzeros'));up_s.append(f.get_tensor(p+'up_proj.scales'))
            down_qw.append(f.get_tensor(p+'down_proj.qweight'));down_qz.append(f.get_tensor(p+'down_proj.qzeros'));down_s.append(f.get_tensor(p+'down_proj.scales'))
    w13_qw=torch.stack([torch.cat([gate_qw[e],up_qw[e]],dim=1) for e in range(E)])
    w13_qz=torch.stack([torch.cat([gate_qz[e],up_qz[e]],dim=1) for e in range(E)])
    w13_s=torch.stack([torch.cat([gate_s[e],up_s[e]],dim=1) for e in range(E)])
    w2_qw=torch.stack(down_qw);w2_qz=torch.stack(down_qz);w2_s=torch.stack(down_s)
    checks={{'w13_qweight_sha256':thash(w13_qw),'w13_qzeros_sha256':thash(w13_qz),'w13_scales_sha256':thash(w13_s),'w2_qweight_sha256':thash(w2_qw),'w2_qzeros_sha256':thash(w2_qz),'w2_scales_sha256':thash(w2_s)}}
    return tuple(x.cuda(non_blocking=False) for x in (w13_qw,w13_qz,w13_s,w2_qw,w2_qz,w2_s)),checks

def ref_moe(x,w13,w2,ids,tw):
    xf=x.float();w13f=w13.float();w2f=w2.float();y=torch.zeros((x.size(0),K),device=x.device,dtype=torch.float32)
    for j in range(TOPK):
        e=j
        z=xf@w13f[e].transpose(0,1)
        gate,up=z.split(I,dim=1)
        h=F.silu(gate)*up
        y += tw[:,j:j+1].float()*(h@w2f[e].transpose(0,1))
    return y

def main():
    out={{}}
    try:
        install=hp.install_patch(force=True)
        before_free,before_total=torch.cuda.mem_get_info()
        tensors,checks=load_real();w13_qw,w13_qz,w13_s,w2_qw,w2_qz,w2_s=tensors
        w1,s1,zp1=hp._awq_w13_to_packed_nfirst(w13_qw,w13_s,w13_qz)
        w2=hp._awq_w2_to_bf16(w2_qw,w2_s,w2_qz)
        w13_bf=_unpack_and_dequant_int4_awq(w13_qw,w13_s,w13_qz,transpose_output=True,output_dtype=torch.bfloat16)
        dummy=torch.ones(1,device='cuda',dtype=torch.float16)
        par=FusedMoEParallelConfig(tp_size=1,pcp_size=1,dp_size=1,ep_size=1,tp_rank=0,pcp_rank=0,dp_rank=0,ep_rank=0,sp_size=1,use_ep=False,all2all_backend='',enable_eplb=False)
        moe=FusedMoEConfig(num_experts=E,experts_per_token=TOPK,hidden_dim=K,intermediate_size=I,num_local_experts=E,num_logical_experts=E,activation=MoEActivation.SILU,device='cuda',routing_method=RoutingMethodType.TopK,moe_parallel_config=par,in_dtype=torch.bfloat16,intermediate_size_per_partition=I)
        qcfg=make_wna16_moe_quant_config(w1_scale=s1,w2_scale=dummy,group_size=128,num_bits=4,w1_zp=zp1,w2_zp=None)
        experts=hp.R9700HybridWNA16Experts(moe,qcfg)
        kernel=make_wna16_moe_kernel(moe_quant_config=qcfg,moe_config=moe,experts_cls=hp.R9700HybridWNA16Experts,backend=WNA16MoEBackend.TRITON)
        cases=[];torch.manual_seed(20260909)
        ids_base=torch.arange(TOPK,device='cuda',dtype=torch.int64).view(1,-1)
        for dtype in (torch.bfloat16,torch.float16):
            for M in (1,8,16,20):
                x=(torch.randn((M,K),device='cuda')*.05).to(dtype);ids=ids_base.expand(M,-1).contiguous();tw=torch.full((M,TOPK),1.0/TOPK,device='cuda',dtype=torch.float32)
                ws1_shape,ws2_shape,out_shape=experts.workspace_shapes(M,N,K,TOPK,E,E,None,MoEActivation.SILU)
                ws1=torch.empty(ws1_shape,device='cuda',dtype=dtype);ws2=torch.empty(ws2_shape,device='cuda',dtype=dtype);y=torch.empty(out_shape,device='cuda',dtype=dtype)
                experts.apply(output=y,hidden_states=x,w1=w1,w2=w2,topk_weights=tw,topk_ids=ids,activation=MoEActivation.SILU,global_num_experts=E,expert_map=None,a1q_scale=None,a2_scale=None,workspace13=ws1,workspace2=ws2,expert_tokens_meta=None,apply_router_weight_on_input=False)
                torch.cuda.synchronize();ref=ref_moe(x,w13_bf,w2,ids,tw);yf=y.float();diff=(yf-ref).abs();refnorm=float(torch.linalg.vector_norm(ref));reln=float(torch.linalg.vector_norm(yf-ref)/(torch.linalg.vector_norm(ref)+1e-12));cos=float(F.cosine_similarity(yf.flatten(),ref.flatten(),dim=0));finite=bool(torch.isfinite(y).all().item());max_abs=float(diff.max());mean_abs=float(diff.mean());scale=float(ref.abs().max())
                passed=finite and cos>=0.999 and reln<=0.035
                cases.append({{'M':M,'input_dtype':str(dtype),'path':'custom_small_w1' if M<=hp.SMALL_TOKEN_LIMIT else 'generic_wna16_w1_fallback','finite':finite,'cosine':cos,'relative_l2':reln,'max_abs_error':max_abs,'mean_abs_error':mean_abs,'reference_max_abs':scale,'reference_l2':refnorm,'pass':passed}})
        after_free,after_total=torch.cuda.mem_get_info()
        out={{'pass':all(c['pass'] for c in cases),'install':install,'kernel_type':type(kernel).__name__,'experts_type':type(experts).__name__,'source':{{'shard':str(SHARD),'layer':0,'experts':list(range(E)),'w13_qweight_shape':list(w13_qw.shape),'w13_qzeros_shape':list(w13_qz.shape),'w13_scales_shape':list(w13_s.shape),'w2_qweight_shape':list(w2_qw.shape),'w2_qzeros_shape':list(w2_qz.shape),'w2_scales_shape':list(w2_s.shape),'checksums':checks}},'converted':{{'w1_shape':list(w1.shape),'w1_dtype':str(w1.dtype),'w1_scale_shape':list(s1.shape),'w1_zp_shape':list(zp1.shape),'w2_shape':list(w2.shape),'w2_dtype':str(w2.dtype),'correction_shape':list(experts.w1_correction.shape),'correction_dtype':str(experts.w1_correction.dtype)}},'cases':cases,'gpu_memory':{{'free_before_bytes':int(before_free),'free_after_bytes':int(after_free),'total_bytes':int(after_total),'probe_delta_bytes':int(before_free-after_free)}},'truth_boundary':'real layer-0 Qwen AWQ W1/W2 weights with synthetic routed activations; child-process hybrid Experts correctness smoke; resident vLLM server untouched; no E2E serving performance claim'}}
    except Exception as e:
        out={{'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-40000:]}}
    print(json.dumps(out,sort_keys=True))
main()
'''
    )


def main() -> int:
    live = discover_live_container(run=run)
    container = str((live.get("selected") or {}).get("Names") or "")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "truth_boundary": {
            "server_process_patched": False,
            "service_restarted": False,
            "production_runtime_modified": False,
            "real_checkpoint_weights": True,
            "synthetic_routed_activations": True,
            "official_support_claim": False,
        },
    }
    if not container:
        payload.update({"pass": False, "error": "no_vllm_container"})
    else:
        cp = run(["docker", "cp", str(PATCH), f"{container}:/tmp/r9700_wna16_hybrid_patch.py"], 60)
        ex = run(["docker", "exec", container, "python3", "-c", child_code()], 480)
        parsed: dict[str, Any] = {}
        for line in ex["stdout"].splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                parsed = obj
        payload["copy"] = {"returncode": cp["returncode"], "stderr_tail": cp["stderr"][-2000:]}
        payload["probe"] = parsed
        payload["docker_exec"] = {"returncode": ex["returncode"], "stderr_tail": ex["stderr"][-16000:]}
        payload["pass"] = cp["returncode"] == 0 and ex["returncode"] == 0 and bool(parsed.get("pass"))
    payload["probe_sha256"] = digest(payload)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "docs" / "evidence" / f"r9700_wna16_hybrid_real_weight_smoke_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "pass": payload.get("pass"),
        "path": str(path.relative_to(ROOT)),
        "sha256": payload["probe_sha256"],
        "cases": (payload.get("probe") or {}).get("cases", []),
        "error": (payload.get("probe") or {}).get("error"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
