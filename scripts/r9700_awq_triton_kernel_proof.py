#!/usr/bin/env python3
"""Run the real Triton WNA16 GEMM on R9700 using synthetic AutoAWQ data.

The test is isolated inside the already-running vLLM container. It does not
restart the service, load model weights, or alter vLLM files. A candidate
AutoAWQ->Triton layout adapter is compared against a dequantized matmul
reference, then the actual fused_moe_kernel_gptq_awq path is launched.
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
SCHEMA = "hyperloom.r9700.awq_triton_kernel_proof.v1"


def run(argv: list[str], timeout: float = 180.0) -> dict[str, Any]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-120000:], "stderr": proc.stderr[-30000:]}


def digest(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inside_code() -> str:
    return textwrap.dedent(r'''
        import json, traceback, torch
        out={}
        try:
            import triton.language as tl
            from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_wna16_triton_kernel,
                try_get_optimal_moe_config,
            )
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
            from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config

            device=torch.device("cuda")
            torch.manual_seed(20260908)
            E=4
            M=4
            K=256
            N=384
            group_size=128
            top_k=1

            def awq_to_triton(qw, scales, qz):
                E0,K0,Np=qw.shape
                N0=Np*8
                shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device)
                reverse=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
                vals=((qw.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E0,K0,N0)
                nfirst=vals.transpose(1,2).contiguous()
                wbytes=(nfirst[...,0::2] | (nfirst[...,1::2]<<4)).to(torch.uint8).contiguous()
                st=scales.transpose(1,2).contiguous()
                G=qz.shape[1]
                zvals=((qz.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E0,G,N0)
                zt=zvals.transpose(1,2).contiguous()
                zbytes=(zt[:,0::2,:] | (zt[:,1::2,:]<<4)).to(torch.uint8).contiguous()
                return wbytes,st,zbytes

            # Synthetic asymmetric AutoAWQ-like tensors.
            qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=device)
            qz=torch.randint(-(2**31),2**31-1,(E,K//group_size,N//8),dtype=torch.int32,device=device)
            scales=(torch.rand((E,K//group_size,N),device=device,dtype=torch.float32)*0.025+0.002).to(torch.float16)
            B,Bs,Bz=awq_to_triton(qw,scales,qz)
            A=(torch.randn((M,K),device=device,dtype=torch.float32)*0.15).to(torch.bfloat16).contiguous()
            topk_ids=torch.arange(M,device=device,dtype=torch.int64).remainder(E).reshape(M,1)

            # Reference uses vLLM's own AWQ dequantizer.
            ref_weights=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.float32)
            ref=torch.stack([A[i].float() @ ref_weights[int(topk_ids[i,0])].transpose(0,1) for i in range(M)],dim=0)

            # Obtain the same style of config TritonExperts would request.
            dummy_w2=torch.empty((E,K,(N//2)//2),dtype=torch.uint8,device=device)
            qcfg=int4_w4a16_moe_quant_config(
                w1_scale=Bs,
                w2_scale=torch.ones((E,K,(N//2)//group_size),dtype=torch.float16,device=device),
                w1_zp=Bz,
                w2_zp=None,
                block_shape=[0,group_size],
            )
            config_name=qcfg.config_name(A.dtype)
            config=try_get_optimal_moe_config(
                B.size(), dummy_w2.size(), top_k, config_name, M, block_shape=[0,group_size]
            )
            config=dict(config)
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, config["BLOCK_SIZE_M"], E, None
            )
            C=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=device)
            torch.cuda.synchronize()
            invoke_fused_moe_wna16_triton_kernel(
                A,B,C,Bs,Bz,None,
                sorted_token_ids,expert_ids,num_tokens_post_padded,
                False,top_k,config,tl.bfloat16,
                False,True,[0,group_size],
            )
            torch.cuda.synchronize()
            got=C[:,0,:].float()
            diff=(got-ref).abs()
            denom=ref.abs().clamp_min(1e-3)
            rel=(diff/denom)
            cos=torch.nn.functional.cosine_similarity(got.flatten(),ref.flatten(),dim=0)
            finite=bool(torch.isfinite(got).all().item())
            max_abs=float(diff.max().item())
            mean_abs=float(diff.mean().item())
            mean_rel=float(rel.mean().item())
            p99_abs=float(torch.quantile(diff.flatten(),0.99).item())
            allclose=bool(torch.allclose(got,ref,rtol=0.08,atol=0.08))
            passed=finite and allclose and float(cos.item())>0.999
            out={
                "pass":passed,
                "device":torch.cuda.get_device_name(0),
                "torch":torch.__version__,
                "hip":getattr(torch.version,"hip",None),
                "shape":{"E":E,"M":M,"K":K,"N":N,"top_k":top_k,"group_size":group_size},
                "config_name":config_name,
                "config":config,
                "routing":{"topk_ids":topk_ids.cpu().tolist(),"sorted_count":int(sorted_token_ids.numel()),"expert_ids_count":int(expert_ids.numel())},
                "numeric":{"max_abs_error":max_abs,"p99_abs_error":p99_abs,"mean_abs_error":mean_abs,"mean_relative_error":mean_rel,"cosine":float(cos.item()),"allclose_rtol_0p08_atol_0p08":allclose,"finite":finite},
                "interpretation":"real Triton WNA16 kernel matches dequantized AutoAWQ reference within BF16 tolerance" if passed else "kernel result failed numeric gate; do not enable candidate",
            }
        except Exception as exc:
            out={"pass":False,"error":type(exc).__name__+":"+str(exc),"trace":traceback.format_exc()[-20000:]}
        print(json.dumps(out,sort_keys=True))
    ''')


def main() -> int:
    live=discover_live_container(run=run)
    selected=live.get("selected") or {}
    container=str(selected.get("Names") or "")
    payload={
        "schema":SCHEMA,
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "truth_boundary":{
            "synthetic_kernel_test":True,
            "real_r9700_gpu_kernel":True,
            "live_model_weights_touched":False,
            "service_restarted":False,
            "production_runtime_modified":False,
            "official_support_claim":False,
        },
        "container":container,
    }
    if not container:
        payload.update({"pass":False,"error":"no_vllm_container"})
    else:
        result=run(["docker","exec",container,"python3","-c",inside_code()],timeout=240)
        parsed={}
        for line in result.get("stdout","").splitlines():
            try: obj=json.loads(line)
            except Exception: continue
            if isinstance(obj,dict): parsed=obj
        payload["probe"]=parsed
        payload["docker_exec"]={"returncode":result.get("returncode"),"stderr_tail":result.get("stderr","")[-6000:]}
        payload["pass"]=bool(parsed.get("pass")) and result.get("returncode")==0
    payload["probe_sha256"]=digest(payload)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=ROOT/"docs"/"evidence"/f"r9700_awq_triton_kernel_proof_{stamp}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"pass":payload.get("pass"),"output":str(path),"probe_sha256":payload["probe_sha256"],"probe":payload.get("probe",{})},sort_keys=True))
    return 0 if payload.get("pass") else 2


if __name__=="__main__":
    raise SystemExit(main())
