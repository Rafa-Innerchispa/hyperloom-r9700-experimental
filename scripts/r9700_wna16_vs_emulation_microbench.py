#!/usr/bin/env python3
"""Microbenchmark real Triton WNA16 vs current dequantized Triton emulation.

Uses Qwen3 MoE dimensions on a reduced synthetic expert set. The running vLLM
service remains untouched. Concurrency/serving is not part of this benchmark;
this isolates the first expert GEMM backend so any speed difference is kernel-
attributable rather than a serving effect.
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
SCHEMA = "hyperloom.r9700.wna16_vs_emulation_microbench.v1"


def run(argv: list[str], timeout: float = 300.0) -> dict[str, Any]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-140000:], "stderr": proc.stderr[-40000:]}


def digest(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inside_code() -> str:
    return textwrap.dedent(r'''
        import json, math, statistics, time, traceback, torch
        out={}
        try:
            import triton.language as tl
            from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_wna16_triton_kernel,
                invoke_fused_moe_triton_kernel,
                try_get_optimal_moe_config,
            )
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
            from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config

            device=torch.device("cuda")
            torch.manual_seed(20260908)
            E=8
            K=2048
            N=1536  # Qwen3 gate+up output = 2 * moe_intermediate_size(768)
            N2=768
            group_size=128
            top_k=8

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

            qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=device)
            qz=torch.randint(-(2**31),2**31-1,(E,K//group_size,N//8),dtype=torch.int32,device=device)
            scales=(torch.rand((E,K//group_size,N),device=device,dtype=torch.float32)*0.02+0.002).to(torch.float16)
            Bq,Bs,Bz=awq_to_triton(qw,scales,qz)
            Bbf=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.bfloat16)

            dummy_w2_q=torch.empty((E,K,N2//2),dtype=torch.uint8,device=device)
            dummy_w2_bf=torch.empty((E,K,N2),dtype=torch.bfloat16,device=device)
            qcfg=int4_w4a16_moe_quant_config(
                w1_scale=Bs,
                w2_scale=torch.ones((E,K,N2//group_size),dtype=torch.float16,device=device),
                w1_zp=Bz,
                w2_zp=None,
                block_shape=[0,group_size],
            )
            qname=qcfg.config_name(torch.bfloat16)

            def bench_one(M):
                A=(torch.randn((M,K),device=device,dtype=torch.float32)*0.10).to(torch.bfloat16).contiguous()
                base=torch.arange(top_k,device=device,dtype=torch.int64)
                topk_ids=torch.stack([(base+i).remainder(E) for i in range(M)],dim=0).contiguous()

                qconfig=dict(try_get_optimal_moe_config(Bq.size(),dummy_w2_q.size(),top_k,qname,M,block_shape=[0,group_size]))
                econfig=dict(try_get_optimal_moe_config(Bbf.size(),dummy_w2_bf.size(),top_k,None,M))
                qsorted,qexperts,qpost=moe_align_block_size(topk_ids,qconfig["BLOCK_SIZE_M"],E,None)
                esorted,eexperts,epost=moe_align_block_size(topk_ids,econfig["BLOCK_SIZE_M"],E,None)
                Cq=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=device)
                Ce=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=device)

                def qcall():
                    invoke_fused_moe_wna16_triton_kernel(
                        A,Bq,Cq,Bs,Bz,None,qsorted,qexperts,qpost,
                        False,top_k,qconfig,tl.bfloat16,False,True,[0,group_size]
                    )

                def ecall():
                    invoke_fused_moe_triton_kernel(
                        A,Bbf,Ce,None,None,None,esorted,eexperts,epost,
                        False,top_k,econfig,tl.bfloat16,
                        False,False,False,False,False,None,None
                    )

                # correctness gate before timing
                qcall(); ecall(); torch.cuda.synchronize()
                dq=(Cq.float()-Ce.float()).abs()
                cosine=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0).item())
                close=bool(torch.allclose(Cq.float(),Ce.float(),rtol=0.08,atol=0.08))

                # Short warmup then alternating timed rounds to reduce drift.
                for _ in range(3): qcall(); ecall()
                torch.cuda.synchronize()
                reps=max(6, min(18, 72//max(M,1)))
                rounds=[]
                for order in (("q","e"),("e","q"),("q","e")):
                    row={}
                    for label in order:
                        fn=qcall if label=="q" else ecall
                        torch.cuda.synchronize()
                        t0=time.perf_counter()
                        for _ in range(reps): fn()
                        torch.cuda.synchronize()
                        row[label]=(time.perf_counter()-t0)*1000.0/reps
                    rounds.append(row)
                q_ms=[r["q"] for r in rounds]
                e_ms=[r["e"] for r in rounds]
                qmed=float(statistics.median(q_ms))
                emed=float(statistics.median(e_ms))
                speedup=emed/qmed if qmed>0 else None
                gain=(emed-qmed)/emed*100.0 if emed>0 else None
                return {
                    "M":M,
                    "reps_per_round":reps,
                    "qconfig":qconfig,
                    "emulation_config":econfig,
                    "correctness":{"allclose":close,"cosine":cosine,"max_abs_error":float(dq.max().item()),"mean_abs_error":float(dq.mean().item())},
                    "triton_wna16_ms_rounds":q_ms,
                    "emulation_bf16_ms_rounds":e_ms,
                    "triton_wna16_median_ms":qmed,
                    "emulation_bf16_median_ms":emed,
                    "speedup_x":speedup,
                    "latency_reduction_percent":gain,
                    "pass":close and cosine>0.999,
                }

            rows=[]
            for M in (1,2,4,8,16):
                rows.append(bench_one(M))
            valid=[r for r in rows if r["pass"] and r["speedup_x"] is not None]
            speedups=[r["speedup_x"] for r in valid]
            gains=[r["latency_reduction_percent"] for r in valid]
            out={
                "pass":len(valid)==len(rows),
                "device":torch.cuda.get_device_name(0),
                "torch":torch.__version__,
                "hip":getattr(torch.version,"hip",None),
                "scope":"first MoE expert GEMM microbenchmark; not end-to-end serving",
                "shape":{"E_reduced":E,"K":K,"N_gate_up":N,"N_down":N2,"top_k":top_k,"group_size":group_size},
                "rows":rows,
                "summary":{
                    "median_speedup_x":float(statistics.median(speedups)) if speedups else None,
                    "median_latency_reduction_percent":float(statistics.median(gains)) if gains else None,
                    "min_speedup_x":float(min(speedups)) if speedups else None,
                    "max_speedup_x":float(max(speedups)) if speedups else None,
                },
                "truth_boundary":"Compares real WNA16 Triton int4 GEMM against current emulation backend's pre-dequantized BF16 Triton GEMM on synthetic Qwen3-shaped data. Does not include model load or serving scheduler.",
            }
        except Exception as exc:
            out={"pass":False,"error":type(exc).__name__+":"+str(exc),"trace":traceback.format_exc()[-24000:]}
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
            "real_r9700_gpu_kernel":True,
            "synthetic_qwen3_shaped_data":True,
            "serving_benchmark":False,
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
        result=run(["docker","exec",container,"python3","-c",inside_code()],timeout=420)
        parsed={}
        for line in result.get("stdout","").splitlines():
            try: obj=json.loads(line)
            except Exception: continue
            if isinstance(obj,dict): parsed=obj
        payload["probe"]=parsed
        payload["docker_exec"]={"returncode":result.get("returncode"),"stderr_tail":result.get("stderr","")[-8000:]}
        payload["pass"]=bool(parsed.get("pass")) and result.get("returncode")==0
    payload["probe_sha256"]=digest(payload)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=ROOT/"docs"/"evidence"/f"r9700_wna16_vs_emulation_microbench_{stamp}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"pass":payload.get("pass"),"output":str(path),"probe_sha256":payload["probe_sha256"],"summary":(payload.get("probe") or {}).get("summary"),"rows":(payload.get("probe") or {}).get("rows")},sort_keys=True))
    return 0 if payload.get("pass") else 2


if __name__=="__main__":
    raise SystemExit(main())
