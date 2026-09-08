#!/usr/bin/env python3
"""Synthetic correctness proof for an AutoAWQ -> Triton WNA16 MoE layout adapter.

Runs inside the live vLLM container but does not touch model weights or restart
anything. It compares a candidate packed-layout conversion against vLLM's own
AutoAWQ dequantization reference on synthetic tensors.
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
SCHEMA = "hyperloom.r9700.awq_triton_layout_proof.v1"


def run(argv: list[str], timeout: float = 120.0) -> dict[str, Any]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-100000:], "stderr": proc.stderr[-20000:]}


def stable_hash(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inside_code() -> str:
    return textwrap.dedent(r'''
        import json, traceback, torch
        out={}
        try:
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
            from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER

            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            torch.manual_seed(20260908)

            def adapter(qw, scales, qz):
                # Mirrors the already-used dense AutoAWQ conversion semantics,
                # generalized to the leading expert dimension.
                E,K,Np=qw.shape
                N=Np*8
                shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device)
                reverse=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
                vals=((qw.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E,K,N)
                nfirst=vals.transpose(1,2).contiguous()  # E,N,K
                assert K%2==0
                wbytes=(nfirst[...,0::2] | (nfirst[...,1::2]<<4)).to(torch.uint8).contiguous()
                st=scales.transpose(1,2).contiguous()  # E,N,G
                zbytes=None
                if qz is not None:
                    G=qz.shape[1]
                    zvals=((qz.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E,G,N)
                    zt=zvals.transpose(1,2).contiguous()  # E,N,G
                    assert N%2==0
                    zbytes=(zt[:,0::2,:] | (zt[:,1::2,:]<<4)).to(torch.uint8).contiguous()
                return wbytes,st,zbytes

            def decode_triton(wbytes, st, zbytes, group_size):
                E,N,K2=wbytes.shape
                K=K2*2
                low=(wbytes&0xF).to(torch.int16)
                high=((wbytes>>4)&0xF).to(torch.int16)
                nfirst=torch.stack((low,high),dim=-1).reshape(E,N,K)
                q=nfirst.transpose(1,2).contiguous()  # E,K,N
                scales=st.transpose(1,2).contiguous()  # E,G,N
                if zbytes is None:
                    q=q-8
                else:
                    zl=(zbytes&0xF).to(torch.int16)
                    zh=((zbytes>>4)&0xF).to(torch.int16)
                    zt=torch.stack((zl,zh),dim=2).reshape(E,N,-1)
                    zp=zt.transpose(1,2).contiguous()  # E,G,N
                    q=q-zp.repeat_interleave(group_size,dim=1)
                return q.to(torch.float32)*scales.repeat_interleave(group_size,dim=1).to(torch.float32)

            def one_case(name,E,K,N,group_size):
                assert K%group_size==0 and N%8==0 and K%2==0 and N%2==0
                # Random packed int32 values are valid nibble containers and avoid
                # baking the candidate packer into the test fixture.
                qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=device)
                qz=torch.randint(-(2**31),2**31-1,(E,K//group_size,N//8),dtype=torch.int32,device=device)
                scales=(torch.rand((E,K//group_size,N),device=device,dtype=torch.float32)*0.09+0.001).to(torch.float16)
                ref=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=False,output_dtype=torch.float32)
                wbytes,st,zbytes=adapter(qw,scales,qz)
                got=decode_triton(wbytes,st,zbytes,group_size)
                diff=(ref-got).abs()
                return {
                    "name":name,
                    "shape":{"E":E,"K":K,"N":N,"group_size":group_size},
                    "device":str(device),
                    "max_abs_error":float(diff.max().item()),
                    "mean_abs_error":float(diff.mean().item()),
                    "exact":bool(torch.equal(ref,got)),
                    "allclose":bool(torch.allclose(ref,got,rtol=0,atol=0)),
                    "weight_bytes_shape":list(wbytes.shape),
                    "scale_shape":list(st.shape),
                    "zero_bytes_shape":list(zbytes.shape),
                }

            cases=[
                one_case("w13_like",2,256,384,128),
                one_case("w2_like",2,384,256,128),
                one_case("qwen3_r9700_reduced_ratio",1,2048,768,128),
            ]
            out["torch"]={"version":torch.__version__,"hip":getattr(torch.version,"hip",None),"device":str(device)}
            out["cases"]=cases
            out["pass"]=all(c["allclose"] for c in cases)
            out["interpretation"]="candidate AWQ->Triton packed layout preserves vLLM AutoAWQ dequantized values" if out["pass"] else "candidate layout does not match vLLM AutoAWQ reference; do not enable Triton"
        except Exception as exc:
            out["error"]=type(exc).__name__+":"+str(exc)
            out["trace"]=traceback.format_exc()[-16000:]
            out["pass"]=False
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
            "synthetic_layout_test":True,
            "live_model_weights_touched":False,
            "service_restarted":False,
            "kernel_enabled":False,
            "official_support_claim":False,
        },
        "container":container,
    }
    if not container:
        payload["pass"]=False
        payload["error"]="no_vllm_container"
    else:
        result=run(["docker","exec",container,"python3","-c",inside_code()],timeout=180)
        parsed={}
        for line in result.get("stdout","").splitlines():
            try:
                obj=json.loads(line)
            except Exception:
                continue
            if isinstance(obj,dict):
                parsed=obj
        payload["probe"]=parsed
        payload["docker_exec"]={"returncode":result.get("returncode"),"stderr_tail":result.get("stderr","")[-4000:]}
        payload["pass"]=bool(parsed.get("pass")) and result.get("returncode")==0
    payload["probe_sha256"]=stable_hash(payload)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=ROOT/"docs"/"evidence"/f"r9700_awq_triton_layout_proof_{stamp}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"pass":payload.get("pass"),"output":str(path),"probe_sha256":payload["probe_sha256"],"probe":payload.get("probe",{})},sort_keys=True))
    return 0 if payload.get("pass") else 2


if __name__=="__main__":
    raise SystemExit(main())
