#!/usr/bin/env python3
"""Validate the gfx1201 packed-INT4 W1 kernel against real Qwen3-Coder AWQ expert weights.

This probe intentionally loads only layer 0 / experts 0..7 from the local model
checkpoint while the resident vLLM server remains untouched. Activations and
routing are synthetic; weights are real checkpoint tensors.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.wna16_real_weight_layer_probe.v1"


def run(argv: list[str], timeout: float = 480.0) -> dict[str, Any]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": p.returncode, "stdout": p.stdout[-320000:], "stderr": p.stderr[-80000:]}


def digest(payload: dict[str, Any]) -> str:
    c = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    c.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inside_code() -> str:
    return textwrap.dedent(r'''
import glob, json, os, statistics, traceback, torch
import triton, triton.language as tl
from safetensors import safe_open
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

MODEL_DIR = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
LAYER = 0
EXPERTS = list(range(8))
TOPK = 8
CFG = {"BM":16,"BN":128,"nw":4,"waves":4,"mi":None}

@triton.jit
def moe_corr(A,B,C,S,R,sorted_ids,expert_ids,npost_ptr,
             N:tl.constexpr,K:tl.constexpr,EM,num_valid,
             sam,sak,sbe,sbk,sbn,scm,scn,sse,ssk,ssn,sre,srk,srn,
             top_k:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GK:tl.constexpr,GM:tl.constexpr):
    pid=tl.program_id(0);npm=tl.cdiv(EM,BM);npn=tl.cdiv(N,BN);npg=GM*npn
    gid=pid//npg;first=gid*GM;gsm=tl.minimum(npm-first,GM)
    pm=first+((pid%npg)%gsm);pn=(pid%npg)//gsm;npost=tl.load(npost_ptr)
    if pm*BM>=npost:return
    sid=pm*BM+tl.arange(0,BM).to(tl.int64);tok=tl.load(sorted_ids+sid).to(tl.int64);tm=tok<num_valid
    exp=tl.load(expert_ids+pm).to(tl.int64);on=pn*BN+tl.arange(0,BN).to(tl.int64);nm=on<N
    if exp==-1:
        tl.store(C+tok[:,None]*scm+on[None,:]*scn,0.0,mask=tm[:,None]&nm[None,:]);return
    ok=tl.arange(0,BK).to(tl.int64);acc=tl.zeros((BM,BN),tl.float32)
    for g in tl.range(0,K,GK):
        gi=g//GK
        sc=tl.load(S+exp*sse+on*ssn+gi*ssk,mask=nm,other=0.).to(tl.float32)
        corr=tl.load(R+exp*sre+on*srn+gi*srk,mask=nm,other=0.).to(tl.float32)
        qacc=tl.zeros((BM,BN),tl.float32);asum=tl.zeros((BM,),tl.float32)
        for sub in tl.static_range(0,GK,BK):
            kk=g+sub+ok
            a=tl.load(A+(tok[:,None]//top_k)*sam+kk[None,:]*sak,mask=tm[:,None]&(kk[None,:]<K),other=0.).to(tl.bfloat16)
            pb=tl.load(B+exp*sbe+(kk[:,None]//2)*sbk+on[None,:]*sbn,mask=(kk[:,None]<K)&nm[None,:],other=0)
            q=((pb>>((kk[:,None]%2)*4))&15).to(tl.bfloat16)
            qacc=tl.dot(a,q,acc=qacc);asum+=tl.sum(a.to(tl.float32),axis=1)
        acc+=qacc*sc[None,:]-asum[:,None]*corr[None,:]
    tl.store(C+tok[:,None]*scm+on[None,:]*scn,acc.to(tl.bfloat16),mask=tm[:,None]&nm[None,:])

def awq_to_triton(qw,scales,qz):
    E,K,Np=qw.shape;N=Np*8
    shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device)
    rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
    vals=((qw.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,K,N)
    nt=vals.transpose(1,2).contiguous()
    B=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous()
    S=scales.transpose(1,2).contiguous();G=qz.shape[1]
    zv=((qz.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous()
    R=(zv.float()*S.float()).to(torch.float16).contiguous()
    return B,S,R

def launch(A,B,C,S,R,sid,se,npad,topk,cfg):
    K=A.size(1);N=B.size(1);EM=sid.numel();grid=(triton.cdiv(EM,cfg['BM'])*triton.cdiv(N,cfg['BN']),)
    kw={'num_warps':cfg['nw'],'num_stages':1,'waves_per_eu':cfg['waves']}
    moe_corr[grid](A,B,C,S,R,sid,se,npad,N,K,EM,A.size(0)*topk,
        A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),
        S.stride(0),S.stride(2),S.stride(1),R.stride(0),R.stride(2),R.stride(1),
        top_k=topk,BM=cfg['BM'],BN=cfg['BN'],BK=32,GK=128,GM=1,**kw)

def find_shard(keys):
    for fp in sorted(glob.glob(MODEL_DIR+'/*.safetensors')):
        with safe_open(fp, framework='pt', device='cpu') as sf:
            ks=set(sf.keys())
            if all(k in ks for k in keys):
                return fp
    raise RuntimeError('required expert tensors not found in one shard')

def load_real_w1():
    gate0=f'model.layers.{LAYER}.mlp.experts.0.gate_proj.qweight'
    up0=f'model.layers.{LAYER}.mlp.experts.0.up_proj.qweight'
    fp=find_shard([gate0,up0])
    qws=[];qzs=[];scs=[]
    meta=[]
    with safe_open(fp, framework='pt', device='cpu') as sf:
        for e in EXPERTS:
            p=f'model.layers.{LAYER}.mlp.experts.{e}'
            gq=sf.get_tensor(p+'.gate_proj.qweight');uq=sf.get_tensor(p+'.up_proj.qweight')
            gz=sf.get_tensor(p+'.gate_proj.qzeros');uz=sf.get_tensor(p+'.up_proj.qzeros')
            gs=sf.get_tensor(p+'.gate_proj.scales');us=sf.get_tensor(p+'.up_proj.scales')
            qws.append(torch.cat([gq,uq],dim=1));qzs.append(torch.cat([gz,uz],dim=1));scs.append(torch.cat([gs,us],dim=1))
            meta.append({'expert':e,'gate_qweight':list(gq.shape),'up_qweight':list(uq.shape),'scale_dtype':str(gs.dtype)})
    return fp, torch.stack(qws), torch.stack(qzs), torch.stack(scs), meta

def measure_pair(fa,fb,reps=24,rounds=21):
    for _ in range(6): fa(); fb()
    torch.cuda.synchronize(); av=[];bv=[];rat=[]
    for i in range(rounds):
        order=('a','b') if i%2==0 else ('b','a')
        vals={}
        for lab in order:
            fn=fa if lab=='a' else fb
            st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(reps): fn()
            en.record();en.synchronize();vals[lab]=float(st.elapsed_time(en))/reps
        av.append(vals['a']);bv.append(vals['b']);rat.append(vals['b']/vals['a'])
    return av,bv,rat

def main():
    out={}
    try:
        torch.manual_seed(20260908)
        dev=torch.cuda.get_device_name(0)
        if 'R9700' not in dev: raise RuntimeError('expected R9700, got '+dev)
        free0,total0=torch.cuda.mem_get_info()
        fp,qw_cpu,qz_cpu,sc_cpu,meta=load_real_w1()
        source={'shard':os.path.basename(fp),'layer':LAYER,'experts':EXPERTS,'qweight_shape':list(qw_cpu.shape),'qzeros_shape':list(qz_cpu.shape),'scales_shape':list(sc_cpu.shape),'scales_dtype':str(sc_cpu.dtype),'expert_meta':meta}
        qw=qw_cpu.to('cuda');qz=qz_cpu.to('cuda');sc=sc_cpu.to('cuda')
        B,S,R=awq_to_triton(qw,sc,qz)
        Bbf=_unpack_and_dequant_int4_awq(qw,sc,qz,transpose_output=True,output_dtype=torch.bfloat16)
        E,K,N=Bbf.shape[0],Bbf.shape[2],Bbf.shape[1]
        dummy=torch.empty((E,K,768),device='cuda',dtype=torch.bfloat16)
        base=torch.arange(TOPK,device='cuda',dtype=torch.int64)
        rows=[]
        for M in (1,2,4,8,16):
            ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous()
            A=(torch.randn((M,K),device='cuda')*.1).to(torch.bfloat16)
            sid,se,npad=moe_align_block_size(ids,CFG['BM'],E,None)
            Ca=torch.zeros((M,TOPK,N),device='cuda',dtype=torch.bfloat16);Cb=torch.zeros_like(Ca)
            ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy.size(),TOPK,None,M))
            esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None)
            fa=lambda:launch(A,B,Ca,S,R,sid,se,npad,TOPK,CFG)
            fb=lambda:invoke_fused_moe_triton_kernel(A,Bbf,Cb,None,None,None,esid,ese,enpad,False,TOPK,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
            fa();fb();torch.cuda.synchronize()
            diff=(Ca.float()-Cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(Ca.float().flatten(),Cb.float().flatten(),dim=0));close=bool(torch.allclose(Ca.float(),Cb.float(),rtol=.08,atol=.08))
            reps=max(12,min(36,72//M));av,bv,rat=measure_pair(fa,fb,reps=reps,rounds=21)
            rows.append({'M':M,'candidate_median_ms':float(statistics.median(av)),'bf16_median_ms':float(statistics.median(bv)),'paired_median_speedup_x':float(statistics.median(rat)),'wins':sum(1 for x in rat if x>1.0),'wins_ge_1p05x':sum(1 for x in rat if x>=1.05),'cosine':cos,'max_abs_error':float(diff.max()),'allclose':close,'candidate_samples_ms':av,'bf16_samples_ms':bv,'paired_speedups_x':rat})
        free1,total1=torch.cuda.mem_get_info()
        out={'pass':all(r['allclose'] and r['cosine']>.999 for r in rows),'device':dev,'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'model_config_dtype':'bfloat16','activation_dtype_used':'torch.bfloat16','source':source,'kernel_config':CFG,'rows':rows,'summary':{'median_of_paired_medians_x':float(statistics.median([r['paired_median_speedup_x'] for r in rows])),'M1_speedup_x':rows[0]['paired_median_speedup_x'],'all_numeric_pass':all(r['allclose'] and r['cosine']>.999 for r in rows),'all_shapes_candidate_wins_majority':all(r['wins']>=11 for r in rows)},'gpu_memory':{'free_before_bytes':int(free0),'free_after_bytes':int(free1),'total_bytes':int(total1),'probe_delta_bytes':int(free0-free1)},'truth_boundary':'real Qwen3-Coder AWQ layer-0 W1 checkpoint weights for experts 0..7; synthetic BF16 activations/routing; resident vLLM process not patched/restarted; not end-to-end serving and not actual live MoE activation-dtype proof'}
    except Exception as e:
        out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
    print(json.dumps(out,sort_keys=True))
main()
''')


def main() -> int:
    live = discover_live_container(run=run)
    container = str((live.get("selected") or {}).get("Names") or "")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": {
            "service_restarted": False,
            "production_runtime_modified": False,
            "real_model_weights": True,
            "synthetic_activations": True,
            "real_r9700_kernel": True,
            "official_support_claim": False,
        },
        "container": container,
    }
    if not container:
        payload.update({"pass": False, "error": "no_vllm_container"})
    else:
        r = run(["docker", "exec", container, "python3", "-c", inside_code()], 480)
        parsed: dict[str, Any] = {}
        for line in r["stdout"].splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                parsed = obj
        payload["probe"] = parsed
        payload["docker_exec"] = {"returncode": r["returncode"], "stderr_tail": r["stderr"][-12000:]}
        payload["pass"] = bool(r["returncode"] == 0 and parsed.get("pass"))
    payload["probe_sha256"] = digest(payload)
    out_dir = ROOT / "docs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"r9700_wna16_real_weight_layer_probe_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": payload.get("pass"), "path": str(out_path.relative_to(ROOT)), "sha256": payload["probe_sha256"], "probe_summary": (payload.get("probe") or {}).get("summary"), "error": (payload.get("probe") or {}).get("error")}, indent=2, sort_keys=True))
    return 0 if payload.get("pass") else 1

if __name__ == "__main__":
    raise SystemExit(main())
