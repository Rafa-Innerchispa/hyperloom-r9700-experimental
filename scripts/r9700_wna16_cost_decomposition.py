#!/usr/bin/env python3
"""Decompose R9700 WNA16 cost into GEMM, dequant, and nibble-unpack overhead.

Uses real Triton kernels on the live R9700 with synthetic Qwen3-shaped tensors.
It does not modify vLLM or model weights. The goal is diagnostic: decide whether
the remaining gap is dominated by packed INT4 unpacking or by scale/zp dequant.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_cost_decomposition.v1'

def run(argv:list[str],timeout:float=300.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-180000:],'stderr':p.stderr[-30000:]}

def digest(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False)); c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inside_code()->str:
    return textwrap.dedent(r'''
import json, statistics, time, traceback, torch
import triton, triton.language as tl
BM=tl.constexpr(16); BN=tl.constexpr(64); BK=tl.constexpr(32); GROUP=tl.constexpr(128)
PY_GROUP=128

@triton.jit
def bf16_k(A,W,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    pn=tl.program_id(0); om=tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k in tl.range(0,K,BK):
        a=tl.load(A+om[:,None]*K+(k+ok)[None,:],mask=(om[:,None]<M)&((k+ok)[None,:]<K),other=0.).to(tl.bfloat16)
        w=tl.load(W+(k+ok)[:,None]*N+on[None,:],mask=((k+ok)[:,None]<K)&(on[None,:]<N),other=0.).to(tl.bfloat16)
        acc=tl.dot(a,w,acc=acc)
    tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))

@triton.jit
def unpacked_k(A,Q,S,Z,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    pn=tl.program_id(0); om=tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for g in tl.range(0,K,GROUP):
        gi=g//GROUP
        sc=tl.load(S+gi*N+on,mask=on<N,other=0.).to(tl.float32)
        zb=tl.load(Z+gi*(N//2)+(on//2),mask=on<N,other=0)
        zp=((zb>>((on%2)*4))&15).to(tl.float32)
        for sub in tl.static_range(0,GROUP,BK):
            kk=g+sub+ok
            a=tl.load(A+om[:,None]*K+kk[None,:],mask=(om[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.bfloat16)
            q=tl.load(Q+kk[:,None]*N+on[None,:],mask=(kk[:,None]<K)&(on[None,:]<N),other=0).to(tl.float32)
            w=((q-zp[None,:])*sc[None,:]).to(tl.bfloat16)
            acc=tl.dot(a,w,acc=acc)
    tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))

@triton.jit
def packed_k(A,Qp,S,Z,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    pn=tl.program_id(0); om=tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for g in tl.range(0,K,GROUP):
        gi=g//GROUP
        sc=tl.load(S+gi*N+on,mask=on<N,other=0.).to(tl.float32)
        zb=tl.load(Z+gi*(N//2)+(on//2),mask=on<N,other=0)
        zp=((zb>>((on%2)*4))&15).to(tl.float32)
        for sub in tl.static_range(0,GROUP,BK):
            kk=g+sub+ok
            a=tl.load(A+om[:,None]*K+kk[None,:],mask=(om[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.bfloat16)
            pb=tl.load(Qp+(kk[:,None]//2)*N+on[None,:],mask=(kk[:,None]<K)&(on[None,:]<N),other=0)
            q=((pb>>((kk[:,None]%2)*4))&15).to(tl.float32)
            w=((q-zp[None,:])*sc[None,:]).to(tl.bfloat16)
            acc=tl.dot(a,w,acc=acc)
    tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))

def bench(fn,reps=40):
    for _ in range(4): fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); vals.append((time.perf_counter()-t)*1000/reps)
    return float(statistics.median(vals)),vals

def main():
  out={}
  try:
    torch.manual_seed(20260908); d='cuda'; M=16;N=1536;K=2048; G=K//PY_GROUP
    A=torch.randn((M,K),device=d,dtype=torch.bfloat16)*.1
    Q=torch.randint(0,16,(K,N),device=d,dtype=torch.uint8)
    Qp=(Q[0::2] | (Q[1::2]<<4)).contiguous()
    S=(torch.rand((G,N),device=d,dtype=torch.float32)*.02+.002).to(torch.float16)
    zp=torch.randint(0,16,(G,N),device=d,dtype=torch.uint8)
    Z=(zp[:,0::2] | (zp[:,1::2]<<4)).contiguous()
    W=((Q.float()-zp.repeat_interleave(PY_GROUP,dim=0).float())*S.repeat_interleave(PY_GROUP,dim=0).float()).to(torch.bfloat16).contiguous()
    C0=torch.empty((M,N),device=d,dtype=torch.bfloat16); C1=torch.empty_like(C0); C2=torch.empty_like(C0)
    grid=(triton.cdiv(N,BN),)
    f0=lambda: bf16_k[grid](A,W,C0,M=M,N=N,K=K,num_warps=2,num_stages=1,waves_per_eu=2)
    f1=lambda: unpacked_k[grid](A,Q,S,Z,C1,M=M,N=N,K=K,num_warps=2,num_stages=1,waves_per_eu=2)
    f2=lambda: packed_k[grid](A,Qp,S,Z,C2,M=M,N=N,K=K,num_warps=2,num_stages=1,waves_per_eu=2)
    f0();f1();f2();torch.cuda.synchronize()
    ref=C0.float(); metrics={}
    for name,C in [('unpacked',C1),('packed',C2)]:
      diff=(C.float()-ref).abs(); metrics[name]={'cosine':float(torch.nn.functional.cosine_similarity(C.float().flatten(),ref.flatten(),dim=0).item()),'max_abs_error':float(diff.max().item()),'allclose':bool(torch.allclose(C.float(),ref,rtol=.08,atol=.08))}
    t0,v0=bench(f0);t1,v1=bench(f1);t2,v2=bench(f2)
    out={'pass':all(x['allclose'] and x['cosine']>.999 for x in metrics.values()),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'M':M,'N':N,'K':K,'group_size':PY_GROUP},'ms':{'bf16_predequant':t0,'int4_unpacked_dequant':t1,'int4_packed_dequant':t2},'ratios':{'unpacked_vs_bf16_x':t0/t1,'packed_vs_bf16_x':t0/t2,'packed_vs_unpacked_x':t1/t2,'unpack_extra_percent':(t2-t1)/t1*100,'dequant_plus_unpacked_over_bf16_percent':(t1-t0)/t0*100},'numeric':metrics,'samples_ms':{'bf16':v0,'unpacked':v1,'packed':v2},'interpretation':'packed_vs_unpacked isolates nibble-unpack overhead; unpacked_vs_bf16 isolates zp/scale dequant overhead plus uint8 loads','truth_boundary':'single-expert synthetic Qwen3-shaped kernel decomposition; no serving claim'}
  except Exception as e: out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-20000:]}
  print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
    if not c: payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',inside_code()],300); parsed={}
        for line in r['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-10000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); p=ROOT/'docs'/'evidence'/f'r9700_wna16_cost_decomposition_{stamp}.json'; p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'probe':payload.get('probe',{})},sort_keys=True)); return 0 if payload.get('pass') else 2
if __name__=='__main__': raise SystemExit(main())
