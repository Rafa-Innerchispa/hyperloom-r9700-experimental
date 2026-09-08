#!/usr/bin/env python3
"""Groupwise algebraic WNA16 kernel experiment for RDNA4/gfx1201.

For an AWQ group with constant scale s[n] and zero-point z[n]:
  A @ ((Q-z)*s) = (A@Q)*s - sum(A)*z*s
This moves scale/zero-point work out of the KxN elementwise dequant path and
reduces it to post-dot vector corrections per group. Q values 0..15 are exactly
representable in BF16. This is tested on the real R9700 with synthetic
Qwen3-shaped tensors and does not patch the running vLLM service.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_groupwise_algebraic_kernel.v1'

def run(argv:list[str],timeout:float=360.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-240000:],'stderr':p.stderr[-40000:]}

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
    pm=tl.program_id(0); pn=tl.program_id(1); om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k in tl.range(0,K,BK):
        a=tl.load(A+om[:,None]*K+(k+ok)[None,:],mask=(om[:,None]<M)&((k+ok)[None,:]<K),other=0.).to(tl.bfloat16)
        w=tl.load(W+(k+ok)[:,None]*N+on[None,:],mask=((k+ok)[:,None]<K)&(on[None,:]<N),other=0.).to(tl.bfloat16)
        acc=tl.dot(a,w,acc=acc)
    tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))

@triton.jit
def packed_dequant_k(A,Qp,S,Z,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1); om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
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

@triton.jit
def algebraic_k(A,Qp,S,Z,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1); om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); ok=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for g in tl.range(0,K,GROUP):
        gi=g//GROUP
        sc=tl.load(S+gi*N+on,mask=on<N,other=0.).to(tl.float32)
        zb=tl.load(Z+gi*(N//2)+(on//2),mask=on<N,other=0)
        zp=((zb>>((on%2)*4))&15).to(tl.float32)
        qacc=tl.zeros((BM,BN),tl.float32)
        asum=tl.zeros((BM,),tl.float32)
        for sub in tl.static_range(0,GROUP,BK):
            kk=g+sub+ok
            a=tl.load(A+om[:,None]*K+kk[None,:],mask=(om[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.bfloat16)
            pb=tl.load(Qp+(kk[:,None]//2)*N+on[None,:],mask=(kk[:,None]<K)&(on[None,:]<N),other=0)
            q=((pb>>((kk[:,None]%2)*4))&15).to(tl.bfloat16)
            qacc=tl.dot(a,q,acc=qacc)
            asum += tl.sum(a.to(tl.float32),axis=1)
        acc += qacc*sc[None,:] - asum[:,None]*(zp*sc)[None,:]
    tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))

def bench(fn,reps):
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
    torch.manual_seed(20260908); d='cuda'; N=1536;K=2048; G=K//PY_GROUP
    Q=torch.randint(0,16,(K,N),device=d,dtype=torch.uint8); Qp=(Q[0::2]|(Q[1::2]<<4)).contiguous()
    S=(torch.rand((G,N),device=d,dtype=torch.float32)*.02+.002).to(torch.float16)
    zp=torch.randint(0,16,(G,N),device=d,dtype=torch.uint8); Z=(zp[:,0::2]|(zp[:,1::2]<<4)).contiguous()
    W=((Q.float()-zp.repeat_interleave(PY_GROUP,dim=0).float())*S.repeat_interleave(PY_GROUP,dim=0).float()).to(torch.bfloat16).contiguous()
    rows=[]
    configs=[{'num_warps':nw,'num_stages':1,'waves_per_eu':w} for nw in (1,2,4) for w in (0,1,2,4)]
    for cfg in configs:
      M=16; A=(torch.randn((M,K),device=d,dtype=torch.float32)*.1).to(torch.bfloat16); C=torch.empty((M,N),device=d,dtype=torch.bfloat16); grid=(triton.cdiv(M,16),triton.cdiv(N,64))
      fn=lambda: algebraic_k[grid](A,Qp,S,Z,C,M=M,N=N,K=K,**cfg)
      try:
        fn(); torch.cuda.synchronize(); ms,_=bench(fn,18); rows.append({'config':cfg,'ms':ms,'finite':bool(torch.isfinite(C).all().item())})
      except Exception as e: rows.append({'config':cfg,'finite':False,'error':type(e).__name__+':'+str(e)})
    valid=[r for r in rows if r.get('finite')]; best=min(valid,key=lambda r:r['ms']); cfg=best['config']; comps=[]
    for M in (1,2,4,8,16,32):
      A=(torch.randn((M,K),device=d,dtype=torch.float32)*.1).to(torch.bfloat16); C0=torch.empty((M,N),device=d,dtype=torch.bfloat16); C1=torch.empty_like(C0); C2=torch.empty_like(C0); grid=(triton.cdiv(M,16),triton.cdiv(N,64))
      f0=lambda: bf16_k[grid](A,W,C0,M=M,N=N,K=K,num_warps=2,num_stages=1,waves_per_eu=2)
      f1=lambda: packed_dequant_k[grid](A,Qp,S,Z,C1,M=M,N=N,K=K,num_warps=2,num_stages=1,waves_per_eu=2)
      f2=lambda: algebraic_k[grid](A,Qp,S,Z,C2,M=M,N=N,K=K,**cfg)
      f0();f1();f2();torch.cuda.synchronize(); ref=C0.float(); metrics={}
      for name,C in [('packed',C1),('algebraic',C2)]:
        diff=(C.float()-ref).abs(); metrics[name]={'cosine':float(torch.nn.functional.cosine_similarity(C.float().flatten(),ref.flatten(),dim=0).item()),'max_abs_error':float(diff.max().item()),'mean_abs_error':float(diff.mean().item()),'allclose':bool(torch.allclose(C.float(),ref,rtol=.08,atol=.08))}
      reps=max(8,min(30,64//max(1,M))); t0,_=bench(f0,reps); t1,_=bench(f1,reps); t2,_=bench(f2,reps)
      comps.append({'M':M,'bf16_ms':t0,'packed_dequant_ms':t1,'algebraic_ms':t2,'algebraic_vs_bf16_x':t0/t2,'algebraic_vs_current_packed_x':t1/t2,'latency_reduction_vs_current_packed_percent':(t1-t2)/t1*100,'numeric':metrics})
    alg=[x['algebraic_vs_current_packed_x'] for x in comps]; ref=[x['algebraic_vs_bf16_x'] for x in comps]
    out={'pass':all(x['numeric']['algebraic']['allclose'] and x['numeric']['algebraic']['cosine']>.999 for x in comps),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'N':N,'K':K,'group_size':PY_GROUP},'tuning_rows':rows,'best':best,'comparisons':comps,'summary':{'median_speedup_vs_current_packed_x':float(statistics.median(alg)),'max_speedup_vs_current_packed_x':float(max(alg)),'median_speedup_vs_bf16_x':float(statistics.median(ref)),'max_speedup_vs_bf16_x':float(max(ref)),'beats_current_packed_all':bool(min(alg)>1.0),'beats_bf16_any':bool(max(ref)>1.0)},'truth_boundary':'single-expert synthetic Qwen3-shaped algebraic AWQ kernel; no serving or official-support claim'}
  except Exception as e: out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
  print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
    if not c: payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',inside_code()],360); parsed={}
        for line in r['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-12000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); p=ROOT/'docs'/'evidence'/f'r9700_wna16_groupwise_algebraic_kernel_{stamp}.json'; p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'best':(payload.get('probe') or {}).get('best'),'summary':(payload.get('probe') or {}).get('summary'),'comparisons':(payload.get('probe') or {}).get('comparisons')},sort_keys=True)); return 0 if payload.get('pass') else 2
if __name__=='__main__': raise SystemExit(main())
