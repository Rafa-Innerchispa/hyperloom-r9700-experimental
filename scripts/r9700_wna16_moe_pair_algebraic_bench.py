#!/usr/bin/env python3
"""Benchmark algebraic packed-INT4 WNA16 for both Qwen3 MoE GEMM shapes on R9700."""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1];SCHEMA='hyperloom.r9700.wna16_moe_pair_algebraic_bench.v1'
def run(argv:list[str],timeout:float=420)->dict[str,Any]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-280000:],'stderr':p.stderr[-50000:]}
def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None);return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def inside_code()->str:return textwrap.dedent(r'''
import json,statistics,time,traceback,torch
import triton,triton.language as tl
@triton.jit
def alg(A,B,C,S,Z,N:tl.constexpr,K:tl.constexpr,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GK:tl.constexpr):
 pm=tl.program_id(0);pn=tl.program_id(1);om=pm*BM+tl.arange(0,BM);on=pn*BN+tl.arange(0,BN);ok=tl.arange(0,BK);mm=om<M;nn=on<N;acc=tl.zeros((BM,BN),tl.float32)
 for g in tl.range(0,K,GK):
  gi=g//GK;sc=tl.load(S+gi*N+on,mask=nn,other=0.).to(tl.float32);zb=tl.load(Z+gi*(N//2)+(on//2),mask=nn,other=0);zp=((zb>>((on%2)*4))&15).to(tl.float32);qacc=tl.zeros((BM,BN),tl.float32);asum=tl.zeros((BM,),tl.float32)
  for sub in tl.static_range(0,GK,BK):
   kk=g+sub+ok;a=tl.load(A+om[:,None]*K+kk[None,:],mask=mm[:,None]&(kk[None,:]<K),other=0.).to(tl.bfloat16);pb=tl.load(B+(kk[:,None]//2)*N+on[None,:],mask=(kk[:,None]<K)&nn[None,:],other=0);q=((pb>>((kk[:,None]%2)*4))&15).to(tl.bfloat16);qacc=tl.dot(a,q,acc=qacc);asum+=tl.sum(a.to(tl.float32),axis=1)
  acc+=qacc*sc[None,:]-asum[:,None]*(zp*sc)[None,:]
 tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=mm[:,None]&nn[None,:])
@triton.jit
def bf(A,W,C,N:tl.constexpr,K:tl.constexpr,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 pm=tl.program_id(0);pn=tl.program_id(1);om=pm*BM+tl.arange(0,BM);on=pn*BN+tl.arange(0,BN);ok=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
 for k in tl.range(0,K,BK):
  kk=k+ok;a=tl.load(A+om[:,None]*K+kk[None,:],mask=(om[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.bfloat16);w=tl.load(W+kk[:,None]*N+on[None,:],mask=(kk[:,None]<K)&(on[None,:]<N),other=0.).to(tl.bfloat16);acc=tl.dot(a,w,acc=acc)
 tl.store(C+om[:,None]*N+on[None,:],acc.to(tl.bfloat16),mask=(om[:,None]<M)&(on[None,:]<N))
def weights(K,N,G=128):
 q=torch.randint(0,16,(K,N),device='cuda',dtype=torch.uint8);qp=(q[0::2]|(q[1::2]<<4)).contiguous();ng=K//G;s=(torch.rand((ng,N),device='cuda')*.02+.002).to(torch.float16);zp=torch.randint(0,16,(ng,N),device='cuda',dtype=torch.uint8);z=(zp[:,0::2]|(zp[:,1::2]<<4)).contiguous();w=((q.float()-zp.repeat_interleave(G,0).float())*s.repeat_interleave(G,0).float()).to(torch.bfloat16).contiguous();return qp,s,z,w
def bench(fn,reps):
 for _ in range(3):fn()
 torch.cuda.synchronize();v=[]
 for _ in range(5):
  t=time.perf_counter()
  for _ in range(reps):fn()
  torch.cuda.synchronize();v.append((time.perf_counter()-t)*1000/reps)
 return float(statistics.median(v))
def section(label,K,N,Ms):
 qp,s,z,w=weights(K,N);rows=[]
 for M in Ms:
  A=(torch.randn((M,K),device='cuda')*.1).to(torch.bfloat16);ca=torch.empty((M,N),device='cuda',dtype=torch.bfloat16);cb=torch.empty_like(ca);grid=(triton.cdiv(M,16),triton.cdiv(N,64));fa=lambda:alg[grid](A,qp,ca,s,z,N=N,K=K,M=M,BM=16,BN=64,BK=32,GK=128,num_warps=4,num_stages=1,waves_per_eu=1);fb=lambda:bf[grid](A,w,cb,N=N,K=K,M=M,BM=16,BN=64,BK=32,num_warps=2,num_stages=1,waves_per_eu=2);fa();fb();torch.cuda.synchronize();d=(ca.float()-cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(ca.float().flatten(),cb.float().flatten(),dim=0));r=max(5,min(24,64//max(1,M)));ta=bench(fa,r);tb=bench(fb,r);rows.append({'M':M,'algebraic_ms':ta,'bf16_ms':tb,'speedup_x':tb/ta,'allclose':bool(torch.allclose(ca.float(),cb.float(),rtol=.08,atol=.08)),'cosine':cos,'max_abs_error':float(d.max())})
 sp=[x['speedup_x'] for x in rows];return {'label':label,'K':K,'N':N,'rows':rows,'median_speedup_x':float(statistics.median(sp)),'max_speedup_x':float(max(sp)),'min_speedup_x':float(min(sp))}
def main():
 try:
  torch.manual_seed(20260908);w1=section('w1',2048,1536,[1,2,4,8,16,32]);w2=section('w2',768,2048,[8,16,32,64,128]);ok=all(r['allclose'] and r['cosine']>.999 for sec in (w1,w2) for r in sec['rows']);out={'pass':ok,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'w1':w1,'w2':w2,'decode_operating_point':{'w1_M':1,'w2_M':8},'truth_boundary':'single-expert synthetic Qwen3 MoE pair benchmark; no running vLLM patch or serving claim'}
 except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
 print(json.dumps(out,sort_keys=True))
main()
''')
def main()->int:
 live=discover_live_container(run=run);c=str((live.get('selected') or {}).get('Names') or '');payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
 if not c:payload.update({'pass':False,'error':'no_vllm_container'})
 else:
  r=run(['docker','exec',c,'python3','-c',inside_code()],420);parsed={}
  for line in r['stdout'].splitlines():
   try:o=json.loads(line)
   except Exception:continue
   if isinstance(o,dict):parsed=o
  payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-12000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
 payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_moe_pair_algebraic_bench_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'probe':payload.get('probe',{})},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
