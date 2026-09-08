#!/usr/bin/env python3
"""Validate the groupwise algebraic AWQ optimization in a routed MoE kernel.

Uses vLLM's routing/alignment primitives and Qwen3-shaped AutoAWQ tensors on the
real R9700. The candidate keeps INT4 weights packed and applies the identity
A@((Q-z)*s) = (A@Q)*s - sum(A)*z*s per 128-wide AWQ group.
No running vLLM files or model weights are modified.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_moe_algebraic_kernel.v1'

def run(argv:list[str],timeout:float=420.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-260000:],'stderr':p.stderr[-50000:]}

def digest(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False)); c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inside_code()->str:
    return textwrap.dedent(r'''
import json, statistics, time, traceback, torch
import triton, triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

@triton.jit
def moe_alg(A,B,C,S,Z,sorted_ids,expert_ids,npost_ptr,N:tl.constexpr,K:tl.constexpr,EM,num_valid,
            sam,sak,sbe,sbk,sbn,scm,scn,sse,ssk,ssn,sze,szk,szn,top_k:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GK:tl.constexpr,GM:tl.constexpr):
    pid=tl.program_id(0); npm=tl.cdiv(EM,BM); npn=tl.cdiv(N,BN); npg=GM*npn
    gid=pid//npg; first=gid*GM; gsm=tl.minimum(npm-first,GM)
    pm=first+((pid%npg)%gsm); pn=(pid%npg)//gsm
    npost=tl.load(npost_ptr)
    if pm*BM>=npost: return
    sid=pm*BM+tl.arange(0,BM).to(tl.int64)
    tok=tl.load(sorted_ids+sid).to(tl.int64); tmask=tok<num_valid
    exp=tl.load(expert_ids+pm).to(tl.int64)
    on=pn*BN+tl.arange(0,BN).to(tl.int64); nmask=on<N
    if exp==-1:
        tl.store(C+tok[:,None]*scm+on[None,:]*scn,0.0,mask=tmask[:,None]&nmask[None,:]); return
    ok=tl.arange(0,BK).to(tl.int64); acc=tl.zeros((BM,BN),tl.float32)
    for g in tl.range(0,K,GK):
        gi=g//GK
        sc=tl.load(S+exp*sse+on*ssn+gi*ssk,mask=nmask,other=0.).to(tl.float32)
        zb=tl.load(Z+exp*sze+(on//2)*szn+gi*szk,mask=nmask,other=0)
        zp=((zb>>((on%2)*4))&15).to(tl.float32)
        qacc=tl.zeros((BM,BN),tl.float32); asum=tl.zeros((BM,),tl.float32)
        for sub in tl.static_range(0,GK,BK):
            kk=g+sub+ok
            a=tl.load(A+(tok[:,None]//top_k)*sam+kk[None,:]*sak,mask=tmask[:,None]&(kk[None,:]<K),other=0.).to(tl.bfloat16)
            pb=tl.load(B+exp*sbe+(kk[:,None]//2)*sbk+on[None,:]*sbn,mask=(kk[:,None]<K)&nmask[None,:],other=0)
            q=((pb>>((kk[:,None]%2)*4))&15).to(tl.bfloat16)
            qacc=tl.dot(a,q,acc=qacc); asum += tl.sum(a.to(tl.float32),axis=1)
        acc += qacc*sc[None,:] - asum[:,None]*(zp*sc)[None,:]
    tl.store(C+tok[:,None]*scm+on[None,:]*scn,acc.to(tl.bfloat16),mask=tmask[:,None]&nmask[None,:])

def awq_to_triton(qw,scales,qz):
    E,K,Np=qw.shape; N=Np*8
    sh=torch.arange(0,32,4,dtype=torch.int32,device=qw.device); rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
    vals=((qw.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,K,N); nt=vals.transpose(1,2).contiguous(); B=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous()
    S=scales.transpose(1,2).contiguous(); G=qz.shape[1]
    zv=((qz.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous(); Z=(zv[:,0::2,:]|(zv[:,1::2,:]<<4)).to(torch.uint8).contiguous()
    return B,S,Z

def launch(A,B,C,S,Z,sid,se,npad,top_k,cfg):
    EM=sid.numel(); N=B.size(1); K=A.size(1); grid=(triton.cdiv(EM,cfg['BM'])*triton.cdiv(N,cfg['BN']),)
    moe_alg[grid](A,B,C,S,Z,sid,se,npad,N,K,EM,A.size(0)*top_k,A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),S.stride(0),S.stride(2),S.stride(1),Z.stride(0),Z.stride(2),Z.stride(1),top_k=top_k,BM=cfg['BM'],BN=cfg['BN'],BK=32,GK=128,GM=1,num_warps=cfg['num_warps'],num_stages=1,waves_per_eu=cfg['waves_per_eu'])

def main():
  out={}
  try:
    torch.manual_seed(20260908); d='cuda'; E=8;K=2048;N=1536;G=128;top_k=8
    qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=d); qz=torch.randint(-(2**31),2**31-1,(E,K//G,N//8),dtype=torch.int32,device=d)
    scales=(torch.rand((E,K//G,N),device=d,dtype=torch.float32)*.02+.002).to(torch.float16); B,S,Z=awq_to_triton(qw,scales,qz)
    Bbf=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.bfloat16); dummy_w2=torch.empty((E,K,768),dtype=torch.bfloat16,device=d)
    configs=[{'BM':16,'BN':64,'num_warps':nw,'waves_per_eu':w} for nw in (2,4) for w in (0,1,2,4)]
    tune=[]; M=8
    base=torch.arange(top_k,device=d,dtype=torch.int64); ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous(); A=(torch.randn((M,K),device=d,dtype=torch.float32)*.1).to(torch.bfloat16)
    for cfg in configs:
      sid,se,npad=moe_align_block_size(ids,cfg['BM'],E,None); C=torch.zeros((M,top_k,N),device=d,dtype=torch.bfloat16); fn=lambda:launch(A,B,C,S,Z,sid,se,npad,top_k,cfg)
      try:
        for _ in range(3):fn()
        torch.cuda.synchronize(); vals=[]
        for _ in range(4):
          t=time.perf_counter()
          for _ in range(10):fn()
          torch.cuda.synchronize(); vals.append((time.perf_counter()-t)*100)
        tune.append({'config':cfg,'ms':float(statistics.median(vals)),'finite':bool(torch.isfinite(C).all().item())})
      except Exception as e:tune.append({'config':cfg,'finite':False,'error':type(e).__name__+':'+str(e)})
    best=min([x for x in tune if x.get('finite')],key=lambda x:x['ms']); cfg=best['config']; comps=[]
    for M in (1,2,4,8,16,32):
      ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous(); A=(torch.randn((M,K),device=d,dtype=torch.float32)*.1).to(torch.bfloat16)
      sid,se,npad=moe_align_block_size(ids,cfg['BM'],E,None); Cq=torch.zeros((M,top_k,N),device=d,dtype=torch.bfloat16); Ce=torch.zeros_like(Cq)
      ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy_w2.size(),top_k,None,M)); esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None)
      q=lambda:launch(A,B,Cq,S,Z,sid,se,npad,top_k,cfg)
      e=lambda:invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,esid,ese,enpad,False,top_k,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
      for _ in range(3):q();e()
      torch.cuda.synchronize(); diff=(Cq.float()-Ce.float()).abs(); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0).item()); close=bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08))
      reps=max(4,min(16,64//max(1,M))); qv=[];ev=[]
      for order in (('q','e'),('e','q'),('q','e')):
       for label in order:
        fn=q if label=='q' else e; torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(reps):fn()
        torch.cuda.synchronize();(qv if label=='q' else ev).append((time.perf_counter()-t)*1000/reps)
      qm=float(statistics.median(qv));em=float(statistics.median(ev));comps.append({'M':M,'algebraic_ms':qm,'emulation_bf16_ms':em,'speedup_vs_emulation_x':em/qm,'latency_reduction_percent':(em-qm)/em*100,'allclose':close,'cosine':cos,'max_abs_error':float(diff.max().item())})
    sp=[x['speedup_vs_emulation_x'] for x in comps]
    out={'pass':all(x['allclose'] and x['cosine']>.999 for x in comps),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'E_reduced':E,'K':K,'N':N,'group_size':G,'top_k':top_k},'best':best,'tuning':tune,'comparisons':comps,'summary':{'median_speedup_vs_emulation_x':float(statistics.median(sp)),'max_speedup_vs_emulation_x':float(max(sp)),'min_speedup_vs_emulation_x':float(min(sp)),'beats_emulation_any':bool(max(sp)>1),'beats_emulation_median':bool(statistics.median(sp)>1)},'truth_boundary':'routed synthetic Qwen3-shaped MoE first GEMM; emulation reference uses pre-dequantized BF16 weights; no serving or official-support claim'}
  except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
  print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
    if not c:payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',inside_code()],420);parsed={}
        for line in r['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-12000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_moe_algebraic_kernel_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'best':(payload.get('probe') or {}).get('best'),'summary':(payload.get('probe') or {}).get('summary'),'comparisons':(payload.get('probe') or {}).get('comparisons')},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
