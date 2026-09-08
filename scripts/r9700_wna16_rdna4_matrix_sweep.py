#!/usr/bin/env python3
"""Sweep matrix-shaped RDNA4 WNA16 Triton candidates on a real R9700.

The current group128 kernel is numerically correct but slower than the
pre-dequantized BF16 micro-GEMM reference. This experiment keeps the same
AutoAWQ packed INT4 layout and varies the amount of K work consumed by each
matrix dot (32/64/128), tile sizes, warps, stages, and waves_per_eu.

Safety/truth boundary:
- runs in docker exec against synthetic Qwen3-shaped tensors;
- does not patch vLLM, model weights, or the production service;
- a candidate is eligible only if it passes finite/allclose/cosine gates;
- performance is a kernel microbenchmark, not an end-to-end serving claim.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_rdna4_matrix_sweep.v1'

def run(argv:list[str],timeout:float=720.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-300000:],'stderr':p.stderr[-60000:]}

def digest(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False)); c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inside_code()->str:
    return textwrap.dedent(r'''
import json, statistics, time, traceback, torch
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

@triton.jit
def kernel(a_ptr,b_ptr,c_ptr,b_scale_ptr,b_zp_ptr,sorted_token_ids_ptr,expert_ids_ptr,num_tokens_post_padded_ptr,
           N:tl.constexpr,K:tl.constexpr,EM,num_valid_tokens,
           stride_am,stride_ak,stride_be,stride_bk,stride_bn,stride_cm,stride_cn,
           stride_bse,stride_bsk,stride_bsn,stride_bze,stride_bzk,stride_bzn,
           top_k:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,SUB_K:tl.constexpr,GROUP_K:tl.constexpr,GROUP_M:tl.constexpr):
    pid=tl.program_id(0)
    npm=tl.cdiv(EM,BLOCK_M); npn=tl.cdiv(N,BLOCK_N); npg=GROUP_M*npn
    gid=pid//npg; first=gid*GROUP_M; gsm=tl.minimum(npm-first,GROUP_M)
    pm=first+((pid%npg)%gsm); pn=(pid%npg)//gsm
    npost=tl.load(num_tokens_post_padded_ptr)
    if pm*BLOCK_M>=npost: return
    sid=pm*BLOCK_M+tl.arange(0,BLOCK_M).to(tl.int64)
    tok=tl.load(sorted_token_ids_ptr+sid).to(tl.int64)
    tmask=tok<num_valid_tokens
    exp=tl.load(expert_ids_ptr+pm).to(tl.int64)
    on=pn*BLOCK_N+tl.arange(0,BLOCK_N).to(tl.int64); nmask=on<N
    if exp==-1:
        cp=c_ptr+tok[:,None]*stride_cm+on[None,:]*stride_cn
        tl.store(cp,0.0,mask=tmask[:,None]&nmask[None,:]); return
    acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    ok=tl.arange(0,SUB_K).to(tl.int64)
    for g in tl.range(0,K,GROUP_K):
        gi=g//GROUP_K
        sp=b_scale_ptr+exp*stride_bse+on*stride_bsn+gi*stride_bsk
        scale=tl.load(sp,mask=nmask,other=0.0).to(tl.float32)
        zpp=b_zp_ptr+exp*stride_bze+(on//2)*stride_bzn+gi*stride_bzk
        zpb=tl.load(zpp,mask=nmask,other=0); zp=((zpb>>((on%2)*4))&0xF).to(tl.float32)
        for sub in tl.static_range(0,GROUP_K,SUB_K):
            kk=g+sub+ok
            ap=a_ptr+(tok[:,None]//top_k)*stride_am+kk[None,:]*stride_ak
            a=tl.load(ap,mask=tmask[:,None]&(kk[None,:]<K),other=0.0)
            bp=b_ptr+exp*stride_be+(kk[:,None]//2)*stride_bk+on[None,:]*stride_bn
            pb=tl.load(bp,mask=(kk[:,None]<K)&nmask[None,:],other=0)
            q=((pb>>((kk[:,None]%2)*4))&0xF).to(tl.float32)
            b=((q-zp[None,:])*scale[None,:]).to(tl.bfloat16)
            acc=tl.dot(a,b,acc=acc)
    cp=c_ptr+tok[:,None]*stride_cm+on[None,:]*stride_cn
    tl.store(cp,acc.to(tl.bfloat16),mask=tmask[:,None]&nmask[None,:])

def awq_to_triton(qw,scales,qz):
    E,K,Np=qw.shape; N=Np*8
    shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device)
    rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
    vals=((qw.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,K,N)
    nt=vals.transpose(1,2).contiguous(); wb=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous()
    st=scales.transpose(1,2).contiguous(); G=qz.shape[1]
    zv=((qz.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous()
    zb=(zv[:,0::2,:]|(zv[:,1::2,:]<<4)).to(torch.uint8).contiguous()
    return wb,st,zb

def launch(A,B,C,S,Z,sid,se,npad,top_k,cfg):
    EM=sid.numel(); N=B.size(1); K=A.size(1)
    grid=(triton.cdiv(EM,cfg['BLOCK_M'])*triton.cdiv(N,cfg['BLOCK_N']),)
    kernel[grid](A,B,C,S,Z,sid,se,npad,N,K,EM,A.size(0)*top_k,
        A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),
        S.stride(0),S.stride(2),S.stride(1),Z.stride(0),Z.stride(2),Z.stride(1),top_k=top_k,
        BLOCK_M=cfg['BLOCK_M'],BLOCK_N=cfg['BLOCK_N'],SUB_K=cfg['SUB_K'],GROUP_K=128,GROUP_M=1,
        num_warps=cfg['num_warps'],num_stages=cfg['num_stages'],waves_per_eu=cfg['waves_per_eu'])

def main():
  out={}
  try:
    torch.manual_seed(20260908); d=torch.device('cuda'); E=8;K=2048;N=1536;G=128;top_k=8
    qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=d)
    qz=torch.randint(-(2**31),2**31-1,(E,K//G,N//8),dtype=torch.int32,device=d)
    scales=(torch.rand((E,K//G,N),device=d,dtype=torch.float32)*.02+.002).to(torch.float16)
    B,S,Z=awq_to_triton(qw,scales,qz)
    Bbf=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.bfloat16)
    dummy_w2=torch.empty((E,K,768),dtype=torch.bfloat16,device=d)
    def ids_for(M):
      base=torch.arange(top_k,device=d,dtype=torch.int64)
      return torch.stack([(base+i)%E for i in range(M)],0).contiguous()
    def setup(M,bm):
      A=(torch.randn((M,K),device=d,dtype=torch.float32)*.1).to(torch.bfloat16).contiguous(); ids=ids_for(M)
      sid,se,npad=moe_align_block_size(ids,bm,E,None); return A,ids,sid,se,npad
    configs=[]
    # Bounded focus set. Earlier sweeps showed BLOCK_M=16 and BLOCK_N>=64
    # are the useful region on gfx1201. Keep each run short enough to persist
    # evidence before any MCP/session interruption.
    for sk in (32,64,128):
      for bm in (16,):
       for bn in (64,128):
        for nw in (2,4):
         for ns in (1,):
          for w in (0,1,2): configs.append({'SUB_K':sk,'BLOCK_M':bm,'BLOCK_N':bn,'num_warps':nw,'num_stages':ns,'waves_per_eu':w})
    rows=[]; M=8
    for cfg in configs:
      A,ids,sid,se,npad=setup(M,cfg['BLOCK_M']); C=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=d)
      try:
        launch(A,B,C,S,Z,sid,se,npad,top_k,cfg); torch.cuda.synchronize()
        finite=bool(torch.isfinite(C).all().item())
        reps=5; samples=[]
        for _ in range(2):
          torch.cuda.synchronize(); t=time.perf_counter()
          for _ in range(reps): launch(A,B,C,S,Z,sid,se,npad,top_k,cfg)
          torch.cuda.synchronize(); samples.append((time.perf_counter()-t)*1000/reps)
        rows.append({'config':cfg,'ms':float(statistics.median(samples)),'finite':finite})
      except Exception as e: rows.append({'config':cfg,'finite':False,'error':type(e).__name__+':'+str(e)})
    viable=sorted([r for r in rows if r.get('finite')],key=lambda r:r['ms'])[:12]
    finalists=[]
    for cand in viable:
      cfg=cand['config']; A,ids,sid,se,npad=setup(8,cfg['BLOCK_M']); Cq=torch.zeros((8,top_k,N),dtype=torch.bfloat16,device=d); Ce=torch.zeros_like(Cq)
      ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy_w2.size(),top_k,None,8)); esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None)
      launch(A,B,Cq,S,Z,sid,se,npad,top_k,cfg); invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,esid,ese,enpad,False,top_k,ecfg,tl.bfloat16,False,False,False,False,False,None,None); torch.cuda.synchronize()
      diff=(Cq.float()-Ce.float()).abs(); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0).item())
      good=bool(torch.isfinite(Cq).all().item()) and bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08)) and cos>.999
      if good: finalists.append({'config':cfg,'tune_ms':cand['ms'],'cosine':cos,'max_abs_error':float(diff.max().item())})
    if not finalists: raise RuntimeError('no numerically valid finalist')
    best=min(finalists,key=lambda r:r['tune_ms']); cfg=best['config']; comps=[]
    for M in (1,2,4,8,16,32,64):
      A,ids,sid,se,npad=setup(M,cfg['BLOCK_M']); Cq=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=d); Ce=torch.zeros_like(Cq)
      ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy_w2.size(),top_k,None,M)); esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None)
      def q(): launch(A,B,Cq,S,Z,sid,se,npad,top_k,cfg)
      def e(): invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,esid,ese,enpad,False,top_k,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
      for _ in range(3): q(); e()
      torch.cuda.synchronize(); diff=(Cq.float()-Ce.float()).abs(); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0).item())
      reps=max(4,min(20,64//max(1,M))); qs=[]; es=[]
      for order in (('q','e'),('e','q'),('q','e')):
       for lab in order:
        fn=q if lab=='q' else e; torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); (qs if lab=='q' else es).append((time.perf_counter()-t)*1000/reps)
      qm=float(statistics.median(qs)); em=float(statistics.median(es)); comps.append({'M':M,'q_ms':qm,'emulation_ms':em,'speedup_x':em/qm,'latency_reduction_percent':(em-qm)/em*100,'cosine':cos,'max_abs_error':float(diff.max().item()),'allclose':bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08))})
    speeds=[x['speedup_x'] for x in comps]
    out={'pass':all(x['allclose'] and x['cosine']>.999 for x in comps),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'E_reduced':E,'K':K,'N':N,'group_size':G,'top_k':top_k},'searched_configs':len(configs),'top_viable':viable,'numeric_finalists':finalists,'best':best,'comparisons':comps,'summary':{'median_speedup_x':float(statistics.median(speeds)),'max_speedup_x':float(max(speeds)),'min_speedup_x':float(min(speeds)),'beats_reference_any':bool(max(speeds)>1.0),'beats_reference_median':bool(statistics.median(speeds)>1.0)},'truth_boundary':'Synthetic Qwen3-shaped first MoE GEMM kernel sweep; BF16 reference is already dequantized; no serving or official-support claim.'}
  except Exception as e: out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
  print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
    if not c: payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',inside_code()],720); parsed={}
        for line in r['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-12000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); p=ROOT/'docs'/'evidence'/f'r9700_wna16_rdna4_matrix_sweep_{stamp}.json'; p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'best':(payload.get('probe') or {}).get('best'),'summary':(payload.get('probe') or {}).get('summary'),'comparisons':(payload.get('probe') or {}).get('comparisons')},sort_keys=True)); return 0 if payload.get('pass') else 2
if __name__=='__main__': raise SystemExit(main())
