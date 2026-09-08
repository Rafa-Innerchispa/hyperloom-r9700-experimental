#!/usr/bin/env python3
"""Tune a routed RDNA4 WNA16 kernel with load-time zero_point*scale correction.

For each AWQ group/output channel, the zero-point and scale are weight constants.
This experiment precomputes correction = zero_point * scale once before timing,
so the request hot path no longer loads/unpacks qzero or multiplies zp*scale.
The candidate remains packed INT4 and is compared to the existing routed BF16
pre-dequantized vLLM-style reference on the real R9700.

No running vLLM files, model weights, or service configuration are modified.
"""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_precomputed_correction_tuning.v1'

def run(argv:list[str],timeout:float=480.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-320000:],'stderr':p.stderr[-60000:]}

def digest(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inside_code()->str:
    return textwrap.dedent(r'''
import json,statistics,traceback,torch
import triton,triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel,try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

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
    shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device);rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
    vals=((qw.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,K,N);nt=vals.transpose(1,2).contiguous();B=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous()
    S=scales.transpose(1,2).contiguous();G=qz.shape[1]
    zv=((qz.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous()
    # FP16 is intentional: correction has the same storage class as scale and
    # is re-expanded to FP32 in the hot path. Numeric gates decide eligibility.
    R=(zv.float()*S.float()).to(torch.float16).contiguous()
    return B,S,R,zv

def launch(A,B,C,S,R,sid,se,npad,topk,cfg):
    K=A.size(1);N=B.size(1);EM=sid.numel();grid=(triton.cdiv(EM,cfg['BM'])*triton.cdiv(N,cfg['BN']),)
    kw={'num_warps':cfg['nw'],'num_stages':1,'waves_per_eu':cfg['waves']}
    if cfg.get('mi') is not None:kw['matrix_instr_nonkdim']=cfg['mi']
    moe_corr[grid](A,B,C,S,R,sid,se,npad,N,K,EM,A.size(0)*topk,
        A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),
        S.stride(0),S.stride(2),S.stride(1),R.stride(0),R.stride(2),R.stride(1),
        top_k=topk,BM=cfg['BM'],BN=cfg['BN'],BK=32,GK=128,GM=1,**kw)

def event_ms(fn,reps):
    for _ in range(4):fn()
    torch.cuda.synchronize();vals=[]
    for _ in range(7):
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
        for _ in range(reps):fn()
        en.record();en.synchronize();vals.append(float(st.elapsed_time(en))/reps)
    return float(statistics.median(vals)),vals

def main():
  out={}
  try:
    torch.manual_seed(20260908);E=8;K=2048;N=1536;G=128;topk=8;M=1
    qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device='cuda');qz=torch.randint(-(2**31),2**31-1,(E,K//G,N//8),dtype=torch.int32,device='cuda');scales=(torch.rand((E,K//G,N),device='cuda')*.02+.002).to(torch.float16)
    B,S,R,zv=awq_to_triton(qw,scales,qz);Bbf=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.bfloat16);dummy=torch.empty((E,K,768),device='cuda',dtype=torch.bfloat16)
    base=torch.arange(topk,device='cuda',dtype=torch.int64);ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous();A=(torch.randn((M,K),device='cuda')*.1).to(torch.bfloat16)
    configs=[]
    for bn in (64,128):
      for nw in (1,2,4):
       for waves in (0,2,4):
        for mi in (None,16):configs.append({'BM':16,'BN':bn,'nw':nw,'waves':waves,'mi':mi})
    tune=[]
    for cfg in configs:
      sid,se,npad=moe_align_block_size(ids,cfg['BM'],E,None);C=torch.zeros((M,topk,N),device='cuda',dtype=torch.bfloat16);fn=lambda:launch(A,B,C,S,R,sid,se,npad,topk,cfg)
      try:
        fn();torch.cuda.synchronize();ms,samples=event_ms(fn,12);tune.append({'config':cfg,'valid':bool(torch.isfinite(C).all()),'ms':ms,'samples_ms':samples})
      except Exception as e:tune.append({'config':cfg,'valid':False,'error':type(e).__name__+':'+str(e)})
    viable=sorted([r for r in tune if r.get('valid')],key=lambda r:r['ms'])[:8]
    ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy.size(),topk,None,M));esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None)
    finals=[]
    for row in viable:
      cfg=row['config'];sid,se,npad=moe_align_block_size(ids,cfg['BM'],E,None);Ca=torch.zeros((M,topk,N),device='cuda',dtype=torch.bfloat16);Cb=torch.zeros_like(Ca);fa=lambda:launch(A,B,Ca,S,R,sid,se,npad,topk,cfg);fb=lambda:invoke_fused_moe_triton_kernel(A,Bbf,Cb,None,None,None,esid,ese,enpad,False,topk,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
      for _ in range(6):fa();fb()
      torch.cuda.synchronize();diff=(Ca.float()-Cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(Ca.float().flatten(),Cb.float().flatten(),dim=0));close=bool(torch.allclose(Ca.float(),Cb.float(),rtol=.08,atol=.08));ams,asamp=event_ms(fa,24);bms,bsamp=event_ms(fb,24);finals.append({'config':cfg,'candidate_ms':ams,'bf16_ms':bms,'speedup_x':bms/ams,'latency_reduction_percent':(bms-ams)/bms*100,'cosine':cos,'max_abs_error':float(diff.max()),'allclose':close,'candidate_samples_ms':asamp,'bf16_samples_ms':bsamp})
    numeric=[r for r in finals if r['allclose'] and r['cosine']>.999];best=max(numeric,key=lambda r:r['speedup_x']) if numeric else None
    # Validate the best config over the small-M region with fresh tensors and paired ordering.
    rows=[]
    if best:
      cfg=best['config']
      for M2 in (1,2,4,8,16):
        ids2=torch.stack([(base+i)%E for i in range(M2)],0).contiguous();A2=(torch.randn((M2,K),device='cuda')*.1).to(torch.bfloat16);sid,se,npad=moe_align_block_size(ids2,cfg['BM'],E,None);Ca=torch.zeros((M2,topk,N),device='cuda',dtype=torch.bfloat16);Cb=torch.zeros_like(Ca);ecfg2=dict(try_get_optimal_moe_config(Bbf.size(),dummy.size(),topk,None,M2));esid2,ese2,enpad2=moe_align_block_size(ids2,ecfg2['BLOCK_SIZE_M'],E,None);fa=lambda:launch(A2,B,Ca,S,R,sid,se,npad,topk,cfg);fb=lambda:invoke_fused_moe_triton_kernel(A2,Bbf,Cb,None,None,None,esid2,ese2,enpad2,False,topk,ecfg2,tl.bfloat16,False,False,False,False,False,None,None)
        for _ in range(6):fa();fb()
        torch.cuda.synchronize();diff=(Ca.float()-Cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(Ca.float().flatten(),Cb.float().flatten(),dim=0));close=bool(torch.allclose(Ca.float(),Cb.float(),rtol=.08,atol=.08));reps=max(12,min(36,72//M2));av=[];bv=[]
        for order in (('a','b'),('b','a'),('a','b'),('b','a'),('a','b')):
          for lab in order:
            fn=fa if lab=='a' else fb;st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
            for _ in range(reps):fn()
            en.record();en.synchronize();(av if lab=='a' else bv).append(float(st.elapsed_time(en))/reps)
        am=float(statistics.median(av));bm=float(statistics.median(bv));rows.append({'M':M2,'candidate_ms':am,'bf16_ms':bm,'speedup_x':bm/am,'latency_reduction_percent':(bm-am)/bm*100,'cosine':cos,'max_abs_error':float(diff.max()),'allclose':close,'candidate_samples_ms':av,'bf16_samples_ms':bv})
    correction_bytes=int(R.numel()*R.element_size());scale_bytes=int(S.numel()*S.element_size())
    sp=[r['speedup_x'] for r in rows]
    out={'pass':bool(rows) and all(r['allclose'] and r['cosine']>.999 for r in rows),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'E_reduced':E,'K':K,'N':N,'group_size':G,'top_k':topk},'searched_configs':len(configs),'tuning':tune,'finalists':finals,'best_M1':best,'small_m_rows':rows,'summary':{'median_speedup_x':float(statistics.median(sp)) if sp else None,'max_speedup_x':float(max(sp)) if sp else None,'min_speedup_x':float(min(sp)) if sp else None,'M1_speedup_x':rows[0]['speedup_x'] if rows else None,'beats_bf16_median':bool(sp and statistics.median(sp)>1.0),'M1_beats_1p05x':bool(rows and rows[0]['speedup_x']>=1.05)},'correction_storage':{'reduced_E_bytes':correction_bytes,'reduced_E_mib':correction_bytes/2**20,'scale_bytes':scale_bytes,'correction_vs_scale_ratio':correction_bytes/scale_bytes,'full_model_W1_estimate_note':'FP16 correction has the same element count and storage as the existing W1 scale tensor; multiply by actual layers/experts only when deriving full-model storage.'},'matrix_instr_nonkdim_supported':any(r.get('valid') and r['config'].get('mi')==16 for r in tune),'truth_boundary':'routed synthetic Qwen3 W1 with load-time FP16 correction; paired HIP-event microbenchmark; no running vLLM patch or serving claim'}
  except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
  print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run);c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
    if not c:payload.update({'pass':False,'error':'no_vllm_container'})
    else:
      r=run(['docker','exec',c,'python3','-c',inside_code()],480);parsed={}
      for line in r['stdout'].splitlines():
        try:o=json.loads(line)
        except Exception:continue
        if isinstance(o,dict):parsed=o
      payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-16000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_precomputed_correction_tuning_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'summary':(payload.get('probe') or {}).get('summary'),'best_M1':(payload.get('probe') or {}).get('best_M1'),'small_m_rows':(payload.get('probe') or {}).get('small_m_rows'),'matrix_instr_nonkdim_supported':(payload.get('probe') or {}).get('matrix_instr_nonkdim_supported'),'correction_storage':(payload.get('probe') or {}).get('correction_storage')},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
