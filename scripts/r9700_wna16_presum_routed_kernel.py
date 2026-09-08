#!/usr/bin/env python3
"""R9700 routed WNA16 algebraic kernel with precomputed activation group sums.

The prior algebraic kernel recomputed sum(A_group) in every output-N tile even
though that value is independent of expert and N. For W1 decode the same hidden
row is also reused across top-k experts. This experiment precomputes one
[M, K/group] FP32 group-sum tensor, then reuses it from all routed expert tiles.
Timing includes the group-sum prepass.
"""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1];SCHEMA='hyperloom.r9700.wna16_presum_routed_kernel.v1'
def run(argv:list[str],timeout:float=420)->dict[str,Any]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-300000:],'stderr':p.stderr[-50000:]}
def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None);return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def inside_code()->str:return textwrap.dedent(r'''
import json,statistics,time,traceback,torch
import triton,triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel,try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
@triton.jit
def group_sum(A,AS,M:tl.constexpr,K:tl.constexpr,GK:tl.constexpr):
 pid=tl.program_id(0);m=pid//(K//GK);g=pid%(K//GK);kk=g*GK+tl.arange(0,GK);a=tl.load(A+m*K+kk,mask=(m<M)&(kk<K),other=0.).to(tl.float32);tl.store(AS+m*(K//GK)+g,tl.sum(a,axis=0),mask=m<M)
@triton.jit
def moe_alg(A,B,C,S,Z,AS,sorted_ids,expert_ids,npost_ptr,N:tl.constexpr,K:tl.constexpr,EM,num_valid,sam,sak,sbe,sbk,sbn,scm,scn,sse,ssk,ssn,sze,szk,szn,top_k:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GK:tl.constexpr,GM:tl.constexpr):
 pid=tl.program_id(0);npm=tl.cdiv(EM,BM);npn=tl.cdiv(N,BN);npg=GM*npn;gid=pid//npg;first=gid*GM;gsm=tl.minimum(npm-first,GM);pm=first+((pid%npg)%gsm);pn=(pid%npg)//gsm;npost=tl.load(npost_ptr)
 if pm*BM>=npost:return
 sid=pm*BM+tl.arange(0,BM).to(tl.int64);tok=tl.load(sorted_ids+sid).to(tl.int64);tm=tok<num_valid;exp=tl.load(expert_ids+pm).to(tl.int64);on=pn*BN+tl.arange(0,BN).to(tl.int64);nm=on<N
 if exp==-1:tl.store(C+tok[:,None]*scm+on[None,:]*scn,0.,mask=tm[:,None]&nm[None,:]);return
 ok=tl.arange(0,BK).to(tl.int64);acc=tl.zeros((BM,BN),tl.float32);ng=K//GK
 for g in tl.range(0,K,GK):
  gi=g//GK;sc=tl.load(S+exp*sse+on*ssn+gi*ssk,mask=nm,other=0.).to(tl.float32);zb=tl.load(Z+exp*sze+(on//2)*szn+gi*szk,mask=nm,other=0);zp=((zb>>((on%2)*4))&15).to(tl.float32);asum=tl.load(AS+(tok//top_k)*ng+gi,mask=tm,other=0.).to(tl.float32);qacc=tl.zeros((BM,BN),tl.float32)
  for sub in tl.static_range(0,GK,BK):
   kk=g+sub+ok;a=tl.load(A+(tok[:,None]//top_k)*sam+kk[None,:]*sak,mask=tm[:,None]&(kk[None,:]<K),other=0.).to(tl.bfloat16);pb=tl.load(B+exp*sbe+(kk[:,None]//2)*sbk+on[None,:]*sbn,mask=(kk[:,None]<K)&nm[None,:],other=0);q=((pb>>((kk[:,None]%2)*4))&15).to(tl.bfloat16);qacc=tl.dot(a,q,acc=qacc)
  acc+=qacc*sc[None,:]-asum[:,None]*(zp*sc)[None,:]
 tl.store(C+tok[:,None]*scm+on[None,:]*scn,acc.to(tl.bfloat16),mask=tm[:,None]&nm[None,:])
def awq(qw,s,qz):
 E,K,Np=qw.shape;N=Np*8;sh=torch.arange(0,32,4,dtype=torch.int32,device=qw.device);rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device);v=((qw.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,K,N);nt=v.transpose(1,2).contiguous();B=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous();S=s.transpose(1,2).contiguous();G=qz.shape[1];zv=((qz.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous();Z=(zv[:,0::2,:]|(zv[:,1::2,:]<<4)).to(torch.uint8).contiguous();return B,S,Z
def launch(A,B,C,S,Z,AS,sid,se,npad,topk,cfg):
 K=A.size(1);N=B.size(1);EM=sid.numel();group_sum[(A.size(0)*(K//128),)](A,AS,M=A.size(0),K=K,GK=128,num_warps=1);grid=(triton.cdiv(EM,cfg['BM'])*triton.cdiv(N,cfg['BN']),);moe_alg[grid](A,B,C,S,Z,AS,sid,se,npad,N,K,EM,A.size(0)*topk,A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),S.stride(0),S.stride(2),S.stride(1),Z.stride(0),Z.stride(2),Z.stride(1),top_k=topk,BM=cfg['BM'],BN=cfg['BN'],BK=32,GK=128,GM=1,num_warps=cfg['nw'],num_stages=1,waves_per_eu=cfg['waves'])
def main():
 try:
  torch.manual_seed(20260908);E=8;K=2048;N=1536;topk=8;gs=128;qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device='cuda');qz=torch.randint(-(2**31),2**31-1,(E,K//gs,N//8),dtype=torch.int32,device='cuda');sc=(torch.rand((E,K//gs,N),device='cuda')*.02+.002).to(torch.float16);B,S,Z=awq(qw,sc,qz);Bbf=_unpack_and_dequant_int4_awq(qw,sc,qz,transpose_output=True,output_dtype=torch.bfloat16);dummy=torch.empty((E,K,768),device='cuda',dtype=torch.bfloat16);base=torch.arange(topk,device='cuda',dtype=torch.int64);cfg={'BM':16,'BN':64,'nw':2,'waves':4};rows=[]
  for M in (1,2,4,8,16,32):
   ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous();A=(torch.randn((M,K),device='cuda')*.1).to(torch.bfloat16);AS=torch.empty((M,K//gs),device='cuda',dtype=torch.float32);sid,se,npad=moe_align_block_size(ids,cfg['BM'],E,None);Ca=torch.zeros((M,topk,N),device='cuda',dtype=torch.bfloat16);Cb=torch.zeros_like(Ca);ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy.size(),topk,None,M));esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None);fa=lambda:launch(A,B,Ca,S,Z,AS,sid,se,npad,topk,cfg);fb=lambda:invoke_fused_moe_triton_kernel(A,Bbf,Cb,None,None,None,esid,ese,enpad,False,topk,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
   for _ in range(6):fa();fb()
   torch.cuda.synchronize();d=(Ca.float()-Cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(Ca.float().flatten(),Cb.float().flatten(),dim=0));reps=max(8,min(30,96//max(1,M)));av=[];bv=[]
   for _ in range(9):
    for label,fn,target in [('a',fa,av),('b',fb,bv),('b',fb,bv),('a',fa,av)]:
     torch.cuda.synchronize();st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
     for _ in range(reps):fn()
     en.record();en.synchronize();target.append(float(st.elapsed_time(en))/reps)
   am=float(statistics.median(av));bm=float(statistics.median(bv));rows.append({'M':M,'presum_algebraic_ms':am,'bf16_ms':bm,'speedup_x':bm/am,'latency_reduction_percent':(bm-am)/bm*100,'cosine':cos,'max_abs_error':float(d.max()),'allclose':bool(torch.allclose(Ca.float(),Cb.float(),rtol=.08,atol=.08)),'algebraic_samples_ms':av,'bf16_samples_ms':bv})
  sp=[r['speedup_x'] for r in rows];out={'pass':all(r['allclose'] and r['cosine']>.999 for r in rows),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'config':cfg,'rows':rows,'summary':{'median_speedup_x':float(statistics.median(sp)),'max_speedup_x':float(max(sp)),'min_speedup_x':float(min(sp)),'M1_speedup_x':rows[0]['speedup_x']},'truth_boundary':'routed synthetic Qwen3 W1; timing includes activation-group-sum prepass; HIP events; no running vLLM patch'}
 except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
 print(json.dumps(out,sort_keys=True))
main()
''')
def main()->int:
 live=discover_live_container(run=run);c=str((live.get('selected') or {}).get('Names') or '');payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'model_weights_touched':False,'real_r9700_kernel':True,'synthetic_data':True,'official_support_claim':False},'container':c}
 if not c:payload.update({'pass':False,'error':'no_vllm_container'})
 else:
  r=run(['docker','exec',c,'python3','-c',inside_code()],420);parsed={}
  for line in r['stdout'].splitlines():
   try:o=json.loads(line)
   except Exception:continue
   if isinstance(o,dict):parsed=o
  payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-12000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
 payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_presum_routed_kernel_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'summary':(payload.get('probe') or {}).get('summary'),'rows':(payload.get('probe') or {}).get('rows')},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
