#!/usr/bin/env python3
"""Validate precomputed-correction W1 kernel at the actual vLLM activation dtype.

The live server is launched with --dtype float16. The current AWQ emulation
backend dequantizes expert weights to BF16, while Triton compute_type follows the
FP16 hidden-state dtype. This benchmark mirrors that combination: FP16 A/output,
BF16 fallback weights, tl.float16 compute, routed top_k=8.
"""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1];SCHEMA='hyperloom.r9700.wna16_correction_fp16_runtime.v1'
def run(argv:list[str],timeout:float=420)->dict[str,Any]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-320000:],'stderr':p.stderr[-50000:]}
def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None);return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def inside_code()->str:return textwrap.dedent(r'''
import json,statistics,traceback,torch
import triton,triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel,try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
@triton.jit
def k(A,B,C,S,R,sids,eids,npad,N:tl.constexpr,K:tl.constexpr,EM,nvalid,sam,sak,sbe,sbk,sbn,scm,scn,sse,ssk,ssn,sre,srk,srn,topk:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GK:tl.constexpr):
 pid=tl.program_id(0);npm=tl.cdiv(EM,BM);npn=tl.cdiv(N,BN);pm=pid//npn;pn=pid%npn;npost=tl.load(npad)
 if pm*BM>=npost:return
 sid=pm*BM+tl.arange(0,BM).to(tl.int64);tok=tl.load(sids+sid).to(tl.int64);tm=tok<nvalid;exp=tl.load(eids+pm).to(tl.int64);on=pn*BN+tl.arange(0,BN).to(tl.int64);nm=on<N
 if exp==-1:tl.store(C+tok[:,None]*scm+on[None,:]*scn,0.,mask=tm[:,None]&nm[None,:]);return
 ok=tl.arange(0,BK).to(tl.int64);acc=tl.zeros((BM,BN),tl.float32)
 for g in tl.range(0,K,GK):
  gi=g//GK;sc=tl.load(S+exp*sse+on*ssn+gi*ssk,mask=nm,other=0.).to(tl.float32);corr=tl.load(R+exp*sre+on*srn+gi*srk,mask=nm,other=0.).to(tl.float32);qacc=tl.zeros((BM,BN),tl.float32);asum=tl.zeros((BM,),tl.float32)
  for sub in tl.static_range(0,GK,BK):
   kk=g+sub+ok;a=tl.load(A+(tok[:,None]//topk)*sam+kk[None,:]*sak,mask=tm[:,None]&(kk[None,:]<K),other=0.).to(tl.float16);pb=tl.load(B+exp*sbe+(kk[:,None]//2)*sbk+on[None,:]*sbn,mask=(kk[:,None]<K)&nm[None,:],other=0);q=((pb>>((kk[:,None]%2)*4))&15).to(tl.float16);qacc=tl.dot(a,q,acc=qacc);asum+=tl.sum(a.to(tl.float32),axis=1)
  acc+=qacc*sc[None,:]-asum[:,None]*corr[None,:]
 tl.store(C+tok[:,None]*scm+on[None,:]*scn,acc.to(tl.float16),mask=tm[:,None]&nm[None,:])
def convert(qw,sc,qz):
 E,K,Np=qw.shape;N=Np*8;sh=torch.arange(0,32,4,dtype=torch.int32,device=qw.device);rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device);v=((qw.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,K,N);nt=v.transpose(1,2).contiguous();B=(nt[...,0::2]|(nt[...,1::2]<<4)).to(torch.uint8).contiguous();S=sc.transpose(1,2).contiguous();G=qz.shape[1];zv=((qz.unsqueeze(-1)>>sh)&15)[...,rev].reshape(E,G,N).transpose(1,2).contiguous();R=(zv.float()*S.float()).to(torch.float16).contiguous();return B,S,R
def launch(A,B,C,S,R,sid,se,npad,topk):
 K=A.size(1);N=B.size(1);EM=sid.numel();grid=(triton.cdiv(EM,16)*triton.cdiv(N,128),);k[grid](A,B,C,S,R,sid,se,npad,N,K,EM,A.size(0)*topk,A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),S.stride(0),S.stride(2),S.stride(1),R.stride(0),R.stride(2),R.stride(1),topk=topk,BM=16,BN=128,BK=32,GK=128,num_warps=4,num_stages=1,waves_per_eu=4)
def timed(fn,reps):
 st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
 for _ in range(reps):fn()
 en.record();en.synchronize();return float(st.elapsed_time(en))/reps
def main():
 try:
  torch.manual_seed(20260908);E=8;K=2048;N=1536;G=128;topk=8;qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device='cuda');qz=torch.randint(-(2**31),2**31-1,(E,K//G,N//8),dtype=torch.int32,device='cuda');sc=(torch.rand((E,K//G,N),device='cuda')*.02+.002).to(torch.float16);B,S,R=convert(qw,sc,qz);Bbf=_unpack_and_dequant_int4_awq(qw,sc,qz,transpose_output=True,output_dtype=torch.bfloat16);dummy=torch.empty((E,K,768),device='cuda',dtype=torch.bfloat16);base=torch.arange(topk,device='cuda',dtype=torch.int64);rows=[]
  for M in (1,2,4,8,16):
   ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous();A=(torch.randn((M,K),device='cuda')*.1).to(torch.float16);sid,se,npad=moe_align_block_size(ids,16,E,None);Ca=torch.zeros((M,topk,N),device='cuda',dtype=torch.float16);Cb=torch.zeros_like(Ca);ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy.size(),topk,None,M));esid,ese,enpad=moe_align_block_size(ids,ecfg['BLOCK_SIZE_M'],E,None);fa=lambda:launch(A,B,Ca,S,R,sid,se,npad,topk);fb=lambda:invoke_fused_moe_triton_kernel(A,Bbf,Cb,None,None,None,esid,ese,enpad,False,topk,ecfg,tl.float16,False,False,False,False,False,None,None)
   for _ in range(10):fa();fb()
   torch.cuda.synchronize();diff=(Ca.float()-Cb.float()).abs();cos=float(torch.nn.functional.cosine_similarity(Ca.float().flatten(),Cb.float().flatten(),dim=0));reps=max(12,min(36,72//M));pairs=[]
   for i in range(21):
    if i%2==0:am=timed(fa,reps);bm=timed(fb,reps)
    else:bm=timed(fb,reps);am=timed(fa,reps)
    pairs.append({'round':i,'candidate_ms':am,'bf16_ms':bm,'speedup_x':bm/am,'candidate_first':i%2==0})
   ratios=[p['speedup_x'] for p in pairs];ams=[p['candidate_ms'] for p in pairs];bms=[p['bf16_ms'] for p in pairs];rows.append({'M':M,'candidate_median_ms':float(statistics.median(ams)),'bf16_median_ms':float(statistics.median(bms)),'paired_speedup_median_x':float(statistics.median(ratios)),'paired_speedup_min_x':float(min(ratios)),'paired_speedup_max_x':float(max(ratios)),'wins':sum(x>1 for x in ratios),'wins_ge_1p05x':sum(x>=1.05 for x in ratios),'rounds':21,'cosine':cos,'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),'allclose':bool(torch.allclose(Ca.float(),Cb.float(),rtol=.08,atol=.08)),'pairs':pairs})
  out={'pass':all(r['allclose'] and r['cosine']>.999 for r in rows),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'runtime_dtype':'float16','fallback_weight_dtype':'bfloat16','rows':rows,'M1_gate':{'median_ge_1p05x':rows[0]['paired_speedup_median_x']>=1.05,'wins_ge_15_of_21':rows[0]['wins']>=15,'passes':rows[0]['paired_speedup_median_x']>=1.05 and rows[0]['wins']>=15},'truth_boundary':'mirrors live --dtype float16 compute with BF16 emulation weights; 21 paired HIP-event rounds; no service patch'}
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
 payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_correction_fp16_runtime_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'M1_gate':(payload.get('probe') or {}).get('M1_gate'),'rows':[{k:r.get(k) for k in ['M','candidate_median_ms','bf16_median_ms','paired_speedup_median_x','paired_speedup_min_x','paired_speedup_max_x','wins','wins_ge_1p05x','rounds','cosine','max_abs_error','allclose']} for r in (payload.get('probe') or {}).get('rows',[])]},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
