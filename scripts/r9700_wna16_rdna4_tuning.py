#!/usr/bin/env python3
"""Bounded RDNA4 tuning sweep for the proven Triton WNA16 AutoAWQ path.

Synthetic Qwen3-shaped first-MoE-GEMM only. Tests a small block grid, validates
numerics for every candidate, then compares the best valid candidate against
current BF16 emulation over several token-batch sizes. No service restart.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA="hyperloom.r9700.wna16_rdna4_tuning.v1"

def run(argv:list[str],timeout:float=420.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {"returncode":p.returncode,"stdout":p.stdout[-180000:],"stderr":p.stderr[-40000:]}

def digest(x:dict[str,Any])->str:
    y=json.loads(json.dumps(x,sort_keys=True,allow_nan=False)); y.pop("probe_sha256",None)
    return hashlib.sha256(json.dumps(y,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def code()->str:
    return textwrap.dedent(r'''
import json,time,traceback,torch
out={}
try:
 import triton.language as tl
 from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
 from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
 from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel,invoke_fused_moe_triton_kernel,try_get_optimal_moe_config
 from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
 E,K,N,N2,gs,topk=8,2048,1536,768,128,8
 dev=torch.device('cuda'); torch.manual_seed(20260908)
 def adapt(qw,s,qz):
  E0,K0,Np=qw.shape; N0=Np*8; shifts=torch.arange(0,32,4,dtype=torch.int32,device=dev); rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=dev)
  v=((qw.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E0,K0,N0).transpose(1,2).contiguous(); wb=(v[...,0::2]|(v[...,1::2]<<4)).to(torch.uint8).contiguous()
  st=s.transpose(1,2).contiguous(); G=qz.shape[1]; z=((qz.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E0,G,N0).transpose(1,2).contiguous(); zb=(z[:,0::2,:]|(z[:,1::2,:]<<4)).to(torch.uint8).contiguous(); return wb,st,zb
 qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=dev); qz=torch.randint(-(2**31),2**31-1,(E,K//gs,N//8),dtype=torch.int32,device=dev); s=(torch.rand((E,K//gs,N),device=dev)*.02+.002).to(torch.float16)
 Bq,Bs,Bz=adapt(qw,s,qz); Bbf=_unpack_and_dequant_int4_awq(qw,s,qz,True,torch.bfloat16); d2q=torch.empty((E,K,N2//2),dtype=torch.uint8,device=dev); d2b=torch.empty((E,K,N2),dtype=torch.bfloat16,device=dev)
 def setup(M):
  A=(torch.randn((M,K),device=dev)*.1).to(torch.bfloat16).contiguous(); base=torch.arange(topk,device=dev,dtype=torch.int64); ids=torch.stack([(base+i)%E for i in range(M)]).contiguous()
  ec=dict(try_get_optimal_moe_config(Bbf.size(),d2b.size(),topk,None,M)); es,ee,ep=moe_align_block_size(ids,ec['BLOCK_SIZE_M'],E,None); Ce=torch.zeros((M,topk,N),dtype=torch.bfloat16,device=dev)
  def ecall(): invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,es,ee,ep,False,topk,ec,tl.bfloat16,False,False,False,False,False,None,None)
  ecall(); torch.cuda.synchronize(); return A,ids,ec,Ce,ecall
 A,ids,ec,Ce,ecall=setup(8)
 configs=[]
 for bn in (32,64,128):
  for bk in (32,64,128): configs.append({'BLOCK_SIZE_M':16,'GROUP_SIZE_M':1,'SPLIT_K':1,'BLOCK_SIZE_N':bn,'BLOCK_SIZE_K':bk})
 tuning=[]
 for c in configs:
  row={'config':c}
  try:
   qs,qe,qp=moe_align_block_size(ids,c['BLOCK_SIZE_M'],E,None); Cq=torch.zeros_like(Ce)
   def qcall(): invoke_fused_moe_wna16_triton_kernel(A,Bq,Cq,Bs,Bz,None,qs,qe,qp,False,topk,c,tl.bfloat16,False,True,[0,gs])
   qcall(); torch.cuda.synchronize(); diff=(Cq.float()-Ce.float()).abs(); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0)); close=bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08))
   for _ in range(2): qcall()
   torch.cuda.synchronize(); reps=8; t0=time.perf_counter();
   for _ in range(reps): qcall()
   torch.cuda.synchronize(); ms=(time.perf_counter()-t0)*1000/reps
   row.update({'valid':close and cos>.999,'ms':ms,'cosine':cos,'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean())})
  except Exception as ex:
   row.update({'valid':False,'error':type(ex).__name__+':'+str(ex)})
   try: torch.cuda.synchronize()
   except Exception: pass
  tuning.append(row)
 valid=[r for r in tuning if r.get('valid')]
 best=min(valid,key=lambda r:r['ms']) if valid else None
 compare=[]
 if best:
  bc=best['config']
  for M in (1,8,16,32,64):
   A,ids,ec,Ce,ecall=setup(M); qs,qe,qp=moe_align_block_size(ids,bc['BLOCK_SIZE_M'],E,None); Cq=torch.zeros_like(Ce)
   def qcall(): invoke_fused_moe_wna16_triton_kernel(A,Bq,Cq,Bs,Bz,None,qs,qe,qp,False,topk,bc,tl.bfloat16,False,True,[0,gs])
   qcall(); ecall(); torch.cuda.synchronize(); diff=(Cq.float()-Ce.float()).abs(); close=bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08)); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0))
   reps=6 if M<=16 else 3
   def timing(fn):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t=time.perf_counter();
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)*1000/reps
   qms=timing(qcall); ems=timing(ecall); compare.append({'M':M,'q_ms':qms,'emulation_ms':ems,'speedup_x':ems/qms,'latency_reduction_percent':(ems-qms)/ems*100,'allclose':close,'cosine':cos,'max_abs_error':float(diff.max())})
 out={'pass':bool(best) and all(x['allclose'] and x['cosine']>.999 for x in compare),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'shape':{'E_reduced':E,'K':K,'N':N,'top_k':topk,'group_size':gs},'tuning_at_M8':tuning,'best':best,'best_vs_emulation':compare,'truth_boundary':'bounded block tuning for first MoE GEMM only; synthetic Qwen3 dimensions, no serving claim'}
except Exception as ex:
 out={'pass':False,'error':type(ex).__name__+':'+str(ex),'trace':traceback.format_exc()[-24000:]}
print(json.dumps(out,sort_keys=True))
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    p={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'truth_boundary':{'service_restarted':False,'model_weights_touched':False,'production_runtime_modified':False,'real_r9700_kernel':True,'synthetic_data':True}}
    if not c: p.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',code()],420); parsed={}
        for line in r.get('stdout','').splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        p['probe']=parsed;p['docker_exec']={'returncode':r.get('returncode'),'stderr_tail':r.get('stderr','')[-8000:]};p['pass']=bool(parsed.get('pass')) and r.get('returncode')==0
    p['probe_sha256']=digest(p); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); path=ROOT/'docs'/'evidence'/f'r9700_wna16_rdna4_tuning_{stamp}.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':p.get('pass'),'output':str(path),'probe_sha256':p['probe_sha256'],'best':(p.get('probe') or {}).get('best'),'best_vs_emulation':(p.get('probe') or {}).get('best_vs_emulation'),'tuning':(p.get('probe') or {}).get('tuning_at_M8')},sort_keys=True)); return 0 if p.get('pass') else 2
if __name__=='__main__': raise SystemExit(main())
