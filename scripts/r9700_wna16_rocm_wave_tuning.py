#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
def run(a,timeout=420):
 p=subprocess.run(a,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-160000:],'stderr':p.stderr[-30000:]}
def code(): return textwrap.dedent(r'''
import json,time,traceback,torch
out={}
try:
 import triton.language as tl
 from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
 from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
 from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel,invoke_fused_moe_triton_kernel,try_get_optimal_moe_config
 from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
 dev=torch.device('cuda');torch.manual_seed(20260908);E,K,N,N2,gs,topk,M=8,2048,1536,768,128,8,8
 shifts=torch.arange(0,32,4,dtype=torch.int32,device=dev);rev=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=dev)
 qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=dev);qz=torch.randint(-(2**31),2**31-1,(E,K//gs,N//8),dtype=torch.int32,device=dev);s=(torch.rand((E,K//gs,N),device=dev)*.02+.002).to(torch.float16)
 v=((qw.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,K,N).transpose(1,2).contiguous();Bq=(v[...,0::2]|(v[...,1::2]<<4)).to(torch.uint8).contiguous();Bs=s.transpose(1,2).contiguous();z=((qz.unsqueeze(-1)>>shifts)&15)[...,rev].reshape(E,K//gs,N).transpose(1,2).contiguous();Bz=(z[:,0::2,:]|(z[:,1::2,:]<<4)).to(torch.uint8).contiguous();Bbf=_unpack_and_dequant_int4_awq(qw,s,qz,True,torch.bfloat16)
 A=(torch.randn((M,K),device=dev)*.1).to(torch.bfloat16).contiguous();base=torch.arange(topk,device=dev);ids=torch.stack([(base+i)%E for i in range(M)]).long().contiguous();d2b=torch.empty((E,K,N2),dtype=torch.bfloat16,device=dev)
 ec=dict(try_get_optimal_moe_config(Bbf.size(),d2b.size(),topk,None,M));es,ee,ep=moe_align_block_size(ids,ec['BLOCK_SIZE_M'],E,None);Ce=torch.zeros((M,topk,N),dtype=torch.bfloat16,device=dev)
 def ecall(): invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,es,ee,ep,False,topk,ec,tl.bfloat16,False,False,False,False,False,None,None)
 ecall();torch.cuda.synchronize();
 def timing(fn,reps=10):
  for _ in range(2):fn()
  torch.cuda.synchronize();t=time.perf_counter()
  for _ in range(reps):fn()
  torch.cuda.synchronize();return (time.perf_counter()-t)*1000/reps
 ems=timing(ecall)
 rows=[]
 for nw in (1,2,4,8):
  for w in (0,1,2,4):
   c={'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':1,'SPLIT_K':1,'num_warps':nw,'num_stages':2,'waves_per_eu':w};row={'config':c}
   try:
    qs,qe,qp=moe_align_block_size(ids,16,E,None);C=torch.zeros_like(Ce)
    def f():invoke_fused_moe_wna16_triton_kernel(A,Bq,C,Bs,Bz,None,qs,qe,qp,False,topk,c,tl.bfloat16,False,True,[0,gs])
    f();torch.cuda.synchronize();d=(C.float()-Ce.float()).abs();cos=float(torch.nn.functional.cosine_similarity(C.float().flatten(),Ce.float().flatten(),dim=0));ok=bool(torch.allclose(C.float(),Ce.float(),rtol=.08,atol=.08)) and cos>.999
    ms=timing(f,8);row.update({'valid':ok,'ms':ms,'speedup_vs_emulation_x':ems/ms,'latency_reduction_percent':(ems-ms)/ems*100,'cosine':cos,'max_abs_error':float(d.max())})
   except Exception as ex: row.update({'valid':False,'error':type(ex).__name__+':'+str(ex)})
   rows.append(row)
 valid=[r for r in rows if r.get('valid')];best=min(valid,key=lambda r:r['ms']) if valid else None
 out={'pass':bool(best),'device':torch.cuda.get_device_name(0),'emulation_ms':ems,'rows':rows,'best':best,'truth_boundary':'M=8 first Qwen3-shaped MoE GEMM; AMD ROCm tuning knobs only; no serving claim'}
except Exception as ex:out={'pass':False,'error':type(ex).__name__+':'+str(ex),'trace':traceback.format_exc()[-20000:]}
print(json.dumps(out,sort_keys=True))
''')
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');p={'schema':'hyperloom.r9700.wna16_rocm_wave_tuning.v1','captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'truth_boundary':{'service_restarted':False,'model_weights_touched':False,'production_runtime_modified':False}}
 r=run(['docker','exec',c,'python3','-c',code()]);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 p['probe']=o;p['pass']=bool(o.get('pass')) and r['returncode']==0;p['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-6000:]};q=json.loads(json.dumps(p,sort_keys=True));p['probe_sha256']=hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_wna16_rocm_wave_tuning_{stamp}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':p['pass'],'output':str(path),'probe_sha256':p['probe_sha256'],'emulation_ms':o.get('emulation_ms'),'best':o.get('best'),'rows':o.get('rows')},sort_keys=True));return 0 if p['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
