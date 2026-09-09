#!/usr/bin/env python3
"""Real-weight correctness smoke for the stock-layout-preserving R9700 hybrid."""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_stock_layout_real_weight_smoke.v1'
PATCH=ROOT/'scripts'/'r9700_wna16_hybrid_patch.py'
MODEL='/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ'

def run(argv:list[str],timeout:float=480.0)->dict[str,Any]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-360000:],'stderr':p.stderr[-80000:]}
def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None);return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def code():
 return textwrap.dedent(r'''
import json,sys,traceback,torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0,'/tmp')
import r9700_wna16_hybrid_patch as hp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig,FusedMoEParallelConfig,RoutingMethodType
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import make_wna16_moe_quant_config,_unpack_and_dequant_int4_awq,make_wna16_moe_kernel,WNA16MoEBackend
MODEL=Path("/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ");SHARD=MODEL/'model-00001-of-00006.safetensors';E=8;K=2048;I=768;TOPK=8

def load():
 gq=[];gz=[];gs=[];uq=[];uz=[];us=[];dq=[];dz=[];ds=[]
 with safe_open(str(SHARD),framework='pt',device='cpu') as f:
  for e in range(E):
   p=f'model.layers.0.mlp.experts.{e}.'
   gq.append(f.get_tensor(p+'gate_proj.qweight'));gz.append(f.get_tensor(p+'gate_proj.qzeros'));gs.append(f.get_tensor(p+'gate_proj.scales'))
   uq.append(f.get_tensor(p+'up_proj.qweight'));uz.append(f.get_tensor(p+'up_proj.qzeros'));us.append(f.get_tensor(p+'up_proj.scales'))
   dq.append(f.get_tensor(p+'down_proj.qweight'));dz.append(f.get_tensor(p+'down_proj.qzeros'));ds.append(f.get_tensor(p+'down_proj.scales'))
 w13q=torch.stack([torch.cat([gq[e],uq[e]],1) for e in range(E)]);w13z=torch.stack([torch.cat([gz[e],uz[e]],1) for e in range(E)]);w13s=torch.stack([torch.cat([gs[e],us[e]],1) for e in range(E)])
 return tuple(x.cuda() for x in (w13q,w13z,w13s,torch.stack(dq),torch.stack(dz),torch.stack(ds)))

def ref_moe(x,w13,w2,ids,tw):
 out=torch.zeros((x.size(0),K),device='cuda',dtype=torch.float32);xf=x.float();w13f=w13.float();w2f=w2.float()
 for m in range(x.size(0)):
  for j in range(TOPK):
   e=int(ids[m,j]);z=xf[m:m+1]@w13f[e].transpose(0,1);gate,up=z.split(I,1);h=F.silu(gate)*up;out[m:m+1]+=tw[m,j].float()*(h@w2f[e].transpose(0,1))
 return out

def main():
 out={}
 try:
  install=hp.install_patch(force=True);free0,total0=torch.cuda.mem_get_info();w13q,w13z,w13s,w2q,w2z,w2s=load()
  w1,s1,z1=hp._awq_w13_to_packed_nfirst(w13q,w13s,w13z);w2,ss2,zz2=hp._awq_w13_to_packed_nfirst(w2q,w2s,w2z)
  w13bf=_unpack_and_dequant_int4_awq(w13q,w13s,w13z,transpose_output=True,output_dtype=torch.bfloat16)
  w2tmp=_unpack_and_dequant_int4_awq(w2q,w2s,w2z,transpose_output=False,output_dtype=torch.bfloat16);w2bf=w2tmp.permute(0,2,1).contiguous()
  par=FusedMoEParallelConfig(tp_size=1,pcp_size=1,dp_size=1,ep_size=1,tp_rank=0,pcp_rank=0,dp_rank=0,ep_rank=0,sp_size=1,use_ep=False,all2all_backend='',enable_eplb=False)
  moe=FusedMoEConfig(num_experts=E,experts_per_token=TOPK,hidden_dim=K,intermediate_size=I,num_local_experts=E,num_logical_experts=E,activation=MoEActivation.SILU,device='cuda',routing_method=RoutingMethodType.TopK,moe_parallel_config=par,in_dtype=torch.float16,intermediate_size_per_partition=I)
  qcfg=make_wna16_moe_quant_config(w1_scale=s1,w2_scale=ss2,group_size=128,num_bits=4,w1_zp=z1,w2_zp=zz2)
  experts=hp.R9700HybridWNA16Experts(moe,qcfg);kernel=make_wna16_moe_kernel(moe_quant_config=qcfg,moe_config=moe,experts_cls=hp.R9700HybridWNA16Experts,backend=WNA16MoEBackend.TRITON)
  torch.manual_seed(20260909);base=torch.arange(TOPK,device='cuda',dtype=torch.int64);cases=[]
  for M in (1,8,20):
   x=(torch.randn((M,K),device='cuda')*.05).to(torch.float16);ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous();tw=torch.full((M,TOPK),1/TOPK,device='cuda',dtype=torch.float32)
   ws1s,ws2s,outs=experts.workspace_shapes(M,w1.size(1),K,TOPK,E,E,None,MoEActivation.SILU);ws1=torch.empty(ws1s,device='cuda',dtype=torch.float16);ws2=torch.empty(ws2s,device='cuda',dtype=torch.float16);y=torch.empty(outs,device='cuda',dtype=torch.float16)
   experts.apply(output=y,hidden_states=x,w1=w1,w2=w2,topk_weights=tw,topk_ids=ids,activation=MoEActivation.SILU,global_num_experts=E,expert_map=None,a1q_scale=None,a2_scale=None,workspace13=ws1,workspace2=ws2,expert_tokens_meta=None,apply_router_weight_on_input=False);torch.cuda.synchronize();path=experts.last_path
   ref=ref_moe(x,w13bf,w2bf,ids,tw);yf=y.float();cos=float(F.cosine_similarity(yf.flatten(),ref.flatten(),dim=0));rel=float(torch.linalg.vector_norm(yf-ref)/(torch.linalg.vector_norm(ref)+1e-12));finite=bool(torch.isfinite(y).all());passed=finite and cos>=.999 and rel<=.02
   cases.append({'M':M,'path':path,'finite':finite,'cosine':cos,'relative_l2':rel,'max_abs_error':float((yf-ref).abs().max()),'pass':passed})
  free1,total1=torch.cuda.mem_get_info();out={'pass':all(c['pass'] for c in cases) and cases[0]['path']=='custom_small_w1_stock_w2' and cases[1]['path']=='custom_small_w1_stock_w2' and cases[2]['path']=='stock_full_fallback','install':install,'experts_type':type(experts).__name__,'kernel_type':type(kernel).__name__,'converted':{'w1_shape':list(w1.shape),'w1_dtype':str(w1.dtype),'w1_scale_shape':list(s1.shape),'w1_zp_shape':list(z1.shape),'w2_shape':list(w2.shape),'w2_dtype':str(w2.dtype),'w2_scale_shape':list(ss2.shape),'w2_zp_shape':list(zz2.shape),'correction_shape':list(experts.w1_correction.shape)},'cases':cases,'gpu_memory':{'free_before_bytes':int(free0),'free_after_bytes':int(free1),'total_bytes':int(total1),'probe_delta_bytes':int(free0-free1)},'truth_boundary':'actual Qwen layer-0 AWQ W1/W2 weights; FP16 routed activations; small W1 custom, W2 stock WNA16, M20 complete stock fallback; resident vLLM untouched; no E2E serving performance claim'}
 except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-40000:]}
 print(json.dumps(out,sort_keys=True))
main()
''')
def main()->int:
 live=discover_live_container(run=run);c=str((live.get('selected') or {}).get('Names') or '');payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c}
 if not c:payload.update({'pass':False,'error':'no_vllm_container'})
 else:
  cp=run(['docker','cp',str(PATCH),f'{c}:/tmp/r9700_wna16_hybrid_patch.py'],60);ex=run(['docker','exec',c,'python3','-c',code()],480);parsed={}
  for line in ex['stdout'].splitlines():
   try:o=json.loads(line)
   except Exception:continue
   if isinstance(o,dict):parsed=o
  payload.update({'copy':{'returncode':cp['returncode']},'probe':parsed,'docker_exec':{'returncode':ex['returncode'],'stderr_tail':ex['stderr'][-16000:]},'pass':cp['returncode']==0 and ex['returncode']==0 and bool(parsed.get('pass'))})
 payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_wna16_stock_layout_real_weight_smoke_{stamp}.json';path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'path':str(path.relative_to(ROOT)),'sha256':payload['probe_sha256'],'cases':(payload.get('probe') or {}).get('cases'),'converted':(payload.get('probe') or {}).get('converted'),'error':(payload.get('probe') or {}).get('error')},indent=2,sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
