#!/usr/bin/env python3
"""Contract smoke for the process-local R9700 hybrid Experts implementation.

Copies only the experimental Python module into /tmp of the running container,
installs its monkeypatch in that child interpreter, constructs vLLM's real
FusedMoEConfig/FusedMoEQuantConfig, and calls Experts.apply for both a small
custom-W1 case and a larger generic-WNA16 fallback case. The vLLM server process
is not patched, restarted, or imported into the child interpreter's state.
"""
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1];SCHEMA='hyperloom.r9700.wna16_hybrid_contract_smoke.v1'
PATCH=ROOT/'scripts'/'r9700_wna16_hybrid_patch.py'
def run(argv:list[str],timeout:float=360,input_text:str|None=None)->dict[str,Any]:
 p=subprocess.run(argv,input=input_text,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-260000:],'stderr':p.stderr[-50000:]}
def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None);return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def child_code()->str:return textwrap.dedent(r'''
import json,sys,traceback,torch
sys.path.insert(0,'/tmp')
import r9700_wna16_hybrid_patch as hp
out={}
try:
 info=hp.install_patch(force=True)
 from vllm.model_executor.layers.fused_moe.activation import MoEActivation
 from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig,FusedMoEParallelConfig,RoutingMethodType
 from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import make_wna16_moe_quant_config,make_wna16_moe_kernel,WNA16MoEBackend
 E=8;K=2048;N=1536;I=768;G=16;topk=8
 par=FusedMoEParallelConfig(tp_size=1,pcp_size=1,dp_size=1,ep_size=1,tp_rank=0,pcp_rank=0,dp_rank=0,ep_rank=0,sp_size=1,use_ep=False,all2all_backend='',enable_eplb=False)
 moe=FusedMoEConfig(num_experts=E,experts_per_token=topk,hidden_dim=K,intermediate_size=I,num_local_experts=E,num_logical_experts=E,activation=MoEActivation.SILU,device='cuda',routing_method=RoutingMethodType.TopK,moe_parallel_config=par,in_dtype=torch.bfloat16,intermediate_size_per_partition=I)
 w1=torch.randint(0,256,(E,N,K//2),device='cuda',dtype=torch.uint8)
 s=(torch.rand((E,N,G),device='cuda')*.02+.002).to(torch.float16)
 zp=torch.randint(0,256,(E,N//2,G),device='cuda',dtype=torch.uint8)
 w2=(torch.randn((E,K,I),device='cuda')*.02).to(torch.bfloat16)
 dummy=torch.ones(1,device='cuda',dtype=torch.float16)
 qcfg=make_wna16_moe_quant_config(w1_scale=s,w2_scale=dummy,group_size=128,num_bits=4,w1_zp=zp,w2_zp=None)
 experts=hp.R9700HybridWNA16Experts(moe,qcfg)
 cases=[]
 for M in (1,20):
  x=(torch.randn((M,K),device='cuda')*.05).to(torch.bfloat16);ids=torch.stack([torch.arange(topk,device='cuda',dtype=torch.int64)%E for _ in range(M)],0);tw=torch.full((M,topk),1.0/topk,device='cuda',dtype=torch.float32)
  ws1_shape,ws2_shape,out_shape=experts.workspace_shapes(M,N,K,topk,E,E,None,MoEActivation.SILU)
  ws1=torch.empty(ws1_shape,device='cuda',dtype=torch.bfloat16);ws2=torch.empty(ws2_shape,device='cuda',dtype=torch.bfloat16);y=torch.empty(out_shape,device='cuda',dtype=torch.bfloat16)
  experts.apply(output=y,hidden_states=x,w1=w1,w2=w2,topk_weights=tw,topk_ids=ids,activation=MoEActivation.SILU,global_num_experts=E,expert_map=None,a1q_scale=None,a2_scale=None,workspace13=ws1,workspace2=ws2,expert_tokens_meta=None,apply_router_weight_on_input=False)
  torch.cuda.synchronize();cases.append({'M':M,'finite':bool(torch.isfinite(y).all().item()),'mean_abs':float(y.float().abs().mean().item()),'max_abs':float(y.float().abs().max().item()),'output_dtype':str(y.dtype),'path':'custom_small_w1' if M<=hp.SMALL_TOKEN_LIMIT else 'generic_wna16_w1_fallback'})
 # Verify make_wna16_moe_kernel accepts the monkeypatched Experts class exactly.
 kernel=make_wna16_moe_kernel(moe_quant_config=qcfg,moe_config=moe,experts_cls=hp.R9700HybridWNA16Experts,backend=WNA16MoEBackend.TRITON)
 out={'pass':all(c['finite'] for c in cases),'install':info,'cases':cases,'kernel_type':type(kernel).__name__,'experts_type':type(experts).__name__,'correction_shape':list(experts.w1_correction.shape),'correction_dtype':str(experts.w1_correction.dtype),'truth_boundary':'synthetic class-contract smoke only; no live server patch and no serving benchmark'}
except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
print(json.dumps(out,sort_keys=True))
''')
def main()->int:
 live=discover_live_container(run=run);c=str((live.get('selected') or {}).get('Names') or '');payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'server_process_patched':False,'service_restarted':False,'production_runtime_modified':False,'synthetic_data':True,'real_r9700_kernel':True,'official_support_claim':False},'container':c}
 if not c:payload.update({'pass':False,'error':'no_vllm_container'})
 else:
  cp=run(['docker','cp',str(PATCH),f'{c}:/tmp/r9700_wna16_hybrid_patch.py'],60);ex=run(['docker','exec',c,'python3','-c',child_code()],360);parsed={}
  for line in ex['stdout'].splitlines():
   try:o=json.loads(line)
   except Exception:continue
   if isinstance(o,dict):parsed=o
  payload['copy']={'returncode':cp['returncode'],'stderr_tail':cp['stderr'][-2000:]};payload['probe']=parsed;payload['docker_exec']={'returncode':ex['returncode'],'stderr_tail':ex['stderr'][-12000:]};payload['pass']=cp['returncode']==0 and ex['returncode']==0 and bool(parsed.get('pass'))
 payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_hybrid_contract_smoke_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'probe':payload.get('probe',{})},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
