#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from r9700_awq_backend_probe import discover_live_container
from r9700_aiter_flydsl_direct_smoke import run
ROOT=Path(__file__).resolve().parents[1]
def code():return textwrap.dedent(r'''
import json,time,traceback,torch
out={}
try:
 import aiter
 from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1,flydsl_a16w4_gemm2
 from aiter.ops.shuffle import shuffle_weight
 dev=torch.device('cuda');torch.manual_seed(20260908);E,H,I,topk,M,gs=8,2048,768,8,2,32;bm,bn,bk,bn2,bk2=16,64,128,128,128
 def to_fly(vals):
  sh=shuffle_weight(vals.contiguous(),layout=(16,16));u=(sh.contiguous().view(-1).to(torch.int16)&15).to(torch.uint8).view(-1,8);z=torch.empty((u.shape[0],4),device=dev,dtype=torch.uint8);z[:,0]=u[:,0]|(u[:,4]<<4);z[:,1]=u[:,1]|(u[:,5]<<4);z[:,2]=u[:,2]|(u[:,6]<<4);z[:,3]=u[:,3]|(u[:,7]<<4);return z.view(-1).to(torch.int8)
 def fs(s):E0,G,N=s.shape;return s.view(E0,G//2,2,N).permute(0,1,3,2).contiguous().view(-1)
 w13v=torch.randint(-8,8,(E,2*I,H),dtype=torch.int8,device=dev);w2v=torch.randint(-8,8,(E,H,I),dtype=torch.int8,device=dev);w13=to_fly(w13v);w2=to_fly(w2v);s13=(torch.rand((E,H//gs,2*I),device=dev)*.015+.002).to(torch.bfloat16);s2=(torch.rand((E,I//gs,H),device=dev)*.015+.002).to(torch.bfloat16);s13f=fs(s13);s2f=fs(s2)
 x=(torch.randn((M,H),device=dev)*.1).to(torch.bfloat16);scores=torch.randn((M,E),device=dev);tv,ids=torch.topk(scores,k=topk,dim=1);tw=torch.softmax(tv,dim=1).float();ids=ids.int().contiguous();tw=tw.float().contiguous()
 # Force-load JIT op, then bypass the broken aiter Python wrapper and invoke torch op schema directly.
 ws_size=aiter.moe_sorting_opus_get_workspace_size(M,E,topk,0);maxpad=int(ids.numel()+E*bm-topk);maxblocks=(maxpad+bm-1)//bm;sid=torch.empty(maxpad,dtype=torch.int32,device=dev);sw=torch.empty(maxpad,dtype=torch.float32,device=dev);se=torch.empty(maxblocks,dtype=torch.int32,device=dev);nv=torch.empty(2,dtype=torch.int32,device=dev);mbuf=torch.empty((M,H),dtype=torch.float16,device=dev);workspace=torch.empty(ws_size,dtype=torch.uint8,device=dev) if ws_size>0 else None
 torch.ops.aiter.moe_sorting_opus_fwd(ids,tw,sid,sw,se,nv,mbuf,E,bm,None,None,workspace,0,None,None,None);torch.cuda.synchronize();nv1=nv[:1].contiguous();ss=int(sid.numel())
 inter=torch.empty((ss,I),device=dev,dtype=torch.bfloat16);torch.cuda.synchronize();t=time.perf_counter();flydsl_a16w4_gemm1(a_bf16=x.contiguous(),w1_u8=w13.view(torch.uint8).contiguous(),w1_scale_u8=s13f.view(torch.uint8).contiguous().view(-1),sorted_expert_ids=se.contiguous(),cumsum_tensor=nv1.int(),m_indices=sid.int().contiguous(),inter_sorted_bf16=inter,n_tokens=M,NE=E,D_HIDDEN=H,D_INTER=I,topk=topk,tile_m=bm,tile_n=bn,tile_k=bk,waves_per_eu=None,act='silu',w_dtype='int4',w_layout='standard');y=torch.zeros((M,H),device=dev,dtype=torch.bfloat16);flydsl_a16w4_gemm2(inter_sorted_bf16=inter,w2_u8=w2.view(torch.uint8).contiguous(),w2_scale_u8=s2f.view(torch.uint8).contiguous().view(-1),sorted_expert_ids=se.contiguous(),cumsum_tensor=nv1.int(),sorted_token_ids=sid,sorted_weights=sw.view(-1).contiguous(),flat_out=y.view(-1),M_logical=M,max_sorted=ss,NE=E,D_HIDDEN=H,D_INTER=I,topk=topk,tile_m=bm,tile_n=bn2,tile_k=bk2,waves_per_eu=None,w_dtype='int4');torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
 # Reference
 W13=w13v.float()*s13.transpose(1,2).repeat_interleave(gs,2).float();W2=w2v.float()*s2.transpose(1,2).repeat_interleave(gs,2).float();ref=[]
 for m in range(M):
  acc=torch.zeros(H,device=dev)
  for j in range(topk):
   e=int(ids[m,j]);h=x[m].float()@W13[e].t();g,u=h[:I],h[I:];acc+=tw[m,j]*(torch.nn.functional.silu(g)*u@W2[e].t())
  ref.append(acc)
 ref=torch.stack(ref);d=(y.float()-ref).abs();cos=float(torch.nn.functional.cosine_similarity(y.float().flatten(),ref.flatten(),dim=0));close=bool(torch.allclose(y.float(),ref,rtol=.12,atol=.12));out={'pass':close and cos>.995,'device':torch.cuda.get_device_name(0),'kernel_ms_first_compiled_run':ms,'workspace_bytes':ws_size,'routing':{'maxpad':maxpad,'num_valid_ids':nv.cpu().tolist(),'maxblocks':maxblocks},'numeric':{'allclose':close,'cosine':cos,'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean()),'p99_abs_error':float(torch.quantile(d.flatten(),.99))},'interpretation':'ABI-bypassed AITER FlyDSL W4A16 full two-stage kernel executed on gfx1201' if close and cos>.995 else 'AITER path launched but numeric gate failed'}
except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-26000:]}
print(json.dumps(out,sort_keys=True))
''')
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');p={'schema':'hyperloom.r9700.aiter_flydsl_abi_patch_smoke.v1','captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'forced_env':{'GPU_ARCHS':'gfx1201','AITER_GPU_ARCHS':'gfx1201'},'truth_boundary':{'experimental_wrapper_bypass':True,'persistent_runtime_changed':False,'service_restarted':False,'synthetic_symmetric_int4':True,'autoawq_zero_points_supported':False}}
 r=run(['docker','exec','-e','GPU_ARCHS=gfx1201','-e','AITER_GPU_ARCHS=gfx1201',c,'python3','-c',code()],420);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 p['probe']=o;p['pass']=bool(o.get('pass')) and r['returncode']==0;p['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-10000:]};p['probe_sha256']=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_aiter_flydsl_abi_patch_smoke_{stamp}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':p['pass'],'output':str(path),'probe_sha256':p['probe_sha256'],'probe':o,'stderr_tail':r['stderr'][-5000:]},sort_keys=True));return 0 if p['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
