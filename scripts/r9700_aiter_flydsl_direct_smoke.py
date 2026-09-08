#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,textwrap
from datetime import datetime,timezone
from pathlib import Path
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
def run(a,timeout=420):
 p=subprocess.run(a,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-180000:],'stderr':p.stderr[-40000:]}
def code():return textwrap.dedent(r'''
import json,time,traceback,torch
out={}
try:
 from aiter.fused_moe import moe_sorting
 from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1,flydsl_a16w4_gemm2
 from aiter.ops.shuffle import shuffle_weight
 dev=torch.device('cuda');torch.manual_seed(20260908)
 E,H,I,topk,M,gs=8,2048,768,8,2,32
 # Build symmetric signed int4 values, GPTQ pack along K, then FlyDSL shuffle/pack.
 def pack_gptq(vals):
  # vals [E,N,K] signed [-8,7] -> [E,K//8,N] int32 standard nibble order
  u=(vals.to(torch.int16)&15).to(torch.int32); E0,N,K=u.shape; x=u.transpose(1,2).contiguous().reshape(E0,K//8,8,N); shifts=(torch.arange(8,device=dev,dtype=torch.int32)*4).view(1,1,8,1); return torch.sum(x<<shifts,dim=2,dtype=torch.int32)
 def to_fly(vals):
  sh=shuffle_weight(vals.contiguous(),layout=(16,16));flat=sh.contiguous().view(-1).to(torch.int16);u=(flat&15).to(torch.uint8).view(-1,8);z=torch.empty((u.shape[0],4),device=dev,dtype=torch.uint8);z[:,0]=u[:,0]|(u[:,4]<<4);z[:,1]=u[:,1]|(u[:,5]<<4);z[:,2]=u[:,2]|(u[:,6]<<4);z[:,3]=u[:,3]|(u[:,7]<<4);return z.view(-1).to(torch.int8)
 w13v=torch.randint(-8,8,(E,2*I,H),dtype=torch.int8,device=dev);w2v=torch.randint(-8,8,(E,H,I),dtype=torch.int8,device=dev)
 w13=to_fly(w13v);w2=to_fly(w2v)
 s13=(torch.rand((E,H//gs,2*I),device=dev)*.015+.002).to(torch.bfloat16);s2=(torch.rand((E,I//gs,H),device=dev)*.015+.002).to(torch.bfloat16)
 def fly_scale(s):
  E0,G,N=s.shape;return s.view(E0,G//2,2,N).permute(0,1,3,2).contiguous().view(-1).contiguous()
 s13f=fly_scale(s13);s2f=fly_scale(s2)
 x=(torch.randn((M,H),device=dev)*.1).to(torch.bfloat16);scores=torch.randn((M,E),device=dev);vals,ids=torch.topk(scores,k=topk,dim=1);tw=torch.softmax(vals,dim=1).float();tile_m,tile_n,tile_k,tile_n2,tile_k2=16,64,128,128,128
 ids32=ids.int();tw32=tw.float();sorted_ids,sorted_w,sorted_e,nvalid,_=moe_sorting(ids32,tw32,E,H,torch.float16,tile_m);nvalid=nvalid[:1].contiguous();sorted_ids=sorted_ids.contiguous();sorted_w=sorted_w.contiguous();sorted_e=sorted_e.contiguous();ss=int(sorted_ids.numel())
 inter=torch.empty((ss,I),device=dev,dtype=torch.bfloat16);empty=torch.empty((0,),device=dev,dtype=torch.uint8)
 torch.cuda.synchronize();t0=time.perf_counter()
 flydsl_a16w4_gemm1(a_bf16=x.contiguous(),w1_u8=w13.view(torch.uint8).contiguous(),w1_scale_u8=s13f.view(torch.uint8).contiguous().view(-1),sorted_expert_ids=sorted_e,cumsum_tensor=nvalid.int().contiguous(),m_indices=sorted_ids.int().contiguous(),inter_sorted_bf16=inter,n_tokens=M,NE=E,D_HIDDEN=H,D_INTER=I,topk=topk,tile_m=tile_m,tile_n=tile_n,tile_k=tile_k,waves_per_eu=None,act='silu',w_dtype='int4',w_layout='standard')
 out2=torch.zeros((M,H),device=dev,dtype=torch.bfloat16)
 flydsl_a16w4_gemm2(inter_sorted_bf16=inter,w2_u8=w2.view(torch.uint8).contiguous(),w2_scale_u8=s2f.view(torch.uint8).contiguous().view(-1),sorted_expert_ids=sorted_e,cumsum_tensor=nvalid.int().contiguous(),sorted_token_ids=sorted_ids,sorted_weights=sorted_w.view(-1).contiguous(),flat_out=out2.view(-1),M_logical=M,max_sorted=ss,NE=E,D_HIDDEN=H,D_INTER=I,topk=topk,tile_m=tile_m,tile_n=tile_n2,tile_k=tile_k2,waves_per_eu=None,w_dtype='int4')
 torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
 # Reference full two-stage MoE from original signed values/scales.
 w13scale=s13.transpose(1,2).repeat_interleave(gs,dim=2).float();w2scale=s2.transpose(1,2).repeat_interleave(gs,dim=2).float();W13=w13v.float()*w13scale;W2=w2v.float()*w2scale
 ref=[]
 for m in range(M):
  acc=torch.zeros((H,),device=dev,dtype=torch.float32)
  for j in range(topk):
   e=int(ids[m,j]);h=x[m].float()@W13[e].t();gate,up=h[:I],h[I:];mid=torch.nn.functional.silu(gate)*up;y=mid@W2[e].t();acc+=tw[m,j]*y
  ref.append(acc)
 ref=torch.stack(ref);d=(out2.float()-ref).abs();cos=float(torch.nn.functional.cosine_similarity(out2.float().flatten(),ref.flatten(),dim=0));close=bool(torch.allclose(out2.float(),ref,rtol=.12,atol=.12));out={'pass':close and cos>.995,'device':torch.cuda.get_device_name(0),'elapsed_first_compile_and_run_ms':elapsed,'shape':{'E':E,'H':H,'I':I,'top_k':topk,'M':M,'group_size':gs},'numeric':{'allclose':close,'cosine':cos,'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean()),'p99_abs_error':float(torch.quantile(d.flatten(),.99))},'routing':{'sorted_size':ss,'nvalid':nvalid.cpu().tolist()},'interpretation':'direct AITER FlyDSL W4A16 two-stage kernel executed and matched reference' if close and cos>.995 else 'kernel executed but numeric gate failed'}
except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-24000:]}
print(json.dumps(out,sort_keys=True))
''')
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');p={'schema':'hyperloom.r9700.aiter_flydsl_direct_smoke.v1','captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'truth_boundary':{'synthetic_symmetric_int4':True,'autoawq_zero_points_supported':False,'service_restarted':False,'model_weights_touched':False}}
 r=run(['docker','exec',c,'python3','-c',code()],420);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 p['probe']=o;p['pass']=bool(o.get('pass')) and r['returncode']==0;p['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-8000:]};raw=json.dumps(p,sort_keys=True,separators=(',',':')).encode();p['probe_sha256']=hashlib.sha256(raw).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_aiter_flydsl_direct_smoke_{stamp}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':p['pass'],'output':str(path),'probe_sha256':p['probe_sha256'],'probe':o},sort_keys=True));return 0 if p['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
