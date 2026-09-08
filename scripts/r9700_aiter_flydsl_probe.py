#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,hashlib,textwrap
from datetime import datetime,timezone
from pathlib import Path
from r9700_awq_backend_probe import discover_live_container
ROOT=Path(__file__).resolve().parents[1]
def run(a,timeout=180):
 p=subprocess.run(a,capture_output=True,text=True,timeout=timeout,check=False);return {'returncode':p.returncode,'stdout':p.stdout[-100000:],'stderr':p.stderr[-30000:]}
def code():return textwrap.dedent(r'''
import json,traceback,inspect
out={}
try:
 import torch,aiter
 from vllm._aiter_ops import rocm_aiter_ops,is_aiter_found_and_supported
 out['runtime']={'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'device':torch.cuda.get_device_name(0),'aiter_version':getattr(aiter,'__version__',None),'aiter_found_supported':bool(is_aiter_found_and_supported())}
 flags={}
 for name in ('is_fused_moe_enabled','is_rdna_aiter_enabled','is_aiter_enabled','is_triton_gemm_enabled'):
  fn=getattr(rocm_aiter_ops,name,None)
  if callable(fn):
   try:flags[name]=bool(fn())
   except Exception as e:flags[name]='error:'+type(e).__name__
 out['aiter_flags']=flags
 try:
  from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1,flydsl_a16w4_gemm2
  out['flydsl_import']={'ok':True,'gemm1':str(flydsl_a16w4_gemm1),'gemm2':str(flydsl_a16w4_gemm2)}
 except Exception as e:out['flydsl_import']={'ok':False,'error':type(e).__name__+':'+str(e)}
 try:
  from vllm.model_executor.layers.fused_moe.fused_flydsl_moe import fused_flydsl_moe_impl,_FLYDSL_MOE_DEFAULT_CONFIG
  out['vllm_flydsl']={'ok':True,'default_config':_FLYDSL_MOE_DEFAULT_CONFIG.get(8),'signature':str(inspect.signature(fused_flydsl_moe_impl))}
 except Exception as e:out['vllm_flydsl']={'ok':False,'error':type(e).__name__+':'+str(e)}
 out['pass']=bool(out.get('flydsl_import',{}).get('ok')) and bool(out.get('vllm_flydsl',{}).get('ok'))
except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-16000:]}
print(json.dumps(out,sort_keys=True))
''')
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');p={'schema':'hyperloom.r9700.aiter_flydsl_probe.v1','captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'truth_boundary':{'read_only':True,'service_restarted':False,'model_weights_touched':False}}
 r=run(['docker','exec',c,'python3','-c',code()]);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 p['probe']=o;p['pass']=bool(o.get('pass')) and r['returncode']==0;p['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-6000:]};raw=json.dumps(p,sort_keys=True,separators=(',',':')).encode();p['probe_sha256']=hashlib.sha256(raw).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_aiter_flydsl_probe_{stamp}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':p['pass'],'output':str(path),'probe_sha256':p['probe_sha256'],'probe':o},sort_keys=True));return 0 if p['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
