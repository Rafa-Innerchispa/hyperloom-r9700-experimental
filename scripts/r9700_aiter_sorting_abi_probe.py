#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,inspect,textwrap
from r9700_awq_backend_probe import discover_live_container
from r9700_aiter_flydsl_direct_smoke import run

def inner(): return textwrap.dedent(r'''
import json,inspect,traceback,torch
out={}
try:
 import aiter,aiter.fused_moe as fm
 x=torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]],device='cuda',dtype=torch.int32);w=torch.full((2,8),1/8,device='cuda',dtype=torch.float32)
 try: fm.moe_sorting(x,w,8,2048,torch.float16,16)
 except Exception as e: out['expected_wrapper_error']=type(e).__name__+':'+str(e)
 op=torch.ops.aiter.moe_sorting_opus_fwd
 out['schemas']=str(getattr(op,'_schemas',None));out['wrapper_source']=inspect.getsource(fm._moe_sorting_impl);out['public_source']=inspect.getsource(fm.moe_sorting)
 for name in ('moe_sorting_opus_get_workspace_size','moe_sorting_opus_fwd'):
  obj=getattr(aiter,name,None);out[name]=str(obj)
 out['pass']=True
except Exception as e:out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-16000:]}
print(json.dumps(out,sort_keys=True))
''')
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');r=run(['docker','exec','-e','GPU_ARCHS=gfx1201','-e','AITER_GPU_ARCHS=gfx1201',c,'python3','-c',inner()],180);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 print(json.dumps({'returncode':r['returncode'],'probe':o,'stderr_tail':r['stderr'][-4000:]},sort_keys=True));return 0 if o.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
