#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from r9700_awq_backend_probe import discover_live_container
from r9700_aiter_flydsl_direct_smoke import run,code
ROOT=Path(__file__).resolve().parents[1]
def main():
 live=discover_live_container(run=run);c=str((live.get('selected')or{}).get('Names')or'');p={'schema':'hyperloom.r9700.aiter_flydsl_gfx1201_retry.v1','captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'forced_env':{'GPU_ARCHS':'gfx1201','AITER_GPU_ARCHS':'gfx1201'},'truth_boundary':{'process_scoped_env_only':True,'persistent_runtime_changed':False,'service_restarted':False,'synthetic_symmetric_int4':True,'autoawq_zero_points_supported':False}}
 r=run(['docker','exec','-e','GPU_ARCHS=gfx1201','-e','AITER_GPU_ARCHS=gfx1201',c,'python3','-c',code()],420);o={}
 for line in r['stdout'].splitlines():
  try:x=json.loads(line)
  except:continue
  if isinstance(x,dict):o=x
 p['probe']=o;p['pass']=bool(o.get('pass')) and r['returncode']==0;p['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-10000:]};p['probe_sha256']=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_aiter_flydsl_gfx1201_retry_{stamp}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':p['pass'],'output':str(path),'probe_sha256':p['probe_sha256'],'probe':o,'stderr_tail':r['stderr'][-4000:]},sort_keys=True));return 0 if p['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
