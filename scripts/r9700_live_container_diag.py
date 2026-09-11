from __future__ import annotations
import json, subprocess

def run(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    return {'rc':p.returncode,'stdout':p.stdout[-20000:],'stderr':p.stderr[-4000:]}

out={
 'docker_ps':run(['docker','ps','-a','--format','{{.Names}}\t{{.Status}}\t{{.Image}}']),
 'stable_logs':run(['docker','logs','--tail','120','inneros-vllm-canary-rocm10']),
 'test_logs':run(['docker','logs','--tail','120','hyperloom-r9700-unified-pilot']),
 'rocm_pids':run(['rocm-smi','--showpids','--showmemuse','--json']),
 'vram':run(['rocm-smi','--showmeminfo','vram','--json']),
}
print(json.dumps(out,indent=2,sort_keys=True))
