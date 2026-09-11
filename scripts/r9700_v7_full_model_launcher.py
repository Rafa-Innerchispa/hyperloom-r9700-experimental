#!/usr/bin/env python3
"""Launch the clean full-model HyperLoom v7 candidate on the physical R9700.

Precondition: the stock systemd unit must already be stopped through authorized
host ops. The candidate starts in a fresh process with the versioned v7 patch
installed by an explicit Python entrypoint before vLLM imports/model loading.
"""
from __future__ import annotations
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SERVICE="inneros-vllm-canary-rocm10.service"; STOCK="inneros-vllm-canary-rocm10"; CANDIDATE="hyperloom-r9700-v7-candidate"
IMAGE="rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"; MODEL_PATH="/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"; MODEL_MOUNT="/home/rlopez/inneros/inneros_core/var/local_models:/models"
PATCH=ROOT/"scripts"/"r9700_wna16_hybrid_patch_v7_clean.py"; ENTRY=ROOT/"scripts"/"r9700_v7_server_entry.py"; EXPECTED="eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269"
def run(a,t=90): return subprocess.run(a,capture_output=True,text=True,timeout=t,check=False)
def service_state():
 p=run(["systemctl","--user","is-active",SERVICE],20); return p.stdout.strip() or "unknown"
def running(n):
 p=run(["docker","inspect","-f","{{.State.Running}}",n],20); return p.returncode==0 and p.stdout.strip()=="true"
def used_vram():
 p=run(["rocm-smi","--showmeminfo","vram","--json"],20)
 try:return int(json.loads(p.stdout)["card0"]["VRAM Total Used Memory (B)"])
 except Exception:return -1
def wait_vram(timeout=150):
 s=time.monotonic(); vals=[]
 while time.monotonic()-s<timeout:
  v=used_vram(); vals.append(v)
  if 0<=v<5*1024**3:return {"ok":True,"wait_sec":time.monotonic()-s,"last":v}
  time.sleep(2)
 return {"ok":False,"wait_sec":time.monotonic()-s,"samples_tail":vals[-10:]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity():
 p=run(["docker","inspect",CANDIDATE],30)
 if p.returncode:return {"ok":False,"stderr":p.stderr[-3000:]}
 r=json.loads(p.stdout)[0]; env=r.get("Config",{}).get("Env",[]) or []
 return {"ok":True,"image":r.get("Config",{}).get("Image"),"started_at":r.get("State",{}).get("StartedAt"),"pid":r.get("State",{}).get("Pid"),"running":r.get("State",{}).get("Running"),"relevant_env":sorted(x for x in env if x.startswith("GPU_MAX_HW_QUEUES=") or x.startswith("HYPERLOOM_R9700_") or x.startswith("VLLM_ROCM_USE_AITER")),"cmd":r.get("Config",{}).get("Cmd",[]),"mounts":[{"source":m.get("Source"),"destination":m.get("Destination"),"rw":m.get("RW")} for m in r.get("Mounts",[]) if m.get("Destination") in {"/tmp/r9700_wna16_hybrid_patch.py","/tmp/r9700_v7_server_entry.py"}]}
def main():
 stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); out=ROOT/"docs"/"evidence"/f"r9700_v7_full_model_launch_{stamp}.json"; out.parent.mkdir(parents=True,exist_ok=True)
 d={"schema":"hyperloom.r9700.v7_full_model_launch.v2","captured_at_utc":datetime.now(timezone.utc).isoformat(),"candidate":CANDIDATE,"model":MODEL,"image":IMAGE,"expected_patch_sha256":EXPECTED,"service_state_before":service_state(),"stock_container_running_before":running(STOCK),"truth_boundary":"launch identity only; no performance claim without separate measurement and path evidence"}
 if not PATCH.exists() or not ENTRY.exists():d.update(ok=False,error="v7 patch or explicit entrypoint missing")
 else:
  obs=sha(PATCH); d["observed_patch_sha256"]=obs
  if obs!=EXPECTED:d.update(ok=False,error="v7 patch SHA mismatch")
  elif d["service_state_before"] in {"active","activating","reloading"} or d["stock_container_running_before"]:d.update(ok=False,error="stock systemd/container must be stopped before candidate launch")
  else:
   run(["docker","rm","-f",CANDIDATE],30); free=wait_vram(); d["vram_free"]=free
   if not free.get("ok"):d.update(ok=False,error="VRAM did not return below clean-launch threshold")
   else:
    a=["docker","run","-d","--name",CANDIDATE,"--network","host","--ipc","host","--device=/dev/kfd","--device=/dev/dri","--group-add","video","--security-opt","label=disable","-v",MODEL_MOUNT,"-v",f"{PATCH}:/tmp/r9700_wna16_hybrid_patch.py:ro","-v",f"{ENTRY}:/tmp/r9700_v7_server_entry.py:ro","-e","GPU_MAX_HW_QUEUES=1","-e","HYPERLOOM_R9700_EVIDENCE_FILE=/tmp/r9700_candidate_paths.jsonl",IMAGE,"python3","/tmp/r9700_v7_server_entry.py","--model",MODEL_PATH,"--served-model-name",MODEL,"--host","127.0.0.1","--port","8000","--max-model-len","8192","--gpu-memory-utilization","0.82","--dtype","float16","--trust-remote-code"]
    p=run(a,45); d["launch_returncode"]=p.returncode; d["container_id"]=p.stdout.strip()
    if p.returncode:d.update(ok=False,error="docker run failed",stderr_tail=p.stderr[-5000:])
    else:d["identity"]=identity(); d["ok"]=True
 out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); print(str(out)); print(json.dumps(d,indent=2,sort_keys=True)); return 0 if d.get("ok") else 2
if __name__=="__main__":raise SystemExit(main())
