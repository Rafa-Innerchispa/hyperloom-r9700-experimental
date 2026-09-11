from __future__ import annotations
import datetime as dt,hashlib,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
IMAGE='rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0'
MODEL_PATH='/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ'
MODEL='QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ'
PORT=8011
NAME='hyperloom-r9700-p3-int4-repack-p8011'
OLD='hyperloom-r9700-p3-int4-repack'
SERVICE='inneros-vllm-canary-rocm10.service'
OVERLAY=ROOT/'var'/'r9700_phase3_vllm43389_overlay'
FILES={
'vllm/model_executor/layers/fused_moe/config.py':'18af9f7414b4a9e7620fd998f26b58b3294e58a5951af99a3923135509edc8a6',
'vllm/model_executor/layers/fused_moe/experts/triton_moe.py':'beec848e3a6b76c362ec81ecf35e8a515f14fa5deb959aff7f166b6e9e499201',
'vllm/model_executor/layers/fused_moe/fused_moe.py':'8de93930b8f7071741404fb27190273cd798d447fa32e274eda36e9499e0eb71',
'vllm/model_executor/layers/fused_moe/oracle/int_wna16.py':'6fd057e0f0fff1bdfad3a8c70f34f7fe352ab4c2f920ba13466ae2eb79096a24',
'vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py':'0f526fa4910b98fa8e64dea6b9c54b43d310ce47e91fa8366ab6e4e907d9a35d'}
def run(a,t=30):return subprocess.run(a,text=True,capture_output=True,timeout=t)
def shaf(p):return hashlib.sha256(p.read_bytes()).hexdigest()
svc=run(['systemctl','--user','is-active',SERVICE]); state=svc.stdout.strip()
if state=='active': raise RuntimeError('stock systemd service must be inactive before isolated benchmark')
for n in (OLD,NAME): run(['docker','rm','-f',n],t=30)
mounts=[]
for rel,expected in FILES.items():
 p=OVERLAY/rel
 if not p.exists() or shaf(p)!=expected: raise RuntimeError(f'overlay hash mismatch: {rel}')
 mounts += ['-v',f'{p}:/opt/python/lib/python3.14/site-packages/{rel}:ro']
# Wait for clean VRAM.
last=None; ok=False; t0=time.monotonic()
while time.monotonic()-t0<120:
 p=run(['rocm-smi','--showmeminfo','vram','--json'])
 try:
  o=json.loads(p.stdout); last=int(o['card0']['VRAM Total Used Memory (B)'])
  if last<1_000_000_000: ok=True; break
 except Exception: pass
 time.sleep(1)
if not ok: raise RuntimeError(f'VRAM did not return to clean state: {last}')
cmd=['docker','run','-d','--name',NAME,'--network','host','--ipc','host','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--security-opt','label=disable','-e','GPU_MAX_HW_QUEUES=1','-v',str(ROOT.parent.parent.parent.parent/'local_models')+':/models']
# The known model store lives at inneros_core/var/local_models, resolve robustly from this worktree.
model_store=pathlib.Path('/home/rlopez/inneros/inneros_core/var/local_models')
cmd=['docker','run','-d','--name',NAME,'--network','host','--ipc','host','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--security-opt','label=disable','-e','GPU_MAX_HW_QUEUES=1','-v',f'{model_store}:/models']+mounts+[IMAGE,'python3','-m','vllm.entrypoints.openai.api_server','--model',MODEL_PATH,'--served-model-name',MODEL,'--host','127.0.0.1','--port',str(PORT),'--max-model-len','8192','--gpu-memory-utilization','0.82','--dtype','float16','--trust-remote-code']
p=run(cmd,t=60)
out={'schema':'hyperloom.r9700.phase3.isolated_port_launch.v1','captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'name':NAME,'port':PORT,'launch_rc':p.returncode,'container_id':p.stdout.strip(),'stderr':p.stderr[-2000:],'stock_service_state_before':state,'vram_used_before':last,'image':IMAGE,'model':MODEL,'GPU_MAX_HW_QUEUES':'1','overlay_hashes':FILES,'runtime_patch_sha256':'3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d','ok':p.returncode==0}
ev=ROOT/'docs'/'evidence'; ev.mkdir(parents=True,exist_ok=True); q=ev/f"r9700_phase3_isolated_port_launch_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"; q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(str(q)); print(json.dumps(out,indent=2,sort_keys=True)); raise SystemExit(0 if out['ok'] else 1)
