from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import socket
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_ROOT = pathlib.Path('/home/rlopez/inneros/inneros_core/var/local_execution/worktrees/Rafa-Innerchispa__hyperloom-r9700-experimental/chatgpt__r9700-amd-final-runtime-20260910')
IMAGE = 'rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0'
MODEL_PATH = '/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ'
MODEL = 'QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ'
PORT = 18017
NAME = 'hyperloom-r9700-p4-s3-verbose-p18017'
SERVICE = 'inneros-vllm-canary-rocm10.service'
OVERLAY = RUNTIME_ROOT / 'var' / 'r9700_phase3_vllm43389_overlay'
CONFIG = RUNTIME_ROOT / 'scripts' / 'r9700_phase3_tuned_moe_config.json'
CONFIG_SHA = '8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6'
PATCH_SHA = '3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d'
FILES = {
    'vllm/model_executor/layers/fused_moe/config.py': '18af9f7414b4a9e7620fd998f26b58b3294e58a5951af99a3923135509edc8a6',
    'vllm/model_executor/layers/fused_moe/experts/triton_moe.py': 'beec848e3a6b76c362ec81ecf35e8a515f14fa5deb959aff7f166b6e9e499201',
    'vllm/model_executor/layers/fused_moe/fused_moe.py': '8de93930b8f7071741404fb27190273cd798d447fa32e274eda36e9499e0eb71',
    'vllm/model_executor/layers/fused_moe/oracle/int_wna16.py': '6fd057e0f0fff1bdfad3a8c70f34f7fe352ab4c2f920ba13466ae2eb79096a24',
    'vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py': '0f526fa4910b98fa8e64dea6b9c54b43d310ce47e91fa8366ab6e4e907d9a35d',
}


def run(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


state = run(['systemctl', '--user', 'is-active', SERVICE]).stdout.strip()
if state == 'active':
    raise RuntimeError('stock systemd service must be inactive before S3 isolated benchmark')

inspect = run(['docker', 'inspect', '-f', '{{.Name}} {{.State.Status}}', NAME], timeout=10)
if inspect.returncode == 0:
    if not inspect.stdout.strip().startswith(f'/{NAME} '):
        raise RuntimeError(f'refusing unexpected container identity: {inspect.stdout.strip()!r}')
    rm = run(['docker', 'rm', '-f', NAME], timeout=30)
    if rm.returncode != 0:
        raise RuntimeError(f'failed to remove prior S3 candidate: {rm.stderr[-1000:]}')

sock = socket.socket()
try:
    sock.bind(('127.0.0.1', PORT))
except OSError as exc:
    raise RuntimeError(f'port {PORT} occupied after exact-container cleanup: {exc}') from exc
finally:
    sock.close()

mounts: list[str] = []
for rel, expected in FILES.items():
    path = OVERLAY / rel
    if not path.exists() or sha256(path) != expected:
        raise RuntimeError(f'overlay hash mismatch: {rel}')
    mounts += ['-v', f'{path}:/opt/python/lib/python3.14/site-packages/{rel}:ro']
if not CONFIG.exists() or sha256(CONFIG) != CONFIG_SHA:
    raise RuntimeError('S3 config missing or hash mismatch')

last_vram = None
clean = False
t0 = time.monotonic()
while time.monotonic() - t0 < 120:
    proc = run(['rocm-smi', '--showmeminfo', 'vram', '--json'])
    try:
        last_vram = int(json.loads(proc.stdout)['card0']['VRAM Total Used Memory (B)'])
        if last_vram < 1_000_000_000:
            clean = True
            break
    except Exception:
        pass
    time.sleep(1)
if not clean:
    raise RuntimeError(f'VRAM not clean: {last_vram}')

model_store = pathlib.Path('/home/rlopez/inneros/inneros_core/var/local_models')
config_name = 'E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json'
config_dst = f'/opt/python/lib/python3.14/site-packages/vllm/model_executor/layers/fused_moe/configs/{config_name}'
cmd = [
    'docker', 'run', '-d', '--name', NAME,
    '--network', 'host', '--ipc', 'host',
    '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'video',
    '--security-opt', 'label=disable', '-e', 'GPU_MAX_HW_QUEUES=1',
    '-v', f'{model_store}:/models:ro', '-v', f'{CONFIG}:{config_dst}:ro',
] + mounts + [
    IMAGE, 'python3', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', MODEL_PATH, '--served-model-name', MODEL,
    '--host', '127.0.0.1', '--port', str(PORT),
    '--max-model-len', '8192', '--gpu-memory-utilization', '0.82',
    '--dtype', 'float16', '--trust-remote-code',
    '--jit-monitor-mode', 'warn', '--jit-monitor-verbose',
]
launch = run(cmd, timeout=60)
out = {
    'schema': 'hyperloom.r9700.phase4.s3_verbose_launch.v1',
    'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'name': NAME, 'port': PORT, 'launch_rc': launch.returncode,
    'container_id': launch.stdout.strip(), 'stderr': launch.stderr[-2000:],
    'stock_service_state_before': state, 'vram_used_before': last_vram,
    'image': IMAGE, 'model': MODEL, 'GPU_MAX_HW_QUEUES': '1',
    'runtime_patch_sha256': PATCH_SHA, 'hybrid_config_sha256': CONFIG_SHA,
    'overlay_hashes': FILES, 'jit_monitor_verbose': True,
    'ok': launch.returncode == 0,
}
ev = ROOT / 'docs' / 'evidence'; ev.mkdir(parents=True, exist_ok=True)
path = ev / f"r9700_phase4_s3_verbose_launch_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(path); print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out['ok'] else 1)
