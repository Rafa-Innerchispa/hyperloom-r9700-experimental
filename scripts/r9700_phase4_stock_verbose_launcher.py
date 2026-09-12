from __future__ import annotations

import datetime as dt
import json
import pathlib
import socket
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL_PATH = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
PORT = 18016
NAME = "hyperloom-r9700-p4-stock-verbose-p18016"
SERVICE = "inneros-vllm-canary-rocm10.service"


def run(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)


# Safety order matters: only remove our exact disposable container, then prove
# that no unrelated process still owns the benchmark port.
svc = run(["systemctl", "--user", "is-active", SERVICE])
service_state = svc.stdout.strip()
if service_state == "active":
    raise RuntimeError("stock systemd service must be inactive before isolated Phase4 benchmark")

inspect = run(["docker", "inspect", "-f", "{{.Name}} {{.State.Status}}", NAME], timeout=10)
removed_previous = False
if inspect.returncode == 0:
    previous = inspect.stdout.strip()
    if not previous.startswith(f"/{NAME} "):
        raise RuntimeError(f"refusing to remove unexpected container identity: {previous!r}")
    rm = run(["docker", "rm", "-f", NAME], timeout=30)
    if rm.returncode != 0:
        raise RuntimeError(f"failed to remove previous disposable container: {rm.stderr[-1000:]}")
    removed_previous = True

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", PORT))
except OSError as exc:
    raise RuntimeError(
        f"benchmark port {PORT} remains occupied after exact-container cleanup; refusing to disturb unrelated owner: {exc}"
    ) from exc
finally:
    sock.close()

last_vram = None
clean = False
t0 = time.monotonic()
while time.monotonic() - t0 < 120:
    proc = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    try:
        obj = json.loads(proc.stdout)
        last_vram = int(obj["card0"]["VRAM Total Used Memory (B)"])
        if last_vram < 1_000_000_000:
            clean = True
            break
    except Exception:
        pass
    time.sleep(1)
if not clean:
    raise RuntimeError(f"VRAM did not return to clean state: {last_vram}")

model_store = pathlib.Path("/home/rlopez/inneros/inneros_core/var/local_models")
cmd = [
    "docker", "run", "-d",
    "--name", NAME,
    "--network", "host",
    "--ipc", "host",
    "--device", "/dev/kfd",
    "--device", "/dev/dri",
    "--group-add", "video",
    "--security-opt", "label=disable",
    "-e", "GPU_MAX_HW_QUEUES=1",
    "-v", f"{model_store}:/models",
    IMAGE,
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--served-model-name", MODEL,
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.82",
    "--dtype", "float16",
    "--trust-remote-code",
    "--jit-monitor-mode", "warn",
    "--jit-monitor-verbose",
]
launch = run(cmd, timeout=60)
out = {
    "schema": "hyperloom.r9700.phase4.stock_verbose_launch.v2",
    "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "name": NAME,
    "port": PORT,
    "launch_rc": launch.returncode,
    "container_id": launch.stdout.strip(),
    "stderr": launch.stderr[-2000:],
    "stock_service_state_before": service_state,
    "removed_previous_exact_container": removed_previous,
    "vram_used_before": last_vram,
    "image": IMAGE,
    "model": MODEL,
    "GPU_MAX_HW_QUEUES": "1",
    "jit_monitor_mode": "warn",
    "jit_monitor_verbose": True,
    "ok": launch.returncode == 0,
}
ev = ROOT / "docs" / "evidence"
ev.mkdir(parents=True, exist_ok=True)
path = ev / f"r9700_phase4_stock_verbose_launch_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(path)
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out["ok"] else 1)
