#!/usr/bin/env python3
"""Launch a clean full-model HyperLoom v7 candidate on the physical R9700.

Precondition: the stock systemd unit must already be stopped through authorized
host ops. This script never mutates that unit and refuses to launch if the stock
container is running. The candidate starts in a fresh process with the v7 patch
installed before vLLM model loading via a read-only .pth bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = "inneros-vllm-canary-rocm10.service"
STOCK = "inneros-vllm-canary-rocm10"
CANDIDATE = "hyperloom-r9700-v7-candidate"
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_PATH = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_MOUNT = "/home/rlopez/inneros/inneros_core/var/local_models:/models"
PATCH = ROOT / "scripts" / "r9700_wna16_hybrid_patch_v7_clean.py"
BOOTSTRAP = ROOT / "scripts" / "r9700_v7_candidate_bootstrap.pth"
EXPECTED_PATCH_SHA256 = "eb6ae784d4cf9c7c9a956a09405b5d9ff479b88cd2cc575e294f4b85a9ff4269"


def run(argv: list[str], timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def service_state() -> str:
    p = run(["systemctl", "--user", "is-active", SERVICE], 20)
    return p.stdout.strip() or "unknown"


def container_running(name: str) -> bool:
    p = run(["docker", "inspect", "-f", "{{.State.Running}}", name], 20)
    return p.returncode == 0 and p.stdout.strip() == "true"


def used_vram() -> int:
    p = run(["rocm-smi", "--showmeminfo", "vram", "--json"], 20)
    try:
        return int(json.loads(p.stdout)["card0"]["VRAM Total Used Memory (B)"])
    except Exception:
        return -1


def wait_vram_free(timeout: float = 120.0) -> dict[str, object]:
    started = time.monotonic()
    samples: list[int] = []
    while time.monotonic() - started < timeout:
        value = used_vram()
        samples.append(value)
        if 0 <= value < 5 * 1024**3:
            return {"ok": True, "wait_sec": time.monotonic() - started, "last": value}
        time.sleep(2)
    return {"ok": False, "wait_sec": time.monotonic() - started, "samples_tail": samples[-10:]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity() -> dict[str, object]:
    p = run(["docker", "inspect", CANDIDATE], 30)
    if p.returncode:
        return {"ok": False, "stderr": p.stderr[-3000:]}
    row = json.loads(p.stdout)[0]
    env = row.get("Config", {}).get("Env", []) or []
    relevant = sorted(
        item
        for item in env
        if item.startswith("GPU_MAX_HW_QUEUES=")
        or item.startswith("HYPERLOOM_R9700_")
        or item.startswith("VLLM_ROCM_USE_AITER")
    )
    return {
        "ok": True,
        "image": row.get("Config", {}).get("Image"),
        "started_at": row.get("State", {}).get("StartedAt"),
        "pid": row.get("State", {}).get("Pid"),
        "running": row.get("State", {}).get("Running"),
        "relevant_env": relevant,
        "cmd": row.get("Config", {}).get("Cmd", []),
        "mounts": [
            {"source": m.get("Source"), "destination": m.get("Destination"), "rw": m.get("RW")}
            for m in row.get("Mounts", [])
            if m.get("Destination") in {
                "/tmp/r9700_wna16_hybrid_patch.py",
                "/opt/python/lib/python3.14/site-packages/r9700_hyperloom_candidate.pth",
            }
        ],
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "docs" / "evidence" / f"r9700_v7_full_model_launch_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": "hyperloom.r9700.v7_full_model_launch.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE,
        "model": MODEL,
        "image": IMAGE,
        "expected_patch_sha256": EXPECTED_PATCH_SHA256,
        "service_state_before": service_state(),
        "stock_container_running_before": container_running(STOCK),
        "truth_boundary": "launch identity only; no performance claim without separate measurement and path evidence",
    }

    if not PATCH.exists() or not BOOTSTRAP.exists():
        payload.update({"ok": False, "error": "v7 patch or bootstrap missing"})
    else:
        observed = sha256(PATCH)
        payload["observed_patch_sha256"] = observed
        if observed != EXPECTED_PATCH_SHA256:
            payload.update({"ok": False, "error": "v7 patch SHA mismatch"})
        elif payload["service_state_before"] in {"active", "activating", "reloading"} or payload["stock_container_running_before"]:
            payload.update({"ok": False, "error": "stock systemd/container must be stopped before candidate launch"})
        else:
            run(["docker", "rm", "-f", CANDIDATE], 30)
            free = wait_vram_free()
            payload["vram_free"] = free
            if not free.get("ok"):
                payload.update({"ok": False, "error": "VRAM did not return below clean-launch threshold"})
            else:
                argv = [
                    "docker", "run", "-d", "--name", CANDIDATE,
                    "--network", "host", "--ipc", "host",
                    "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
                    "--security-opt", "label=disable",
                    "-v", MODEL_MOUNT,
                    "-v", f"{PATCH}:/tmp/r9700_wna16_hybrid_patch.py:ro",
                    "-v", f"{BOOTSTRAP}:/opt/python/lib/python3.14/site-packages/r9700_hyperloom_candidate.pth:ro",
                    "-e", "GPU_MAX_HW_QUEUES=1",
                    "-e", "HYPERLOOM_R9700_EVIDENCE_FILE=/tmp/r9700_candidate_paths.jsonl",
                    IMAGE,
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", MODEL_PATH,
                    "--served-model-name", MODEL,
                    "--host", "127.0.0.1", "--port", "8000",
                    "--max-model-len", "8192",
                    "--gpu-memory-utilization", "0.82",
                    "--dtype", "float16",
                    "--trust-remote-code",
                ]
                p = run(argv, 45)
                payload["launch_returncode"] = p.returncode
                payload["container_id"] = p.stdout.strip()
                if p.returncode:
                    payload.update({"ok": False, "error": "docker run failed", "stderr_tail": p.stderr[-5000:]})
                else:
                    payload["identity"] = identity()
                    payload["ok"] = True

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
