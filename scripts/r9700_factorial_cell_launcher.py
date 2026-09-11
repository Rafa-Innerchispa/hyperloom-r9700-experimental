#!/usr/bin/env python3
"""Launch one clean process-isolated R9700 serving factorial cell.

IMPORTANT: this script deliberately does not stop/start the stock systemd unit.
The caller must use the authorized host-ops plane to stop
`inneros-vllm-canary-rocm10.service` before launch and restore it after the
candidate is removed. This prevents systemd auto-restart from contaminating the
GPU while a candidate is loading or being benchmarked.

The script exits immediately after starting and identifying the test container.
Readiness, measurement, removal, and stock restore are separate explicit gates.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STABLE_SERVICE = "inneros-vllm-canary-rocm10.service"
STABLE_CONTAINER = "inneros-vllm-canary-rocm10"
TEST = "hyperloom-r9700-unified-pilot"
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_PATH = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_MOUNT = "/home/rlopez/inneros/inneros_core/var/local_models:/models"
OVERLAY = ROOT / "scripts" / "rocm_v027_rdna4_unified_overlay.py"


def run(argv: list[str], *, timeout: float = 90.0, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if check and proc.returncode:
        raise RuntimeError(
            f"rc={proc.returncode} argv={argv!r} stderr={proc.stderr[-4000:]}"
        )
    return proc


def stock_service_state() -> str:
    proc = run(["systemctl", "--user", "is-active", STABLE_SERVICE], timeout=20)
    return proc.stdout.strip() or "unknown"


def stock_container_running() -> bool:
    proc = run(
        ["docker", "inspect", "-f", "{{.State.Running}}", STABLE_CONTAINER],
        timeout=20,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def used_vram() -> int:
    probe = run(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=20)
    try:
        return int(json.loads(probe.stdout)["card0"]["VRAM Total Used Memory (B)"])
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


def make_overlay() -> None:
    src = run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            IMAGE,
            "/opt/python/lib/python3.14/site-packages/vllm/platforms/rocm.py",
        ],
        timeout=90,
        check=True,
    ).stdout
    old = (
        "    if is_aiter_found_and_supported():\n"
        "        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n"
        "    backends.append(AttentionBackendEnum.TRITON_ATTN)\n"
    )
    new = (
        "    if is_aiter_found_and_supported():\n"
        "        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n"
        "    elif on_gfx12x() and getattr(rocm_aiter_ops, \"_AITER_ENABLED\", False):\n"
        "        backends.insert(0, AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n"
        "    backends.append(AttentionBackendEnum.TRITON_ATTN)\n"
    )
    if old not in src:
        raise RuntimeError("ROCm attention backend block no longer matches controlled image")
    OVERLAY.write_text(src.replace(old, new, 1), encoding="utf-8")


def inspect_test() -> dict[str, object]:
    proc = run(["docker", "inspect", TEST], timeout=30, check=True)
    row = json.loads(proc.stdout)[0]
    env = row.get("Config", {}).get("Env", []) or []
    relevant = sorted(
        item
        for item in env
        if item.startswith("VLLM_ROCM_USE_AITER")
        or item.startswith("GPU_MAX_HW_QUEUES=")
    )
    return {
        "image": row.get("Config", {}).get("Image"),
        "started_at": row.get("State", {}).get("StartedAt"),
        "pid": row.get("State", {}).get("Pid"),
        "running": row.get("State", {}).get("Running"),
        "relevant_env": relevant,
        "cmd": row.get("Config", {}).get("Cmd", []),
    }


def launch(cell: str) -> int:
    if cell not in {"stock_queue1", "unified_defaultq"}:
        raise ValueError(f"unsupported factorial cell: {cell}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "docs" / "evidence" / f"r9700_factorial_{cell}_launch_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": "hyperloom.r9700.serving_factorial_launch.v2",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": cell,
        "model": MODEL,
        "image": IMAGE,
        "stock_service_state_before": stock_service_state(),
        "stock_container_running_before": stock_container_running(),
        "truth_boundary": "launch identity only; benchmark evidence is captured separately",
    }

    if payload["stock_service_state_before"] in {"active", "activating", "reloading"} or payload["stock_container_running_before"]:
        payload["ok"] = False
        payload["error"] = "stock service/container must be stopped via authorized host ops before candidate launch"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    run(["docker", "rm", "-f", TEST], timeout=30)
    free = wait_vram_free()
    payload["vram_free"] = free
    if not free.get("ok"):
        payload["ok"] = False
        payload["error"] = "VRAM did not return below clean-launch threshold"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    docker_args = [
        "docker",
        "run",
        "-d",
        "--name",
        TEST,
        "--network",
        "host",
        "--ipc",
        "host",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--security-opt",
        "label=disable",
        "-v",
        MODEL_MOUNT,
    ]

    if cell == "stock_queue1":
        docker_args += ["-e", "GPU_MAX_HW_QUEUES=1"]
    else:
        make_overlay()
        docker_args += [
            "-v",
            f"{OVERLAY}:/opt/python/lib/python3.14/site-packages/vllm/platforms/rocm.py:ro",
            "-e",
            "VLLM_ROCM_USE_AITER=1",
            "-e",
            "VLLM_ROCM_USE_AITER_MHA=0",
            "-e",
            "VLLM_ROCM_USE_AITER_MOE=0",
            "-e",
            "VLLM_ROCM_USE_AITER_LINEAR=0",
            "-e",
            "VLLM_ROCM_USE_AITER_RMSNORM=0",
            "-e",
            "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1",
        ]

    docker_args += [
        IMAGE,
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.82",
        "--dtype",
        "float16",
        "--trust-remote-code",
    ]
    if cell == "unified_defaultq":
        docker_args += ["--attention-backend", "ROCM_AITER_UNIFIED_ATTN"]

    launched = run(docker_args, timeout=40)
    payload["launch_rc"] = launched.returncode
    payload["container_id"] = launched.stdout.strip()
    if launched.returncode:
        payload["ok"] = False
        payload["launch_stderr"] = launched.stderr[-5000:]
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 4

    payload["candidate_identity"] = inspect_test()
    payload["ok"] = True
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
