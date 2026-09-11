#!/usr/bin/env python3
"""Run one process-isolated R9700 serving factorial cell and restore stock.

This helper exists to separate the two knobs that were accidentally combined in
the 2026-09-10 Unified Attention campaign:

* AITER RDNA4 Unified Attention selection
* GPU_MAX_HW_QUEUES=1

Supported cells intentionally cover only the two missing cells. Existing evidence
already covers stock/default and Unified-Attention/queue1.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STABLE = "inneros-vllm-canary-rocm10"
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


def wait_health(timeout: float = 420.0) -> dict[str, object]:
    started = time.monotonic()
    last_error = ""
    attempts = 0
    while time.monotonic() - started < timeout:
        attempts += 1
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                ids = [row.get("id") for row in payload.get("data", []) if isinstance(row, dict)]
                if response.status == 200 and MODEL in ids:
                    return {
                        "ok": True,
                        "ready_sec": time.monotonic() - started,
                        "attempts": attempts,
                        "models": ids,
                    }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    return {
        "ok": False,
        "ready_sec": time.monotonic() - started,
        "attempts": attempts,
        "last_error": last_error,
    }


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
    proc = run(["docker", "inspect", TEST], timeout=30)
    if proc.returncode:
        return {"ok": False, "stderr": proc.stderr[-3000:]}
    row = json.loads(proc.stdout)[0]
    env = row.get("Config", {}).get("Env", []) or []
    relevant = sorted(
        item
        for item in env
        if item.startswith("VLLM_ROCM_USE_AITER") or item.startswith("GPU_MAX_HW_QUEUES=")
    )
    return {
        "ok": True,
        "image": row.get("Config", {}).get("Image"),
        "started_at": row.get("State", {}).get("StartedAt"),
        "pid": row.get("State", {}).get("Pid"),
        "relevant_env": relevant,
        "cmd": row.get("Config", {}).get("Cmd", []),
    }


def restore_stock() -> dict[str, object]:
    removed = run(["docker", "rm", "-f", TEST], timeout=40)
    started = run(["docker", "start", STABLE], timeout=40)
    health = wait_health()
    return {
        "remove_candidate_rc": removed.returncode,
        "start_stock_rc": started.returncode,
        "health": health,
    }


def run_measurement(cell: str, stamp: str) -> dict[str, object]:
    proc = run(["python3", str(ROOT / "scripts" / "r9700_active_measure.py")], timeout=180)
    result: dict[str, object] = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-16000:],
        "stderr_tail": proc.stderr[-6000:],
    }
    if proc.returncode:
        return result
    first_line = proc.stdout.splitlines()[0].strip() if proc.stdout.splitlines() else ""
    source = Path(first_line)
    if source.is_file():
        dest = ROOT / "docs" / "evidence" / f"r9700_factorial_{cell}_{stamp}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        result["source"] = str(source)
        result["evidence"] = str(dest.relative_to(ROOT))
    return result


def launch(cell: str) -> int:
    if cell not in {"stock_queue1", "unified_defaultq"}:
        raise ValueError(f"unsupported factorial cell: {cell}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "docs" / "evidence" / f"r9700_factorial_{cell}_run_{stamp}.json"
    payload: dict[str, object] = {
        "schema": "hyperloom.r9700.serving_factorial_cell.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": cell,
        "model": MODEL,
        "image": IMAGE,
        "truth_boundary": "process-isolated causal gate; not a full-model HyperLoom promotion by itself",
    }

    run(["docker", "rm", "-f", TEST], timeout=30)
    stable_running = (
        run(["docker", "inspect", "-f", "{{.State.Running}}", STABLE], timeout=20).stdout.strip()
        == "true"
    )
    payload["stock_was_running"] = stable_running
    if stable_running:
        stop = run(["docker", "stop", "-t", "20", STABLE], timeout=50)
        payload["stock_stop_rc"] = stop.returncode

    free = wait_vram_free()
    payload["vram_free"] = free
    if not free.get("ok"):
        payload["restore"] = restore_stock()
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

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
        payload["launch_stderr"] = launched.stderr[-5000:]
        payload["restore"] = restore_stock()
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    health = wait_health()
    payload["candidate_health"] = health
    payload["candidate_identity"] = inspect_test()
    logs = run(["docker", "logs", TEST], timeout=30)
    payload["candidate_log_tail"] = (logs.stdout + "\n" + logs.stderr)[-20000:]

    if health.get("ok"):
        payload["measurement"] = run_measurement(cell, stamp)

    payload["restore"] = restore_stock()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({
        "cell": cell,
        "candidate_health": health,
        "measurement": payload.get("measurement"),
        "restore": payload["restore"],
    }, indent=2, sort_keys=True))

    measurement_ok = isinstance(payload.get("measurement"), dict) and payload["measurement"].get("returncode") == 0
    restore_ok = bool((payload["restore"] or {}).get("health", {}).get("ok"))
    return 0 if health.get("ok") and measurement_ok and restore_ok else 4
