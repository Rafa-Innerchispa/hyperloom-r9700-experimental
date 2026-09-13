#!/usr/bin/env python3
"""Fail-closed preflight for starting the R9700 S3 canary service."""
from __future__ import annotations

import hashlib
import json
import pathlib
import socket
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "var" / "r9700_phase5_canary_bundle"
MANIFEST = BUNDLE / "manifest.json"
PORT = 18018
STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"
CANARY_CONTAINER = "inneros-vllm-hyperloom-s3-canary"
S3_CONFIG_SHA256 = "8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6"
CONFIG_NAME = "E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json"
FILES = {
    "vllm/model_executor/layers/fused_moe/config.py": "18af9f7414b4a9e7620fd998f26b58b3294e58a5951af99a3923135509edc8a6",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "beec848e3a6b76c362ec81ecf35e8a515f14fa5deb959aff7f166b6e9e499201",
    "vllm/model_executor/layers/fused_moe/fused_moe.py": "8de93930b8f7071741404fb27190273cd798d447fa32e274eda36e9499e0eb71",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": "6fd057e0f0fff1bdfad3a8c70f34f7fe352ab4c2f920ba13466ae2eb79096a24",
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py": "0f526fa4910b98fa8e64dea6b9c54b43d310ce47e91fa8366ab6e4e907d9a35d",
}


def run(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checks: dict[str, object] = {}

    svc = run(["systemctl", "--user", "is-active", STOCK_SERVICE])
    stock_state = svc.stdout.strip()
    checks["stock_service_state"] = stock_state
    checks["stock_service_inactive"] = stock_state != "active"

    existing = run(["docker", "inspect", "-f", "{{.State.Status}}", CANARY_CONTAINER])
    checks["existing_canary_container"] = existing.stdout.strip() if existing.returncode == 0 else None
    checks["no_existing_canary_container"] = existing.returncode != 0

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", PORT))
        checks["port_free"] = True
    except OSError as exc:
        checks["port_free"] = False
        checks["port_error"] = str(exc)
    finally:
        sock.close()

    vram = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    try:
        used = int(json.loads(vram.stdout)["card0"]["VRAM Total Used Memory (B)"])
    except Exception:
        used = -1
    checks["vram_used_bytes"] = used
    checks["vram_clean_under_1gb"] = 0 <= used < 1_000_000_000

    checks["manifest_exists"] = MANIFEST.exists()
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            checks["manifest_pass"] = bool(manifest.get("pass"))
        except Exception as exc:
            checks["manifest_pass"] = False
            checks["manifest_error"] = f"{type(exc).__name__}: {exc}"
    else:
        checks["manifest_pass"] = False

    file_checks: dict[str, bool] = {}
    for rel, expected in FILES.items():
        path = BUNDLE / "overlay" / rel
        file_checks[rel] = path.exists() and sha(path) == expected
    checks["overlay_hashes"] = file_checks
    checks["overlay_all_match"] = all(file_checks.values())

    cfg = BUNDLE / "config" / CONFIG_NAME
    checks["config_hash_match"] = cfg.exists() and sha(cfg) == S3_CONFIG_SHA256

    required = [
        checks["stock_service_inactive"],
        checks["no_existing_canary_container"],
        checks["port_free"],
        checks["vram_clean_under_1gb"],
        checks["manifest_exists"],
        checks["manifest_pass"],
        checks["overlay_all_match"],
        checks["config_hash_match"],
    ]
    out = {"schema": "hyperloom.r9700.phase5.canary_preflight.v1", "port": PORT, "checks": checks, "pass": all(required)}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
