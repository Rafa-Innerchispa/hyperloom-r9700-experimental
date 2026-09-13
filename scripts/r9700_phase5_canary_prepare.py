#!/usr/bin/env python3
"""Prepare the R9700 S3 canary entirely from versioned repo sources.

This deliberately does not mutate the operational vLLM service. It rebuilds the
exact vLLM #43389 overlay against the pinned ROCm10 image, verifies every
patched-file hash plus the S3 config hash, and emits a deployment manifest for
the subsequent isolated canary launcher.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "r9700_phase3_build_vllm43389_overlay.py"
OVERLAY = ROOT / "var" / "r9700_phase3_vllm43389_overlay"
CONFIG = ROOT / "docs" / "evidence" / "r9700_phase3_tuned_moe_config_s3_20260912.json"
OUT = ROOT / "var" / "r9700_phase5_canary_manifest.json"
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
PATCH_SHA = "3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d"
CONFIG_SHA = "8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6"
FILES = {
    "vllm/model_executor/layers/fused_moe/config.py": "18af9f7414b4a9e7620fd998f26b58b3294e58a5951af99a3923135509edc8a6",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "beec848e3a6b76c362ec81ecf35e8a515f14fa5deb959aff7f166b6e9e499201",
    "vllm/model_executor/layers/fused_moe/fused_moe.py": "8de93930b8f7071741404fb27190273cd798d447fa32e274eda36e9499e0eb71",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": "6fd057e0f0fff1bdfad3a8c70f34f7fe352ab4c2f920ba13466ae2eb79096a24",
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py": "0f526fa4910b98fa8e64dea6b9c54b43d310ce47e91fa8366ab6e4e907d9a35d",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str, code: int = 2) -> int:
    print(json.dumps({"schema": "hyperloom.r9700.phase5.prepare.v1", "pass": False, "error": message}, indent=2))
    return code


def main() -> int:
    if not BUILDER.exists():
        return fail(f"missing builder: {BUILDER}")
    if not CONFIG.exists():
        return fail(f"missing versioned S3 config: {CONFIG}")
    if sha256(CONFIG) != CONFIG_SHA:
        return fail(f"S3 config SHA mismatch: {sha256(CONFIG)}")

    proc = subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, text=True, capture_output=True, timeout=300)
    if proc.returncode != 0:
        return fail("overlay builder failed: " + (proc.stderr or proc.stdout)[-4000:], 3)

    observed: dict[str, str] = {}
    for rel, expected in FILES.items():
        path = OVERLAY / rel
        if not path.exists():
            return fail(f"overlay file missing: {rel}", 4)
        digest = sha256(path)
        observed[rel] = digest
        if digest != expected:
            return fail(f"overlay SHA mismatch for {rel}: {digest}", 5)

    patch_file = OVERLAY / "vllm43389_runtime.patch"
    if not patch_file.exists() or sha256(patch_file) != PATCH_SHA:
        return fail("runtime patch missing or SHA mismatch", 6)

    manifest = {
        "schema": "hyperloom.r9700.phase5.canary_manifest.v1",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass": True,
        "image": IMAGE,
        "model": MODEL,
        "runtime_patch_sha256": PATCH_SHA,
        "s3_config_path": str(CONFIG.relative_to(ROOT)),
        "s3_config_sha256": CONFIG_SHA,
        "overlay_path": str(OVERLAY.relative_to(ROOT)),
        "overlay_hashes": observed,
        "source": "reconstructed from versioned repo builder + pinned upstream commit; no historical worktree dependency",
        "next_gate": "isolated canary launch -> readiness warmup -> C1/C4/fresh-prefix 6K -> rollback",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
