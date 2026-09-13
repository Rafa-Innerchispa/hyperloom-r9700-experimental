#!/usr/bin/env python3
"""Build a deterministic deploy bundle for the validated R9700 S3 canary.

This script is safe to run while the stock service is active. It does not touch
serving processes or GPU state. It rebuilds the #43389 source overlay from the
pinned ROCm10 image/upstream patch, verifies every selected file hash, copies
only the verified files plus the canonical S3 config into a fresh Phase5 bundle,
and emits a machine-readable manifest.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "r9700_phase3_build_vllm43389_overlay.py"
SOURCE_OVERLAY = ROOT / "var" / "r9700_phase3_vllm43389_overlay"
BUNDLE = ROOT / "var" / "r9700_phase5_canary_bundle"
BUNDLE_OVERLAY = BUNDLE / "overlay"
CONFIG_SRC = ROOT / "docs" / "evidence" / "r9700_phase3_tuned_moe_config_s3_20260912.json"
CONFIG_NAME = "E=128,N=768,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json"
CONFIG_DST = BUNDLE / "config" / CONFIG_NAME
MANIFEST = BUNDLE / "manifest.json"
EVIDENCE = ROOT / "docs" / "evidence" / "r9700_phase5_canary_bundle_manifest.json"

IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
RUNTIME_PATCH_SHA256 = "3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d"
S3_CONFIG_SHA256 = "8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6"
FILES = {
    "vllm/model_executor/layers/fused_moe/config.py": "18af9f7414b4a9e7620fd998f26b58b3294e58a5951af99a3923135509edc8a6",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "beec848e3a6b76c362ec81ecf35e8a515f14fa5deb959aff7f166b6e9e499201",
    "vllm/model_executor/layers/fused_moe/fused_moe.py": "8de93930b8f7071741404fb27190273cd798d447fa32e274eda36e9499e0eb71",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py": "6fd057e0f0fff1bdfad3a8c70f34f7fe352ab4c2f920ba13466ae2eb79096a24",
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py": "0f526fa4910b98fa8e64dea6b9c54b43d310ce47e91fa8366ab6e4e907d9a35d",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> int:
    print(json.dumps({"schema": "hyperloom.r9700.phase5.canary_bundle.v1", "pass": False, "error": message}, indent=2))
    return 2


def main() -> int:
    if not BUILDER.exists():
        return fail(f"missing builder: {BUILDER}")
    if not CONFIG_SRC.exists():
        return fail(f"missing canonical S3 config: {CONFIG_SRC}")
    if sha256(CONFIG_SRC) != S3_CONFIG_SHA256:
        return fail(f"S3 config hash mismatch: {sha256(CONFIG_SRC)}")

    build = subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, text=True, capture_output=True, timeout=300)
    if build.returncode != 0:
        return fail(f"overlay builder failed rc={build.returncode}: {build.stderr[-5000:]}")

    source_patch = SOURCE_OVERLAY / "vllm43389_runtime.patch"
    if not source_patch.exists() or sha256(source_patch) != RUNTIME_PATCH_SHA256:
        return fail("runtime patch missing or hash mismatch after builder")

    verified: dict[str, dict[str, object]] = {}
    for rel, expected in FILES.items():
        src = SOURCE_OVERLAY / rel
        if not src.exists():
            return fail(f"overlay file missing: {rel}")
        actual = sha256(src)
        if actual != expected:
            return fail(f"overlay hash mismatch {rel}: {actual}")
        verified[rel] = {"sha256": actual, "bytes": src.stat().st_size}

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE_OVERLAY.mkdir(parents=True, exist_ok=True)
    CONFIG_DST.parent.mkdir(parents=True, exist_ok=True)

    for rel in FILES:
        src = SOURCE_OVERLAY / rel
        dst = BUNDLE_OVERLAY / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(dst) != FILES[rel]:
            return fail(f"post-copy hash mismatch: {rel}")
    shutil.copy2(CONFIG_SRC, CONFIG_DST)
    if sha256(CONFIG_DST) != S3_CONFIG_SHA256:
        return fail("post-copy S3 config hash mismatch")

    out = {
        "schema": "hyperloom.r9700.phase5.canary_bundle.v1",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "image": IMAGE,
        "runtime_patch_sha256": RUNTIME_PATCH_SHA256,
        "s3_config_sha256": S3_CONFIG_SHA256,
        "config_name": CONFIG_NAME,
        "bundle_root": str(BUNDLE),
        "files": verified,
        "builder": str(BUILDER.relative_to(ROOT)),
        "builder_rebuilds_from_pinned_source": True,
        "stock_runtime_mutated": False,
        "pass": True,
    }
    MANIFEST.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    EVIDENCE.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
