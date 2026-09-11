#!/usr/bin/env python3
"""Build and verify an isolated vLLM #43389 overlay for the ROCm10 R9700 image."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
RUNTIME_VLLM_COMMIT = "f46a9dfe2c5f57bebbd29556cbbb25eabd874226"
UPSTREAM_COMMIT = "f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3"
PATCH_URL = f"https://github.com/vllm-project/vllm/commit/{UPSTREAM_COMMIT}.patch"
EXPECTED_RUNTIME_PATCH_SHA256 = "3935c4dc60cfa6448e18aea1ef9fa6b00a0b46de294c59b7518a012138f5630d"
FILES = (
    "vllm/model_executor/layers/fused_moe/config.py",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
    "vllm/model_executor/layers/fused_moe/fused_moe.py",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
    "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py",
)
OUT_ROOT = ROOT / "var" / "r9700_phase3_vllm43389_overlay"
PATCH_FILE = OUT_ROOT / "vllm43389_runtime.patch"
MANIFEST_FILE = ROOT / "docs/evidence/r9700_phase3_vllm43389_overlay_manifest.json"


def run(argv: list[str], *, cwd: pathlib.Path | None = None, timeout: float = 120.0):
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_cat(rel: str) -> str:
    proc = run([
        "docker", "run", "--rm", "--entrypoint", "cat", IMAGE,
        f"/opt/python/lib/python3.14/site-packages/{rel}",
    ], timeout=90)
    if proc.returncode:
        raise RuntimeError(f"cannot copy {rel}: {proc.stderr[-3000:]}")
    return proc.stdout


def filter_patch(text: str) -> str:
    allowed = set(FILES)
    selected: list[str] = []
    current: list[str] = []
    keep = False
    for line in text.splitlines(True):
        if line.startswith("diff --git "):
            if current and keep:
                selected.extend(current)
            current = [line]
            parts = line.strip().split()
            target = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else ""
            keep = target in allowed
        elif current:
            current.append(line)
    if current and keep:
        selected.extend(current)
    return "".join(selected)


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    base_hashes: dict[str, str] = {}
    for rel in FILES:
        path = OUT_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("moe_wna16_utils.py"):
            continue
        path.write_text(image_cat(rel), encoding="utf-8")
        base_hashes[rel] = sha(path)

    # Critical: isolate git discovery from the parent HyperLoom worktree.
    init = run(["git", "init", "-q"], cwd=OUT_ROOT, timeout=30)
    if init.returncode:
        raise RuntimeError(f"git init failed: {init.stderr[-2000:]}")

    request = urllib.request.Request(PATCH_URL, headers={"User-Agent": "hyperloom-r9700-phase3"})
    with urllib.request.urlopen(request, timeout=60) as response:
        full_patch = response.read().decode("utf-8")
    runtime_patch = filter_patch(full_patch)
    runtime_sha = hashlib.sha256(runtime_patch.encode()).hexdigest()
    if runtime_sha != EXPECTED_RUNTIME_PATCH_SHA256:
        raise RuntimeError(f"runtime patch SHA mismatch: {runtime_sha}")
    PATCH_FILE.write_text(runtime_patch, encoding="utf-8")

    check = run(["git", "apply", "--check", "--verbose", str(PATCH_FILE)], cwd=OUT_ROOT)
    if check.returncode:
        raise RuntimeError(f"patch check failed: {check.stderr[-8000:]}")
    apply = run(["git", "apply", str(PATCH_FILE)], cwd=OUT_ROOT)
    if apply.returncode:
        raise RuntimeError(f"patch apply failed: {apply.stderr[-8000:]}")

    patched: dict[str, dict[str, object]] = {}
    for rel in FILES:
        path = OUT_ROOT / rel
        if not path.exists():
            raise RuntimeError(f"patched file missing: {rel}")
        compile_probe = run(["python3", "-m", "py_compile", str(path)], timeout=30)
        if compile_probe.returncode:
            raise RuntimeError(f"py_compile failed {rel}: {compile_probe.stderr[-2000:]}")
        patched[rel] = {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "py_compile_rc": compile_probe.returncode,
        }

    manifest = {
        "schema": "hyperloom.r9700.phase3.vllm43389_overlay.v3",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "image": IMAGE,
        "runtime_vllm_commit": RUNTIME_VLLM_COMMIT,
        "upstream_pr": 43389,
        "upstream_commit": UPSTREAM_COMMIT,
        "runtime_patch_sha256": runtime_sha,
        "runtime_patch_bytes": len(runtime_patch.encode()),
        "base_hashes": base_hashes,
        "patched_files": patched,
        "pass": True,
        "truth_boundary": "exact source overlay and syntax verified; no numerical/performance claim yet",
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
