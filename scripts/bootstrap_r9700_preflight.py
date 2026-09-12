#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

UPSTREAM = "https://github.com/AMD-AGI/Hyperloom.git"
UPSTREAM_COMMIT = "9ae79d6a8c9fec7ed041735e70fb19ef39850813"


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout={p.stdout[-4000:]}\nstderr={p.stderr[-4000:]}")
    return p.stdout


def patch_files(root: Path) -> None:
    identity = root / "src/hyperloom/common/gpu_identity.py"
    text = identity.read_text()
    needle = '    "mi355x": ("gfx950", 256),\n'
    if '"r9700": ("gfx1201", 64)' not in text:
        text = text.replace(needle, needle + '    "r9700": ("gfx1201", 64),\n')
    identity.write_text(text)

    gpu_types = root / "src/hyperloom/inference_optimizer/gpu_types.py"
    text = gpu_types.read_text()
    needle = '    "gfx950": "mi355x",\n'
    if '"gfx1201": "r9700"' not in text:
        text = text.replace(needle, needle + '    "gfx1201": "r9700",\n')
    text = text.replace(
        '"""Return mi300x|mi308x|mi325x|mi355x or None if undetectable."""',
        '"""Return a supported AMD GPU type or None if undetectable."""',
    )
    gpu_types.write_text(text)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="hyperloom-r9700-live-"))
    try:
        run(["git", "clone", "--filter=blob:none", UPSTREAM, str(work)])
        run(["git", "checkout", UPSTREAM_COMMIT], cwd=work)
        patch_files(work)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(work / "src")
        env["HYPERLOOM_BENCHMARK_BACKEND"] = "bypass"
        code = r'''
import json, os, subprocess
from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES
from hyperloom.inference_optimizer.gpu_types import _autodetect_gpu_type, _resolve_amd_gpu_type, amd_gpu_dispatch_identity
from hyperloom.orchestrator.actions.executors.benchmark_backend import BypassBackend, resolve_backend_name
p = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10)
detected = _autodetect_gpu_type()
resolved = _resolve_amd_gpu_type("r9700")
identity = amd_gpu_dispatch_identity("r9700")
lifecycle = BypassBackend().lifecycle_eligibility({"framework":"vllm","envs":{"PORT":8888},"profiler":{"torch_profiler":{"enabled":False}}})
print(json.dumps({
  "rocm_smi_product": p.stdout.strip(),
  "autodetected_gpu_type": detected,
  "resolved_gpu_type": resolved,
  "dispatch_identity": list(identity) if identity else None,
  "registered_identity": list(AMD_GPU_DISPATCH_IDENTITIES.get("r9700", ())),
  "backend": resolve_backend_name(),
  "bypass_lifecycle": lifecycle,
}, indent=2, sort_keys=True))
assert detected == "r9700", detected
assert resolved == "r9700"
assert identity == ("gfx1201", 64)
assert resolve_backend_name() == "bypass"
assert lifecycle and lifecycle.get("eligible") is True
'''
        p = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, text=True, capture_output=True, timeout=120)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr, file=sys.stderr)
            return p.returncode
        print(json.dumps({"ok": True, "upstream_commit": UPSTREAM_COMMIT, "mode": "patched-live-preflight", "gpu_type": "r9700", "gfx": "gfx1201", "backend": "bypass"}, indent=2))
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
