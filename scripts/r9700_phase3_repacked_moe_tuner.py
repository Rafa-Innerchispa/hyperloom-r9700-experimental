#!/usr/bin/env python3
"""Bounded R9700 tuner for the Phase-3 repacked INT4 MoE path.

This script is intentionally self-contained and conservative. It mirrors the
runtime layout introduced by vLLM PR #43389 (N-packed int32 + interleave),
then tunes only the token counts that matter to the observed serving regimes.
It never edits the installed vLLM tree.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
MODEL_STORE = pathlib.Path("/home/rlopez/inneros/inneros_core/var/local_models")
OVERLAY = ROOT / "var" / "r9700_phase3_vllm43389_overlay"
OUT = ROOT / "docs" / "evidence" / "r9700_phase3_repacked_moe_tuner.json"
BATCHES = [1, 2, 4, 8, 16, 32, 64]


def mount_args() -> list[str]:
    base = "/opt/python/lib/python3.14/site-packages"
    rels = [
        "vllm/model_executor/layers/fused_moe/config.py",
        "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
        "vllm/model_executor/layers/fused_moe/fused_moe.py",
        "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
        "vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py",
    ]
    out: list[str] = []
    for rel in rels:
        src = OVERLAY / rel
        if not src.exists():
            raise RuntimeError(f"missing overlay file: {src}")
        out += ["-v", f"{src}:{base}/{rel}:ro"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    if not MODEL_STORE.exists():
        raise RuntimeError(f"model store missing: {MODEL_STORE}")
    benchmark_url = (
        "https://raw.githubusercontent.com/vllm-project/vllm/"
        "f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3/benchmarks/kernels/benchmark_moe.py"
    )
    bench = ROOT / "var" / "r9700_phase3_benchmark_moe_43389.py"
    bench.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    req = urllib.request.Request(benchmark_url, headers={"User-Agent": "hyperloom-r9700-phase3"})
    with urllib.request.urlopen(req, timeout=60) as r:
        bench.write_bytes(r.read())

    save_dir = ROOT / "var" / "r9700_phase3_tuned_configs"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video", "--ipc=host",
        "-e", "GPU_MAX_HW_QUEUES=1",
        "-v", "/home/rlopez/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{MODEL_STORE}:/models:ro",
        "-v", f"{bench}:/tmp/benchmark_moe.py:ro",
        "-v", f"{save_dir}:/tuned",
    ] + mount_args() + [
        IMAGE,
        "python3", "/tmp/benchmark_moe.py",
        "--model", MODEL,
        "--tp-size", "1",
        "--dtype", "int4_w4a16",
        "--batch-size", *[str(x) for x in BATCHES],
        "--tune",
        "--trust-remote-code",
        "--save-dir", "/tuned",
    ]
    record = {
        "schema": "hyperloom.r9700.phase3.repacked_moe_tuner.v2",
        "candidate": "vllm-pr-43389-repacked-int4",
        "upstream_commit": "f5d8dc25b7fc89c4ced724ba9f37a3f4555191e3",
        "batch_sizes": BATCHES,
        "model_store": str(MODEL_STORE),
        "command": cmd,
        "dry_run": ns.dry_run,
    }
    if ns.dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    t0 = time.time()
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=7200)
    record.update({
        "returncode": p.returncode,
        "elapsed_sec": time.time() - t0,
        "stdout_tail": p.stdout[-20000:],
        "stderr_tail": p.stderr[-20000:],
    })
    configs = []
    for fp in sorted(save_dir.glob("*.json")):
        try:
            configs.append({"name": fp.name, "json": json.loads(fp.read_text())})
        except Exception as exc:
            configs.append({"name": fp.name, "error": repr(exc)})
    record["configs"] = configs
    record["pass"] = p.returncode == 0 and bool(configs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"returncode": p.returncode, "elapsed_sec": record["elapsed_sec"], "configs": [x.get("name") for x in configs], "pass": record["pass"]}, indent=2))
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
