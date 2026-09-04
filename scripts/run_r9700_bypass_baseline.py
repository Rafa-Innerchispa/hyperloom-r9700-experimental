#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

HYPERLOOM_REPO = "https://github.com/AMD-AGI/Hyperloom.git"
HYPERLOOM_COMMIT = "9ae79d6a8c9fec7ed041735e70fb19ef39850813"
INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
BASE_URL = "http://127.0.0.1:8000"


def cmd(argv, *, cwd=None, env=None, timeout=300, check=True):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed {argv}:\nstdout={p.stdout[-8000:]}\nstderr={p.stderr[-8000:]}")
    return p


def patch_hyperloom(root: Path) -> None:
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
    gpu_types.write_text(text)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="hl-r9700-bypass-"))
    hl = root / "Hyperloom"
    ix = root / "InferenceX"
    out = root / "out"
    try:
        health = cmd(["curl", "-fsS", f"{BASE_URL}/health"], check=False, timeout=10)
        models = cmd(["curl", "-fsS", f"{BASE_URL}/v1/models"], check=False, timeout=10)
        if health.returncode != 0 or models.returncode != 0:
            print(json.dumps({"ok": False, "stage": "vllm_health", "health_rc": health.returncode, "models_rc": models.returncode}))
            return 2

        cmd(["git", "clone", "--filter=blob:none", HYPERLOOM_REPO, str(hl)], timeout=180)
        cmd(["git", "checkout", HYPERLOOM_COMMIT], cwd=hl)
        patch_hyperloom(hl)
        cmd(["git", "clone", "--depth", "1", INFERENCEX_REPO, str(ix)], timeout=180)

        bench_py = ix / "utils/bench_serving/benchmark_serving.py"
        probe = cmd([sys.executable, str(bench_py), "--help"], cwd=ix, check=False, timeout=60)
        if probe.returncode != 0:
            print(json.dumps({
                "ok": False,
                "stage": "inferencex_python_dependencies",
                "returncode": probe.returncode,
                "stderr_tail": probe.stderr[-4000:],
                "inferencex": str(ix),
            }, indent=2))
            return 3

        out.mkdir(parents=True, exist_ok=True)
        config = root / "baseline_r9700.yaml"
        config.write_text(textwrap.dedent(f"""
        benchmark:
          framework: vllm
          model: {MODEL}
          inferencex_path: {ix}
          precision: float16
          timeout_seconds: 300
          runner_type: r9700
          envs:
            TP: 1
            CONC: 1
            ISL: 32
            OSL: 32
            RANDOM_RANGE_RATIO: 0.0
            NUM_PROMPTS: 2
            NUM_WARMUPS: 1
            RUN_EVAL: false
            PORT: 8000
        """).strip() + "\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(hl / "src")
        env["HYPERLOOM_BENCHMARK_BACKEND"] = "bypass"
        env["BENCHMARK_BASE_URL"] = BASE_URL
        env["MAGPIE_RUN_PHASE"] = "client"
        env["INFERENCEX_PATH"] = str(ix)
        env["GPU_TYPE"] = "r9700"
        env["TARGET_GPU_TYPE"] = "r9700"

        run = cmd([
            sys.executable,
            "-m",
            "hyperloom.orchestrator.actions.executors.bypass_runner",
            "benchmark",
            "--benchmark-config",
            str(config),
            "--output-dir",
            str(out),
            "--run-mode",
            "local",
        ], cwd=hl, env=env, check=False, timeout=360)

        reports = list(out.rglob("benchmark_report.json"))
        result_files = list(out.rglob("inferencex_result.json")) + list(out.rglob("*.json"))
        report_payload = None
        if reports:
            try:
                report_payload = json.loads(reports[0].read_text())
            except Exception:
                report_payload = {"path": str(reports[0]), "parse": "failed"}

        summary = {
            "ok": run.returncode == 0 and bool(reports),
            "stage": "hyperloom_bypass_baseline",
            "gpu_type": "r9700",
            "gfx": "gfx1201",
            "backend": "bypass",
            "remote_base_url": BASE_URL,
            "model": MODEL,
            "hyperloom_commit": HYPERLOOM_COMMIT,
            "returncode": run.returncode,
            "stdout_tail": run.stdout[-6000:],
            "stderr_tail": run.stderr[-6000:],
            "report_paths": [str(p) for p in reports],
            "json_artifacts": [str(p) for p in result_files[:20]],
            "benchmark_report": report_payload,
        }
        print(json.dumps(summary, indent=2, default=str))
        return 0 if summary["ok"] else 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
