#!/usr/bin/env python3
"""Collect a sanitized R9700/vLLM launch manifest without restarting services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
SCHEMA = "hyperloom.r9700.vllm_rocm10.launch_manifest.v1"

_SECRET_RE = re.compile(
    r"(?i)(api[-_]?key|token|secret|password|authorization)(=|\s+)[^\s]+|"
    r"(crsr_[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+)"
)


def redact(text: str) -> str:
    """Remove obvious credentials from process or command text."""
    return _SECRET_RE.sub(lambda match: match.group(1) + match.group(2) + "REDACTED" if match.group(2) else "REDACTED", text)


def stable_json_sha256(payload: dict[str, Any]) -> str:
    """Hash a JSON payload after removing its own hash field."""
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("manifest_sha256", None)
    data = json.dumps(clone, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _run(argv: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": redact(completed.stdout)[:20000],
            "stderr": redact(completed.stderr)[:8000],
        }
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "returncode": None, "error": type(exc).__name__}


def _request_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer local"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_model(base_url: str, request_json: Callable[[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    request = request_json or (lambda url: _request_json(url, timeout=10.0))
    payload = request(base_url.rstrip("/") + "/models")
    models = payload.get("data") if isinstance(payload, dict) else None
    selected = models[0] if isinstance(models, list) and models and isinstance(models[0], dict) else {}
    return {
        "raw": payload,
        "selected_model": selected.get("id"),
        "root": selected.get("root"),
        "max_model_len": selected.get("max_model_len"),
    }


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def collect_docker_vllm(run: Callable[[list[str]], dict[str, Any]] = _run) -> list[dict[str, Any]]:
    result = run(["docker", "ps", "--format", "{{json .}}"])
    rows = parse_json_lines(str(result.get("stdout") or ""))
    selected: list[dict[str, Any]] = []
    for row in rows:
        blob = " ".join(str(row.get(key, "")) for key in ("Image", "Names", "Command"))
        if re.search(r"\b(vllm|rocm)\b", blob, flags=re.IGNORECASE):
            selected.append({key: redact(str(row.get(key, ""))) for key in ("ID", "Image", "Names", "Command", "Ports")})
    return selected


def collect_vllm_processes(run: Callable[[list[str]], dict[str, Any]] = _run) -> list[dict[str, Any]]:
    result = run(["ps", "-eo", "pid,ppid,cmd", "--sort=-%mem"])
    rows: list[dict[str, Any]] = []
    for line in str(result.get("stdout") or "").splitlines():
        if "vllm.entrypoints.openai.api_server" not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        rows.append({"pid": parts[0], "ppid": parts[1], "cmd": redact(parts[2])})
    return rows


def collect_rocm_smi(run: Callable[[list[str]], dict[str, Any]] = _run) -> dict[str, Any]:
    result = run(["rocm-smi", "--showproductname", "--showdriverversion", "--showtemp", "--showuse", "--showmemuse", "--showpower", "--json"])
    if result.get("returncode") != 0:
        return {"available": False, "probe": result}
    try:
        payload = json.loads(str(result.get("stdout") or "{}"))
    except json.JSONDecodeError:
        return {"available": False, "probe": result}
    return {"available": True, "payload": payload}


def collect_container_versions(
    container_name: str,
    run: Callable[[list[str]], dict[str, Any]] = _run,
) -> dict[str, Any]:
    code = (
        "import json, platform\n"
        "payload={'python': platform.python_version()}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_version']=torch.__version__\n"
        " payload['torch_hip']=getattr(torch.version,'hip',None)\n"
        "except Exception as exc:\n"
        " payload['torch_error']=type(exc).__name__\n"
        "try:\n"
        " import vllm\n"
        " payload['vllm_version']=getattr(vllm,'__version__',None)\n"
        "except Exception as exc:\n"
        " payload['vllm_error']=type(exc).__name__\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = run(["docker", "exec", container_name, "python3", "-c", code])
    if result.get("returncode") != 0:
        return {"available": False, "probe": result}
    try:
        return {"available": True, "payload": json.loads(str(result.get("stdout") or "{}"))}
    except json.JSONDecodeError:
        return {"available": False, "probe": result}


def build_manifest(
    *,
    base_url: str = DEFAULT_BASE_URL,
    request_json: Callable[[str], dict[str, Any]] | None = None,
    run: Callable[[list[str]], dict[str, Any]] = _run,
) -> dict[str, Any]:
    model = discover_model(base_url, request_json=request_json)
    containers = collect_docker_vllm(run=run)
    container_name = next((row["Names"] for row in containers if "vllm" in row.get("Names", "").lower()), "")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "truth_boundary": {
            "support_status": "experimental InnerChispa compatibility path; not official AMD Hyperloom R9700 support",
            "hardware_label": "AMD Radeon AI PRO R9700 workstation GPU, gfx1201",
            "claim_status": "measured_runtime_manifest",
            "concurrency_result_class": "serving/concurrency scaling, not a GEAK/Arbor kernel optimization claim",
        },
        "endpoint": {"base_url": base_url.rstrip("/"), "models_path": "/models"},
        "model": model,
        "docker": {"vllm_containers": containers, "selected_container": container_name},
        "runtime_versions": collect_container_versions(container_name, run=run) if container_name else {"available": False},
        "gpu": collect_rocm_smi(run=run),
        "launch_processes": collect_vllm_processes(run=run),
        "awq_observability": {
            "model_name_contains_awq": "AWQ" in str(model.get("selected_model") or "").upper(),
            "backend_observed": "not directly exposed by OpenAI /models; inspect vLLM logs or profiler before claiming a specific AWQ kernel",
            "vllm_use_triton_awq_required": "unknown_from_manifest",
        },
        "safety": {
            "service_restarted": False,
            "secrets_redacted": True,
            "network_scope": "local endpoint only",
        },
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    return manifest


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "docs" / "evidence" / f"r9700_vllm_rocm10_launch_manifest_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    manifest = build_manifest(base_url=args.base_url)
    output = Path(args.output) if args.output else default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "schema": manifest["schema"],
        "model": manifest["model"].get("selected_model"),
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
