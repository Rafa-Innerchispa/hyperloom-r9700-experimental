#!/usr/bin/env python3
"""Read-only host preflight for the isolated vLLM 0.29 R9700 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docs" / "evidence" / "vllm_v0_29_0_r9700_awq_triton_candidate.patch"
PATCH_SHA256 = "372d5b73c27536910a615eb0138231514747bde23d5d6a2a5460a2e41fa3c6c9"
DEFAULT_MODEL_DIR = Path(
    "/home/rlopez/inneros/inneros_core/var/local_models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"
)
DEFAULT_PORT = 18029
CANDIDATE_CONTAINER = "hyperloom-r9700-vllm029-gfx1201-candidate"
PRESERVED_SERVICES = (
    "inneros-vllm-canary-rocm10.service",
    "inneros-vllm-hyperloom-s3-production.service",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(argv: list[str], *, timeout: float = 20.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": type(exc).__name__,
        }
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-8000:],
    }


def port_is_free(port: int) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()
    return True, ""


def service_state(
    service: str,
    *,
    run: Callable[..., dict[str, Any]] = run_command,
) -> str:
    probe = run(["systemctl", "--user", "is-active", service])
    state = str(probe.get("stdout") or "").strip()
    if state:
        return state
    if probe.get("returncode") == 3:
        return "inactive"
    return "unknown"


def container_running(
    name: str,
    *,
    run: Callable[..., dict[str, Any]] = run_command,
) -> bool:
    probe = run(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return probe.get("returncode") == 0 and str(probe.get("stdout") or "").strip() == "true"


def collect_preflight(
    *,
    candidate_port: int = DEFAULT_PORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    patch_path: Path = PATCH,
    run: Callable[..., dict[str, Any]] = run_command,
    path_exists: Callable[[Path], bool] | None = None,
    port_probe: Callable[[int], tuple[bool, str]] = port_is_free,
) -> dict[str, Any]:
    exists = path_exists or (lambda path: path.exists())
    reasons: list[str] = []

    patch_exists = exists(patch_path)
    observed_patch_sha: str | None = None
    if patch_exists:
        try:
            observed_patch_sha = sha256_file(patch_path)
        except OSError:
            reasons.append("patch_unreadable")
    else:
        reasons.append("patch_missing")
    if observed_patch_sha is not None and observed_patch_sha != PATCH_SHA256:
        reasons.append("patch_sha256_mismatch")

    model_exists = exists(model_dir)
    if not model_exists:
        reasons.append("model_dir_missing")

    device_kfd = exists(Path("/dev/kfd"))
    device_dri = exists(Path("/dev/dri"))
    if not device_kfd:
        reasons.append("device_kfd_missing")
    if not device_dri:
        reasons.append("device_dri_missing")

    rocm_smi = run(["rocm-smi", "--showproductname", "--showdriverversion", "--json"])
    rocminfo = run(["rocminfo"])
    rocm_blob = "\n".join(
        [
            str(rocm_smi.get("stdout") or ""),
            str(rocm_smi.get("stderr") or ""),
            str(rocminfo.get("stdout") or ""),
            str(rocminfo.get("stderr") or ""),
        ]
    ).lower()
    gpu_name_proved = "r9700" in rocm_blob or "radeon ai pro r9700" in rocm_blob
    gfx1201_proved = "gfx1201" in rocm_blob
    if rocm_smi.get("returncode") != 0:
        reasons.append("rocm_smi_unavailable")
    if rocminfo.get("returncode") != 0:
        reasons.append("rocminfo_unavailable")
    if not gpu_name_proved:
        reasons.append("r9700_identity_unproved")
    if not gfx1201_proved:
        reasons.append("gfx1201_identity_unproved")

    port_free, port_error = port_probe(candidate_port)
    if not port_free:
        reasons.append("candidate_port_busy")

    candidate_running = container_running(CANDIDATE_CONTAINER, run=run)
    if candidate_running:
        reasons.append("candidate_container_already_running")

    services = {service: service_state(service, run=run) for service in PRESERVED_SERVICES}
    active_services = [
        service
        for service, state in services.items()
        if state in {"active", "activating", "reloading"}
    ]
    if active_services:
        reasons.append("preserved_runtime_active_requires_separate_host_control")

    return {
        "schema": "hyperloom.r9700.vllm029.gpu_preflight.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": not reasons,
        "reasons": reasons,
        "truth_boundary": (
            "Read-only preflight. It does not stop services, remove containers, load a model, "
            "or prove that the vLLM 0.29 GPU runtime works."
        ),
        "patch": {
            "path": str(patch_path),
            "exists": patch_exists,
            "expected_sha256": PATCH_SHA256,
            "observed_sha256": observed_patch_sha,
        },
        "model": {"path": str(model_dir), "exists": model_exists},
        "devices": {"/dev/kfd": device_kfd, "/dev/dri": device_dri},
        "gpu": {
            "r9700_identity_proved": gpu_name_proved,
            "gfx1201_identity_proved": gfx1201_proved,
            "rocm_smi_returncode": rocm_smi.get("returncode"),
            "rocminfo_returncode": rocminfo.get("returncode"),
        },
        "isolation": {
            "candidate_port": candidate_port,
            "port_free": port_free,
            "port_error": port_error,
            "candidate_container": CANDIDATE_CONTAINER,
            "candidate_container_running": candidate_running,
        },
        "preserved_services": services,
        "active_preserved_services": active_services,
        "runtime_mutation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--patch", type=Path, default=PATCH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = collect_preflight(
        candidate_port=args.candidate_port,
        model_dir=args.model_dir,
        patch_path=args.patch,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
