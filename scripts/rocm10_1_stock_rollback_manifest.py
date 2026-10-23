#!/usr/bin/env python3
"""Capture a redacted rollback manifest for the active stock R9700 service.

This command is read-only. It does not stop, restart, rename, remove, pull, build,
or recreate any container. The intended rollback during an exclusive ROCm 10.1
window is to preserve the existing stock container and start it again by name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

STOCK_CONTAINER = "inneros-vllm-qwen3-coder-30b-awq"
STOCK_BASE_URL = "http://127.0.0.1:8000/v1"
SCHEMA = "hyperloom.rocm10_1.stock_rollback_manifest.v1"

_SECRET_KEY_RE = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|authorization|credential|private[_-]?key)")
_SECRET_VALUE_RE = re.compile(r"(?i)(?:sk-|gh[pousr]_|crsr_|bearer\s+)[A-Za-z0-9._-]+")


def redact_text(value: str) -> str:
    return _SECRET_VALUE_RE.sub("REDACTED", value)


def redact_environment(values: list[str]) -> list[str]:
    redacted: list[str] = []
    for item in values:
        key, separator, value = item.partition("=")
        if separator and _SECRET_KEY_RE.search(key):
            redacted.append(f"{key}=REDACTED")
        else:
            redacted.append(redact_text(item))
    return redacted


def stable_json_sha256(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clone.pop("manifest_sha256", None)
    encoded = json.dumps(
        clone,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run(argv: list[str], timeout_seconds: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _request_json(url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer local"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def tcp_listening(host: str = "127.0.0.1", port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _docker_json(
    argv: list[str],
    runner: Callable[[list[str], float], subprocess.CompletedProcess[str]],
) -> Any:
    result = runner(argv, 20.0)
    if result.returncode != 0:
        raise RuntimeError(f"docker_probe_failed:{redact_text(result.stderr.strip())}")
    return json.loads(result.stdout)


def capture_manifest(
    *,
    container_name: str = STOCK_CONTAINER,
    base_url: str = STOCK_BASE_URL,
    runner: Callable[[list[str], float], subprocess.CompletedProcess[str]] = _run,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
    port_probe: Callable[[], bool] = tcp_listening,
) -> dict[str, Any]:
    inspected = _docker_json(["docker", "inspect", container_name], runner)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise RuntimeError("stock_container_inspect_shape_invalid")
    container = inspected[0]

    image_ref = str(container.get("Config", {}).get("Image") or "")
    image_id = str(container.get("Image") or "")
    image_inspected = _docker_json(["docker", "image", "inspect", image_id], runner)
    image = image_inspected[0] if isinstance(image_inspected, list) and image_inspected else {}

    state = container.get("State") or {}
    config = container.get("Config") or {}
    host_config = container.get("HostConfig") or {}
    network_settings = container.get("NetworkSettings") or {}
    mounts = container.get("Mounts") or []

    model_probe: dict[str, Any]
    try:
        model_probe = request_json(base_url.rstrip("/") + "/models")
    except Exception as exc:  # noqa: BLE001
        model_probe = {"ok": False, "error": type(exc).__name__}

    safe_mounts = [
        {
            "type": mount.get("Type"),
            "source": redact_text(str(mount.get("Source") or "")),
            "destination": mount.get("Destination"),
            "mode": mount.get("Mode"),
            "rw": mount.get("RW"),
        }
        for mount in mounts
    ]

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "container": {
            "name": container_name,
            "id": container.get("Id"),
            "created": container.get("Created"),
            "running": bool(state.get("Running")),
            "status": state.get("Status"),
            "started_at": state.get("StartedAt"),
            "image_reference": image_ref,
            "image_id": image_id,
            "entrypoint": config.get("Entrypoint"),
            "command": config.get("Cmd"),
            "working_dir": config.get("WorkingDir"),
            "user": config.get("User"),
            "environment": redact_environment(list(config.get("Env") or [])),
            "restart_policy": host_config.get("RestartPolicy"),
            "network_mode": host_config.get("NetworkMode"),
            "ipc_mode": host_config.get("IpcMode"),
            "security_opt": host_config.get("SecurityOpt"),
            "devices": host_config.get("Devices"),
            "group_add": host_config.get("GroupAdd"),
            "port_bindings": host_config.get("PortBindings"),
            "published_ports": network_settings.get("Ports"),
            "mounts": safe_mounts,
        },
        "image": {
            "id": image.get("Id"),
            "repo_tags": image.get("RepoTags"),
            "repo_digests": image.get("RepoDigests"),
            "created": image.get("Created"),
            "size_bytes": image.get("Size"),
            "architecture": image.get("Architecture"),
            "os": image.get("Os"),
        },
        "endpoint": {
            "base_url": base_url.rstrip("/"),
            "tcp_listening": bool(port_probe()),
            "models": model_probe,
        },
        "rollback": {
            "strategy": "preserve_existing_container_then_start_by_name",
            "container_must_not_be_removed": True,
            "start_argv": ["docker", "start", container_name],
            "verify_url": base_url.rstrip("/") + "/models",
            "success_conditions": [
                "container_state_running_true",
                "tcp_127_0_0_1_8000_listening",
                "models_endpoint_returns_expected_qwen_model",
            ],
        },
        "safety": {
            "mutates_host": False,
            "secrets_redacted": True,
            "does_not_stop_or_restart": True,
            "does_not_pull_or_build": True,
            "does_not_remove_container_or_image": True,
        },
    }
    payload["manifest_sha256"] = stable_json_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default=STOCK_CONTAINER)
    parser.add_argument("--base-url", default=STOCK_BASE_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = capture_manifest(
            container_name=args.container,
            base_url=args.base_url,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": redact_text(str(exc)),
                    "mutates_host": False,
                },
                sort_keys=True,
            )
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "container": manifest["container"]["name"],
                "running": manifest["container"]["running"],
                "image_id": manifest["container"]["image_id"],
                "endpoint_listening": manifest["endpoint"]["tcp_listening"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(args.output),
                "mutates_host": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
