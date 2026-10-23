from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "rocm10_1_stock_rollback_manifest.py"
SPEC = importlib.util.spec_from_file_location("rocm10_1_stock_rollback_manifest", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def completed(argv, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def fake_runner(argv: list[str], _timeout: float):
    if argv == ["docker", "inspect", MOD.STOCK_CONTAINER]:
        return completed(
            argv,
            json.dumps(
                [
                    {
                        "Id": "container-id",
                        "Created": "2026-10-01T00:00:00Z",
                        "Image": "sha256:stock-image",
                        "State": {
                            "Running": True,
                            "Status": "running",
                            "StartedAt": "2026-10-01T00:01:00Z",
                        },
                        "Config": {
                            "Image": "rocm/vllm:stock",
                            "Entrypoint": ["python3"],
                            "Cmd": ["-m", "vllm.entrypoints.openai.api_server"],
                            "WorkingDir": "/workspace",
                            "User": "",
                            "Env": [
                                "MODEL=QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
                                "API_KEY=sk-super-secret",
                                "ROCM_ATTN=1",
                            ],
                        },
                        "HostConfig": {
                            "RestartPolicy": {"Name": "unless-stopped"},
                            "NetworkMode": "host",
                            "IpcMode": "host",
                            "SecurityOpt": ["seccomp=unconfined"],
                            "Devices": [
                                {
                                    "PathOnHost": "/dev/kfd",
                                    "PathInContainer": "/dev/kfd",
                                }
                            ],
                            "GroupAdd": ["video", "render"],
                            "PortBindings": None,
                        },
                        "NetworkSettings": {"Ports": {}},
                        "Mounts": [
                            {
                                "Type": "bind",
                                "Source": "/models/qwen",
                                "Destination": "/models/qwen",
                                "Mode": "ro",
                                "RW": False,
                            }
                        ],
                    }
                ]
            ),
        )
    if argv == ["docker", "image", "inspect", "sha256:stock-image"]:
        return completed(
            argv,
            json.dumps(
                [
                    {
                        "Id": "sha256:stock-image",
                        "RepoTags": ["rocm/vllm:stock"],
                        "RepoDigests": ["rocm/vllm@sha256:digest"],
                        "Created": "2026-09-30T00:00:00Z",
                        "Size": 123456789,
                        "Architecture": "amd64",
                        "Os": "linux",
                    }
                ]
            ),
        )
    raise AssertionError(argv)


def test_manifest_preserves_stock_identity_and_redacts_secrets() -> None:
    manifest = MOD.capture_manifest(
        runner=fake_runner,
        request_json=lambda _url: {
            "data": [{"id": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"}]
        },
        port_probe=lambda: True,
    )

    assert manifest["schema"] == MOD.SCHEMA
    assert manifest["container"]["running"] is True
    assert manifest["container"]["image_id"] == "sha256:stock-image"
    assert manifest["image"]["repo_digests"] == ["rocm/vllm@sha256:digest"]
    assert "API_KEY=REDACTED" in manifest["container"]["environment"]
    assert "sk-super-secret" not in json.dumps(manifest)
    assert manifest["rollback"]["start_argv"] == [
        "docker",
        "start",
        MOD.STOCK_CONTAINER,
    ]
    assert manifest["rollback"]["container_must_not_be_removed"] is True
    assert manifest["safety"]["mutates_host"] is False
    assert manifest["manifest_sha256"] == MOD.stable_json_sha256(manifest)


def test_models_probe_failure_is_recorded_without_failing_manifest() -> None:
    def failed_request(_url: str):
        raise TimeoutError("endpoint timeout")

    manifest = MOD.capture_manifest(
        runner=fake_runner,
        request_json=failed_request,
        port_probe=lambda: True,
    )

    assert manifest["endpoint"]["tcp_listening"] is True
    assert manifest["endpoint"]["models"] == {
        "ok": False,
        "error": "TimeoutError",
    }


def test_environment_redaction_keeps_non_secret_runtime_flags() -> None:
    values = MOD.redact_environment(
        [
            "ROCM_ATTN=1",
            "GPU_MAX_HW_QUEUES=1",
            "PASSWORD=human-secret",
            "AUTHORIZATION=Bearer abcdef",
        ]
    )
    assert "ROCM_ATTN=1" in values
    assert "GPU_MAX_HW_QUEUES=1" in values
    assert "PASSWORD=REDACTED" in values
    assert "AUTHORIZATION=REDACTED" in values


def test_docker_probe_failure_fails_closed() -> None:
    def broken_runner(argv: list[str], _timeout: float):
        return completed(argv, stderr="not found", returncode=1)

    try:
        MOD.capture_manifest(
            runner=broken_runner,
            request_json=lambda _url: {},
            port_probe=lambda: False,
        )
    except RuntimeError as exc:
        assert str(exc).startswith("docker_probe_failed")
    else:
        raise AssertionError("capture_manifest should fail closed")
