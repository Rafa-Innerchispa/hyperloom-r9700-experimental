from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import r9700_vllm029_gpu_preflight as preflight


def healthy_run(argv: list[str], **_kwargs: Any) -> dict[str, Any]:
    if argv[:2] == ["rocm-smi", "--showproductname"]:
        return {
            "returncode": 0,
            "stdout": '{"card0":{"Card series":"AMD Radeon AI PRO R9700"}}',
            "stderr": "",
        }
    if argv == ["rocminfo"]:
        return {"returncode": 0, "stdout": "Name: gfx1201\n", "stderr": ""}
    if argv[:2] == ["docker", "inspect"]:
        return {"returncode": 1, "stdout": "", "stderr": "not found"}
    if argv[:3] == ["systemctl", "--user", "is-active"]:
        return {"returncode": 3, "stdout": "inactive\n", "stderr": ""}
    raise AssertionError(f"unexpected command: {argv}")


def test_healthy_preflight_passes_without_runtime_mutation():
    result = preflight.collect_preflight(
        run=healthy_run,
        path_exists=lambda _path: True,
        port_probe=lambda _port: (True, ""),
    )

    assert result["pass"] is True, result
    assert result["reasons"] == []
    assert result["runtime_mutation"] is False
    assert result["patch"]["observed_sha256"] == preflight.PATCH_SHA256
    assert result["gpu"]["r9700_identity_proved"] is True
    assert result["gpu"]["gfx1201_identity_proved"] is True
    assert result["isolation"]["port_free"] is True
    assert result["active_preserved_services"] == []


def test_busy_candidate_port_fails_closed():
    result = preflight.collect_preflight(
        run=healthy_run,
        path_exists=lambda _path: True,
        port_probe=lambda _port: (False, "already in use"),
    )

    assert result["pass"] is False
    assert "candidate_port_busy" in result["reasons"]
    assert result["runtime_mutation"] is False


def test_active_preserved_service_is_reported_not_stopped():
    def run(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            if argv[-1] == preflight.PRESERVED_SERVICES[0]:
                return {"returncode": 0, "stdout": "active\n", "stderr": ""}
        return healthy_run(argv, **kwargs)

    result = preflight.collect_preflight(
        run=run,
        path_exists=lambda _path: True,
        port_probe=lambda _port: (True, ""),
    )

    assert result["pass"] is False
    assert "preserved_runtime_active_requires_separate_host_control" in result["reasons"]
    assert preflight.PRESERVED_SERVICES[0] in result["active_preserved_services"]
    assert result["runtime_mutation"] is False


def test_unproved_gpu_identity_fails_closed():
    def run(argv: list[str], **_kwargs: Any) -> dict[str, Any]:
        if argv[0] == "rocm-smi":
            return {"returncode": 0, "stdout": "unknown gpu", "stderr": ""}
        if argv == ["rocminfo"]:
            return {"returncode": 0, "stdout": "Name: gfx000\n", "stderr": ""}
        if argv[:2] == ["docker", "inspect"]:
            return {"returncode": 1, "stdout": "", "stderr": "not found"}
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return {"returncode": 3, "stdout": "inactive\n", "stderr": ""}
        raise AssertionError(f"unexpected command: {argv}")

    result = preflight.collect_preflight(
        run=run,
        path_exists=lambda _path: True,
        port_probe=lambda _port: (True, ""),
    )

    assert result["pass"] is False
    assert "r9700_identity_unproved" in result["reasons"]
    assert "gfx1201_identity_unproved" in result["reasons"]


def test_missing_devices_and_model_fail_closed(tmp_path: Path):
    result = preflight.collect_preflight(
        run=healthy_run,
        model_dir=tmp_path / "missing-model",
        path_exists=lambda path: path == preflight.PATCH,
        port_probe=lambda _port: (True, ""),
    )

    assert result["pass"] is False
    assert "model_dir_missing" in result["reasons"]
    assert "device_kfd_missing" in result["reasons"]
    assert "device_dri_missing" in result["reasons"]
