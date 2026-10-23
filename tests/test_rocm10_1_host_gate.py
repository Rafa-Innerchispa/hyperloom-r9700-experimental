from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "rocm10_1_host_gate.py"
SPEC = importlib.util.spec_from_file_location("rocm10_1_host_gate", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def snapshot(**overrides):
    values = {
        "root_total_gib": 466.0,
        "root_used_gib": 380.0,
        "root_free_gib": 86.0,
        "root_used_percent": 81.55,
        "memory_total_gib": 30.0,
        "memory_available_gib": 9.0,
        "swap_total_gib": 8.0,
        "swap_used_gib": 8.0,
        "swap_io_kib_per_second": 0.0,
        "load_1m": 10.0,
        "cpu_count": 16,
        "image_present": True,
        "image_id": "sha256:rocm101",
        "stock_container_running": True,
        "stock_endpoint_healthy": True,
        "candidate_port_free": True,
        "cleanup_active": False,
    }
    values.update(overrides)
    return MOD.HostSnapshot(**values)


def test_current_46_gib_snapshot_stays_on_hold() -> None:
    report = MOD.evaluate_snapshot(
        snapshot(
            root_used_gib=401.0,
            root_free_gib=46.0,
            root_used_percent=90.0,
        )
    )
    assert report.ready is False
    assert any(item.startswith("root_free_below_gate") for item in report.blockers)
    assert any(item.startswith("root_usage_above_gate") for item in report.blockers)
    assert report.image_pull_allowed is False
    assert report.mutates_host is False


def test_full_but_idle_swap_is_warning_when_memory_is_healthy() -> None:
    report = MOD.evaluate_snapshot(snapshot())
    assert report.ready is True
    assert report.blockers == []
    assert any(item.startswith("swap_nearly_full_but_idle") for item in report.warnings)
    assert report.next_action.startswith("capture_stock_rollback_manifest")


def test_active_swap_pressure_blocks_candidate_window() -> None:
    report = MOD.evaluate_snapshot(snapshot(swap_io_kib_per_second=4096.0))
    assert report.ready is False
    assert any(item.startswith("active_swap_pressure") for item in report.blockers)


def test_missing_image_does_not_authorize_pull() -> None:
    report = MOD.evaluate_snapshot(snapshot(image_present=False, image_id=None))
    assert report.ready is False
    assert "rocm10_1_vllm_image_not_present_do_not_pull_until_disk_gate_passes" in report.blockers
    assert report.image_pull_allowed is False


def test_cleanup_or_occupied_candidate_port_blocks_launch() -> None:
    report = MOD.evaluate_snapshot(snapshot(cleanup_active=True, candidate_port_free=False))
    assert report.ready is False
    assert "storage_cleanup_still_active" in report.blockers
    assert f"candidate_port_{MOD.CANDIDATE_PORT}_not_free" in report.blockers


def test_stock_health_is_required_before_exclusive_window() -> None:
    report = MOD.evaluate_snapshot(snapshot(stock_container_running=False, stock_endpoint_healthy=False))
    assert report.ready is False
    assert "stock_container_not_running" in report.blockers
    assert "stock_endpoint_8000_not_healthy" in report.blockers
