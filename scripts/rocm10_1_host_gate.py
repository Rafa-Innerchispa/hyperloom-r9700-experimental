#!/usr/bin/env python3
"""Read-only host gate for the ROCm 10.1 R9700 candidate window.

The gate intentionally performs no image pull, container start/stop, service restart,
file deletion, or swap mutation. It only observes the host and returns a fail-closed
JSON decision that can be archived before an exclusive 30B candidate window.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

VLLM_IMAGE = "rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0"
STOCK_CONTAINER = "inneros-vllm-qwen3-coder-30b-awq"
STOCK_HOST = "127.0.0.1"
STOCK_PORT = 8000
CANDIDATE_PORT = 18029


@dataclass(frozen=True)
class GateThresholds:
    min_root_free_gib: float = 70.0
    max_root_used_percent: float = 85.0
    min_memory_available_gib: float = 6.0
    swap_warning_percent: float = 90.0
    max_swap_io_kib_per_second: float = 1024.0


@dataclass(frozen=True)
class HostSnapshot:
    root_total_gib: float
    root_used_gib: float
    root_free_gib: float
    root_used_percent: float
    memory_total_gib: float
    memory_available_gib: float
    swap_total_gib: float
    swap_used_gib: float
    swap_io_kib_per_second: float
    load_1m: float
    cpu_count: int
    image_present: bool
    image_id: str | None
    stock_container_running: bool
    stock_endpoint_healthy: bool
    candidate_port_free: bool
    cleanup_active: bool


@dataclass(frozen=True)
class GateReport:
    ready: bool
    blockers: list[str]
    warnings: list[str]
    next_action: str
    mutates_host: bool
    image_pull_allowed: bool
    exclusive_gpu_window_required: bool
    thresholds: GateThresholds
    snapshot: HostSnapshot


def _gib(value: int | float) -> float:
    return round(float(value) / (1024.0**3), 3)


def _run(argv: list[str], timeout_seconds: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        token = raw_value.strip().split()[0] if raw_value.strip() else "0"
        try:
            values[key] = int(token)
        except ValueError:
            continue
    return values


def _read_vmstat(path: Path = Path("/proc/vmstat")) -> tuple[int, int]:
    values: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 2:
            continue
        if parts[0] in {"pswpin", "pswpout"}:
            values[parts[0]] = int(parts[1])
    return values.get("pswpin", 0), values.get("pswpout", 0)


def sample_swap_io_kib_per_second(
    interval_seconds: float = 1.0,
    reader: Callable[[], tuple[int, int]] = _read_vmstat,
    sleeper: Callable[[float], None] = time.sleep,
) -> float:
    if interval_seconds <= 0:
        return 0.0
    before_in, before_out = reader()
    sleeper(interval_seconds)
    after_in, after_out = reader()
    page_kib = float(os.sysconf("SC_PAGE_SIZE")) / 1024.0
    pages = max(0, after_in - before_in) + max(0, after_out - before_out)
    return round((pages * page_kib) / interval_seconds, 3)


def tcp_listening(host: str, port: int, timeout_seconds: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        return sock.connect_ex((host, port)) == 0


def docker_image_identity(
    image: str = VLLM_IMAGE,
    runner: Callable[[list[str], float], subprocess.CompletedProcess[str]] = _run,
) -> tuple[bool, str | None]:
    try:
        result = runner(["docker", "image", "inspect", "--format", "{{.Id}}", image], 20.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None
    image_id = result.stdout.strip() if result.returncode == 0 else ""
    return bool(image_id), image_id or None


def docker_container_running(
    container_name: str = STOCK_CONTAINER,
    runner: Callable[[list[str], float], subprocess.CompletedProcess[str]] = _run,
) -> bool:
    try:
        result = runner(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            20.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def collect_host_snapshot(
    *,
    cleanup_active: bool = False,
    swap_sample_seconds: float = 1.0,
) -> HostSnapshot:
    disk = shutil.disk_usage("/")
    meminfo = _read_key_values(Path("/proc/meminfo"))
    mem_total_kib = meminfo.get("MemTotal", 0)
    mem_available_kib = meminfo.get("MemAvailable", 0)
    swap_total_kib = meminfo.get("SwapTotal", 0)
    swap_free_kib = meminfo.get("SwapFree", 0)
    image_present, image_id = docker_image_identity()

    root_used_percent = (disk.used / disk.total * 100.0) if disk.total else 100.0
    swap_used_kib = max(0, swap_total_kib - swap_free_kib)

    return HostSnapshot(
        root_total_gib=_gib(disk.total),
        root_used_gib=_gib(disk.used),
        root_free_gib=_gib(disk.free),
        root_used_percent=round(root_used_percent, 2),
        memory_total_gib=round(mem_total_kib / 1024.0**2, 3),
        memory_available_gib=round(mem_available_kib / 1024.0**2, 3),
        swap_total_gib=round(swap_total_kib / 1024.0**2, 3),
        swap_used_gib=round(swap_used_kib / 1024.0**2, 3),
        swap_io_kib_per_second=sample_swap_io_kib_per_second(swap_sample_seconds),
        load_1m=round(os.getloadavg()[0], 3),
        cpu_count=max(1, os.cpu_count() or 1),
        image_present=image_present,
        image_id=image_id,
        stock_container_running=docker_container_running(),
        stock_endpoint_healthy=tcp_listening(STOCK_HOST, STOCK_PORT),
        candidate_port_free=not tcp_listening(STOCK_HOST, CANDIDATE_PORT),
        cleanup_active=cleanup_active,
    )


def evaluate_snapshot(
    snapshot: HostSnapshot,
    thresholds: GateThresholds = GateThresholds(),
) -> GateReport:
    blockers: list[str] = []
    warnings: list[str] = []

    if snapshot.root_free_gib < thresholds.min_root_free_gib:
        blockers.append(f"root_free_below_gate:{snapshot.root_free_gib:.3f}GiB<{thresholds.min_root_free_gib:.3f}GiB")
    if snapshot.root_used_percent > thresholds.max_root_used_percent:
        blockers.append(
            f"root_usage_above_gate:{snapshot.root_used_percent:.2f}%>{thresholds.max_root_used_percent:.2f}%"
        )
    if snapshot.memory_available_gib < thresholds.min_memory_available_gib:
        blockers.append(
            f"memory_available_below_gate:{snapshot.memory_available_gib:.3f}GiB<"
            f"{thresholds.min_memory_available_gib:.3f}GiB"
        )
    if snapshot.cleanup_active:
        blockers.append("storage_cleanup_still_active")
    if not snapshot.image_present:
        blockers.append("rocm10_1_vllm_image_not_present_do_not_pull_until_disk_gate_passes")
    if not snapshot.stock_container_running:
        blockers.append("stock_container_not_running")
    if not snapshot.stock_endpoint_healthy:
        blockers.append("stock_endpoint_8000_not_healthy")
    if not snapshot.candidate_port_free:
        blockers.append(f"candidate_port_{CANDIDATE_PORT}_not_free")

    swap_used_percent = snapshot.swap_used_gib / snapshot.swap_total_gib * 100.0 if snapshot.swap_total_gib > 0 else 0.0
    if swap_used_percent >= thresholds.swap_warning_percent:
        if snapshot.swap_io_kib_per_second > thresholds.max_swap_io_kib_per_second:
            blockers.append(
                f"active_swap_pressure:{snapshot.swap_io_kib_per_second:.3f}KiB/s>"
                f"{thresholds.max_swap_io_kib_per_second:.3f}KiB/s"
            )
        else:
            warnings.append(
                f"swap_nearly_full_but_idle:{swap_used_percent:.2f}% used; "
                f"io={snapshot.swap_io_kib_per_second:.3f}KiB/s"
            )

    if snapshot.load_1m > snapshot.cpu_count * 1.5:
        warnings.append(f"high_load:{snapshot.load_1m:.3f} on {snapshot.cpu_count} CPUs")

    ready = not blockers
    next_action = (
        "capture_stock_rollback_manifest_then_open_bounded_exclusive_gpu_window"
        if ready
        else "keep_rocm10_1_on_hold_and_resolve_blockers_without_pulling_or_building"
    )
    return GateReport(
        ready=ready,
        blockers=blockers,
        warnings=warnings,
        next_action=next_action,
        mutates_host=False,
        image_pull_allowed=False,
        exclusive_gpu_window_required=True,
        thresholds=thresholds,
        snapshot=snapshot,
    )


def _load_snapshot(path: Path) -> HostSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HostSnapshot(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        help="Evaluate a sanitized HostSnapshot JSON instead of probing the host.",
    )
    parser.add_argument(
        "--cleanup-active",
        action="store_true",
        help="Fail closed while Cursor/storage cleanup is still running.",
    )
    parser.add_argument("--swap-sample-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    snapshot = (
        _load_snapshot(args.snapshot_json)
        if args.snapshot_json
        else collect_host_snapshot(
            cleanup_active=args.cleanup_active,
            swap_sample_seconds=args.swap_sample_seconds,
        )
    )
    report = evaluate_snapshot(snapshot)
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
