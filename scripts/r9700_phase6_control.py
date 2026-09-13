#!/usr/bin/env python3
"""Phase 6 controlled promotion for HyperLoom R9700 S3.

The public API contract stays on 127.0.0.1:8000. Exactly one backend owns the
GPU and port at a time: stock (default) or the production S3 service.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"
S3_SERVICE = "inneros-vllm-hyperloom-s3-production.service"
GUARD_TIMER = "inneros-vllm-hyperloom-phase6-guard.timer"
MODEL_ID = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
API_MODELS_URL = "http://127.0.0.1:8000/v1/models"
SCHEMA = "hyperloom.r9700.phase6.routing.v1"
VRAM_CLEAN_BYTES = 1_000_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path() -> pathlib.Path:
    raw = os.environ.get("HYPERLOOM_PHASE6_STATE")
    if raw:
        return pathlib.Path(raw)
    return pathlib.Path.home() / ".local/state/hyperloom-r9700/phase6_state.json"


def lock_path() -> pathlib.Path:
    root = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    return root / "hyperloom-r9700-phase6.lock"


def default_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "desired_backend": "stock",
        "active_backend": "stock",
        "validated": False,
        "guard_failures": 0,
        "updated_at": now(),
        "reason": "stock_default",
    }


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = default_state()
        data["reason"] = "state_unreadable_fail_closed"
    if data.get("schema") != SCHEMA:
        data = default_state()
        data["reason"] = "state_schema_mismatch_fail_closed"
    return data


def save_state(data: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["schema"] = SCHEMA
    data["updated_at"] = now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(argv: list[str], *, check: bool = False, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if check and cp.returncode != 0:
        raise RuntimeError(f"command_failed rc={cp.returncode}: {' '.join(argv)}: {cp.stderr.strip()}")
    return cp


def service_state(name: str) -> str:
    cp = run(["systemctl", "--user", "is-active", name], timeout=20)
    return cp.stdout.strip() or "inactive"


def set_service(action: str, name: str, *, check: bool = True, timeout: int = 720) -> None:
    run(["systemctl", "--user", action, name], check=check, timeout=timeout)


def model_ready(timeout: float = 5.0) -> tuple[bool, dict[str, Any]]:
    try:
        with urllib.request.urlopen(API_MODELS_URL, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            ids = [str(x.get("id")) for x in payload.get("data", []) if isinstance(x, dict)]
            return resp.status == 200 and MODEL_ID in ids, {"http_status": resp.status, "model_ids": ids}
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}


def vram_used_bytes() -> int:
    cp = run(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=30)
    if cp.returncode != 0:
        return -1
    try:
        return int(json.loads(cp.stdout)["card0"]["VRAM Total Used Memory (B)"])
    except Exception:
        return -1


def wait_for(predicate, *, timeout: float, interval: float = 1.0, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise RuntimeError(f"timeout_waiting_for_{description}")


def wait_service_inactive(name: str, timeout: float = 60.0) -> None:
    wait_for(lambda: service_state(name) != "active", timeout=timeout, description=f"{name}_inactive")


def wait_vram_clean(timeout: float = 90.0) -> None:
    wait_for(lambda: 0 <= vram_used_bytes() < VRAM_CLEAN_BYTES, timeout=timeout, description="vram_clean")


def verify_backend(service: str) -> dict[str, Any]:
    svc = service_state(service)
    ready, details = model_ready()
    result = {"service": service, "service_state": svc, "ready": ready, "readiness": details}
    result["pass"] = svc == "active" and ready
    return result


@contextmanager
def exclusive_lock():
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("phase6_lock_busy") from exc
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def fallback_locked(reason: str) -> dict[str, Any]:
    set_service("disable", GUARD_TIMER, check=False, timeout=30)
    set_service("stop", GUARD_TIMER, check=False, timeout=30)
    set_service("stop", S3_SERVICE, check=False, timeout=90)
    wait_service_inactive(S3_SERVICE, timeout=60)
    wait_vram_clean(timeout=120)
    set_service("start", STOCK_SERVICE, check=True, timeout=720)
    verification = verify_backend(STOCK_SERVICE)
    if not verification["pass"]:
        state = default_state()
        state.update({"desired_backend": "stock", "active_backend": "unknown", "validated": False,
                      "reason": "fallback_failed", "fallback_reason": reason, "verification": verification})
        save_state(state)
        raise RuntimeError("stock_fallback_readiness_failed")
    state = default_state()
    state.update({"desired_backend": "stock", "active_backend": "stock", "validated": True,
                  "reason": "automatic_fallback" if reason != "manual_rollback" else "manual_rollback",
                  "fallback_reason": reason, "verification": verification})
    save_state(state)
    return state


def promote(*, dry_run: bool = False) -> dict[str, Any]:
    plan = {
        "schema": "hyperloom.r9700.phase6.promotion_plan.v1",
        "from": "stock",
        "to": "hyperloom_s3",
        "public_endpoint": API_MODELS_URL.rsplit("/v1/models", 1)[0],
        "steps": [
            "verify_stock_ready",
            "stop_stock",
            "wait_stock_inactive",
            "wait_vram_clean",
            "start_s3_production_on_port_8000",
            "verify_s3_systemd_and_model_identity",
            "enable_phase6_guard",
            "commit_active_state",
        ],
        "fallback_on_any_failure": True,
    }
    if dry_run:
        return {"dry_run": True, "plan": plan, "state": load_state()}

    with exclusive_lock():
        stock = verify_backend(STOCK_SERVICE)
        if not stock["pass"]:
            raise RuntimeError("stock_not_ready_before_promotion")
        state = load_state()
        state.update({"desired_backend": "hyperloom_s3", "active_backend": "stock", "validated": False,
                      "reason": "promotion_in_progress", "guard_failures": 0})
        save_state(state)
        try:
            set_service("stop", STOCK_SERVICE, check=True, timeout=90)
            wait_service_inactive(STOCK_SERVICE, timeout=60)
            wait_vram_clean(timeout=120)
            set_service("start", S3_SERVICE, check=True, timeout=720)
            verification = verify_backend(S3_SERVICE)
            if not verification["pass"]:
                raise RuntimeError("s3_readiness_failed")
            set_service("enable", GUARD_TIMER, check=True, timeout=30)
            set_service("start", GUARD_TIMER, check=True, timeout=30)
            state.update({"active_backend": "hyperloom_s3", "validated": True, "reason": "promotion_validated",
                          "guard_failures": 0, "verification": verification})
            save_state(state)
            return state
        except Exception as exc:
            fallback_locked(f"promotion_failure:{type(exc).__name__}:{exc}")
            raise


def rollback(*, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "plan": ["stop_guard", "stop_s3", "wait_vram_clean", "start_stock", "verify_stock"]}
    with exclusive_lock():
        return fallback_locked("manual_rollback")


def guard_once(*, failure_threshold: int = 3) -> dict[str, Any]:
    with exclusive_lock():
        state = load_state()
        if state.get("active_backend") != "hyperloom_s3":
            return {"action": "noop", "reason": "stock_or_non_s3_active", "state": state}
        svc = service_state(S3_SERVICE)
        ready, details = model_ready()
        if svc != "active":
            recovered = fallback_locked(f"guard_service_state:{svc}")
            return {"action": "fallback", "trigger": "service_not_active", "state": recovered}
        if ready:
            state["guard_failures"] = 0
            state["last_guard"] = {"pass": True, "readiness": details, "at": now()}
            save_state(state)
            return {"action": "healthy", "state": state}
        failures = int(state.get("guard_failures", 0)) + 1
        state["guard_failures"] = failures
        state["last_guard"] = {"pass": False, "readiness": details, "at": now()}
        save_state(state)
        if failures >= failure_threshold:
            recovered = fallback_locked(f"guard_readiness_failures:{failures}")
            return {"action": "fallback", "trigger": "readiness_threshold", "state": recovered}
        return {"action": "degraded", "failures": failures, "threshold": failure_threshold, "state": state}


def status() -> dict[str, Any]:
    ready, details = model_ready()
    return {
        "schema": "hyperloom.r9700.phase6.status.v1",
        "state": load_state(),
        "services": {"stock": service_state(STOCK_SERVICE), "hyperloom_s3": service_state(S3_SERVICE),
                     "guard_timer": service_state(GUARD_TIMER)},
        "public_endpoint": {"url": API_MODELS_URL, "ready": ready, "details": details},
        "vram_used_bytes": vram_used_bytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("promote")
    p.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("rollback")
    r.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("guard-once")
    g.add_argument("--failure-threshold", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.cmd == "status":
            out = status()
        elif args.cmd == "promote":
            out = promote(dry_run=args.dry_run)
        elif args.cmd == "rollback":
            out = rollback(dry_run=args.dry_run)
        else:
            out = guard_once(failure_threshold=max(1, args.failure_threshold))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        code = 4 if str(exc) == "phase6_lock_busy" else 2
        print(json.dumps({"schema": "hyperloom.r9700.phase6.error.v1", "error": str(exc)}, indent=2), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
