#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "scripts" / "systemd" / "inneros-vllm-hyperloom-s3-canary.service"
CONFIG = ROOT / "docs" / "evidence" / "r9700_phase3_tuned_moe_config_s3_20260912.json"
PREPARE = ROOT / "scripts" / "r9700_phase5_prepare_canary.py"
PREFLIGHT = ROOT / "scripts" / "r9700_phase5_canary_preflight.py"
READINESS = ROOT / "scripts" / "r9700_readiness_warmup.py"
EXPECTED_CONFIG_SHA = "8b63443060479c3edf254556f93a31e251cd4d4e510ac391d2482d8228d4d3e6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checks: dict[str, bool] = {}
    checks["canonical_config_exists"] = CONFIG.exists()
    checks["canonical_config_hash"] = CONFIG.exists() and sha(CONFIG) == EXPECTED_CONFIG_SHA
    checks["phase5_sources_exist"] = all(p.exists() for p in (UNIT, PREPARE, PREFLIGHT, READINESS))

    unit = UNIT.read_text(encoding="utf-8") if UNIT.exists() else ""
    checks["unit_separate_container"] = "inneros-vllm-hyperloom-s3-canary" in unit
    checks["unit_port_18018"] = "--port 18018" in unit
    checks["unit_has_preflight"] = "r9700_phase5_canary_preflight.py" in unit
    checks["unit_has_readiness"] = "r9700_readiness_warmup.py --base-url http://127.0.0.1:18018" in unit
    checks["unit_queue1"] = "GPU_MAX_HW_QUEUES=1" in unit
    checks["unit_jit_monitor"] = "--jit-monitor-mode warn" in unit
    checks["unit_does_not_auto_conflict_stock"] = "Conflicts=" not in unit
    checks["unit_does_not_stop_stock"] = "systemctl --user stop inneros-vllm-canary-rocm10.service" not in unit

    preflight = PREFLIGHT.read_text(encoding="utf-8") if PREFLIGHT.exists() else ""
    checks["preflight_checks_stock"] = 'STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"' in preflight and 'checks["stock_service_inactive"]' in preflight
    checks["preflight_checks_vram"] = 'checks["vram_clean_under_1gb"]' in preflight
    checks["preflight_checks_overlay"] = 'checks["overlay_all_match"]' in preflight
    checks["preflight_checks_config"] = 'checks["config_hash_match"]' in preflight

    prepare = PREPARE.read_text(encoding="utf-8") if PREPARE.exists() else ""
    checks["prepare_uses_pinned_builder"] = "r9700_phase3_build_vllm43389_overlay.py" in prepare
    checks["prepare_checks_patch"] = "RUNTIME_PATCH_SHA256" in prepare
    checks["prepare_checks_config"] = "S3_CONFIG_SHA256" in prepare
    checks["prepare_checks_postcopy"] = "post-copy hash mismatch" in prepare
    checks["prepare_declares_no_stock_mutation"] = '"stock_runtime_mutated": False' in prepare

    out = {"schema": "hyperloom.r9700.phase5.packaging_selftest.v1", "checks": checks, "pass": all(checks.values())}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
