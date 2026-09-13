from __future__ import annotations

import hashlib
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


def test_canonical_s3_config_hash_is_stable() -> None:
    assert CONFIG.exists()
    assert sha(CONFIG) == EXPECTED_CONFIG_SHA


def test_phase5_sources_exist() -> None:
    for path in (UNIT, PREPARE, PREFLIGHT, READINESS):
        assert path.exists(), path


def test_unit_is_separate_and_fail_closed() -> None:
    text = UNIT.read_text(encoding="utf-8")
    assert "inneros-vllm-hyperloom-s3-canary" in text
    assert "--port 18018" in text
    assert "r9700_phase5_canary_preflight.py" in text
    assert "r9700_readiness_warmup.py --base-url http://127.0.0.1:18018" in text
    assert "GPU_MAX_HW_QUEUES=1" in text
    assert "--jit-monitor-mode warn" in text
    assert "Conflicts=" not in text
    assert "systemctl --user stop inneros-vllm-canary-rocm10.service" not in text


def test_preflight_checks_stock_and_clean_vram() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert 'STOCK_SERVICE = "inneros-vllm-canary-rocm10.service"' in text
    assert 'checks["stock_service_inactive"]' in text
    assert 'checks["vram_clean_under_1gb"]' in text
    assert 'checks["overlay_all_match"]' in text
    assert 'checks["config_hash_match"]' in text


def test_prepare_rebuilds_and_hash_verifies() -> None:
    text = PREPARE.read_text(encoding="utf-8")
    assert "r9700_phase3_build_vllm43389_overlay.py" in text
    assert "RUNTIME_PATCH_SHA256" in text
    assert "S3_CONFIG_SHA256" in text
    assert "post-copy hash mismatch" in text
    assert '"stock_runtime_mutated": False' in text
