from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "r9700_current_upstream_candidate_plan.py"


def run_plan(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_default_plan_is_isolated_and_non_executing() -> None:
    proc = run_plan()

    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["pass"] is True
    assert payload["candidate"]["port"] == 8011
    assert payload["production_immutability"]["port"] == 8000
    assert payload["production_immutability"]["must_not_be_mutated"] is True
    assert "Planning artifact only" in payload["truth_boundary"]
    assert [stage["id"] for stage in payload["stages"]] == [
        "source",
        "isolation",
        "load",
        "correctness",
        "path_evidence",
        "performance",
        "promotion",
    ]


def test_production_port_is_rejected() -> None:
    proc = run_plan("--candidate-port", "8000")

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["pass"] is False
    assert "candidate_port_collides_with_production" in payload["errors"]


def test_production_container_is_rejected() -> None:
    proc = run_plan(
        "--candidate-container",
        "inneros-vllm-hyperloom-s3-production",
    )

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["pass"] is False
    assert "candidate_container_collides_with_production" in payload["errors"]
