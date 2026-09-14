from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "r9700_upstream_delta_static.py"


def test_upstream_delta_static_guard_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["pass"] is True
    assert payload["network_access"] is False
    assert payload["live_runtime_mutation"] is False
    assert payload["checks"]["hyperloom_identity_patch"]["pass"] is True
    assert payload["checks"]["phase3_overlay_lineage"]["pass"] is True
    assert payload["checks"]["s3_guarded_hybrid"]["pass"] is True
    assert payload["decisions"]["s3_small_m_w1_hybrid"] == "retain"
