#!/usr/bin/env python3
"""Probe AITER FlyDSL MoE sorting on gfx1201 without using the broken Opus ABI path.

The environment switch is set before importing aiter.fused_moe. This is an isolated
probe executed inside the resident vLLM container. It does not patch or restart vLLM.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from r9700_awq_backend_probe import discover_live_container

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.aiter_flydsl_sorting_bypass.v1"


def _run(argv: list[str], timeout: float = 180.0) -> dict[str, Any]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def _inside() -> str:
    return textwrap.dedent(r'''
import os
os.environ['AITER_USE_FLYDSL_MOE_SORTING'] = '1'
os.environ['AITER_USE_CK_MOE_SORTING'] = '0'
import json, traceback, torch
import aiter.fused_moe as fm

out = {
    'device': torch.cuda.get_device_name(0),
    'flydsl_available': bool(fm.is_flydsl_available()),
    'use_flydsl_sorting': bool(getattr(fm, '_USE_FLYDSL_MOE_SORTING', False)),
    'use_ck_sorting': bool(getattr(fm, '_USE_CK_MOE_SORTING', False)),
    'sort_backend': str(getattr(fm, '_MOE_SORT_BACKEND', '')),
}
try:
    torch.manual_seed(20260908)
    token_num, E, topk, model_dim, block_m = 4, 8, 2, 2048, 16
    topk_ids = torch.tensor([[0,1],[1,2],[2,3],[3,4]], dtype=torch.int32, device='cuda')
    topk_weights = torch.rand((token_num, topk), dtype=torch.float32, device='cuda')
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    result = fm.moe_sorting(topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = result
    torch.cuda.synchronize()
    out.update({
        'pass': True,
        'sorted_ids_shape': list(sorted_ids.shape),
        'sorted_weights_shape': list(sorted_weights.shape),
        'sorted_expert_ids_shape': list(sorted_expert_ids.shape),
        'num_valid_ids': int(num_valid_ids.reshape(-1)[0].item()),
        'moe_buf_shape': list(moe_buf.shape) if hasattr(moe_buf, 'shape') else None,
        'sorted_ids_head': [int(x) for x in sorted_ids[:min(16, sorted_ids.numel())].tolist()],
        'finite_weights': bool(torch.isfinite(sorted_weights).all().item()),
    })
except Exception as exc:
    out.update({'pass': False, 'error': f'{type(exc).__name__}:{exc}', 'trace': traceback.format_exc()[-24000:]})
print(json.dumps(out, sort_keys=True))
''')


def _sha(payload: dict[str, Any]) -> str:
    c = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    c.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    live = discover_live_container(run=_run)
    container = str((live.get("selected") or {}).get("Names") or "")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "truth_boundary": {
            "production_runtime_modified": False,
            "service_restarted": False,
            "isolated_child_process": True,
            "official_support_claim": False,
        },
    }
    if not container:
        payload.update({"pass": False, "error": "no_vllm_container"})
    else:
        r = _run(["docker", "exec", container, "python3", "-c", _inside()], 180)
        parsed: dict[str, Any] = {}
        for line in r["stdout"].splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                parsed = obj
        payload["probe"] = parsed
        payload["docker_exec"] = {"returncode": r["returncode"], "stderr_tail": r["stderr"][-12000:]}
        payload["pass"] = bool(parsed.get("pass")) and r["returncode"] == 0
    payload["probe_sha256"] = _sha(payload)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "docs" / "evidence" / f"r9700_aiter_flydsl_sorting_bypass_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": payload.get("pass"), "output": str(out), "probe_sha256": payload["probe_sha256"], "probe": payload.get("probe")}, sort_keys=True))
    return 0 if payload.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
