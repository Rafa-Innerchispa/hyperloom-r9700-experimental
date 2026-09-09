#!/usr/bin/env python3
"""Reversible full-model smoke for the R9700 stock-layout WNA16 hybrid.

The resident container is benchmarked in stock state, then bootstrapped once with
an ephemeral .pth hook that installs the process-local patch before vLLM model
loading. After one deterministic request and path-evidence capture, the hook is
neutralized from outside the container and the original stock process is
restored and health-checked. No site-package source file is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hyperloom.r9700.stock_layout_live_candidate_smoke.v1"
CONTAINER = "inneros-vllm-canary-rocm10"
BASE = "http://127.0.0.1:8000/v1"
MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
PATCH = ROOT / "scripts" / "r9700_wna16_hybrid_patch.py"
PTH = "/opt/python/lib/python3.14/site-packages/r9700_hyperloom_candidate.pth"
REMOTE_PATCH = "/tmp/r9700_wna16_hybrid_patch.py"
PATH_EVIDENCE = "/tmp/r9700_candidate_paths.jsonl"


def run(argv: list[str], timeout: float = 120.0) -> dict[str, Any]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "argv": argv}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": type(exc).__name__ + ":" + str(exc), "argv": argv}


def health(timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        ids = [x.get("id") for x in payload.get("data", []) if isinstance(x, dict)]
        return {"ok": r.status == 200 and MODEL in ids, "status": r.status, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__ + ":" + str(exc)}


def wait_health(timeout_sec: float = 360.0) -> dict[str, Any]:
    started = time.perf_counter(); attempts = 0; last: dict[str, Any] = {}
    while time.perf_counter() - started < timeout_sec:
        attempts += 1; last = health()
        if last.get("ok"):
            return {"ok": True, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}
        time.sleep(2)
    return {"ok": False, "ready_sec": time.perf_counter() - started, "attempts": attempts, "health": last}


def container_state() -> dict[str, Any]:
    r = run(["docker", "inspect", CONTAINER], 30)
    if r.get("returncode") != 0:
        return {"ok": False, "stderr": str(r.get("stderr") or "")[-1000:]}
    row = json.loads(r["stdout"])[0]
    return {"ok": True, "running": bool(row.get("State", {}).get("Running")), "started_at": row.get("State", {}).get("StartedAt"), "pid": row.get("State", {}).get("Pid")}


def request_once(label: str) -> dict[str, Any]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly this token and nothing else: R9700_HYBRID_CHECK"}],
        "temperature": 0,
        "max_tokens": 32,
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read().decode())
        text = str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
        return {"ok": r.status == 200, "label": label, "status": r.status, "elapsed_sec": time.perf_counter() - started, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_len": len(text), "exact_expected": text.strip() == "R9700_HYBRID_CHECK", "usage": payload.get("usage")}
    except Exception as exc:
        return {"ok": False, "label": label, "elapsed_sec": time.perf_counter() - started, "error": type(exc).__name__ + ":" + str(exc)}


def parse_paths(text: str) -> list[dict[str, Any]]:
    rows=[]
    for line in text.splitlines():
        try:o=json.loads(line)
        except Exception:continue
        if isinstance(o,dict):rows.append(o)
    return rows


def digest(payload: dict[str, Any]) -> str:
    clean=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False));clean.pop("probe_sha256",None)
    return hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    captured=datetime.now(timezone.utc);out:dict[str,Any]={"schema":SCHEMA,"captured_at_utc":captured.isoformat(),"container":CONTAINER,"truth_boundary":{"full_model_candidate_load":True,"process_local_patch":True,"site_package_source_overwritten":False,"stock_restore_required":True,"not_e2e_performance_campaign":True}}
    temp_pth=None;empty_path=None;candidate_hook_written=False
    try:
        if not PATCH.exists():raise RuntimeError("patch_missing")
        if not health().get("ok"):raise RuntimeError("stock_service_not_healthy_before")
        out["stock_identity_before"]=container_state();out["baseline_before"]=request_once("stock_before")
        if not out["baseline_before"].get("ok"):raise RuntimeError("baseline_before_request_failed")
        exists=run(["docker","exec",CONTAINER,"test","-e",PTH],20)
        if exists.get("returncode")==0:raise RuntimeError("candidate_pth_already_exists")
        cp=run(["docker","cp",str(PATCH),f"{CONTAINER}:{REMOTE_PATCH}"],60)
        if cp.get("returncode")!=0:raise RuntimeError("patch_copy_failed")
        run(["docker","exec",CONTAINER,"rm","-f",PATH_EVIDENCE],20)
        hook='import sys,os; sys.path.insert(0,"/tmp"); os.environ["HYPERLOOM_R9700_EVIDENCE_FILE"]="/tmp/r9700_candidate_paths.jsonl"; import r9700_wna16_hybrid_patch as _r9700p; _r9700p.install_patch(force=True)\n'
        fd,temp_pth=tempfile.mkstemp(prefix="r9700_candidate_",suffix=".pth",dir=str(ROOT));os.close(fd);Path(temp_pth).write_text(hook,encoding="utf-8")
        cp_hook=run(["docker","cp",temp_pth,f"{CONTAINER}:{PTH}"],60)
        if cp_hook.get("returncode")!=0:raise RuntimeError("candidate_hook_copy_failed")
        candidate_hook_written=True
        restart=run(["docker","restart","--time","30",CONTAINER],90);out["candidate_restart"]={"returncode":restart.get("returncode"),"stderr_tail":str(restart.get("stderr") or "")[-1000:]}
        if restart.get("returncode")!=0:raise RuntimeError("candidate_restart_failed")
        ready=wait_health();out["candidate_health"]=ready;out["candidate_identity"]=container_state()
        if not ready.get("ok"):
            logs=run(["docker","logs","--tail","240",CONTAINER],30);out["candidate_logs_tail"]=(str(logs.get("stdout") or "")+str(logs.get("stderr") or ""))[-30000:]
            raise RuntimeError("candidate_service_not_healthy")
        out["candidate_request"]=request_once("candidate")
        if not out["candidate_request"].get("ok"):raise RuntimeError("candidate_request_failed")
        ev=run(["docker","exec",CONTAINER,"cat",PATH_EVIDENCE],30);rows=parse_paths(str(ev.get("stdout") or ""));out["path_evidence"]=rows;out["path_counts"]={p:sum(1 for r in rows if r.get("path")==p) for p in sorted({str(r.get("path")) for r in rows})}
        out["custom_path_seen"]=any(r.get("path")=="custom_small_w1_stock_w2" for r in rows);out["stock_fallback_seen"]=any(r.get("path")=="stock_full_fallback" for r in rows)
        out["candidate_output_matches_stock_hash"]=out["candidate_request"].get("text_sha256")==out["baseline_before"].get("text_sha256")
        if not out["custom_path_seen"]:raise RuntimeError("candidate_custom_path_not_observed")
        out["pass"]=True
    except Exception as exc:
        out["pass"]=False;out["error"]=type(exc).__name__+":"+str(exc)
        logs=run(["docker","logs","--tail","180",CONTAINER],30);out.setdefault("candidate_logs_tail",(str(logs.get("stdout") or "")+str(logs.get("stderr") or ""))[-24000:])
    finally:
        # Neutralize the bootstrap from outside the container. docker cp works
        # even if the candidate process exited and the container is stopped.
        if candidate_hook_written:
            fd,empty_path=tempfile.mkstemp(prefix="r9700_empty_",dir=str(ROOT));os.close(fd);Path(empty_path).write_text("",encoding="utf-8")
            neutral=run(["docker","cp",empty_path,f"{CONTAINER}:{PTH}"],60);out["restore_neutralize_returncode"]=neutral.get("returncode")
            state=container_state()
            action=["docker","restart","--time","30",CONTAINER] if state.get("running") else ["docker","start",CONTAINER]
            rr=run(action,90);out["restore_start_returncode"]=rr.get("returncode");ready=wait_health();out["restore_health"]=ready;out["stock_identity_after"]=container_state()
            if ready.get("ok"):
                run(["docker","exec",CONTAINER,"rm","-f",PTH,REMOTE_PATCH,PATH_EVIDENCE],30)
                out["baseline_after"]=request_once("stock_after")
            else:
                out["pass"]=False;out["restore_failure"]=True
            check=run(["docker","exec",CONTAINER,"test","!","-e",PTH],20) if ready.get("ok") else {"returncode":None}
            out["pth_removed"]=check.get("returncode")==0
            if not out.get("pth_removed") or not ready.get("ok"):out["pass"]=False
        for fp in (temp_pth,empty_path):
            if fp:
                try:Path(fp).unlink(missing_ok=True)
                except Exception:pass
    out["probe_sha256"]=digest(out);stamp=captured.strftime("%Y%m%dT%H%M%SZ");path=ROOT/'docs'/'evidence'/f'r9700_stock_layout_live_candidate_smoke_{stamp}.json';path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"pass":out.get("pass"),"path":str(path.relative_to(ROOT)),"sha256":out["probe_sha256"],"candidate_health":out.get("candidate_health"),"path_counts":out.get("path_counts"),"custom_path_seen":out.get("custom_path_seen"),"candidate_output_matches_stock_hash":out.get("candidate_output_matches_stock_hash"),"restore_health":out.get("restore_health"),"pth_removed":out.get("pth_removed"),"error":out.get("error")},indent=2,sort_keys=True));return 0 if out.get("pass") else 2

if __name__=="__main__":raise SystemExit(main())
