from __future__ import annotations
import json, subprocess, time

SERVICE = "inneros-vllm-hyperloom-s3-canary.service"
TIMEOUT = 360

def run(args):
    return subprocess.run(args, text=True, capture_output=True, timeout=15)

start = time.monotonic(); attempts = 0; last = {}
while time.monotonic() - start < TIMEOUT:
    attempts += 1
    active = run(["systemctl", "--user", "is-active", SERVICE])
    show = run(["systemctl", "--user", "show", SERVICE, "-p", "ActiveState", "-p", "SubState", "-p", "Result"])
    last = {"is_active": active.stdout.strip(), "is_active_rc": active.returncode, "show": show.stdout.strip()}
    if active.stdout.strip() == "active":
        print(json.dumps({"ok": True, "attempts": attempts, "elapsed_sec": time.monotonic()-start, **last}, indent=2)); raise SystemExit(0)
    if "ActiveState=failed" in show.stdout or active.stdout.strip() == "failed":
        print(json.dumps({"ok": False, "failed": True, "attempts": attempts, "elapsed_sec": time.monotonic()-start, **last}, indent=2)); raise SystemExit(2)
    time.sleep(5)
print(json.dumps({"ok": False, "timeout": True, "attempts": attempts, "elapsed_sec": time.monotonic()-start, **last}, indent=2)); raise SystemExit(3)
