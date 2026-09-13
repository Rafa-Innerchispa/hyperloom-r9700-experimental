from __future__ import annotations
import json
import subprocess
import time

NAME = "hyperloom-r9700-p4-s3-verbose-p18017"

def run(argv, timeout=30):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)

inspect = run(["docker", "inspect", "-f", "{{.Name}} {{.State.Status}} {{.State.ExitCode}}", NAME])
removed = False
remove = None
if inspect.returncode == 0:
    remove = run(["docker", "rm", "-f", NAME], timeout=60)
    removed = remove.returncode == 0

time.sleep(2)
vram = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
out = {
    "name": NAME,
    "existed": inspect.returncode == 0,
    "inspect": inspect.stdout.strip(),
    "removed": removed,
    "remove_rc": None if remove is None else remove.returncode,
    "remove_stdout": "" if remove is None else remove.stdout.strip(),
    "remove_stderr": "" if remove is None else remove.stderr.strip(),
    "vram": vram.stdout.strip(),
}
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if (inspect.returncode != 0 or removed) else 2)
