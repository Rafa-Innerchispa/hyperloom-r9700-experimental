#!/usr/bin/env python3
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = [
    "r9700_vllm_wna16_bounded_tuner_20260909T051731Z.json",
    "r9700_tuned_config_live_smoke_20260909T052123Z.json",
    "r9700_tuned_config_live_smoke_20260909T052523Z.json",
]

def run(argv):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[-2000:]}

for name in EVIDENCE:
    path = ROOT / "docs" / "evidence" / name
    try:
        print(f"--- {name} ---")
        print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"file": name, "error": f"{type(exc).__name__}: {exc}"}))

ps = run(["docker", "ps", "--filter", "name=inneros-vllm", "--filter", "status=running", "--format", "{{.Names}}"])
names = [x for x in ps["stdout"].splitlines() if x]
print(json.dumps({"containers": names}))
if names:
    cid = names[0]
    inspect = run(["docker", "inspect", cid, "--format", "{{json .Config.Env}}|{{json .Mounts}}"])
    env_raw, _, mounts_raw = inspect["stdout"].partition("|")
    try: envs = json.loads(env_raw)
    except Exception: envs = []
    prefixes = ("VLLM_ROCM_", "GPU_MAX_HW_QUEUES=", "HIP_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES=")
    filtered_env = [x for x in envs if x.startswith(prefixes)]
    try: mounts = json.loads(mounts_raw)
    except Exception: mounts = []
    safe_mounts = [{"type":m.get("Type"),"destination":m.get("Destination"),"source":m.get("Source")} for m in mounts if m.get("Destination")=="/models"]
    code = "import importlib.metadata as m,glob,site,os,json; vers={};\nfor x in ('amd-aiter','aiter','vllm','torch','triton'):\n try: vers[x]=m.version(x)\n except Exception: pass\nprint(json.dumps(vers,sort_keys=True)); hits=[];\nfor d in site.getsitepackages():\n for p in glob.glob(os.path.join(d,'*.pth')):\n  try:\n   t=open(p,errors='ignore').read();\n   if any(k in t.lower() for k in ('r9700','sameproc','hyperloom')): hits.append({'path':p,'content':t[:300]})\n  except Exception: pass\nprint('PTH_HITS='+json.dumps(hits))"
    inner = run(["docker", "exec", cid, "python", "-c", code])
    print(json.dumps({"container":cid,"filtered_env":filtered_env,"model_mounts":safe_mounts,"versions_and_pth":inner}, indent=2))
