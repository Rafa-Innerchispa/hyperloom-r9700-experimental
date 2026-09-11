from __future__ import annotations
import json,subprocess,time,urllib.request
PORT=18011; NAME='hyperloom-r9700-p3-int4-repack-p18011'; start=time.monotonic(); last=''; attempts=0
while time.monotonic()-start<150:
 attempts+=1
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=4) as r:
   body=r.read().decode()[:1000]
   if r.status==200:
    print(json.dumps({'ok':True,'port':PORT,'status':r.status,'elapsed_sec':time.monotonic()-start,'attempts':attempts,'body':body},indent=2)); raise SystemExit(0)
 except Exception as e:last=f'{type(e).__name__}:{e}'
 p=subprocess.run(['docker','inspect','-f','{{.State.Status}} {{.State.ExitCode}}',NAME],text=True,capture_output=True)
 if p.returncode==0 and not p.stdout.strip().startswith('running'):
  print(json.dumps({'ok':False,'port':PORT,'container':p.stdout.strip(),'last':last,'elapsed_sec':time.monotonic()-start},indent=2)); raise SystemExit(3)
 time.sleep(5)
print(json.dumps({'ok':False,'port':PORT,'container':'still_running','last':last,'elapsed_sec':time.monotonic()-start},indent=2)); raise SystemExit(4)
