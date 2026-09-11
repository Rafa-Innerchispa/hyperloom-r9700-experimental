from __future__ import annotations
import concurrent.futures,datetime as dt,hashlib,json,statistics,subprocess,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MODEL='QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ'; PORT=8011; BASE=f'http://127.0.0.1:{PORT}'
PROMPT='Implement a Python LRU cache with get and put methods. Return only code.'
ROUNDS=8

def one(prompt):
 payload={'model':MODEL,'prompt':prompt,'max_tokens':128,'temperature':0.0,'seed':7,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request(BASE+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 t0=time.monotonic(); first=None; chunks=[]; usage={}
 with urllib.request.urlopen(req,timeout=120) as resp:
  for raw in resp:
   line=raw.decode('utf-8','replace').strip()
   if not line.startswith('data:'):continue
   b=line[5:].strip()
   if b=='[DONE]':break
   try:o=json.loads(b)
   except Exception:continue
   if o.get('usage'):usage=o['usage']
   if o.get('choices'):
    s=o['choices'][0].get('text') or ''
    if s and first is None:first=time.monotonic()
    chunks.append(s)
 end=time.monotonic(); s=''.join(chunks); ct=usage.get('completion_tokens') or 0
 return {'tokens':ct,'elapsed_sec':end-t0,'ttft_sec':None if first is None else first-t0,'decode_tok_s':None if first is None else ct/max(end-first,1e-9),'sha256':hashlib.sha256(s.encode()).hexdigest(),'text_len':len(s)}

def sample_gpu(stop,rows):
 while not stop.is_set():
  p=subprocess.run(['rocm-smi','--showuse','--showclocks','--showpower','--showtemp','--json'],text=True,capture_output=True,timeout=5)
  try: card=json.loads(p.stdout).get('card0',{})
  except Exception: card={}
  rows.append({'t':time.time(),'card0':card})
  stop.wait(0.35)

def c4(round_no):
 telemetry=[]; stop=threading.Event(); th=threading.Thread(target=sample_gpu,args=(stop,telemetry),daemon=True); th.start(); t0=time.monotonic()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
   rows=[f.result() for f in [ex.submit(one,PROMPT+f'\nCase {i}:') for i in range(4)]]
 finally:
  stop.set(); th.join(timeout=2)
 wall=time.monotonic()-t0; total=sum(r['tokens'] for r in rows)
 return {'round':round_no,'wall_sec':wall,'aggregate_tok_s':total/max(wall,1e-9),'rows':rows,'telemetry':telemetry}

svc=subprocess.run(['systemctl','--user','is-active','inneros-vllm-canary-rocm10.service'],text=True,capture_output=True).stdout.strip()
rounds=[]
for i in range(ROUNDS):
 rounds.append(c4(i+1)); time.sleep(1.5)
vals=[r['aggregate_tok_s'] for r in rounds]
hashes=[[x['sha256'] for x in r['rows']] for r in rounds]
out={'schema':'hyperloom.r9700.phase3.c4_stability.v1','captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'port':PORT,'round_count':ROUNDS,'stock_service_state':svc,'rounds':rounds,'aggregate':{'median_tok_s':statistics.median(vals),'min_tok_s':min(vals),'max_tok_s':max(vals),'range_over_median_pct':(max(vals)-min(vals))/statistics.median(vals)*100,'unique_hash_vectors':len({tuple(x) for x in hashes})}}
p=ROOT/'docs'/'evidence'/f"r9700_phase3_c4_stability_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"; p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(str(p)); print(json.dumps({'aggregate':out['aggregate'],'rounds':[{'round':r['round'],'tok_s':r['aggregate_tok_s'],'hashes':[x['sha256'] for x in r['rows']]} for r in rounds]},indent=2))
