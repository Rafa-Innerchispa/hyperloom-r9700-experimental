from __future__ import annotations
import concurrent.futures, datetime as dt, hashlib, json, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL='QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ'
label=sys.argv[1] if len(sys.argv)>1 else 'active'

def req(prompt,max_tokens=96):
    payload={'model':MODEL,'prompt':prompt,'max_tokens':max_tokens,'temperature':0.0,'seed':7,'stream':True,'stream_options':{'include_usage':True}}
    r=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    t0=time.monotonic(); first=None; text=[]; usage={}
    with urllib.request.urlopen(r,timeout=240) as resp:
        status=resp.status
        for raw in resp:
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data:'): continue
            body=line[5:].strip()
            if body=='[DONE]': break
            try:o=json.loads(body)
            except Exception:continue
            if o.get('usage'):usage=o['usage']
            if o.get('choices'):
                s=o['choices'][0].get('text') or ''
                if s and first is None:first=time.monotonic()
                text.append(s)
    end=time.monotonic(); s=''.join(text); ct=usage.get('completion_tokens') or max(1,len(s.split()))
    return {'ok':status==200,'status':status,'elapsed_sec':end-t0,'ttft_sec':None if first is None else first-t0,'completion_tokens':ct,'prompt_tokens':usage.get('prompt_tokens'),'decode_tok_s':None if first is None else ct/max(end-first,1e-9),'text_sha256':hashlib.sha256(s.encode()).hexdigest(),'text_len':len(s)}

def c4(prompt,max_tokens=96):
    t=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: rows=[f.result() for f in [ex.submit(req,prompt+f'\nCase {i}:',max_tokens) for i in range(4)]]
    wall=time.monotonic()-t; total=sum(x['completion_tokens'] for x in rows)
    return {'wall_sec':wall,'completion_tokens':total,'aggregate_tok_s':total/max(wall,1e-9),'rows':rows}

def health():
    try:
      with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as r:return {'ok':r.status==200,'status':r.status,'body':r.read().decode()[:2000]}
    except Exception as e:return {'ok':False,'error':f'{type(e).__name__}: {e}'}
short='Implement a Python LRU cache with get and put methods. Return only code.'
longp=(' x'*6000)+'\nSummarize the repeated marker pattern in one sentence.'
out={'schema':'hyperloom.r9700.active_measure.v1','label':label,'captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'health':health()}
if out['health']['ok']:
    out['correctness']=req('Return exactly a compact Python function add(a,b) that returns a+b.',64)
    out['c1']=req(short,128); out['c4']=c4(short,128); out['long_context']=req(longp,64)
p=ROOT/'docs'/'evidence'/f"r9700_active_measure_{label}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(str(p)); print(json.dumps({k:out.get(k) for k in ('health','correctness','c1','c4','long_context')},indent=2,sort_keys=True))
