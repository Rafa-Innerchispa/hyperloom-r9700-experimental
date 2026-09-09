#!/usr/bin/env python3
"""Aggregate the latest three real-weight custom-vs-stock WNA16 campaigns."""
from __future__ import annotations
import hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_stock_gate_aggregate.v1'

def digest(x):
 c=json.loads(json.dumps(x,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None)
 return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main()->int:
 files=sorted((ROOT/'docs'/'evidence').glob('r9700_wna16_real_weight_vs_stock_*.json'))[-3:]
 out={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':[str(p.relative_to(ROOT)) for p in files]}
 try:
  if len(files)!=3:raise RuntimeError(f'expected 3 campaigns, found {len(files)}')
  docs=[json.loads(p.read_text()) for p in files];probes=[d.get('probe') or {} for d in docs]
  if not all(d.get('pass') and p.get('pass') and p.get('promotion_gate_pass') for d,p in zip(docs,probes)):raise RuntimeError('one or more campaigns did not pass')
  ms=[1,2,4,8,16];rows=[]
  for m in ms:
   rr=[]
   for p in probes:
    r=next(x for x in p['rows'] if x['M']==m);rr.append(r)
   speeds=[float(r['paired_median_speedup_vs_stock_x']) for r in rr]
   wins=[int(r['custom_wins']) for r in rr]
   rows.append({'M':m,'run_speedups_x':speeds,'median_speedup_x':statistics.median(speeds),'min_speedup_x':min(speeds),'max_speedup_x':max(speeds),'run_wins':wins,'all_runs_majority_wins':all(x>=11 for x in wins),'all_runs_numeric_pass':all(r['custom_cosine_vs_ref']>=.999 and r['stock_cosine_vs_ref']>=.999 and r['custom_relative_l2_vs_ref']<=.02 and r['stock_relative_l2_vs_ref']<=.02 for r in rr)})
  campaign_medians=[float(p['summary']['median_of_paired_medians_vs_stock_x']) for p in probes];m1=[float(p['summary']['M1_speedup_vs_stock_x']) for p in probes]
  out.update({'pass':all(r['all_runs_numeric_pass'] and r['all_runs_majority_wins'] for r in rows),'promotion_gate_pass':all(r['min_speedup_x']>1.0 for r in rows) and min(m1)>=1.02,'rows':rows,'summary':{'campaign_median_speedups_x':campaign_medians,'median_campaign_speedup_x':statistics.median(campaign_medians),'min_campaign_speedup_x':min(campaign_medians),'max_campaign_speedup_x':max(campaign_medians),'M1_run_speedups_x':m1,'M1_median_speedup_x':statistics.median(m1),'M1_min_speedup_x':min(m1),'all_three_promotion_pass':True},'truth_boundary':'aggregate of three consecutive child-process microbench campaigns; actual Qwen AWQ W1 weights and live-compatible FP16 contract; not independent full vLLM process starts and not E2E serving proof'})
 except Exception as e:out.update({'pass':False,'promotion_gate_pass':False,'error':type(e).__name__+':'+str(e)})
 out['probe_sha256']=digest(out);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_stock_gate_aggregate_{stamp}.json';p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'pass':out.get('pass'),'promotion_gate_pass':out.get('promotion_gate_pass'),'path':str(p.relative_to(ROOT)),'sha256':out['probe_sha256'],'summary':out.get('summary'),'rows':out.get('rows')},indent=2,sort_keys=True));return 0 if out.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
