#!/usr/bin/env python3
"""Quantify the memory cost of vLLM Int4EmulationTritonExperts on the live R9700 stack.

Read-only probe. It does not allocate GPU tensors, restart vLLM, or modify the model.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.int4_emulation_memory_probe.v1'

def run(argv:list[str],timeout:float=120.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def parse_json_maybe(s:str)->Any:
    try:return json.loads(s)
    except Exception:return None

def sha(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False));c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':c,'truth_boundary':{'read_only':True,'gpu_tensor_allocations':False,'service_restarted':False,'official_support_claim':False}}
    if not c:
        payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        stats=run(['docker','stats','--no-stream','--format','{{json .}}',c],30)
        inspect=run(['docker','inspect',c,'--format','{{json .State}}'],30)
        smi=run(['rocm-smi','--showmeminfo','vram','--json'],30)
        code=textwrap.dedent(r'''
import json
from pathlib import Path
root=Path('/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ')
cfg=json.loads((root/'config.json').read_text())
keys=['model_type','hidden_size','intermediate_size','moe_intermediate_size','num_experts','num_experts_per_tok','num_hidden_layers','decoder_sparse_step','mlp_only_layers','shared_expert_intermediate_size']
print(json.dumps({k:cfg.get(k) for k in keys},sort_keys=True))
''')
        cfg_run=run(['docker','exec',c,'python3','-c',code],30)
        cfg={}
        for line in cfg_run['stdout'].splitlines():
            obj=parse_json_maybe(line)
            if isinstance(obj,dict):cfg=obj
        payload['docker_stats']=parse_json_maybe(stats['stdout']) or {'raw':stats['stdout']}
        payload['docker_state']=parse_json_maybe(inspect['stdout']) or {'raw':inspect['stdout']}
        payload['rocm_smi']=parse_json_maybe(smi['stdout']) or {'raw':smi['stdout']}
        payload['model_config']=cfg
        h=int(cfg.get('hidden_size') or 0); m=int(cfg.get('moe_intermediate_size') or 0); e=int(cfg.get('num_experts') or 0); L=int(cfg.get('num_hidden_layers') or 0)
        sparse_step=int(cfg.get('decoder_sparse_step') or 1); mlp_only=set(cfg.get('mlp_only_layers') or [])
        moe_layers=sum(1 for i in range(L) if i not in mlp_only and (sparse_step<=1 or (i+1)%sparse_step==0)) if L else 0
        per_expert=3*h*m
        total_expert_params=per_expert*e*moe_layers
        group=128
        # Approximate quantized storage: 4-bit weights + fp16 scales + packed 4-bit zero-points per group.
        int4_weight_bytes=total_expert_params/2
        groups=total_expert_params/group if group else 0
        scale_bytes=groups*2
        zp_bytes=groups/2
        packed_bytes=int4_weight_bytes+scale_bytes+zp_bytes
        bf16_bytes=total_expert_params*2
        payload['theoretical_moe_expert_storage']={
            'moe_layers':moe_layers,'experts_per_layer':e,'hidden_size':h,'moe_intermediate_size':m,
            'expert_params_total':total_expert_params,
            'packed_awq_bytes_approx':int(packed_bytes),'packed_awq_gib_approx':packed_bytes/(1024**3),
            'bf16_dequant_bytes':int(bf16_bytes),'bf16_dequant_gib':bf16_bytes/(1024**3),
            'bf16_vs_packed_ratio':(bf16_bytes/packed_bytes if packed_bytes else None),
            'incremental_bytes_if_fully_dequantized':int(max(0,bf16_bytes-packed_bytes)),
            'incremental_gib_if_fully_dequantized':max(0,bf16_bytes-packed_bytes)/(1024**3),
            'note':'Theoretical expert-weight storage only; actual runtime may use HMM/host spill, temporary buffers, or layer-specific handling.'
        }
        payload['pass']=stats['returncode']==0 and smi['returncode']==0 and bool(cfg)
    payload['probe_sha256']=sha(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'docs'/'evidence'/f'r9700_int4_emulation_memory_probe_{stamp}.json';out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(out),'probe_sha256':payload['probe_sha256'],'docker_stats':payload.get('docker_stats'),'theoretical_moe_expert_storage':payload.get('theoretical_moe_expert_storage'),'rocm_smi':payload.get('rocm_smi')},sort_keys=True))
    return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
