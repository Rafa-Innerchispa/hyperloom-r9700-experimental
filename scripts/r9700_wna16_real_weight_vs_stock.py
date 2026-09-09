#!/usr/bin/env python3
"""Compare the R9700 custom W1 correction kernel against stock vLLM Triton WNA16.

Uses actual Qwen3-Coder layer-0 AWQ expert weights, live-compatible FP16
activations, identical packed N-first W1/scales/zero-points, identical routing,
and 21 alternating paired HIP-event rounds per M. The resident vLLM process is
not patched or restarted.
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
SCHEMA = "hyperloom.r9700.wna16_real_weight_vs_stock.v1"
PATCH = ROOT / "scripts" / "r9700_wna16_hybrid_patch.py"
MODEL = "/models/QuantTrio__Qwen3-Coder-30B-A3B-Instruct-AWQ"


def run(argv: list[str], timeout: float = 480.0) -> dict[str, Any]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return {"returncode": p.returncode, "stdout": p.stdout[-360000:], "stderr": p.stderr[-80000:]}


def digest(payload: dict[str, Any]) -> str:
    clean = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    clean.pop("probe_sha256", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def child_code() -> str:
    return textwrap.dedent(
        rf'''
import json,statistics,sys,traceback,torch
import torch.nn.functional as F
import triton,triton.language as tl
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0,'/tmp')
import r9700_wna16_hybrid_patch as hp
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel,try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq

MODEL=Path({MODEL!r});SHARD=MODEL/'model-00001-of-00006.safetensors'
E=8;K=2048;I=768;N=1536;TOPK=8;GROUP=128
CFG={{'BM':16,'BN':128,'BK':32,'GK':128,'num_warps':4,'num_stages':1,'waves_per_eu':4}}
BLOCK_SHAPE=[0,GROUP]

def load_w1():
    qws=[];qzs=[];scs=[]
    with safe_open(str(SHARD),framework='pt',device='cpu') as f:
        for e in range(E):
            p=f'model.layers.0.mlp.experts.{{e}}.'
            gq=f.get_tensor(p+'gate_proj.qweight');uq=f.get_tensor(p+'up_proj.qweight')
            gz=f.get_tensor(p+'gate_proj.qzeros');uz=f.get_tensor(p+'up_proj.qzeros')
            gs=f.get_tensor(p+'gate_proj.scales');us=f.get_tensor(p+'up_proj.scales')
            qws.append(torch.cat([gq,uq],dim=1));qzs.append(torch.cat([gz,uz],dim=1));scs.append(torch.cat([gs,us],dim=1))
    return torch.stack(qws),torch.stack(qzs),torch.stack(scs)

def custom_launch(A,B,C,S,R,sid,se,npad):
    EM=sid.numel();grid=(triton.cdiv(EM,CFG['BM'])*triton.cdiv(N,CFG['BN']),)
    hp._r9700_w1_correction_kernel[grid](
        A,B,C,S,R,sid,se,npad,N,K,EM,A.size(0)*TOPK,
        A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),
        C.stride(1),C.stride(2),S.stride(0),S.stride(2),S.stride(1),
        R.stride(0),R.stride(2),R.stride(1),top_k=TOPK,
        BM=CFG['BM'],BN=CFG['BN'],BK=CFG['BK'],GK=CFG['GK'],USE_FP16=True,
        num_warps=CFG['num_warps'],num_stages=CFG['num_stages'],waves_per_eu=CFG['waves_per_eu'])

def pair(fa,fb,reps,rounds=21):
    for _ in range(8):fa();fb()
    torch.cuda.synchronize();av=[];bv=[];rat=[]
    for i in range(rounds):
        vals={{}}
        for lab in (('a','b') if i%2==0 else ('b','a')):
            fn=fa if lab=='a' else fb
            st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
            for _ in range(reps):fn()
            en.record();en.synchronize();vals[lab]=float(st.elapsed_time(en))/reps
        av.append(vals['a']);bv.append(vals['b']);rat.append(vals['b']/vals['a'])
    return av,bv,rat

def reference(A,Bbf,ids):
    out=torch.empty((A.size(0),TOPK,N),device='cuda',dtype=torch.float32)
    af=A.float()
    for m in range(A.size(0)):
        for j in range(TOPK):
            e=int(ids[m,j]);out[m,j]=af[m]@Bbf[e].float().transpose(0,1)
    return out

def main():
    out={{}}
    try:
        torch.manual_seed(20260909);dev=torch.cuda.get_device_name(0)
        if 'R9700' not in dev:raise RuntimeError('expected R9700, got '+dev)
        free0,total0=torch.cuda.mem_get_info();qw0,qz0,sc0=load_w1();qw=qw0.cuda();qz=qz0.cuda();sc=sc0.cuda()
        B,S,ZP=hp._awq_w13_to_packed_nfirst(qw,sc,qz);R=hp._build_correction(ZP,S)
        Bbf=_unpack_and_dequant_int4_awq(qw,sc,qz,transpose_output=True,output_dtype=torch.bfloat16)
        if list(B.shape)!=[E,N,K//2]:raise RuntimeError('unexpected packed W1 shape '+str(list(B.shape)))
        if list(S.shape)!=[E,N,K//GROUP]:raise RuntimeError('unexpected scale shape '+str(list(S.shape)))
        rows=[];base=torch.arange(TOPK,device='cuda',dtype=torch.int64)
        for M in (1,2,4,8,16):
            ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous();A=(torch.randn((M,K),device='cuda')*.1).to(torch.float16)
            csid,cse,cnpad=moe_align_block_size(ids,CFG['BM'],E,None)
            # Match the live W2 packed shape only for stock config selection; W1 timing is isolated.
            w2_shape=torch.Size((E,K,I//2))
            scfg=dict(try_get_optimal_moe_config(B.size(),w2_shape,TOPK,'int4_w4a16',M,block_shape=BLOCK_SHAPE))
            ssid,sse,snpad=moe_align_block_size(ids,scfg['BLOCK_SIZE_M'],E,None)
            Ca=torch.zeros((M,TOPK,N),device='cuda',dtype=torch.float16);Cb=torch.zeros_like(Ca)
            fa=lambda:custom_launch(A,B,Ca,S,R,csid,cse,cnpad)
            fb=lambda:invoke_fused_moe_wna16_triton_kernel(A,B,Cb,S,ZP,None,ssid,sse,snpad,False,TOPK,scfg,tl.float16,False,True,BLOCK_SHAPE)
            fa();fb();torch.cuda.synchronize();ref=reference(A,Bbf,ids);caf=Ca.float();cbf=Cb.float()
            ca_cos=float(F.cosine_similarity(caf.flatten(),ref.flatten(),dim=0));cb_cos=float(F.cosine_similarity(cbf.flatten(),ref.flatten(),dim=0));cross_cos=float(F.cosine_similarity(caf.flatten(),cbf.flatten(),dim=0))
            ca_rel=float(torch.linalg.vector_norm(caf-ref)/(torch.linalg.vector_norm(ref)+1e-12));cb_rel=float(torch.linalg.vector_norm(cbf-ref)/(torch.linalg.vector_norm(ref)+1e-12));cross_rel=float(torch.linalg.vector_norm(caf-cbf)/(torch.linalg.vector_norm(cbf)+1e-12))
            reps=max(12,min(48,96//M));av,bv,rat=pair(fa,fb,reps,21)
            rows.append({{'M':M,'custom_median_ms':float(statistics.median(av)),'stock_median_ms':float(statistics.median(bv)),'paired_median_speedup_vs_stock_x':float(statistics.median(rat)),'custom_wins':sum(x>1 for x in rat),'custom_wins_ge_1p02x':sum(x>=1.02 for x in rat),'custom_samples_ms':av,'stock_samples_ms':bv,'paired_speedups_x':rat,'custom_cosine_vs_ref':ca_cos,'stock_cosine_vs_ref':cb_cos,'custom_relative_l2_vs_ref':ca_rel,'stock_relative_l2_vs_ref':cb_rel,'custom_vs_stock_cosine':cross_cos,'custom_vs_stock_relative_l2':cross_rel,'stock_config':scfg,'reps_per_round':reps}})
        free1,total1=torch.cuda.mem_get_info();speeds=[r['paired_median_speedup_vs_stock_x'] for r in rows]
        numeric=all(r['custom_cosine_vs_ref']>=.999 and r['stock_cosine_vs_ref']>=.999 and r['custom_relative_l2_vs_ref']<=.02 and r['stock_relative_l2_vs_ref']<=.02 for r in rows)
        stable=all(r['custom_wins']>=11 for r in rows)
        out={{'pass':numeric,'promotion_gate_pass':bool(numeric and stable and statistics.median(speeds)>1.0 and rows[0]['paired_median_speedup_vs_stock_x']>=1.02),'device':dev,'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'source':{{'shard':str(SHARD),'layer':0,'experts':list(range(E)),'original_qweight_shape':list(qw0.shape),'original_qzeros_shape':list(qz0.shape),'original_scales_shape':list(sc0.shape)}},'live_compatible_contract':{{'activation_dtype':'torch.float16','quant_config_name':'int4_w4a16','packed_w1_shape':list(B.shape),'scale_shape':list(S.shape),'zp_shape':list(ZP.shape),'top_k':TOPK,'group_size':GROUP}},'custom_config':CFG,'rows':rows,'summary':{{'median_of_paired_medians_vs_stock_x':float(statistics.median(speeds)),'M1_speedup_vs_stock_x':rows[0]['paired_median_speedup_vs_stock_x'],'min_speedup_vs_stock_x':float(min(speeds)),'max_speedup_vs_stock_x':float(max(speeds)),'all_numeric_pass':numeric,'all_shapes_custom_wins_majority':stable}},'gpu_memory':{{'free_before_bytes':int(free0),'free_after_bytes':int(free1),'total_bytes':int(total1),'probe_delta_bytes':int(free0-free1)}},'truth_boundary':'actual Qwen layer-0 AWQ W1 weights; synthetic FP16 activations/routing matching measured live dtype/layout; custom W1 vs current stock vLLM Triton WNA16 W1; no resident server patch/restart and no E2E serving claim'}}
    except Exception as e:
        out={{'pass':False,'promotion_gate_pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-40000:]}}
    print(json.dumps(out,sort_keys=True))
main()
'''
    )


def main() -> int:
    live=discover_live_container(run=run);container=str((live.get('selected') or {}).get('Names') or '')
    payload:dict[str,Any]={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'container':container,'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'real_checkpoint_weights':True,'synthetic_live_compatible_fp16_activations':True,'official_support_claim':False}}
    if not container:payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        cp=run(['docker','cp',str(PATCH),f'{container}:/tmp/r9700_wna16_hybrid_patch.py'],60);ex=run(['docker','exec',container,'python3','-c',child_code()],480);parsed={}
        for line in ex['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['copy']={'returncode':cp['returncode'],'stderr_tail':cp['stderr'][-2000:]};payload['probe']=parsed;payload['docker_exec']={'returncode':ex['returncode'],'stderr_tail':ex['stderr'][-16000:]};payload['pass']=cp['returncode']==0 and ex['returncode']==0 and bool(parsed.get('pass'))
    payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'docs'/'evidence'/f'r9700_wna16_real_weight_vs_stock_{stamp}.json';path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'promotion_gate_pass':(payload.get('probe') or {}).get('promotion_gate_pass'),'path':str(path.relative_to(ROOT)),'sha256':payload['probe_sha256'],'summary':(payload.get('probe') or {}).get('summary'),'error':(payload.get('probe') or {}).get('error')},indent=2,sort_keys=True));return 0 if payload.get('pass') else 2

if __name__=='__main__':raise SystemExit(main())
