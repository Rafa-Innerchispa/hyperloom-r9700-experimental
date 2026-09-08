#!/usr/bin/env python3
"""RDNA4-specialized AutoAWQ WNA16 MoE kernel experiment.

Optimizes the generic vLLM WNA16 Triton path for the exact Qwen3 AutoAWQ
shape used on Radeon AI PRO R9700: group_size=128, int4 asymmetric weights.
The generic kernel materializes scale/zero-point loads over [BLOCK_K, BLOCK_N]
even though scale/zp are constant across each 128-wide K group. This candidate
loads scale/zp once per N lane and reuses them across four K=32 dot products.

This is an isolated microbenchmark. It does not patch the running vLLM service.
"""
from __future__ import annotations
import hashlib, json, subprocess, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from r9700_awq_backend_probe import discover_live_container

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='hyperloom.r9700.wna16_group128_rdna4_kernel.v1'

def run(argv:list[str],timeout:float=420.0)->dict[str,Any]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-180000:],'stderr':p.stderr[-40000:]}

def digest(payload:dict[str,Any])->str:
    c=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False)); c.pop('probe_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inside_code()->str:
    return textwrap.dedent(r'''
import json, statistics, time, traceback, torch
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.auto_awq import _REVERSE_AWQ_PACK_ORDER
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _unpack_and_dequant_int4_awq
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

@triton.jit
def rdna4_awq_group128_kernel(
    a_ptr,b_ptr,c_ptr,b_scale_ptr,b_zp_ptr,
    sorted_token_ids_ptr,expert_ids_ptr,num_tokens_post_padded_ptr,
    N:tl.constexpr,K:tl.constexpr,EM,num_valid_tokens,
    stride_am,stride_ak,stride_be,stride_bk,stride_bn,
    stride_cm,stride_cn,stride_bse,stride_bsk,stride_bsn,
    stride_bze,stride_bzk,stride_bzn,
    top_k:tl.constexpr,
    BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,
    SUB_K:tl.constexpr,GROUP_K:tl.constexpr,GROUP_SIZE_M:tl.constexpr,
):
    pid=tl.program_id(0)
    num_pid_m=tl.cdiv(EM,BLOCK_SIZE_M); num_pid_n=tl.cdiv(N,BLOCK_SIZE_N)
    num_pid_in_group=GROUP_SIZE_M*num_pid_n
    group_id=pid//num_pid_in_group; first_pid_m=group_id*GROUP_SIZE_M
    group_size_m=tl.minimum(num_pid_m-first_pid_m,GROUP_SIZE_M)
    pid_m=first_pid_m+((pid%num_pid_in_group)%group_size_m)
    pid_n=(pid%num_pid_in_group)//group_size_m
    npost=tl.load(num_tokens_post_padded_ptr)
    if pid_m*BLOCK_SIZE_M>=npost: return
    offs_token_id=pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M).to(tl.int64)
    offs_token=tl.load(sorted_token_ids_ptr+offs_token_id).to(tl.int64)
    token_mask=offs_token<num_valid_tokens
    expert=tl.load(expert_ids_ptr+pid_m).to(tl.int64)
    offs_n=pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N).to(tl.int64)
    nmask=offs_n<N
    if expert==-1:
        cptr=c_ptr+offs_token[:,None]*stride_cm+offs_n[None,:]*stride_cn
        tl.store(cptr,0.0,mask=token_mask[:,None]&nmask[None,:]); return
    acc=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32)
    offs_sub=tl.arange(0,SUB_K).to(tl.int64)
    # Qwen3 AWQ uses group_size=128 exactly. Load one scale/zp per N lane per group,
    # then reuse across four K=32 tiles instead of reloading a [K,N] replica.
    for g in tl.range(0,K,GROUP_K):
        gi=g//GROUP_K
        sp=b_scale_ptr+expert*stride_bse+offs_n*stride_bsn+gi*stride_bsk
        scale=tl.load(sp,mask=nmask,other=0.0).to(tl.float32)
        zpp=b_zp_ptr+expert*stride_bze+(offs_n//2)*stride_bzn+gi*stride_bzk
        zpbyte=tl.load(zpp,mask=nmask,other=0)
        zpshift=(offs_n%2)*4
        zp=((zpbyte>>zpshift)&0xF).to(tl.float32)
        for sub in tl.static_range(0,GROUP_K,SUB_K):
            kk=g+sub+offs_sub
            ap=a_ptr+(offs_token[:,None]//top_k)*stride_am+kk[None,:]*stride_ak
            a=tl.load(ap,mask=token_mask[:,None]&(kk[None,:]<K),other=0.0)
            bp=b_ptr+expert*stride_be+(kk[:,None]//2)*stride_bk+offs_n[None,:]*stride_bn
            packed=tl.load(bp,mask=(kk[:,None]<K)&nmask[None,:],other=0)
            shift=(kk[:,None]%2)*4
            b=((packed>>shift)&0xF).to(tl.float32)
            b=((b-zp[None,:])*scale[None,:]).to(tl.bfloat16)
            acc=tl.dot(a,b,acc=acc)
    out=acc.to(tl.bfloat16)
    cp=c_ptr+offs_token[:,None]*stride_cm+offs_n[None,:]*stride_cn
    tl.store(cp,out,mask=token_mask[:,None]&nmask[None,:])

def awq_to_triton(qw,scales,qz):
    E,K,Np=qw.shape; N=Np*8
    shifts=torch.arange(0,32,4,dtype=torch.int32,device=qw.device)
    reverse=torch.tensor(_REVERSE_AWQ_PACK_ORDER,dtype=torch.long,device=qw.device)
    vals=((qw.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E,K,N)
    nfirst=vals.transpose(1,2).contiguous(); wb=(nfirst[...,0::2]|(nfirst[...,1::2]<<4)).to(torch.uint8).contiguous()
    st=scales.transpose(1,2).contiguous(); G=qz.shape[1]
    zv=((qz.unsqueeze(-1)>>shifts)&0xF)[...,reverse].reshape(E,G,N).transpose(1,2).contiguous()
    zb=(zv[:,0::2,:]|(zv[:,1::2,:]<<4)).to(torch.uint8).contiguous()
    return wb,st,zb

def launch(A,B,C,S,Z,sorted_ids,experts,npost,top_k,cfg):
    EM=sorted_ids.numel(); M=A.size(0); N=B.size(1); K=A.size(1)
    grid=(triton.cdiv(EM,cfg['BLOCK_SIZE_M'])*triton.cdiv(N,cfg['BLOCK_SIZE_N']),)
    rdna4_awq_group128_kernel[grid](
        A,B,C,S,Z,sorted_ids,experts,npost,N,K,EM,M*top_k,
        A.stride(0),A.stride(1),B.stride(0),B.stride(2),B.stride(1),C.stride(1),C.stride(2),
        S.stride(0),S.stride(2),S.stride(1),Z.stride(0),Z.stride(2),Z.stride(1),
        top_k=top_k,BLOCK_SIZE_M=cfg['BLOCK_SIZE_M'],BLOCK_SIZE_N=cfg['BLOCK_SIZE_N'],
        SUB_K=32,GROUP_K=128,GROUP_SIZE_M=cfg.get('GROUP_SIZE_M',1),
        num_warps=cfg.get('num_warps',4),num_stages=cfg.get('num_stages',2),waves_per_eu=cfg.get('waves_per_eu',0))

def main():
    out={}
    try:
        torch.manual_seed(20260908); dev=torch.device('cuda')
        E=8;K=2048;N=1536;G=128;top_k=8
        qw=torch.randint(-(2**31),2**31-1,(E,K,N//8),dtype=torch.int32,device=dev)
        qz=torch.randint(-(2**31),2**31-1,(E,K//G,N//8),dtype=torch.int32,device=dev)
        scales=(torch.rand((E,K//G,N),device=dev,dtype=torch.float32)*0.02+0.002).to(torch.float16)
        B,S,Z=awq_to_triton(qw,scales,qz); Bbf=_unpack_and_dequant_int4_awq(qw,scales,qz,transpose_output=True,output_dtype=torch.bfloat16)
        dummy_w2=torch.empty((E,K,768),dtype=torch.bfloat16,device=dev)
        def setup(M,block_m):
            A=(torch.randn((M,K),device=dev,dtype=torch.float32)*.1).to(torch.bfloat16).contiguous()
            base=torch.arange(top_k,device=dev,dtype=torch.int64); ids=torch.stack([(base+i)%E for i in range(M)],0).contiguous()
            sid,se,npad=moe_align_block_size(ids,block_m,E,None)
            return A,sid,se,npad
        candidates=[]
        for bn in (32,64,128):
          for nw in (2,4,8):
            for ns in (1,2): candidates.append({'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':bn,'GROUP_SIZE_M':1,'num_warps':nw,'num_stages':ns,'waves_per_eu':0})
        M_tune=8; rows=[]
        for cfg in candidates:
            A,sid,se,npad=setup(M_tune,cfg['BLOCK_SIZE_M']); C=torch.zeros((M_tune,top_k,N),dtype=torch.bfloat16,device=dev)
            try:
                for _ in range(2): launch(A,B,C,S,Z,sid,se,npad,top_k,cfg)
                torch.cuda.synchronize(); reps=12; t0=time.perf_counter()
                for _ in range(reps): launch(A,B,C,S,Z,sid,se,npad,top_k,cfg)
                torch.cuda.synchronize(); ms=(time.perf_counter()-t0)*1000/reps
                rows.append({'config':cfg,'ms':ms,'valid':bool(torch.isfinite(C).all())})
            except Exception as e: rows.append({'config':cfg,'valid':False,'error':type(e).__name__+':'+str(e)})
        valid=[r for r in rows if r.get('valid')]; best=min(valid,key=lambda r:r['ms'])
        cfg=best['config']; compares=[]
        for M in (1,2,4,8,16,32):
            A,sid,se,npad=setup(M,cfg['BLOCK_SIZE_M']); Cq=torch.zeros((M,top_k,N),dtype=torch.bfloat16,device=dev); Ce=torch.zeros_like(Cq)
            ecfg=dict(try_get_optimal_moe_config(Bbf.size(),dummy_w2.size(),top_k,None,M))
            esid,ese,enpad=moe_align_block_size(torch.stack([(torch.arange(top_k,device=dev,dtype=torch.int64)+i)%E for i in range(M)],0).contiguous(),ecfg['BLOCK_SIZE_M'],E,None)
            def qcall(): launch(A,B,Cq,S,Z,sid,se,npad,top_k,cfg)
            def ecall(): invoke_fused_moe_triton_kernel(A,Bbf,Ce,None,None,None,esid,ese,enpad,False,top_k,ecfg,tl.bfloat16,False,False,False,False,False,None,None)
            qcall();ecall();torch.cuda.synchronize(); diff=(Cq.float()-Ce.float()).abs(); cos=float(torch.nn.functional.cosine_similarity(Cq.float().flatten(),Ce.float().flatten(),dim=0).item()); close=bool(torch.allclose(Cq.float(),Ce.float(),rtol=.08,atol=.08))
            for _ in range(3): qcall();ecall()
            torch.cuda.synchronize(); reps=max(5,min(16,64//M)); qt=[];et=[]
            for order in (('q','e'),('e','q'),('q','e')):
                for label in order:
                    fn=qcall if label=='q' else ecall; torch.cuda.synchronize();t0=time.perf_counter()
                    for _ in range(reps): fn()
                    torch.cuda.synchronize(); (qt if label=='q' else et).append((time.perf_counter()-t0)*1000/reps)
            qm=float(statistics.median(qt)); em=float(statistics.median(et)); compares.append({'M':M,'q_ms':qm,'emulation_ms':em,'speedup_x':em/qm,'latency_reduction_percent':(em-qm)/em*100,'allclose':close,'cosine':cos,'max_abs_error':float(diff.max().item())})
        out={'pass':all(r['allclose'] and r['cosine']>.999 for r in compares),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':getattr(torch.version,'hip',None),'best_tuning':best,'tuning_rows':rows,'comparisons':compares,'summary':{'median_speedup_x':float(statistics.median([r['speedup_x'] for r in compares])),'max_speedup_x':float(max(r['speedup_x'] for r in compares)),'min_speedup_x':float(min(r['speedup_x'] for r in compares))},'truth_boundary':'Standalone specialized Triton kernel on synthetic Qwen3 AutoAWQ-shaped data; running vLLM service not patched.'}
    except Exception as e: out={'pass':False,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()[-30000:]}
    print(json.dumps(out,sort_keys=True))
main()
''')

def main()->int:
    live=discover_live_container(run=run); c=str((live.get('selected') or {}).get('Names') or '')
    payload={'schema':SCHEMA,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'truth_boundary':{'service_restarted':False,'production_runtime_modified':False,'synthetic_data':True,'official_support_claim':False},'container':c}
    if not c: payload.update({'pass':False,'error':'no_vllm_container'})
    else:
        r=run(['docker','exec',c,'python3','-c',inside_code()],420); parsed={}
        for line in r['stdout'].splitlines():
            try:o=json.loads(line)
            except Exception:continue
            if isinstance(o,dict):parsed=o
        payload['probe']=parsed;payload['docker_exec']={'returncode':r['returncode'],'stderr_tail':r['stderr'][-10000:]};payload['pass']=bool(parsed.get('pass')) and r['returncode']==0
    payload['probe_sha256']=digest(payload);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');p=ROOT/'docs'/'evidence'/f'r9700_wna16_group128_rdna4_kernel_{stamp}.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':payload.get('pass'),'output':str(p),'probe_sha256':payload['probe_sha256'],'best_tuning':(payload.get('probe') or {}).get('best_tuning'),'summary':(payload.get('probe') or {}).get('summary'),'comparisons':(payload.get('probe') or {}).get('comparisons')},sort_keys=True));return 0 if payload.get('pass') else 2
if __name__=='__main__':raise SystemExit(main())
