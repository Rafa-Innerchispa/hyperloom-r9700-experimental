#!/usr/bin/env python3
"""Process-local experimental AutoAWQ MoE backend for R9700/gfx1201.

Design:
- W1 is converted from AutoAWQ to N-first packed INT4 + scales + packed zp.
- A load-time FP16 correction tensor (zero_point * scale) is built once by the
  Experts object and used by the small-token algebraic W1 Triton kernel.
- W1 small-token policy: custom correction kernel for num_tokens <= 16.
- W1 larger-token policy: stock vLLM Triton WNA16.
- W2 always remains on the stock packed Triton WNA16 path.
- AutoAWQ weight conversion/layout is inherited from stock vLLM unchanged.

The patch is installed only in the current Python process. It does not edit
vLLM site-packages. Removing the entrypoint/env returns to stock behavior.
"""
from __future__ import annotations

import json
import os
import mmap
import signal
import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _unpack_and_dequant_int4_awq,
    make_wna16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQMoEMethod,
    _REVERSE_AWQ_PACK_ORDER,
    _replace_or_register_parameter,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import scalar_types

PATCH_NAME = "r9700_autoawq_stock_layout_hybrid_v6"
SMALL_TOKEN_LIMIT = int(os.environ.get("HYPERLOOM_R9700_W1_SMALL_TOKEN_LIMIT", "16"))
EVIDENCE_FILE = os.environ.get("HYPERLOOM_R9700_EVIDENCE_FILE", "")
RUNTIME_GATE_PATH = os.environ.get("HYPERLOOM_R9700_RUNTIME_GATE", "")
_RUNTIME_GATE = None
_RUNTIME_GATE_FD = None
if RUNTIME_GATE_PATH:
    try:
        _RUNTIME_GATE_FD = os.open(RUNTIME_GATE_PATH, os.O_RDONLY)
        _RUNTIME_GATE = mmap.mmap(_RUNTIME_GATE_FD, 1, access=mmap.ACCESS_READ)
    except Exception:
        _RUNTIME_GATE = None

SIGNAL_GATE_ENABLED = os.environ.get("HYPERLOOM_R9700_SIGNAL_GATE", "") == "1"
_RUNTIME_SIGNAL_CUSTOM = os.environ.get("HYPERLOOM_R9700_SIGNAL_GATE_INITIAL", "1") != "0"

def _signal_custom(_signum, _frame):
    global _RUNTIME_SIGNAL_CUSTOM
    _RUNTIME_SIGNAL_CUSTOM = True

def _signal_stock(_signum, _frame):
    global _RUNTIME_SIGNAL_CUSTOM
    _RUNTIME_SIGNAL_CUSTOM = False

if SIGNAL_GATE_ENABLED:
    try:
        signal.signal(signal.SIGUSR1, _signal_custom)
        signal.signal(signal.SIGUSR2, _signal_stock)
    except Exception:
        pass

def _runtime_custom_enabled() -> bool:
    if SIGNAL_GATE_ENABLED:
        return bool(_RUNTIME_SIGNAL_CUSTOM)
    if _RUNTIME_GATE is None:
        return True
    try:
        return _RUNTIME_GATE[0] == 1
    except Exception:
        return True


@triton.jit
def _r9700_w1_correction_kernel(
    A,
    B,
    C,
    S,
    R,
    sorted_ids,
    expert_ids,
    npost_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_se,
    stride_sk,
    stride_sn,
    stride_re,
    stride_rk,
    stride_rn,
    top_k: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GK: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(EM, BM)
    num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n
    pid_n = pid % num_n
    npost = tl.load(npost_ptr)
    if pid_m * BM >= npost:
        return

    sid = pid_m * BM + tl.arange(0, BM).to(tl.int64)
    tok = tl.load(sorted_ids + sid).to(tl.int64)
    token_mask = tok < num_valid
    exp = tl.load(expert_ids + pid_m).to(tl.int64)
    out_n = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    n_mask = out_n < N

    if exp == -1:
        tl.store(
            C + tok[:, None] * stride_cm + out_n[None, :] * stride_cn,
            0.0,
            mask=token_mask[:, None] & n_mask[None, :],
        )
        return

    offs_k = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)

    for group_start in tl.range(0, K, GK):
        gi = group_start // GK
        scale = tl.load(
            S + exp * stride_se + out_n * stride_sn + gi * stride_sk,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        correction = tl.load(
            R + exp * stride_re + out_n * stride_rn + gi * stride_rk,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        qacc = tl.zeros((BM, BN), tl.float32)
        asum = tl.zeros((BM,), tl.float32)

        for sub in tl.static_range(0, GK, BK):
            kk = group_start + sub + offs_k
            a = tl.load(
                A
                + (tok[:, None] // top_k) * stride_am
                + kk[None, :] * stride_ak,
                mask=token_mask[:, None] & (kk[None, :] < K),
                other=0.0,
            )
            packed = tl.load(
                B
                + exp * stride_be
                + (kk[:, None] // 2) * stride_bk
                + out_n[None, :] * stride_bn,
                mask=(kk[:, None] < K) & n_mask[None, :],
                other=0,
            )
            q = ((packed >> ((kk[:, None] % 2) * 4)) & 15)
            if USE_FP16:
                a_dot = a.to(tl.float16)
                q_dot = q.to(tl.float16)
            else:
                a_dot = a.to(tl.bfloat16)
                q_dot = q.to(tl.bfloat16)
            qacc = tl.dot(a_dot, q_dot, acc=qacc)
            asum += tl.sum(a.to(tl.float32), axis=1)

        acc += qacc * scale[None, :] - asum[:, None] * correction[None, :]

    out_ptr = C + tok[:, None] * stride_cm + out_n[None, :] * stride_cn
    if USE_FP16:
        value = acc.to(tl.float16)
    else:
        value = acc.to(tl.bfloat16)
    tl.store(out_ptr, value, mask=token_mask[:, None] & n_mask[None, :])


def _awq_w13_to_packed_nfirst(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AutoAWQ K-first/output-packed int32 -> Triton N-first uint8."""
    E, K, n_packed = qweight.shape
    N = n_packed * 8
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    reverse = torch.tensor(
        _REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=qweight.device
    )

    vals = ((qweight.unsqueeze(-1) >> shifts) & 15)[..., reverse]
    vals = vals.reshape(E, K, N).transpose(1, 2).contiguous()  # E,N,K
    packed = (vals[..., 0::2] | (vals[..., 1::2] << 4)).to(torch.uint8)

    scale_nfirst = scales.transpose(1, 2).contiguous()  # E,N,G
    G = qzeros.shape[1]
    zvals = ((qzeros.unsqueeze(-1) >> shifts) & 15)[..., reverse]
    zvals = zvals.reshape(E, G, N).transpose(1, 2).contiguous()  # E,N,G
    zp_packed = (zvals[:, 0::2, :] | (zvals[:, 1::2, :] << 4)).to(torch.uint8)
    return packed.contiguous(), scale_nfirst, zp_packed.contiguous()


def _awq_w2_to_bf16(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    unpacked = _unpack_and_dequant_int4_awq(
        qweight,
        scales,
        qzeros,
        transpose_output=False,
        output_dtype=torch.bfloat16,
    )
    return unpacked.permute(0, 2, 1).contiguous()


def _build_correction(zp_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed pairwise N zero points -> FP16 zero_point*scale tensor E,N,G."""
    low = (zp_packed & 15).to(torch.uint8)
    high = ((zp_packed >> 4) & 15).to(torch.uint8)
    E, half_N, G = zp_packed.shape
    zp = torch.empty((E, half_N * 2, G), dtype=torch.uint8, device=zp_packed.device)
    zp[:, 0::2, :] = low
    zp[:, 1::2, :] = high
    return (zp.float() * scale.float()).to(torch.float16).contiguous()


class R9700HybridWNA16Experts(TritonWNA16Experts):
    """Stock WNA16 Experts with one guarded RDNA4 small-M W1 substitution."""

    def __init__(self, moe_config, quant_config):
        # Install the A/B signal handlers late, while the actual MoE experts are
        # being constructed in EngineCore. This avoids later vLLM startup code
        # replacing handlers registered too early by sitecustomize.
        if SIGNAL_GATE_ENABLED:
            try:
                signal.signal(signal.SIGUSR1, _signal_custom)
                signal.signal(signal.SIGUSR2, _signal_stock)
            except Exception:
                pass
        super().__init__(moe_config, quant_config)
        if self.quant_config.w1_zp is None:
            raise ValueError("R9700 custom W1 requires asymmetric WNA16 zero points")
        self.w1_correction = _build_correction(
            self.quant_config.w1_zp, self.w1_scale
        )
        self.last_path = "uninitialized"
        self._evidence_paths: set[str] = set()

    def _record_path(self, path, hidden_states, w1, w2, topk_ids) -> None:
        if not EVIDENCE_FILE or path in self._evidence_paths:
            return
        self._evidence_paths.add(path)
        try:
            row = {
                "pid": os.getpid(),
                "path": path,
                "M": int(hidden_states.size(0)),
                "hidden_states_dtype": str(hidden_states.dtype),
                "w1_shape": list(w1.shape),
                "w2_shape": list(w2.shape),
                "topk_shape": list(topk_ids.shape),
                "patch": PATCH_NAME,
            }
            with open(EVIDENCE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            pass

    def _small_w1(
        self,
        hidden_states,
        w1,
        out,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num: int,
    ) -> None:
        K = hidden_states.size(1)
        N = w1.size(1)
        EM = sorted_token_ids.numel()
        grid = (triton.cdiv(EM, 16) * triton.cdiv(N, 128),)
        use_fp16 = hidden_states.dtype == torch.float16
        _r9700_w1_correction_kernel[grid](
            hidden_states,
            w1,
            out,
            self.w1_scale,
            self.w1_correction,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            EM,
            hidden_states.size(0) * top_k_num,
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(2),
            w1.stride(1),
            out.stride(1),
            out.stride(2),
            self.w1_scale.stride(0),
            self.w1_scale.stride(2),
            self.w1_scale.stride(1),
            self.w1_correction.stride(0),
            self.w1_correction.stride(2),
            self.w1_correction.stride(1),
            top_k=top_k_num,
            BM=16,
            BN=128,
            BK=32,
            GK=128,
            USE_FP16=use_fp16,
            num_warps=4,
            num_stages=1,
            waves_per_eu=4,
        )

    def _custom_eligible(self, hidden_states, w1, w2, topk_ids, activation) -> bool:
        return bool(
            hidden_states.size(0) <= SMALL_TOKEN_LIMIT
            and activation == MoEActivation.SILU
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and hidden_states.is_contiguous()
            and w1.dtype == torch.uint8
            and w2.dtype == torch.uint8
            and hidden_states.size(-1) == 2048
            and w1.size(1) == 1536
            and topk_ids.dim() == 2
            and topk_ids.size(1) == 8
            and self.block_shape is not None
            and len(self.block_shape) == 2
            and self.block_shape[0] == 0
            and self.block_shape[1] == 128
        )

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        # Same-process A/B gate. When disabled, use the exact stock implementation
        # while preserving the same process/GPU state to eliminate spawn lottery.
        if not _runtime_custom_enabled():
            self.last_path = "runtime_gate_stock"
            self._record_path(self.last_path, hidden_states, w1, w2, topk_ids)
            return super().apply(
                output=output, hidden_states=hidden_states, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids, activation=activation,
                global_num_experts=global_num_experts, expert_map=expert_map,
                a1q_scale=a1q_scale, a2_scale=a2_scale, workspace13=workspace13,
                workspace2=workspace2, expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Fail safely to the exact stock implementation for every unproven shape,
        # dtype, activation or routing contract.
        if not self._custom_eligible(hidden_states, w1, w2, topk_ids, activation):
            self.last_path = "stock_full_fallback"
            self._record_path(self.last_path, hidden_states, w1, w2, topk_ids)
            return super().apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )
        compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        # Only W1 is replaced, and only in the validated small-M region.
        sorted1, experts1, padded1 = moe_align_block_size(
            topk_ids, 16, global_num_experts, expert_map
        )
        self._small_w1(
            hidden_states,
            w1,
            intermediate_cache1,
            sorted1,
            experts1,
            padded1,
            top_k_num,
        )
        self.last_path = "custom_small_w1_stock_w2"
        self._record_path(self.last_path, hidden_states, w1, w2, topk_ids)

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        # W2 is byte-for-byte stock WNA16 layout and uses the stock kernel.
        # Stock reuses one aligned routing table for W1 and W2. Our custom W1
        # is validated with BM=16, so when stock selects the same block size
        # reuse the exact same alignment instead of launching a second align.
        if int(config["BLOCK_SIZE_M"]) == 16:
            sorted2, experts2, padded2 = sorted1, experts1, padded1
        else:
            sorted2, experts2, padded2 = moe_align_block_size(
                topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
            )
        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted2,
            experts2,
            padded2,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )
        self.moe_sum(intermediate_cache3, output)


class R9700AutoAWQMoEMethod(AutoAWQMoEMethod):
    """Stock AutoAWQ conversion with a guarded R9700 Experts substitution."""

    def __init__(self, quant_config, moe):
        # The current ROCm10/vLLM runtime already selects Triton WNA16 for this
        # R9700/Qwen contract. Keep the complete stock conversion path and only
        # replace the Experts implementation used by the modular kernel.
        super().__init__(quant_config, moe)
        if self.wna16_moe_backend != WNA16MoEBackend.TRITON:
            raise RuntimeError(
                f"{PATCH_NAME} requires stock Triton WNA16 selection; got "
                f"{self.wna16_moe_backend}"
            )
        self.experts_cls = R9700HybridWNA16Experts


def install_patch(force: bool = False) -> dict[str, str]:
    """Install process-local monkeypatches needed by AutoAWQ model loading."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if not force and "R9700" not in device.upper():
        raise RuntimeError(f"{PATCH_NAME} refused non-R9700 device: {device!r}")

    import vllm.model_executor.layers.fused_moe.oracle.int_wna16 as int_wna16
    import vllm.model_executor.layers.quantization.auto_awq as auto_awq

    # make_wna16_moe_kernel constructs its allowed tuple from this module global.
    int_wna16.TritonWNA16Experts = R9700HybridWNA16Experts
    # AutoAWQConfig.get_quant_method resolves this module global at call time.
    auto_awq.AutoAWQMoEMethod = R9700AutoAWQMoEMethod

    return {
        "patch": PATCH_NAME,
        "device": device,
        "auto_awq_method": auto_awq.AutoAWQMoEMethod.__name__,
        "triton_experts": int_wna16.TritonWNA16Experts.__name__,
        "small_token_limit": str(SMALL_TOKEN_LIMIT),
    }
