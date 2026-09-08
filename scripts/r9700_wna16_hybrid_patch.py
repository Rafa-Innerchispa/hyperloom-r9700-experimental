#!/usr/bin/env python3
"""Process-local experimental AutoAWQ MoE backend for R9700/gfx1201.

Design:
- W1 is converted from AutoAWQ to N-first packed INT4 + scales + packed zp.
- A load-time FP16 correction tensor (zero_point * scale) is built once by the
  Experts object and used by the small-token algebraic W1 Triton kernel.
- W1 small-token policy: custom correction kernel for num_tokens <= 16.
- W1 larger-token policy: generic vLLM WNA16 Triton fallback using packed INT4.
- W2 remains BF16 emulation because measured W2 packed/algebraic kernels are
  slower on the current R9700 workload.

The patch is installed only in the current Python process. It does not edit
vLLM site-packages. Removing the entrypoint/env returns to stock behavior.
"""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
    _resize_cache,
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

PATCH_NAME = "r9700_autoawq_hybrid_v1"
SMALL_TOKEN_LIMIT = int(os.environ.get("HYPERLOOM_R9700_W1_SMALL_TOKEN_LIMIT", "16"))


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
    """W1 packed/correction + W1 packed fallback + W2 BF16 hybrid experts."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        if self.quant_config.w1_zp is None:
            raise ValueError("R9700 hybrid W1 requires AWQ zero points")
        self.w1_correction = _build_correction(
            self.quant_config.w1_zp, self.w1_scale
        )

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
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"{PATCH_NAME} currently validates Qwen/SILU only, got {activation}"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(
                f"{PATCH_NAME} supports fp16/bf16 activations, got {hidden_states.dtype}"
            )
        if w1.dtype != torch.uint8:
            raise RuntimeError(f"hybrid W1 must be packed uint8, got {w1.dtype}")
        if w2.dtype != torch.bfloat16:
            raise RuntimeError(f"hybrid W2 must be BF16 fallback, got {w2.dtype}")

        E = w1.size(0)
        num_tokens = hidden_states.size(0)
        N = w1.size(1)  # gate+up output width
        K = hidden_states.size(1)
        top_k_num = topk_ids.size(1)
        if global_num_experts == -1:
            global_num_experts = E

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        act_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, act_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        # Small-token W1 path. This is the only performance-positive region
        # currently promoted. Larger blocks stay packed and use vLLM generic WNA16.
        if num_tokens <= SMALL_TOKEN_LIMIT:
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
        else:
            fake_w2_packed_size = torch.Size(
                (w2.size(0), w2.size(1), w2.size(2) // 2)
            )
            config1 = try_get_optimal_moe_config(
                w1.size(),
                fake_w2_packed_size,
                top_k_num,
                self.quant_config.config_name(hidden_states.dtype),
                num_tokens,
                block_shape=self.block_shape,
            )
            sorted1, experts1, padded1 = moe_align_block_size(
                topk_ids, config1["BLOCK_SIZE_M"], global_num_experts, expert_map
            )
            compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
            invoke_fused_moe_wna16_triton_kernel(
                hidden_states,
                w1,
                intermediate_cache1,
                self.w1_scale,
                self.quant_config.w1_zp,
                None,
                sorted1,
                experts1,
                padded1,
                False,
                top_k_num,
                config1,
                compute_type=compute_type,
                use_int8_w8a16=False,
                use_int4_w4a16=True,
                block_shape=self.block_shape,
            )

        self.activation(activation, intermediate_cache2, intermediate_cache1.view(-1, N))

        # W2 intentionally stays on the current BF16 emulation path.
        # Use an unquantized Triton config and its own alignment block size.
        fake_w1_bf16_size = torch.Size((E, N, K))
        config2 = try_get_optimal_moe_config(
            fake_w1_bf16_size,
            w2.size(),
            top_k_num,
            None,
            num_tokens,
            block_shape=None,
        )
        sorted2, experts2, padded2 = moe_align_block_size(
            topk_ids, config2["BLOCK_SIZE_M"], global_num_experts, expert_map
        )
        compute_type = tl.bfloat16
        # Existing Int4EmulationTritonExperts operates on BF16 weights. Keep
        # this call BF16; if upstream prepare delivers FP16 activations, cast
        # only the W2 activation cache rather than changing packed W1 storage.
        w2_input = intermediate_cache2
        if w2_input.dtype != torch.bfloat16:
            w2_input = w2_input.to(torch.bfloat16)
        invoke_fused_moe_triton_kernel(
            w2_input,
            w2,
            intermediate_cache3,
            None,
            None,
            topk_weights,
            sorted2,
            experts2,
            padded2,
            not apply_router_weight_on_input,
            1,
            config2,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
            B_bias=None,
        )
        self.moe_sum(intermediate_cache3, output)


class R9700AutoAWQMoEMethod(AutoAWQMoEMethod):
    """AutoAWQ method that prepares hybrid W1 packed / W2 BF16 weights."""

    def __init__(self, quant_config, moe):
        # Avoid stock oracle selection because AutoAWQ is deliberately blocked
        # from Triton today. Everything else follows AutoAWQ's own setup.
        from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

        FusedMoEMethodBase.__init__(self, moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("R9700 hybrid currently supports AutoAWQ int4 only")
        if not self.quant_config.zero_point:
            raise ValueError("R9700 hybrid currently targets asymmetric AutoAWQ")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        self.use_marlin = False
        self.wna16_moe_backend = WNA16MoEBackend.TRITON
        self.experts_cls = R9700HybridWNA16Experts

    def process_weights_after_loading(self, layer) -> None:
        w13, w13_scale, w13_zp = _awq_w13_to_packed_nfirst(
            layer.w13_qweight,
            layer.w13_scales,
            layer.w13_qzeros,
        )
        w2 = _awq_w2_to_bf16(
            layer.w2_qweight,
            layer.w2_scales,
            layer.w2_qzeros,
        )
        dummy = torch.ones(1, dtype=torch.float16, device=w13.device)

        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight
        replace_parameter(layer, "w13_scales", w13_scale)
        replace_parameter(layer, "w2_scales", dummy)
        _replace_or_register_parameter(layer, "w13_qzeros", w13_zp)

        # Keep the raw W2 qzero parameter untouched but make it semantically
        # unreachable from the hybrid quant config. This minimizes invasive
        # layer surgery during the first isolated prototype.
        self._setup_kernel(layer)

    def get_fused_moe_quant_config(self, layer):
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            group_size=self.quant_config.group_size,
            num_bits=self.quant_config.weight_bits,
            w1_zp=getattr(layer, "w13_qzeros", None),
            w2_zp=None,
            w1_bias=None,
            w2_bias=None,
        )


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
