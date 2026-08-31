"""
T10-fp16 -- fused Linear + bias + GELU Triton kernel, explicit fp16 compute.

Real profiler trace (journal iter 36, tools/profile_shape8.py) showed shape
#8's actual bottleneck under v_router2's `amp` route: 55.6% fp16 tensor-core
GEMM + 17.1% fp32<->fp16 casting overhead. T10 (candidates/v_triton_fused_ffn.py)
forces `input_precision="ieee"` fp32 accumulation uniformly -- correctness-safe
across every shape, but on #8 it competes against fp16 tensor cores using fp32
math, getting neither the throughput nor the casting savings. Confirmed by a
real @triton.autotune sweep (journal iter 37): 12 configs, correctness held,
but shape 8 was UNCHANGED (~0.83x either way) -- proving this is a precision
mismatch, not a tuning problem.

This variant fixes that directly: explicitly casts x and the weight to fp16
before the GEMM (mirroring what torch.autocast does for a real nn.Linear --
important because autocast only auto-wraps *registered* PyTorch ops, not a
raw Triton kernel call, so calling the fp32 T10 kernel from inside an
autocast context does NOT automatically get it fp16 inputs; the cast has to
be explicit). tl.dot with genuinely-fp16 operands uses fp16 Tensor Cores with
fp32 accumulation by default -- no TF32 ambiguity to guard against (TF32 is
specifically about how *fp32* inputs get truncated; it doesn't apply once
inputs are already fp16), so `input_precision="ieee"` is correctly omitted
here, not overlooked.

GELU is still computed in the fp32 accumulator (erf, exact) before the final
cast to fp16 for storage -- matching T6/T7's established pattern of keeping
reductions/epilogues in fp32 even under an fp16-compute regime.

Scope: standalone candidate, not yet integrated into v_router2's `amp` route.
Test on shape #8 specifically before considering that integration -- if this
doesn't clearly beat both the fp32 T10 (0.83x) and the current `amp` route's
plain nn.Linear+GELU on #8, there's no reason to route anything to it.

CPU fallback: plain fp32 Linear+GELU (same as T10's fallback; fp16 autocast
has no CPU path here).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    _FFN_FP16_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_FFN_FP16_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_linear_gelu_fp16_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # fp32 accumulator regardless of input dtype -- standard mixed-
        # precision recipe (fp16 compute, fp32 accumulate), same as what
        # torch.autocast's own fp16 matmul path does internally.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k0 < K)
            b_mask = (offs_k[:, None] + k0 < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # already fp16 (caller cast it)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # already fp16
            # No input_precision="ieee" here: that flag only concerns how
            # *fp32* inputs are handled (ieee-fp32 vs TF32-truncated). With
            # genuinely fp16 operands there's no TF32 ambiguity -- tl.dot
            # already uses fp16 Tensor Cores with fp32 accumulation.
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

        # Exact erf GELU, computed in the fp32 accumulator (matches T6/T7's
        # established pattern: fp16 compute, fp32 epilogue/reduction).
        inv_sqrt2 = 0.7071067811865476
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * inv_sqrt2))

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

    def fused_linear_gelu_fp16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """x: [..., K] fp32, weight: [N, K] fp32 (nn.Linear layout), bias: [N] fp32.
        Casts x and weight to fp16 explicitly before the GEMM (autocast does
        NOT do this automatically for a raw kernel call), returns fp32 --
        matching what a real autocast-wrapped nn.Linear+GELU would hand back
        to the next (fp32-kept) op in the block."""
        orig_shape = x.shape
        k = orig_shape[-1]
        x2d = x.reshape(-1, k).contiguous().to(torch.float16)
        m = x2d.shape[0]
        n = weight.shape[0]
        w_t = weight.t().contiguous().to(torch.float16)

        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(n, META["BLOCK_N"]))
        _fused_linear_gelu_fp16_kernel[grid](
            x2d, w_t, bias, out,
            m, n, k,
            x2d.stride(0), x2d.stride(1),
            w_t.stride(0), w_t.stride(1),
            out.stride(0), out.stride(1),
        )
        return out.view(*orig_shape[:-1], n)
else:
    def fused_linear_gelu_fp16(x, weight, bias):
        return F.gelu(F.linear(x, weight, bias), approximate="none")


class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, valid_token_mask=None, causal=False):
        b, s, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if valid_token_mask is None:
            context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            neg_inf = float("-inf")
            key_bias = torch.zeros(b, 1, 1, s, dtype=q.dtype, device=x.device).masked_fill(
                ~valid_token_mask[:, None, None, :], neg_inf)
            if causal:
                cb = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(diagonal=1)
                attn_mask = key_bias + torch.zeros(s, s, dtype=q.dtype, device=x.device).masked_fill(cb, neg_inf)
            else:
                attn_mask = key_bias
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        context = context.transpose(1, 2).contiguous().view(b, s, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class OptimizedBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SDPASelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        normed = self.norm2(x)
        hidden = fused_linear_gelu_fp16(normed, self.ffn_in.weight, self.ffn_in.bias)
        x = x + self.ffn_out(hidden)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = OptimizedBlock(config.d_model, config.num_heads, config.ffn_dim)
            blk.norm1 = base_block.norm1
            blk.norm2 = base_block.norm2
            blk.ffn_in = base_block.ffn_in
            blk.ffn_out = base_block.ffn_out
            blk.attention.q_proj = base_block.attention.q_proj
            blk.attention.k_proj = base_block.attention.k_proj
            blk.attention.v_proj = base_block.attention.v_proj
            blk.attention.out_proj = base_block.attention.out_proj
            opt_layers.append(blk)
        self.layers = opt_layers

    def forward(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
