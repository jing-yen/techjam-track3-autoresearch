"""
K1a -- staged block fusion, first stage: the whole FFN sub-block
(`norm2 -> ffn_in -> GELU -> ffn_out -> residual add`) in ONE Triton kernel
launch, per `docs/k1-spec.md`. Motivated by K1's own profiler-adjacent
finding on shape #2 (B=1, S=128, d=128): 374us / ~40 kernels = 9.3us/kernel,
against 0.22us of real math per 128x128x128 GEMM -- ~97% of each kernel's
wall time on this shape is NOT arithmetic. T10 (`v_triton_fused_ffn.py`)
already tried fusing just Linear+GELU and found a hand-rolled Triton GEMM
loses to cuBLASLt on shapes where cuBLASLt's tensor-core path is what's
actually running (the 3 amp-routed shapes). This is a DIFFERENT bet: on the
small S=128 family (NOT amp-routed, still on eager/compile/reduce cuBLAS
calls), the win isn't "beat cuBLAS's math", it's "eliminate 4 kernel
launches (LayerNorm, ffn_in-GEMM, GELU, ffn_out-GEMM, residual-add -- 5
ops) down to 1", where dispatch/HBM-round-trip overhead, not GEMM FLOPs,
is what's measured to dominate.

Fuses 5 ops into 1 kernel per (layer, row-block):
  norm2(x) -> ffn_in(normed) -> erf-GELU -> ffn_out(...) -> x + ...
`ffn_dim == d_model` on all 14 official shapes (confirmed in bench_harness's
OFFICIAL_SHAPES), and both are always a power of 2, so one BLOCK_D=D,
BLOCK_F=F tile covers the full width with no K-loop or masking on the
feature axis -- only the M (row) axis needs masking/tiling. Both weight
matrices (D*F fp32 each) fit resident in shared memory together at
these sizes (see docs/k1-spec.md's SMEM table) well under A100's
164KB/block limit, so there is no staging needed for this stage (K1a),
unlike the QKV-stage table in that spec.

Correctness invariants (docs/k1-spec.md, non-negotiable):
  - GELU is erf-exact (`tl.math.erf`), never a tanh approximation.
  - LayerNorm reduction in fp32 (loads cast up front, before any op).
  - `input_precision="ieee"` on both `tl.dot` calls -- same TF32-default
    bug T10 hit and fixed; not re-derived here, applied directly.
  - Padded rows: this kernel operates purely per-row (no cross-row mixing,
    unlike attention), so a padded row's own residual/FFN output is
    computed and stored normally, then zeroed by the OUTER block wrapper's
    existing masked_fill -- identical to how every other FFN implementation
    in this repo (plain, T7, T10) already handles padding. Not handled
    inside the kernel itself, deliberately, to match that existing pattern.

Attention stays untouched (`F.scaled_dot_product_attention`, same as T7/T10)
-- K1a only replaces the FFN half, per the spec's staged build order
(K1a first; K1b attention fusion only attempted if K1a wins).

Kill criteria (agreed in advance, docs/k1-spec.md): must beat shape #2's
routed `reduce` time (0.374ms) by >20%, or this closes without further
tuning. Falsify on any correctness failure not fixed within 30 minutes.

CPU fallback: plain PyTorch (LayerNorm + Linear + exact-GELU + Linear +
residual), identical math, used when no CUDA.
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
    _K1A_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_K1A_AUTOTUNE_CONFIGS, key=["M", "D", "F"])
    @triton.jit
    def _fused_ffn_block_kernel(
        x_ptr, ln_w_ptr, ln_b_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr,
        out_ptr,
        M, D, F,
        stride_xm, stride_xd,
        stride_w1f, stride_w1d,
        stride_w2d, stride_w2f,
        stride_om, stride_od,
        eps,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        d_idx = tl.arange(0, BLOCK_D)
        f_idx = tl.arange(0, BLOCK_F)

        # ---- load x block, LayerNorm (norm2) in fp32 ----
        x_ptrs = x_ptr + rows[:, None] * stride_xm + d_idx[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / D
        xm = x - mean[:, None]
        var = tl.sum(xm * xm, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = xm * rstd

        ln_w = tl.load(ln_w_ptr + d_idx).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + d_idx).to(tl.float32)
        normed = x_norm * ln_w[None, :] + ln_b[None, :]

        # ---- ffn_in: normed @ W1^T + b1, W1 is [F, D] (nn.Linear layout) ----
        w1_ptrs = w1_ptr + f_idx[:, None] * stride_w1f + d_idx[None, :] * stride_w1d
        w1 = tl.load(w1_ptrs).to(tl.float32)
        h = tl.dot(normed, tl.trans(w1), input_precision="ieee")
        b1 = tl.load(b1_ptr + f_idx).to(tl.float32)
        h = h + b1[None, :]

        # ---- exact erf GELU ----
        inv_sqrt2 = 0.7071067811865476
        h = h * 0.5 * (1.0 + tl.math.erf(h * inv_sqrt2))

        # ---- ffn_out: h @ W2^T + b2, W2 is [D, F] (nn.Linear layout) ----
        w2_ptrs = w2_ptr + d_idx[:, None] * stride_w2d + f_idx[None, :] * stride_w2f
        w2 = tl.load(w2_ptrs).to(tl.float32)
        o = tl.dot(h, tl.trans(w2), input_precision="ieee")
        b2 = tl.load(b2_ptr + d_idx).to(tl.float32)
        o = o + b2[None, :]

        # ---- residual add, store ----
        out = x + o
        out_ptrs = out_ptr + rows[:, None] * stride_om + d_idx[None, :] * stride_od
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])

    def fused_ffn_block(x: torch.Tensor, ln_weight, ln_bias, eps: float,
                         w1: torch.Tensor, b1: torch.Tensor,
                         w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]. Returns x + ffn_out(gelu(ffn_in(layernorm(x)))), fused."""
        orig_shape = x.shape
        d = orig_shape[-1]
        f = w1.shape[0]
        x2d = x.reshape(-1, d).contiguous()
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]),)
        _fused_ffn_block_kernel[grid](
            x2d, ln_weight, ln_bias,
            w1, b1, w2, b2,
            out,
            m, d, f,
            x2d.stride(0), x2d.stride(1),
            w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            out.stride(0), out.stride(1),
            eps,
            BLOCK_D=d, BLOCK_F=f,
        )
        return out.view(*orig_shape)
else:
    def fused_ffn_block(x, ln_weight, ln_bias, eps, w1, b1, w2, b2):
        normed = F.layer_norm(x, (x.shape[-1],), ln_weight, ln_bias, eps)
        h = F.gelu(F.linear(normed, w1, b1), approximate="none")
        return x + F.linear(h, w2, b2)


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
        x = fused_ffn_block(
            x, self.norm2.weight, self.norm2.bias, self.norm2.eps,
            self.ffn_in.weight, self.ffn_in.bias,
            self.ffn_out.weight, self.ffn_out.bias,
        )
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
