"""
K1a-persistent -- same fused FFN block as v_triton_k1a_ffn.py
(norm2 -> ffn_in -> GELU -> ffn_out -> residual, one Triton kernel), but
with a persistent-kernel / grid-stride-loop launch instead of one program
per row-block.

Motivated by two real, measured failures elsewhere in this project, not by
a paper: K1's original megakernel used `grid=(1,)` and was ~11x slower
because it used exactly 1 of A100's 108 SMs (journal iter 35). T17
(`v_triton_addnorm_fused.py`) applied this project's OWN already-proven
AddNorm kernel to the `fused` route and regressed shapes #9/#10/#12 by
~30% (job 778892) -- likely the opposite failure mode: `grid=(cdiv(M,
BLOCK_M),)` for those shapes launches hundreds of small single-purpose
programs, each RE-LOADING the full weight tile from HBM independently, with
per-launch/per-block scheduling overhead that a mature library kernel
(ATen's own LayerNorm, or cuBLAS's GEMM) doesn't pay per-occurrence the
same way. Both are real Triton-kernel-design mistakes with opposite
symptoms -- too few programs vs too many, in this project's own results.

This is the standard fix used by real production kernels (this exact
pattern is Triton's own official persistent-matmul tutorial, and what
vLLM/SGLang's custom kernels do): launch `grid=(min(NUM_SMS,
num_blocks_m),)` -- never more programs than the GPU has SMs, never more
than the work actually needs -- and have each program loop over MULTIPLE
row-blocks via `tl.num_programs`-strided iteration. Two real, distinct
benefits over the non-persistent version, not just "fewer launches":
  1. Occupancy matches hardware size for large M (was 1 for K1's grid=(1,)
     case; was hundreds for T17's grid=(n_rows,) case).
  2. The weight tiles (W1, W2, LayerNorm weight/bias) are loaded ONCE per
     PROGRAM and reused across every row-block that program handles,
     instead of once per row-block -- real, additional HBM-traffic
     reduction beyond just launch-count, unique to this design.

Same correctness invariants as v_triton_k1a_ffn.py (erf GELU,
input_precision="ieee", fp32 LayerNorm reduction, size-guarded fallback for
D/F > 256 -- shape #8's d=1024 caused a real Triton compiler crash/OOM in
the non-persistent version when this guard was missing, see TODO.md K1a).

CPU fallback: plain PyTorch, identical math.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True

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

_NUM_SMS = 108  # A100-80 SM count (confirmed via DeviceProperties dump, K4 diagnostic)
_MAX_SINGLE_TILE_DIM = 256  # same guard as v_triton_k1a_ffn.py -- see docstring


if _HAS_TRITON:
    _K1A_PERSIST_CONFIGS = [
        triton.Config({"BLOCK_M": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_K1A_PERSIST_CONFIGS, key=["M", "D", "F"])
    @triton.jit
    def _persistent_fused_ffn_kernel(
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
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        d_idx = tl.arange(0, BLOCK_D)
        f_idx = tl.arange(0, BLOCK_F)

        # Weights loaded ONCE per program, reused across every row-block
        # this program handles below -- the real, additional saving over
        # the non-persistent kernel (which reloads them per row-block).
        w1_ptrs = w1_ptr + f_idx[:, None] * stride_w1f + d_idx[None, :] * stride_w1d
        w1 = tl.load(w1_ptrs).to(tl.float32)
        b1 = tl.load(b1_ptr + f_idx).to(tl.float32)
        w2_ptrs = w2_ptr + d_idx[:, None] * stride_w2d + f_idx[None, :] * stride_w2f
        w2 = tl.load(w2_ptrs).to(tl.float32)
        b2 = tl.load(b2_ptr + d_idx).to(tl.float32)
        ln_w = tl.load(ln_w_ptr + d_idx).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + d_idx).to(tl.float32)

        num_blocks_m = tl.cdiv(M, BLOCK_M)
        for block_idx in range(pid, num_blocks_m, num_programs):
            rows = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = rows < M

            x_ptrs = x_ptr + rows[:, None] * stride_xm + d_idx[None, :] * stride_xd
            x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

            mean = tl.sum(x, axis=1) / D
            xm = x - mean[:, None]
            var = tl.sum(xm * xm, axis=1) / D
            rstd = 1.0 / tl.sqrt(var + eps)
            x_norm = xm * rstd[:, None]
            normed = x_norm * ln_w[None, :] + ln_b[None, :]

            h = tl.dot(normed, tl.trans(w1), input_precision="ieee")
            h = h + b1[None, :]
            inv_sqrt2 = 0.7071067811865476
            h = h * 0.5 * (1.0 + tl.math.erf(h * inv_sqrt2))

            o = tl.dot(h, tl.trans(w2), input_precision="ieee")
            o = o + b2[None, :]

            out = x + o
            out_ptrs = out_ptr + rows[:, None] * stride_om + d_idx[None, :] * stride_od
            tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])

    def fused_ffn_block(x: torch.Tensor, ln_weight, ln_bias, eps: float,
                         w1: torch.Tensor, b1: torch.Tensor,
                         w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        d = orig_shape[-1]
        f = w1.shape[0]
        if d > _MAX_SINGLE_TILE_DIM or f > _MAX_SINGLE_TILE_DIM:
            normed = F.layer_norm(x, (d,), ln_weight, ln_bias, eps)
            h = F.gelu(F.linear(normed, w1, b1), approximate="none")
            return x + F.linear(h, w2, b2)
        x2d = x.reshape(-1, d).contiguous()
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        grid = lambda META: (min(_NUM_SMS, triton.cdiv(m, META["BLOCK_M"])),)
        _persistent_fused_ffn_kernel[grid](
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
