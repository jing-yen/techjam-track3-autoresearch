"""
T15 + buffer reuse -- standalone test of a standard memory-allocator-overhead
reduction on top of T15's both-boundary AddNorm fusion.

Base mechanism identical to v_triton_addnorm2.py (T15): T7's fused AddNorm
kernel extended to the second residual+norm boundary per layer -- see that
file's docstring for the full argument (real profiler evidence, padding
safety, etc.), unchanged here.

What's different: `fused_add_layernorm`'s wrapper called `torch.empty_like()`
TWICE on every single invocation (8 times per forward pass -- 4 layers x 2
boundaries), discarding the previous allocation every time. CUDA graph
capture (T17, U1') already solves this implicitly for the routes that use
it (a captured graph's private memory pool reuses the same addresses across
replays automatically) -- but this model does NOT use CUDA graphs, so it's
a clean, isolated test of buffer reuse as its OWN lever, standard allocator-
overhead reduction, not novel.

Two output buffers (`self._out_buf`, `self._sum_buf`) are lazily allocated
ONCE per model instance on first use and REUSED (overwritten in place) on
every subsequent call, instead of calling `torch.empty_like()` fresh each
time. Safe by construction, not just assumed: within one model instance,
every `fused_add_layernorm` call uses the exact same (n_rows, n_cols) shape
(batch*seq and d_model never change across layers/boundaries), so one
buffer pair covers every call. The apparent aliasing case -- call K's `sum`
output becomes call K+1's `residual` INPUT, and call K+1 ALSO writes its
own `sum` output into that same buffer -- is safe because the Triton
kernel's `tl.load` for `residual` fully materializes the value into
registers before the later `tl.store` to that same memory location; nothing
re-reads the pointer after the store, the classic safe in-place-update
pattern. Reasoned through here, but not trusted on reasoning alone: real
GPU correctness testing (this file's whole point) is what actually verifies
it, same discipline as every other change in this project.

CPU fallback: plain PyTorch ops, identical math, no buffer reuse (there's
nothing to reuse -- the CPU path never allocates a raw output buffer, it's
ordinary tensor arithmetic).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _fused_add_layernorm_kernel(
        residual_ptr, delta_ptr, weight_ptr, bias_ptr,
        out_ptr, sum_ptr,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        base = row * n_cols

        residual = tl.load(residual_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = residual + delta
        tl.store(sum_ptr + base + col_offsets, x, mask=mask)

        mean = tl.sum(x, axis=0) / n_cols
        xm = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xm * xm, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = xm * rstd

        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_norm * weight + bias
        tl.store(out_ptr + base + col_offsets, y, mask=mask)

    def fused_add_layernorm(residual: torch.Tensor, delta: torch.Tensor,
                             weight: torch.Tensor, bias: torch.Tensor, eps: float,
                             buf_owner=None):
        orig_shape = residual.shape
        n_cols = orig_shape[-1]
        residual2d = residual.reshape(-1, n_cols).contiguous()
        delta2d = delta.reshape(-1, n_cols).contiguous()
        n_rows = residual2d.shape[0]

        if buf_owner is not None:
            key = (n_rows, n_cols, residual2d.dtype)
            cached = getattr(buf_owner, "_addnorm_bufs", None)
            if cached is None or cached[0] != key:
                out = torch.empty_like(residual2d)
                new_sum = torch.empty_like(residual2d)
                buf_owner._addnorm_bufs = (key, out, new_sum)
            else:
                _, out, new_sum = cached
        else:
            out = torch.empty_like(residual2d)
            new_sum = torch.empty_like(residual2d)

        block_size = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        _fused_add_layernorm_kernel[grid](
            residual2d, delta2d, weight, bias, out, new_sum,
            n_cols, eps, BLOCK_SIZE=block_size,
        )
        return out.view(orig_shape), new_sum.view(orig_shape)
else:
    def fused_add_layernorm(residual, delta, weight, bias, eps, buf_owner=None):
        x = residual + delta
        y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return y, x


class _Attention(nn.Module):
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
        return self.out_proj(context)


class _Block(nn.Module):
    """Returns (post_ffn_residual, ffn_delta) UNCOMBINED -- the caller
    fuses their add with the NEXT norm (either the next block's norm1, or
    the model's final_norm), which is exactly the boundary T7 left unfused."""

    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, normed1, valid_token_mask, causal, buf_owner=None):
        attn_out = self.attention(normed1, valid_token_mask, causal)
        normed2, x = fused_add_layernorm(
            x, attn_out, self.norm2.weight, self.norm2.bias, self.norm2.eps,
            buf_owner=buf_owner)
        ffn_out = self.ffn_out(F.gelu(self.ffn_in(normed2), approximate="none"))
        return x, ffn_out


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _Block(config.d_model, config.num_heads, config.ffn_dim)
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
        causal = self.config.causal

        n0 = self.layers[0].norm1
        normed1 = F.layer_norm(x, (x.shape[-1],), n0.weight, n0.bias, n0.eps)

        for i, layer in enumerate(self.layers):
            x, ffn_delta = layer(x, normed1, eff_mask, causal, buf_owner=self)
            if i + 1 < len(self.layers):
                next_norm1 = self.layers[i + 1].norm1
                normed1, x = fused_add_layernorm(
                    x, ffn_delta, next_norm1.weight, next_norm1.bias, next_norm1.eps,
                    buf_owner=self)
            else:
                out, x = fused_add_layernorm(
                    x, ffn_delta, self.final_norm.weight, self.final_norm.bias, self.final_norm.eps,
                    buf_owner=self)

        if has_padding:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out
