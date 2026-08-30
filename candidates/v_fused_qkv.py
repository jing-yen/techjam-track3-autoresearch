"""
Pre-built variant: SDPA + fused QKV projection.

Replaces the three separate q/k/v Linear(d,d) projections with a single
Linear(d, 3d), then splits. Fewer, larger matmuls / fewer kernel launches.
Numerically identical to separate projections (the fused weight is just the row
concatenation of the three), so correctness is essentially exact.

Because the parameter names change (attention.qkv instead of q_proj/k_proj/
v_proj), strict weight copy no longer lines up — so this file sets
STRICT_WEIGHT_COPY=False and provides a custom copy_model_weights() that maps
the baseline's three projections into the fused weight. This also exercises the
harness's non-strict copy path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

# Match the baseline's full-fp32 matmul precision (see v_compile.py).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

STRICT_WEIGHT_COPY = False  # names differ (fused qkv); use copy_model_weights below


class FusedQKVAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, valid_token_mask=None, causal=False):
        b, s, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
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
        self.attention = FusedQKVAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Reuse the baseline's norms/ffn/out_proj by reference; the fused qkv gets
        # its weights via copy_model_weights() below.
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = OptimizedBlock(config.d_model, config.num_heads, config.ffn_dim)
            blk.norm1 = base_block.norm1
            blk.norm2 = base_block.norm2
            blk.ffn_in = base_block.ffn_in
            blk.ffn_out = base_block.ffn_out
            blk.attention.out_proj = base_block.attention.out_proj
            opt_layers.append(blk)
        self.layers = opt_layers

    def forward(self, x, valid_token_mask=None):
        # B1: collapse the all-True (no-padding) mask to None so attention takes
        # the fused is_causal path instead of building an additive mask.
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def copy_model_weights(baseline, optimized):
    """Map the baseline's separate q/k/v projections into the fused qkv weight,
    and copy everything else by matching name + shape."""
    b = baseline.state_dict()
    target = optimized.state_dict()
    new = {}
    n_layers = len(optimized.layers)
    for i in range(n_layers):
        p = f"layers.{i}.attention."
        new[p + "qkv.weight"] = torch.cat(
            [b[p + "q_proj.weight"], b[p + "k_proj.weight"], b[p + "v_proj.weight"]], dim=0)
        new[p + "qkv.bias"] = torch.cat(
            [b[p + "q_proj.bias"], b[p + "k_proj.bias"], b[p + "v_proj.bias"]], dim=0)
    for k, v in b.items():
        if k in target and k not in new and target[k].shape == v.shape:
            new[k] = v
    missing = [k for k in target if k not in new]
    if missing:
        raise RuntimeError(f"fused-qkv copy missing keys: {missing}")
    optimized.load_state_dict(new, strict=True)
