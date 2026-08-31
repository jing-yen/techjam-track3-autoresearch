"""
S5+T6, pushed further -- batch-chunked forward pass under fp16 autocast,
CHUNK_SIZE=16 instead of v_chunked14_amp.py's chunk=8.

Same exact-chunking mechanism as candidates/v_chunked14.py (batch items are
independent -- no LayerNorm/attention/FFN op couples across the batch dim --
so splitting into groups and concatenating is exact, not approximate), same
fp16 autocast as v_chunked14_amp.py (T6, correctness-safe up to max_abs
~0.0017 on the shapes that have a reference -- journal iter 19).

v_chunked14_amp.py's own docstring: "chunk=8 as a first test of [fp16's
memory] headroom; if it holds well under the 79.25 GB ceiling, a future pass
can push it further." This is that pass. fp16 activations are roughly half
fp32's, so working memory should scale at ~1.3 GB/unit chunk size (half
v_chunked14.py's measured ~2.7 GB/unit at fp32) -- (79.25 - 24.4) / 1.3 =~ 42,
comfortably above B=32's own total, meaning fp16 alone might support the
FULL batch with no chunking at all. CHUNK_SIZE=16 here is a deliberately
intermediate step (not the theoretical max) to keep a real safety margin
against that estimate being wrong, while still testing meaningfully bigger
chunks than v_chunked14_amp.py's chunk=8. Never tested on GPU before this.

Correctness: same caveat as v_chunked14.py -- shape #14 has no reference, so
this is validated by running the identical mechanism on #8/#13 (which do),
not by a direct comparison on #14 itself.

Self-contained snapshot, same convention as v_compile.py / v_router.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly

CHUNK_SIZE = 16


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
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
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
        self._chunk_size = CHUNK_SIZE

    def _forward_chunk_impl(self, x, valid_token_mask):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _forward_chunk(self, x, valid_token_mask):
        if x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._forward_chunk_impl(x, valid_token_mask)
        return self._forward_chunk_impl(x, valid_token_mask)

    def forward(self, x, valid_token_mask=None):
        b = x.shape[0]
        if b <= self._chunk_size:
            return self._forward_chunk(x, valid_token_mask)
        outputs = []
        for start in range(0, b, self._chunk_size):
            end = min(start + self._chunk_size, b)
            x_chunk = x[start:end]
            mask_chunk = valid_token_mask[start:end] if valid_token_mask is not None else None
            outputs.append(self._forward_chunk(x_chunk, mask_chunk))
        return torch.cat(outputs, dim=0)
