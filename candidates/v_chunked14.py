"""
S5 -- batch-chunked forward pass, built to clear shape #14's memory ceiling.

Shape #14 (B=32, S=100000, D=1024) OOMs at 73.85 GB of 79.25 GB during the
*optimized* model's own activation memory, not attention -- SDPA/mem-efficient
already eliminates the baseline's O(S^2) score matrix (docs/sdpa_backend_probe.json).
B=32 is 32 independent sequences: nothing in this model couples batch items --
LayerNorm normalizes per-token over d_model, attention is causal/masked within
each sequence only, FFN is per-token. So splitting the batch dimension into
groups and concatenating the outputs is exact, not an approximation: identical
floating-point operations per output element, just fewer "in flight" at once.

Chunk size 4 keeps shape #14's peak at ~35 GB (10.7 GB working + 12.2 GB input
+ 12.2 GB output) against 79.25 GB available -- the measured OOM was only 6 GB
over, chunking clears it by ~44 GB. Full arithmetic: docs/research-shape14.md.

Correctness: shape #14 has no reference (the baseline needs ~18.6 TB to
materialize its score matrix, per B5/PROGRAM.md), so it can't be checked
directly. This candidate instead forces the *same* chunking mechanism on
every shape it's run against, including #8 and #13 -- which DO have a
reference. If chunked-vs-baseline survives the harness's normal correctness
gate there (same gate, no special-casing, no widened tolerance), the chunking
mechanism itself is proven exact, and #14 inherits that proof even though it
can't be checked directly.

Attention path: plain SDPA (candidates/best.py's seed), the fewest moving
parts. This candidate exists to prove batch-chunking is exact and to clear
shape #14's memory ceiling -- not to compete on speed with the four
already-faster candidates in v_router.py on shapes 1-13.

Self-contained snapshot, same convention as v_compile.py / v_router.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly

CHUNK_SIZE = 4


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

    def _forward_chunk(self, x, valid_token_mask):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def forward(self, x, valid_token_mask=None):
        b = x.shape[0]
        if b <= self._chunk_size:
            return self._forward_chunk(x, valid_token_mask)
        # Batch items are independent (no LayerNorm/attention/FFN op couples
        # across the batch dim), so chunking and concatenating is exact.
        outputs = []
        for start in range(0, b, self._chunk_size):
            end = min(start + self._chunk_size, b)
            x_chunk = x[start:end]
            mask_chunk = valid_token_mask[start:end] if valid_token_mask is not None else None
            outputs.append(self._forward_chunk(x_chunk, mask_chunk))
        return torch.cat(outputs, dim=0)
