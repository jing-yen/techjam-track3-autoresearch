"""
S5+L2 -- batch-chunked forward pass for shape #14, chunks issued on 2
concurrent CUDA streams instead of fully sequential.

Same exact-chunking mechanism as v_chunked14.py (batch items are independent,
splitting and concatenating is mathematically exact). Difference: chunks are
still slices of an already-GPU-resident tensor, so there's no host<->device
transfer to hide (the classic "prefetch tile n+1 while tile n computes"
pattern from the kernel-optimization literature doesn't directly apply
here -- there's nothing to prefetch). What alternating 2 streams *can* still
buy is SM occupancy: on a single default stream, chunk N+1's kernels cannot
begin until every one of chunk N's kernels has finished, even if chunk N
alone doesn't saturate the GPU's SMs. Two streams let the scheduler
interleave both chunks' kernels when there's idle capacity.

Memory: S5's chunk=4 estimate is ~10.7 GB working set per chunk. Running 2
chunks concurrently roughly doubles the LIVE working-set overlap (not the
whole peak, which also includes the fixed ~24.4 GB input+output) --
estimated worst case ~45.7 GB against 79.25 GB available, still with
headroom. This is a research probe, not a proven win -- see leaderboard.md/
TODO.md L2 for the measured before/after.

Self-contained snapshot, same convention as v_chunked14.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly

CHUNK_SIZE = 4
N_STREAMS = 2


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
        self._streams = None

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
        if not x.is_cuda:
            outputs = []
            for start in range(0, b, self._chunk_size):
                end = min(start + self._chunk_size, b)
                x_chunk = x[start:end]
                mask_chunk = valid_token_mask[start:end] if valid_token_mask is not None else None
                outputs.append(self._forward_chunk(x_chunk, mask_chunk))
            return torch.cat(outputs, dim=0)

        if self._streams is None:
            self._streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]

        starts = list(range(0, b, self._chunk_size))
        outputs = [None] * len(starts)
        for i, start in enumerate(starts):
            end = min(start + self._chunk_size, b)
            stream = self._streams[i % N_STREAMS]
            with torch.cuda.stream(stream):
                x_chunk = x[start:end]
                mask_chunk = valid_token_mask[start:end] if valid_token_mask is not None else None
                outputs[i] = self._forward_chunk(x_chunk, mask_chunk)
        torch.cuda.synchronize()
        return torch.cat(outputs, dim=0)
