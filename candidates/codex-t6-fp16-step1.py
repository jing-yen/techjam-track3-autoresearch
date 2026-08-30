"""T6: CUDA autocast to float16 while retaining fp32 parameters and inputs.

This candidate deliberately leaves the process-wide TF32 policy untouched so
the baseline continues to run with the organizer/harness defaults.  CUDA
autocast selects float16 for eligible operations; CPU execution stays fp32.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig


STRICT_WEIGHT_COPY = True


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
            key_bias = torch.zeros(
                b, 1, 1, s, dtype=q.dtype, device=x.device
            ).masked_fill(~valid_token_mask[:, None, None, :], neg_inf)
            if causal:
                causal_mask = torch.ones(
                    s, s, dtype=torch.bool, device=x.device
                ).triu(diagonal=1)
                attn_mask = key_bias + torch.zeros(
                    s, s, dtype=q.dtype, device=x.device
                ).masked_fill(causal_mask, neg_inf)
            else:
                attn_mask = key_bias
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
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
        x = x + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(x)), approximate="none")
        )
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            block = OptimizedBlock(
                config.d_model, config.num_heads, config.ffn_dim
            )
            block.norm1 = base_block.norm1
            block.norm2 = base_block.norm2
            block.ffn_in = base_block.ffn_in
            block.ffn_out = base_block.ffn_out
            block.attention.q_proj = base_block.attention.q_proj
            block.attention.k_proj = base_block.attention.k_proj
            block.attention.v_proj = base_block.attention.v_proj
            block.attention.out_proj = base_block.attention.out_proj
            opt_layers.append(block)
        self.layers = opt_layers

    def _forward_impl(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(
            valid_token_mask.all()
        )
        effective_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, effective_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def forward(self, x, valid_token_mask=None):
        if x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._forward_impl(x, valid_token_mask)
        return self._forward_impl(x, valid_token_mask)
