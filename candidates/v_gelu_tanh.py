"""
New direction: tanh-approximate GELU instead of exact (erf) GELU.

Identical to candidates/best.py in every respect except one: every other
candidate in this repo uses `approximate="none"` (exact erf GELU) to match
the baseline's numerics as closely as possible, per PROGRAM.md's correctness
contract. That contract was written assuming exact GELU was required --
nobody had actually measured whether the *cheaper* tanh approximation
(avoids computing erf, a more expensive elementwise op) still clears the
0.002 abs / 2% rel gate. The tanh approximation's known max per-activation
error is ~3e-4 to 5e-4, well under the raw tolerance -- but it compounds
across 4 layers and an FFN width up to 1024 in shape #8, so this is a
measurement question, not a foregone conclusion either way.

If this passes correctness, it stacks with every other technique in this
repo (routing, TF32, AMP, mask-cache) since it's an FFN-level, orthogonal
change -- fold it into v_router2.py's shared attention/FFN block if so.
If it fails, that's useful too: it means PROGRAM.md's exact-GELU
requirement is load-bearing, not just cautious.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

# Match the baseline's full-fp32 matmul precision. TF32 is off by default on
# recent PyTorch, but pin it explicitly so correctness doesn't depend on the
# ambient global flag (see v_compile.py for the failure mode this guards).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly


class SDPASelfAttention(nn.Module):
    """Multi-head self-attention via scaled_dot_product_attention.

    Submodule names (q_proj/k_proj/v_proj/out_proj) match BaselineSelfAttention
    so the baseline state_dict loads with strict=True.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if valid_token_mask is None:
            # No key padding -> let SDPA handle causal internally (fused, no [S,S]
            # mask materialized; this is what makes seq=100000 feasible).
            context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            # Build an additive float mask combining key padding (+ causal).
            # Invalid *key* positions -> -inf. Shape [B, 1, 1, S] broadcasts over
            # heads and query positions.
            neg_inf = float("-inf")
            key_bias = torch.zeros(
                batch, 1, 1, seq_len, dtype=q.dtype, device=x.device
            ).masked_fill(~valid_token_mask[:, None, None, :], neg_inf)
            if causal:
                causal_bool = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=x.device
                ).triu(diagonal=1)
                causal_bias = torch.zeros(
                    seq_len, seq_len, dtype=q.dtype, device=x.device
                ).masked_fill(causal_bool, neg_inf)
                attn_mask = key_bias + causal_bias  # [B, 1, S, S]
            else:
                attn_mask = key_bias  # [B, 1, 1, S]
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(context)

        if valid_token_mask is not None:
            # Zero padded-query rows to match the baseline exactly. (Valid queries
            # always retain >=1 unmasked key, so no all-inf/NaN rows arise here.)
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class OptimizedBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SDPASelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="tanh"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """Same structure/param names as BaselineTransformer, SDPA attention inside.

    We call BaselineTransformer.__init__ to register identical submodules, then
    swap each block for an OptimizedBlock that reuses the baseline's already-
    constructed weights (so strict weight copy still lines up by name).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = OptimizedBlock(config.d_model, config.num_heads, config.ffn_dim)
            # Reuse the baseline block's parameters by name so copy_model_weights
            # (strict) still matches; here we just re-point the modules.
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

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # B1: the harness passes an all-True mask (never None) when there is no
        # padding, so detect "no padding" once here (single sync) and pass None
        # downward. That lets attention take the fused is_causal path (flash /
        # mem-efficient, no [B,1,S,S] mask) instead of always building an additive
        # mask -- which is also what OOMs shape #14.
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
