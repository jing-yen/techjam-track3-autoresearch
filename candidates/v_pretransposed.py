"""
Weight pre-transposition (U5, docs/research-untried.md) -- standalone test.

nn.Linear stores its weight as [out_features, in_features] and computes
x @ W.T + b every single call -- the transpose is not materialized, it's a
stride trick (W.T is a view), but it still means cuBLAS receives the
operands in "NT" layout (first operand Not-transposed, second Transposed)
rather than "NN" (neither transposed). cuBLAS/cuBLASLt maintain separate
tuned kernel selections per transpose-layout combination; NN is generally
the best-covered, most-optimized case since it's the default/most common
GEMM layout. Pre-transposing the weight ONCE at weight-copy time (storing
[in_features, out_features] instead) and computing `x @ W + b` directly
lets cuBLAS pick its NN kernel variant on every call instead of NT.

Real, standard cuBLAS-performance technique -- not novel, flagged
independently in docs/research-untried.md (U5) as "cheap A/B test, never
run." This is that test.

Scope: applied to the plain SDPA eager route (matching best.py's
structure, the simplest baseline for isolating this one variable) across
all Linear layers (Q/K/V/out_proj, ffn_in/ffn_out). Deliberately NOT
combined with T7/T15's AddNorm fusion or any other change, to keep this a
clean, single-variable test -- if it's a real, meaningful win, it can be
combined with the router's other techniques afterward; if not, no
further code should inherit it.

Correctness: mathematically identical to nn.Linear's own x @ W.T + b --
(x @ W.T) and (x @ W_pretransposed) compute the exact same values when
W_pretransposed = W.T.contiguous(), just with the transpose done once at
copy time instead of implicitly on every forward call. No precision
change, no approximation.

CPU fallback: same pre-transposed-weight computation (torch.matmul works
identically on CPU); this isn't a CUDA-specific technique on the Python
side, only the *kernel selection benefit* is CUDA/cuBLAS-specific.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = False  # weight shape is transposed relative to nn.Linear

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


class PreTransposedLinear(nn.Module):
    """Same math as nn.Linear, weight stored as [in_features, out_features]
    (already transposed) so the forward call is a plain x @ W + b -- cuBLAS
    sees an NN-layout GEMM instead of NT."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = PreTransposedLinear(d_model, d_model)
        self.k_proj = PreTransposedLinear(d_model, d_model)
        self.v_proj = PreTransposedLinear(d_model, d_model)
        self.out_proj = PreTransposedLinear(d_model, d_model)

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
        self.ffn_in = PreTransposedLinear(d_model, ffn_dim)
        self.ffn_out = PreTransposedLinear(ffn_dim, d_model)

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
        for _ in self.layers:
            opt_layers.append(OptimizedBlock(config.d_model, config.num_heads, config.ffn_dim))
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


def copy_model_weights(baseline, optimized: "UserOptimizedTransformer") -> None:
    """Transpose each baseline Linear's weight ([out,in] -> [in,out]) once,
    at copy time -- everything else (LayerNorm, biases) copies unchanged."""
    src = baseline.state_dict()
    new = {}
    linear_names = {"attention.q_proj", "attention.k_proj", "attention.v_proj",
                     "attention.out_proj", "ffn_in", "ffn_out"}
    for i in range(len(optimized.layers)):
        p = f"layers.{i}."
        for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            new[p + name] = src[p + name]
        for lin in linear_names:
            new[p + lin + ".weight"] = src[p + lin + ".weight"].t().contiguous()
            new[p + lin + ".bias"] = src[p + lin + ".bias"]
    new["final_norm.weight"] = src["final_norm.weight"]
    new["final_norm.bias"] = src["final_norm.bias"]
    target = dict(optimized.state_dict())
    missing = [k for k in target if k not in new]
    if missing:
        raise RuntimeError(f"pretransposed copy missing keys: {missing}")
    optimized.load_state_dict(new, strict=True)
