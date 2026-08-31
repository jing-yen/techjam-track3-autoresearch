"""
T17 -- apply T7+T15's proven AddNorm fusion (both residual+norm boundaries)
to the `fused` route (shapes #9/#10/#11/#12), which currently has NONE.

Checked directly in v_router2.py: `_FusedBlock.forward` is plain eager
PyTorch -- `x = x + self.attention(...)`, `x = x + self.ffn_out(...)`, two
completely unfused residual-add + LayerNorm pairs, no Triton kernel at all.
Unlike `compile`/`reduce` (which get automatic fusion from
`torch.compile`/Inductor, confirmed via M2's shape #1 trace) and unlike
`best`/`amp` (which already have T7+T15), `fused` gets zero fusion help
from any source. Its only optimization is the fused-QKV projection
(`nn.Linear(d, 3d)` instead of three separate Linears) -- T5/T3's original
contribution, nothing since.

This candidate reuses T7+T15's exact kernel and cross-layer-chaining wiring
(byte-identical to v_triton_addnorm2.py / v_router2.py's
_BestBlockTriton2/_BestTransformer2) but swaps in `_FusedAttention`'s
fused-QKV projection instead of three separate Q/K/V Linears -- the ONLY
structural difference from the already-landed T15 candidate.

Correctness invariants: identical to T7/T15 (erf GELU, fp32 LayerNorm
reduction via the same kernel, padded rows handled the same
already-validated way -- masking only at attention's key-bias and the
final output, never threaded through the fused kernel itself).

CPU fallback: plain PyTorch ops, identical math.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = False  # fused-QKV layout needs a custom weight mapping

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
                             weight: torch.Tensor, bias: torch.Tensor, eps: float):
        orig_shape = residual.shape
        n_cols = orig_shape[-1]
        residual2d = residual.reshape(-1, n_cols).contiguous()
        delta2d = delta.reshape(-1, n_cols).contiguous()
        n_rows = residual2d.shape[0]

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
    def fused_add_layernorm(residual, delta, weight, bias, eps):
        x = residual + delta
        y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return y, x


class _FusedAttention(nn.Module):
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
        return self.out_proj(context)


class _Block(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, normed1, valid_token_mask, causal):
        attn_out = self.attention(normed1, valid_token_mask, causal)
        normed2, x = fused_add_layernorm(
            x, attn_out, self.norm2.weight, self.norm2.bias, self.norm2.eps)
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
            blk.attention.out_proj = base_block.attention.out_proj
            opt_layers.append(blk)
        self.layers = opt_layers

    def forward(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        causal = self.config.causal

        n0 = self.layers[0].norm1
        normed1 = F.layer_norm(x, (x.shape[-1],), n0.weight, n0.bias, n0.eps)

        out = x
        for i, layer in enumerate(self.layers):
            x, ffn_delta = layer(x, normed1, eff_mask, causal)
            if i + 1 < len(self.layers):
                next_norm1 = self.layers[i + 1].norm1
                normed1, x = fused_add_layernorm(
                    x, ffn_delta, next_norm1.weight, next_norm1.bias, next_norm1.eps)
            else:
                out, x = fused_add_layernorm(
                    x, ffn_delta, self.final_norm.weight, self.final_norm.bias, self.final_norm.eps)

        if has_padding:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out


def copy_model_weights(baseline, optimized: "UserOptimizedTransformer") -> None:
    """Fused-QKV weight mapping: baseline's separate q/k/v Linears -> one
    concatenated [3*d, d] weight, matching _FusedAttention.qkv's layout."""
    src = baseline.state_dict()
    new = {}
    for i in range(len(optimized.layers)):
        p = f"layers.{i}."
        for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                     "ffn_in.weight", "ffn_in.bias", "ffn_out.weight", "ffn_out.bias",
                     "attention.out_proj.weight", "attention.out_proj.bias"):
            new[p + name] = src[p + name]
        new[p + "attention.qkv.weight"] = torch.cat([
            src[p + "attention.q_proj.weight"],
            src[p + "attention.k_proj.weight"],
            src[p + "attention.v_proj.weight"],
        ], dim=0)
        new[p + "attention.qkv.bias"] = torch.cat([
            src[p + "attention.q_proj.bias"],
            src[p + "attention.k_proj.bias"],
            src[p + "attention.v_proj.bias"],
        ], dim=0)
    new["final_norm.weight"] = src["final_norm.weight"]
    new["final_norm.bias"] = src["final_norm.bias"]
    missing = [k for k in new if k not in dict(optimized.named_parameters())
               and k not in dict(optimized.named_buffers())]
    target = dict(optimized.state_dict())
    missing = [k for k in target if k not in new]
    if missing:
        raise RuntimeError(f"fused-qkv copy missing keys: {missing}")
    optimized.load_state_dict(new, strict=True)
