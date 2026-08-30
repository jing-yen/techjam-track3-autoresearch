"""
T5 -- per-shape dispatch.

No single candidate wins every shape (leaderboard.md iter 6 + iter 10,
official A100 protocol): compile (max-autotune) takes the small/launch-
overhead shapes hardest (#1/#2/#7), reduce-overhead compile unexpectedly
beats max-autotune on #3/#4/#5 (T1 finding, iter 10), and fused-qkv edges
everything out on #8/#9/#10/#11/#12/#13. Taking the per-shape max over the
four already-validated, already-correct candidates raises the aggregate
median well above any single one, for free -- no new kernel code, just
routing.

Self-contained snapshot (like v_compile.py / v_fused_qkv.py): inlines all
three implementations rather than importing candidates/best.py etc. The
runner ships each candidate to the cluster as a lone file (see runner.py's
run_slurm -- scp destination is a per-job temp path, siblings aren't
copied), so a candidate that imports a sibling by relative __file__ path
breaks off-repo. Learned the hard way: the first version of this file did
exactly that and every shape failed with
FileNotFoundError: .../router_official/best.py.

Unknown shapes (not in the table below, e.g. #6, #14) fall back to
v_compile, the best all-round generalist. This is intentionally NOT a
general heuristic (e.g. "use fused-qkv when d_model is large") -- the three
candidates' relative strengths don't reduce to one clean rule across our
12-shape sample, so the table is measured, not guessed. Extend it as more
shapes get benchmarked.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

# TODO.md S1: the old blanket "disable TF32 at import" here forced full fp32
# for EVERY route target, including best/reduce/fused which never touch
# torch.compile and so can't hit the asymmetric-kernel-selection bug that
# motivated it (Inductor's max-autotune picking a TF32 kernel for the
# candidate while the baseline's eager cuBLAS matmul stayed off-TF32). That
# also silently overrode the harness's own default (allow_tf32=True, matching
# the organizer's config) for the baseline too, since these are process-
# global flags read at import time, before harness's own default is set.
# Scoped now: only the "compile" (max-autotune) target forces full precision;
# everything else runs at whatever the harness/caller configured (default:
# organizer's own TF32-on config), set per-dispatch in __init__ below.

STRICT_WEIGHT_COPY = False  # dispatch target may be the fused-qkv layout


# --------------------------------------------------------------------------- #
# "best" impl -- plain SDPA, param names match the baseline (best.py).
# --------------------------------------------------------------------------- #
class _BestAttention(nn.Module):
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


class _BestBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _BestAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class _BestTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _BestBlock(config.d_model, config.num_heads, config.ffn_dim)
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


# --------------------------------------------------------------------------- #
# "compile" impl -- same attention as _Best*, wrapped in torch.compile.
# --------------------------------------------------------------------------- #
class _CompileTransformerBase(BaselineTransformer):
    """torch.compile-wrapped variant; subclasses set _MODE. Shared by the
    max-autotune ("compile") and reduce-overhead ("reduce") route targets."""
    _MODE = "max-autotune"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _BestBlock(config.d_model, config.num_heads, config.ffn_dim)
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
        self._compiled = None

    def _forward_impl(self, x, valid_token_mask=None):
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def forward(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        if self._compiled is None:
            mode = self._MODE if torch.cuda.is_available() else "default"
            try:
                self._compiled = torch.compile(self._forward_impl, mode=mode)
            except Exception:
                self._compiled = self._forward_impl
        return self._compiled(x, eff_mask)


class _CompileTransformer(_CompileTransformerBase):
    _MODE = "max-autotune"


class _ReduceOverheadTransformer(_CompileTransformerBase):
    _MODE = "reduce-overhead"


# --------------------------------------------------------------------------- #
# "fused" impl -- fused QKV projection (v_fused_qkv.py).
# --------------------------------------------------------------------------- #
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
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class _FusedBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _FusedAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class _FusedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _FusedBlock(config.d_model, config.num_heads, config.ffn_dim)
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
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def _fused_copy(baseline, optimized) -> None:
    b = baseline.state_dict()
    target = optimized.state_dict()
    new = {}
    for i in range(len(optimized.layers)):
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


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
_IMPLS = {
    "best": _BestTransformer,
    "compile": _CompileTransformer,
    "reduce": _ReduceOverheadTransformer,
    "fused": _FusedTransformer,
}

# (batch_size, seq_len, d_model, num_heads) -> best-known implementation.
# Source: leaderboard.md iter 6 (best/compile/fused) + iter 10 (reduce-overhead,
# T1), A100-80, official timing protocol.
_ROUTE = {
    (64, 128, 128, 4):   "compile",  # shape 1  -- 2.02x vs best 1.53x, fused 1.76x, reduce 2.02x
    (1, 128, 128, 4):    "compile",  # shape 2  -- 4.89x vs best 2.33x, fused 2.37x, reduce 5.04x
    (4, 128, 128, 4):    "reduce",   # shape 3  -- 4.83x vs compile 4.24x, best 2.35x, fused 2.36x
    (16, 128, 128, 4):   "reduce",   # shape 4  -- 3.24x vs best 2.17x, compile 2.34x, fused 2.34x
    (128, 128, 128, 4):  "reduce",   # shape 5  -- 2.19x vs fused 1.86x, best 1.66x, compile 1.59x
    (64, 128, 32, 4):    "compile",  # shape 7  -- 3.59x vs reduce 2.79x, best 1.93x, fused 1.99x
    (64, 128, 1024, 4):  "fused",    # shape 8  -- 1.14x vs best 1.09x, compile 1.09x, reduce 1.09x
    (64, 128, 128, 1):   "fused",    # shape 9  -- 1.47x vs best 1.27x, compile 1.22x, reduce 1.21x
    (64, 128, 128, 2):   "fused",    # shape 10 -- 1.70x vs best 1.47x, compile 1.39x, reduce 1.40x
    (64, 128, 128, 16):  "fused",    # shape 11 -- 2.73x vs best 2.45x, compile 2.40x, reduce 2.39x
    (64, 32, 128, 4):    "fused",    # shape 12 -- 2.36x vs best 2.21x, compile 1.91x, reduce 1.94x
    (64, 1024, 128, 4):  "fused",    # shape 13 -- 4.42x vs best 4.16x, compile 4.14x, reduce 4.15x
}
_FALLBACK = "compile"


def copy_model_weights(baseline, optimized: "UserOptimizedTransformer") -> None:
    if optimized._impl_name == "fused":
        _fused_copy(baseline, optimized._impl)
    else:
        import torch_transformer_benchmark as ttb
        ttb.copy_model_weights(baseline, optimized._impl, strict=True)


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Base's own layers/final_norm are dead weight once we delegate.
        del self.layers
        del self.final_norm

        key = (config.batch_size, config.seq_len, config.d_model, config.num_heads)
        self._impl_name = _ROUTE.get(key, _FALLBACK)
        if torch.cuda.is_available():
            # Explicit both ways (not just "disable for compile") because this
            # is a process-global flag and the harness sweeps many shapes/impls
            # through one process -- a prior shape's override would otherwise
            # leak into this one. Non-compile paths restore the harness's own
            # defaults (bench_harness.py --allow-tf32/--matmul-precision,
            # True/"high", matching the organizer's own config) rather than
            # forcing full precision everywhere.
            if self._impl_name == "compile":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
        self._impl = _IMPLS[self._impl_name](config)

    def forward(self, x, valid_token_mask=None):
        return self._impl(x, valid_token_mask)
