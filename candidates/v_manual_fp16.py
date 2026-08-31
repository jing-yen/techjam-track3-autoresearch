"""
T16 -- manually pre-cast fp16 weights, replacing torch.autocast's per-call
weight casting on the amp route.

M2 (tools/profile_shapes.py) found fp32<->fp16 casting (aten::to /
_to_copy / copy_) at a consistent **17.1-17.7%** of CUDA time on BOTH
profiled amp-route shapes (#8 and #13) -- the second-biggest line item
after actual GEMM compute, and completely unaddressed by T7/T15 (which
targeted the LayerNorm/residual side, not this).

Root cause: `_AMPTransformer.forward` (v_router2.py) enters a FRESH
`with torch.autocast(device_type="cuda", dtype=torch.float16):` block on
EVERY call to `forward()` -- not once for the whole benchmark run. Autocast
casts each fp32-stored weight tensor to fp16 on the fly per op call, and
its cache (which avoids re-casting the SAME tensor twice *within* one
autocast region) does not help here, since each of the ~40 weight tensors
in a 4-layer model is used exactly once per forward call: every single
timed call re-casts every weight from fp32 to fp16 from scratch, discarding
the result immediately after. Repeated for 100 timed iterations x 3 rounds,
that is a lot of redundant, deterministic, throwaway work.

Fix: skip the storage detour entirely. Copy the baseline's fp32 weights
into genuinely fp16-dtype nn.Linear parameters ONCE (at weight-copy time),
and drop torch.autocast -- explicit casts only where they're unavoidable
(x <-> fp16 at each Linear/attention boundary, same as autocast already
does for activations; the difference is the WEIGHTS no longer need it).
`nn.Module.load_state_dict`'s underlying `Tensor.copy_` handles the
fp32->fp16 cast transparently, so this is bit-identical to what autocast's
own on-the-fly cast would have produced for the same source weights --
this is not a new numerical regime, just skipping redundant re-derivation
of a value that never changes between calls.

Correctness invariants preserved (matches every other candidate in this
repo): LayerNorm reduction in fp32 (weights kept fp32, inputs cast up
before the reduction, matching the existing T7/T15 fused kernel's own
internal upcast -- reused here unmodified). GELU computed in whatever
precision the input already is post-Linear, exactly matching what
autocast's own op-eligibility rules already do (Linear is fp16-eligible,
elementwise ops downstream inherit that dtype) -- no NEW precision
decision introduced here that the already-validated amp route didn't
already make. SDPA gets fp16 q/k/v (flash-attention-eligible, same as
amp), which accumulates internally in fp32 regardless of input dtype --
inherent to the algorithm, unchanged from the current amp route.

Scope: only the amp-route family (this file mirrors _BestBlockTriton2 /
_AMPTransformer from v_router2.py, both T7+T15 AddNorm boundaries already
built in). Standalone candidate; not yet integrated into v_router2.py --
validate the casting-tax hypothesis on real GPU data first.

CPU fallback: plain fp32 ops (no autocast, no fp16 -- CPU doesn't benefit
from either), identical math to every other candidate's CPU path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match baseline; dtype cast happens via .copy_()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False


# Reused verbatim from v_triton_addnorm2.py / v_router2.py (T7's kernel,
# unmodified) -- casts both operands to fp32 internally regardless of
# their storage dtype, so it is already compatible with fp16 deltas.
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

        out = torch.empty(residual2d.shape, dtype=torch.float32, device=residual.device)
        new_sum = torch.empty_like(out)
        block_size = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        _fused_add_layernorm_kernel[grid](
            residual2d, delta2d, weight, bias, out, new_sum,
            n_cols, eps, BLOCK_SIZE=block_size,
        )
        return out.view(orig_shape), new_sum.view(orig_shape)
else:
    def fused_add_layernorm(residual, delta, weight, bias, eps):
        x = (residual.float() + delta.float())
        y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return y, x


class _Attention16(nn.Module):
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

    def forward(self, normed1_fp32, valid_token_mask=None, causal=False):
        use_fp16 = normed1_fp32.is_cuda
        x = normed1_fp32.to(torch.float16) if use_fp16 else normed1_fp32
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
        out = self.out_proj(context)
        return out.float() if use_fp16 else out


class _Block16(nn.Module):
    """Weights are stored fp16 post-copy (see copy_model_weights below);
    x flows in fp32, cast to fp16 right at each Linear/attention boundary
    and back to fp32 immediately after -- same activation-casting pattern
    torch.autocast already used, minus the redundant per-call WEIGHT cast."""

    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _Attention16(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, normed1, valid_token_mask, causal):
        attn_out = self.attention(normed1, valid_token_mask, causal)
        normed2, x = fused_add_layernorm(
            x, attn_out, self.norm2.weight, self.norm2.bias, self.norm2.eps)

        use_fp16 = normed2.is_cuda
        h = normed2.to(torch.float16) if use_fp16 else normed2
        h = self.ffn_in(h)
        h = F.gelu(h, approximate="none")
        h = self.ffn_out(h)
        ffn_out = h.float() if use_fp16 else h
        return x, ffn_out


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _Block16(config.d_model, config.num_heads, config.ffn_dim)
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
        self._fp16_ready = False

    def _cast_linears_to_fp16(self):
        """Called once after weights are copied in (fp32, from baseline).
        Casts every Linear's weight/bias to fp16 in place -- LayerNorms stay
        fp32 (reduction precision, unchanged invariant)."""
        if not torch.cuda.is_available():
            return
        for blk in self.layers:
            for lin in (blk.attention.q_proj, blk.attention.k_proj,
                        blk.attention.v_proj, blk.attention.out_proj,
                        blk.ffn_in, blk.ffn_out):
                lin.weight.data = lin.weight.data.to(torch.float16)
                lin.bias.data = lin.bias.data.to(torch.float16)
        self._fp16_ready = True

    def forward(self, x, valid_token_mask=None):
        if x.is_cuda and not self._fp16_ready:
            self._cast_linears_to_fp16()

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
