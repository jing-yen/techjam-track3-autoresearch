"""
K1 -- whole-model persistent Triton megakernel, hard-scoped to shape #2 only
(batch=1, seq=128, d_model=128, num_heads=4, head_dim=32, ffn_dim=128,
num_layers=4, causal=True).

Shape #2 is the most launch-overhead-bound shape measured this session:
0.32 TFLOP/s, under 1% of A100's fp32 peak (docs/research-kernel-frontier.md).
The ~40-60 kernel launches a normal eager/compiled forward pass takes for
this shape (per-Linear, per-LayerNorm, SDPA, GELU, residual adds, x4 layers)
are almost entirely overhead: at seq=128/d_model=128, one layer's ENTIRE
attention computation (all 4 heads) and both LayerNorms fit comfortably in
registers/shared memory, with no tiling needed.

**This kernel fuses the ENTIRE 4-layer forward pass into ONE kernel launch**
-- not just one block (K1's originally-scoped target was "4 launches per
forward, one per block"; batch=1 makes going further, to a true single
persistent launch for the whole model, tractable, matching what the cited
literature -- Hazy Research's Llama-1B megakernel -- actually means by
"one kernel", not the more modest per-block version).

Composed from patterns already validated on real A100 hardware this
session, not invented from scratch:
  - GEMM accumulation: same block-tiled `tl.dot` pattern as T10
    (candidates/v_triton_fused_ffn.py), with `input_precision="ieee"` on
    EVERY `tl.dot` call from the start -- T10's first GPU attempt failed
    5/13 shapes because this was omitted (tl.dot defaults to TF32 for fp32
    inputs; journal iter 33). Applying that lesson here up front rather
    than re-discovering it.
  - LayerNorm: same fp32 two-pass mean/var reduction as T7
    (candidates/v_triton_addnorm.py).
  - GELU: same exact-erf epilogue as T10.
  - Softmax: standard numerically-stable max-subtract-exp-sum pattern
    (Triton's own official fused-softmax/attention tutorials), with a
    causal upper-triangular mask applied to the raw scores before the max
    reduction.

Shape-specific hard-coding (deliberate, not generalized): SEQ=128, D=128,
HEADS=4, HEAD_DIM=32, FFN=128, LAYERS=4 are Python constants, not read from
config at trace time -- this kernel is NOT a general per-shape solution,
it targets exactly one shape. Falls back to the standard SDPA/eager path
(matching best.py) for every other shape and on CPU.

Honest status: written and CPU-fallback-tested for STRUCTURAL correctness
only (weight-copy, shapes, control flow) -- the actual Triton kernel
math has NOT been run on a GPU yet. This is a genuinely high-risk, one-shot
attempt per the team's own time-boxing (docs/research-kernel-frontier.md):
build once, test on real hardware, and if it doesn't beat `reduce`'s
0.374ms baseline for shape 2 by a clear margin (>20%, beyond the measured
noise floor), stop and report "path identified" rather than iterating
further -- see TODO.md K1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

STRICT_WEIGHT_COPY = True  # param names match the baseline exactly

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

# Shape #2 exact dimensions -- this kernel is scoped to precisely this shape.
_SEQ = 128
_D = 128
_HEADS = 4
_HEAD_DIM = 32
_FFN = 128
_LAYERS = 4
_TARGET_KEY = (1, _SEQ, _D, _HEADS)  # (batch_size, seq_len, d_model, num_heads)


if _HAS_TRITON:
    @triton.jit
    def _megakernel_fwd(
        x_ptr,                                   # [SEQ, D] input (batch=1, squeezed)
        # Per-layer weights, stacked along dim 0: [LAYERS, ...]
        qkv_w_ptr, qkv_b_ptr,                     # [LAYERS, 3*D, D], [LAYERS, 3*D]
        out_w_ptr, out_b_ptr,                     # [LAYERS, D, D], [LAYERS, D]
        ffn_in_w_ptr, ffn_in_b_ptr,                # [LAYERS, FFN, D], [LAYERS, FFN]
        ffn_out_w_ptr, ffn_out_b_ptr,              # [LAYERS, D, FFN], [LAYERS, D]
        norm1_w_ptr, norm1_b_ptr,                  # [LAYERS, D], [LAYERS, D]
        norm2_w_ptr, norm2_b_ptr,                  # [LAYERS, D], [LAYERS, D]
        final_norm_w_ptr, final_norm_b_ptr,        # [D], [D]
        out_ptr,                                   # [SEQ, D] output
        eps,
        SEQ: tl.constexpr, D: tl.constexpr, HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr, FFN: tl.constexpr, LAYERS: tl.constexpr,
    ):
        # Single-program kernel: batch=1, so one CTA handles the whole
        # forward pass. All tiles below are the FULL SEQ x D (or SEQ x FFN)
        # extent -- no grid/block-index masking needed, this kernel is
        # only ever launched with grid=(1,).
        rows = tl.arange(0, SEQ)
        cols = tl.arange(0, D)

        x = tl.load(x_ptr + rows[:, None] * D + cols[None, :]).to(tl.float32)

        causal_mask = rows[:, None] >= rows[None, :]  # [SEQ, SEQ], True where allowed
        # HEAD_DIM is a plain Python int at trace time (tl.constexpr), not a
        # Triton tensor -- no .to() method. Compute the Python float directly.
        inv_sqrt_head_dim = 1.0 / (HEAD_DIM ** 0.5)

        for layer in range(LAYERS):
            # ---- norm1 ----
            mean1 = tl.sum(x, axis=1) / D
            xm1 = x - mean1[:, None]
            var1 = tl.sum(xm1 * xm1, axis=1) / D
            rstd1 = 1.0 / tl.sqrt(var1 + eps)
            n1w = tl.load(norm1_w_ptr + layer * D + cols).to(tl.float32)
            n1b = tl.load(norm1_b_ptr + layer * D + cols).to(tl.float32)
            normed1 = xm1 * rstd1[:, None] * n1w[None, :] + n1b[None, :]

            # ---- per-head QKV projection + causal attention + out projection ----
            # Triton does not support Python slice syntax on already-
            # materialized tensor VALUES (only on pointer expressions fed to
            # tl.load) -- a combined "compute full Q/K/V, then slice per
            # head" approach hits "unsupported tensor index" at compile time
            # (real error, journal iter 35). Instead compute each head's Q/K/V
            # directly via its own small [SEQ,D]@[D,HEAD_DIM] GEMM, using a
            # pointer-level row offset into qkv_w (still stored
            # [LAYERS, 3D, D], Q/K/V stacked along the output axis) -- no
            # tensor slicing anywhere. Each head's out-projection contribution
            # is accumulated the same way (pointer-offset weight slice,
            # journal iter 35 -- already correct, kept as-is).
            hd = tl.arange(0, HEAD_DIM)
            attn_acc = tl.zeros((SEQ, D), dtype=tl.float32)
            for h in range(HEADS):
                # h is a Python int (HEADS is constexpr, this loop unrolls
                # at trace time), so these are static compile-time offsets.
                q_rows = h * HEAD_DIM + hd
                k_rows = D + h * HEAD_DIM + hd
                v_rows = 2 * D + h * HEAD_DIM + hd

                qw_h = tl.load(qkv_w_ptr + layer * (3 * D) * D + q_rows[None, :] * D + cols[:, None]).to(tl.float32)
                q_h_acc = tl.zeros((SEQ, HEAD_DIM), dtype=tl.float32)
                q_h_acc = tl.dot(normed1, qw_h, q_h_acc, input_precision="ieee")
                q_h = q_h_acc + tl.load(qkv_b_ptr + layer * (3 * D) + q_rows).to(tl.float32)[None, :]

                kw_h = tl.load(qkv_w_ptr + layer * (3 * D) * D + k_rows[None, :] * D + cols[:, None]).to(tl.float32)
                k_h_acc = tl.zeros((SEQ, HEAD_DIM), dtype=tl.float32)
                k_h_acc = tl.dot(normed1, kw_h, k_h_acc, input_precision="ieee")
                k_h = k_h_acc + tl.load(qkv_b_ptr + layer * (3 * D) + k_rows).to(tl.float32)[None, :]

                vw_h = tl.load(qkv_w_ptr + layer * (3 * D) * D + v_rows[None, :] * D + cols[:, None]).to(tl.float32)
                v_h_acc = tl.zeros((SEQ, HEAD_DIM), dtype=tl.float32)
                v_h_acc = tl.dot(normed1, vw_h, v_h_acc, input_precision="ieee")
                v_h = v_h_acc + tl.load(qkv_b_ptr + layer * (3 * D) + v_rows).to(tl.float32)[None, :]

                scores = tl.dot(q_h, tl.trans(k_h), input_precision="ieee") * inv_sqrt_head_dim
                scores = tl.where(causal_mask, scores, float("-inf"))
                row_max = tl.max(scores, axis=1)
                probs = tl.exp(scores - row_max[:, None])
                probs = probs / tl.sum(probs, axis=1)[:, None]
                ctx_h = tl.dot(probs, v_h, input_precision="ieee")  # [SEQ, HEAD_DIM]

                head_rows = h * HEAD_DIM + tl.arange(0, HEAD_DIM)
                ow_h_ptrs = out_w_ptr + layer * D * D + cols[None, :] * D + head_rows[:, None]
                ow_h = tl.load(ow_h_ptrs).to(tl.float32)  # [HEAD_DIM, D]
                attn_acc = tl.dot(ctx_h, ow_h, attn_acc, input_precision="ieee")

            out_bias = tl.load(out_b_ptr + layer * D + cols).to(tl.float32)
            attn_out = attn_acc + out_bias[None, :]

            x = x + attn_out  # residual 1

            # ---- norm2 ----
            mean2 = tl.sum(x, axis=1) / D
            xm2 = x - mean2[:, None]
            var2 = tl.sum(xm2 * xm2, axis=1) / D
            rstd2 = 1.0 / tl.sqrt(var2 + eps)
            n2w = tl.load(norm2_w_ptr + layer * D + cols).to(tl.float32)
            n2b = tl.load(norm2_b_ptr + layer * D + cols).to(tl.float32)
            normed2 = xm2 * rstd2[:, None] * n2w[None, :] + n2b[None, :]

            # ---- FFN: Linear(D,FFN) -> erf GELU -> Linear(FFN,D) ----
            ffn_cols = tl.arange(0, FFN)
            fw1_ptrs = ffn_in_w_ptr + layer * FFN * D + ffn_cols[None, :] * D + cols[:, None]
            fw1 = tl.load(fw1_ptrs).to(tl.float32)  # [D, FFN]
            h_acc = tl.zeros((SEQ, FFN), dtype=tl.float32)
            h_acc = tl.dot(normed2, fw1, h_acc, input_precision="ieee")
            fb1 = tl.load(ffn_in_b_ptr + layer * FFN + ffn_cols).to(tl.float32)
            hidden = h_acc + fb1[None, :]
            inv_sqrt2 = 0.7071067811865476
            hidden = hidden * 0.5 * (1.0 + tl.math.erf(hidden * inv_sqrt2))

            fw2_ptrs = ffn_out_w_ptr + layer * D * FFN + cols[None, :] * FFN + ffn_cols[:, None]
            fw2 = tl.load(fw2_ptrs).to(tl.float32)  # [FFN, D]
            f_acc = tl.zeros((SEQ, D), dtype=tl.float32)
            f_acc = tl.dot(hidden, fw2, f_acc, input_precision="ieee")
            fb2 = tl.load(ffn_out_b_ptr + layer * D + cols).to(tl.float32)
            ffn_out = f_acc + fb2[None, :]

            x = x + ffn_out  # residual 2

        # ---- final norm ----
        mean_f = tl.sum(x, axis=1) / D
        xmf = x - mean_f[:, None]
        varf = tl.sum(xmf * xmf, axis=1) / D
        rstdf = 1.0 / tl.sqrt(varf + eps)
        fnw = tl.load(final_norm_w_ptr + cols).to(tl.float32)
        fnb = tl.load(final_norm_b_ptr + cols).to(tl.float32)
        result = xmf * rstdf[:, None] * fnw[None, :] + fnb[None, :]

        tl.store(out_ptr + rows[:, None] * D + cols[None, :], result.to(out_ptr.dtype.element_ty))
else:
    _megakernel_fwd = None


class SDPASelfAttention(nn.Module):
    """Fallback path (CPU, or GPU shapes other than #2): identical to best.py."""
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

        key = (config.batch_size, config.seq_len, config.d_model, config.num_heads)
        self._use_megakernel = (
            _HAS_TRITON and key == _TARGET_KEY
            and config.ffn_dim == _FFN and config.num_layers == _LAYERS and config.causal
        )

    def _forward_eager(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _forward_megakernel(self, x):
        # x: [1, SEQ, D] -> squeeze batch, kernel operates on [SEQ, D].
        x2d = x[0].contiguous()
        L = len(self.layers)
        device = x.device

        def stack(attr_path):
            vals = []
            for layer in self.layers:
                obj = layer
                for a in attr_path:
                    obj = getattr(obj, a)
                vals.append(obj)
            return torch.stack(vals, dim=0).contiguous()

        qkv_w = torch.cat([
            torch.stack([l.attention.q_proj.weight for l in self.layers], dim=0),
            torch.stack([l.attention.k_proj.weight for l in self.layers], dim=0),
            torch.stack([l.attention.v_proj.weight for l in self.layers], dim=0),
        ], dim=1).contiguous()  # [L, 3D, D]
        qkv_b = torch.cat([
            torch.stack([l.attention.q_proj.bias for l in self.layers], dim=0),
            torch.stack([l.attention.k_proj.bias for l in self.layers], dim=0),
            torch.stack([l.attention.v_proj.bias for l in self.layers], dim=0),
        ], dim=1).contiguous()  # [L, 3D]
        out_w = stack(["attention", "out_proj", "weight"])
        out_b = stack(["attention", "out_proj", "bias"])
        ffn_in_w = stack(["ffn_in", "weight"])
        ffn_in_b = stack(["ffn_in", "bias"])
        ffn_out_w = stack(["ffn_out", "weight"])
        ffn_out_b = stack(["ffn_out", "bias"])
        n1w = stack(["norm1", "weight"])
        n1b = stack(["norm1", "bias"])
        n2w = stack(["norm2", "weight"])
        n2b = stack(["norm2", "bias"])

        out = torch.empty_like(x2d)
        # num_stages=1: default multi-stage pipelining roughly doubles SMEM
        # for double-buffering; the first real GPU attempt needed 278528
        # bytes against A100's 166912-byte limit (journal iter 35) -- a
        # ~67% overshoot. This kernel's tl.dot calls are all single-shot
        # (K=D=128 or K=HEAD_DIM=32, no multi-block K-loop to pipeline), so
        # pipelining buys nothing here anyway; num_stages=1 removes the
        # double-buffering overhead without changing what's computed.
        _megakernel_fwd[(1,)](
            x2d,
            qkv_w, qkv_b, out_w, out_b,
            ffn_in_w, ffn_in_b, ffn_out_w, ffn_out_b,
            n1w, n1b, n2w, n2b,
            self.final_norm.weight, self.final_norm.bias,
            out,
            self.final_norm.eps,
            SEQ=_SEQ, D=_D, HEADS=_HEADS, HEAD_DIM=_HEAD_DIM, FFN=_FFN, LAYERS=L,
            num_stages=1, num_warps=4,
        )
        return out.unsqueeze(0)

    def forward(self, x, valid_token_mask=None):
        no_padding = valid_token_mask is None or bool(valid_token_mask.all())
        if self._use_megakernel and no_padding and x.is_cuda:
            return self._forward_megakernel(x)
        return self._forward_eager(x, valid_token_mask)
