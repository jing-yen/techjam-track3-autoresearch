"""
v_router2 -- consolidated dispatch: B10 mask-cache + T6 AMP (6/8/13).

S2 route change for 9/10 (reduce-overhead instead of fused) was tried and
REVERTED (iter 21): an isolated pairwise test (just shapes 9/10, journal
iter 20) showed reduce beating fused (3.65x/3.98x vs 2.14x/2.37x), but
inside the full 13-shape sweep -- where 5 different shapes now compile a
"reduce" instance in the same process -- it underperformed fused instead
(1.81x/2.01x vs fused's in-sweep 2.09x/2.33x). Likely a torch.compile /
CUDA-graph memory-pool interaction between concurrently-alive compiled
instances that an isolated 2-shape test can't see. Real lesson: route
decisions must be validated inside the actual full sweep, not pairwise --
kept on `fused` for 9/10, same as the original router.

CUDA fp16 autocast target for the three throughput-heavy shapes selected for
the AMP experiment (#6, #8, and #13); confirmed near-optimal by a full
13-shape AMP sweep (journal iter 19) -- AMP only wins on exactly these three.

No single candidate wins every shape (leaderboard.md iter 6 + iter 10,
official A100 protocol): compile (max-autotune) takes the small/launch-
overhead shapes hardest (#1/#2/#7), reduce-overhead compile unexpectedly
beats max-autotune on #3/#4/#5 (T1 finding, iter 10), and fused-qkv edges
everything out on #8/#9/#10/#11/#12/#13. Taking the per-shape max over the
four already-validated, already-correct fp32 candidates raises the aggregate
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

Unknown shapes (not in the table below, e.g. #14) fall back to
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

# Standalone experiment: torch.backends.cudnn.benchmark = True. The single
# most well-known PyTorch perf flag there is -- lets cuDNN benchmark
# multiple algorithm implementations per distinct input shape it sees and
# cache the fastest, instead of a fixed heuristic choice. Checked (grep)
# across every candidate in this repo: set NOWHERE before this file.
# Honest caveat: this model has no conv layers (cuDNN's benchmark flag is
# conv-focused), so payoff here is genuinely uncertain, not assumed --
# that's exactly why this is a real GPU test, not a guess written into
# v_router2.py directly. Zero correctness risk either way: purely a
# backend algorithm-selection hint, changes no computation. Everything
# else in this file is byte-identical to the current leaderboard-best
# v_router2.py (diffed before dispatch to confirm).
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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

# T7: fused residual-add + LayerNorm (AddNorm), confirmed on A100-80
# (journal iter 29): geomean 1.88x -> 2.02x standalone vs plain LayerNorm,
# 13/13 correct. Used ONLY by the eager routes (best/amp) below -- NOT by
# _CompileTransformerBase's compile/reduce routes, which already get their
# own operator fusion from torch.compile/Inductor and where injecting a raw
# Triton kernel call risks graph breaks under tracing. See
# candidates/v_triton_addnorm.py for the original standalone validation.
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


def _effective_mask(module: nn.Module, valid_token_mask: Optional[torch.Tensor]):
    """Return None for a stable all-valid mask without synchronizing every call.

    The harness reuses one mask tensor for warmup and timed forwards.  The first
    call classifies it (and necessarily synchronizes CUDA for ``bool(all())``);
    later calls use only host-visible tensor metadata.  Keeping a strong tensor
    reference prevents Python id reuse, while the version catches normal
    in-place mutations and forces reclassification.
    """
    if valid_token_mask is None:
        return None

    # Tensors allocated under torch.inference_mode() intentionally have no
    # version counter, so in-place changes cannot invalidate a cache safely.
    # Do not cache those.  The harness's timed mask is allocated outside
    # inference mode and remains cacheable; accuracy-trial masks are one-shot.
    try:
        version = valid_token_mask._version
    except RuntimeError:
        return valid_token_mask if not bool(valid_token_mask.all()) else None

    key = (
        id(valid_token_mask),
        valid_token_mask.data_ptr(),
        version,
        tuple(valid_token_mask.shape),
        tuple(valid_token_mask.stride()),
        valid_token_mask.storage_offset(),
        valid_token_mask.device.type,
        valid_token_mask.device.index,
        valid_token_mask.dtype,
    )
    if (
        getattr(module, "_mask_cache_tensor", None) is valid_token_mask
        and getattr(module, "_mask_cache_key", None) == key
    ):
        has_padding = module._mask_cache_has_padding
    else:
        has_padding = not bool(valid_token_mask.all())
        module._mask_cache_tensor = valid_token_mask
        module._mask_cache_key = key
        module._mask_cache_has_padding = has_padding
    return valid_token_mask if has_padding else None



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


class _BestBlockTriton(nn.Module):
    """Same as _BestBlock, but the post-attention residual-add + norm2 step
    runs through the fused Triton AddNorm kernel (T7) instead of two
    separate ops. Used only by the eager routes below (best/amp) -- kept as
    a separate class from _BestBlock so _CompileTransformerBase's
    compile/reduce routes (which construct _BestBlock directly) are
    completely unaffected."""

    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _BestAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, valid_token_mask, causal):
        attn_out = self.attention(self.norm1(x), valid_token_mask, causal)
        normed, x = fused_add_layernorm(
            x, attn_out, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        x = x + self.ffn_out(F.gelu(self.ffn_in(normed), approximate="none"))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class _BestTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _BestBlockTriton(config.d_model, config.num_heads, config.ffn_dim)
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
        eff_mask = _effective_mask(self, valid_token_mask)
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if eff_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


# --------------------------------------------------------------------------- #
# T15: extends T7's fused AddNorm to the SECOND residual+norm boundary
# (ffn_out-add -> next layer's norm1, or -> final_norm for the last layer).
# M2 (tools/profile_shapes.py, shape #13) found this boundary unfused at
# 19.21% of CUDA time -- bigger than T7's own already-fused kernel sitting
# right next to it (9.57%). Standalone validation: candidates/v_triton_addnorm2.py
# (13/13 correct on official shapes AND at --padding-ratio 0.3). Reuses T7's
# exact kernel unmodified; only the block/transformer wiring changes -- see
# that file's docstring for the full padding-safety argument (mirrors T7's
# own already-validated no-intermediate-masking approach).
# --------------------------------------------------------------------------- #
class _BestBlockTriton2(nn.Module):
    """Like _BestBlockTriton, but returns (post_ffn_residual, ffn_delta)
    UNCOMBINED -- the caller fuses their add with the NEXT norm."""

    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _BestAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x, normed1, valid_token_mask, causal):
        attn_out = self.attention(normed1, valid_token_mask, causal)
        normed2, x = fused_add_layernorm(
            x, attn_out, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        ffn_out = self.ffn_out(F.gelu(self.ffn_in(normed2), approximate="none"))
        return x, ffn_out


class _BestTransformer2(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _BestBlockTriton2(config.d_model, config.num_heads, config.ffn_dim)
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

    def _forward_impl(self, x, eff_mask):
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
        return out

    def forward(self, x, valid_token_mask=None):
        eff_mask = _effective_mask(self, valid_token_mask)
        out = self._forward_impl(x, eff_mask)
        if eff_mask is not None:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out


# --------------------------------------------------------------------------- #
# "amp" impl -- plain SDPA under CUDA float16 autocast (T6) + manual CUDA
# graph capture (U1', 2026-08-31: a shape-6 profiler trace found the same
# CPU-dispatch-pressure signal -- "Command Buffer Full" at 39.65% of CPU
# time, docs/research-shape6-profile.md -- that T17 already fixed on the
# fused route via this exact mechanism, +51-244%. Independently
# cross-checked by a collaborator's fact-checker pass, converging on the
# same conclusion without coordination -- docs/research-round1-corrections.md).
# Parameters and inputs remain fp32; autocast chooses eligible mixed-precision
# kernels and unlocks flash SDPA. CPU stays on the ordinary fp32 (no-graph)
# path. Inherits _BestTransformer2's Triton-fused AddNorm block (T7+T15) --
# the fusion kernel casts to fp32 internally for the reduction regardless of
# input dtype, so it is numerically correct whether called under fp32 or
# fp16 autocast; verified empirically (journal T7 AMP-integration entry, T15
# GPU test). The capture happens INSIDE the autocast context: autocast's
# per-op fp16/fp32 dispatch decisions are shape/dtype-driven, not data-value-
# driven, so they are stable across replays with different input VALUES of
# the same shape -- capturing one pass records whichever concrete kernels
# autocast chose, and replay reissues that exact same sequence.
#
# Capture is scoped to the no-padding case only, same restriction and same
# reasoning as T17 (a captured graph is a fixed op sequence; the organizer's
# protocol always runs at padding_ratio=0.0; a padded call falls back to the
# always-correct eager path below).
# --------------------------------------------------------------------------- #
class _AMPTransformer(_BestTransformer2):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._graph = None
        self._static_x = None
        self._static_out = None

    def _autocast_forward_impl(self, x, eff_mask):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self._forward_impl(x, eff_mask)

    def forward(self, x, valid_token_mask=None):
        if not x.is_cuda:
            return super().forward(x, valid_token_mask)

        eff_mask = _effective_mask(self, valid_token_mask)

        if eff_mask is None:
            if self._graph is None:
                self._static_x = x.clone()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._autocast_forward_impl(self._static_x, None)
                torch.cuda.current_stream().wait_stream(s)

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_out = self._autocast_forward_impl(self._static_x, None)

            self._static_x.copy_(x)
            self._graph.replay()
            return self._static_out

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self._forward_impl(x, eff_mask)
        return out.masked_fill(~valid_token_mask[..., None], 0)


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
        eff_mask = _effective_mask(self, valid_token_mask)
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
        eff_mask = _effective_mask(self, valid_token_mask)
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if eff_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


# --------------------------------------------------------------------------- #
# T17 + CUDA graph: fused-QKV attention (unmodified _FusedAttention above) +
# T7/T15's AddNorm kernel at both boundaries + manual CUDA graph capture.
#
# Profiler evidence (journal iter 48-52) showed the AddNorm fusion alone
# regressed shapes #9/#10/#12 by ~30%: genuinely less GPU compute (confirmed
# via a real trace, not inferred) but more CPU-side Triton-launch dispatch
# overhead than these low-compute shapes have concurrent GPU work to hide it
# behind (shape #11's big attention GEMM does have enough, and improved even
# without the graph). CUDA graph capture -- the same mechanism
# torch.compile(mode="reduce-overhead") already gets automatically for
# compile/reduce elsewhere in this file -- eliminates that per-launch
# dispatch cost by replaying a pre-recorded kernel sequence with one cheap
# call. Confirmed on real GPU (job 779394): #9 +59.1%, #10 +59.3%, #11
# +18.0%, #12 +244.5% vs the plain fused route, 13/13 correct.
#
# Capture is scoped to the no-padding case only (see _FusedBlockAddNorm2's
# transformer below) -- a captured graph is a fixed op sequence, so a
# padded call always falls back to the unrestricted eager path instead of
# risking a wrong control-flow branch replaying. Verified separately on
# real GPU at --padding-ratio 0.3 (job 779394, result_padding.json): 4/4
# correct, real (smaller, expected) speedups on the fallback path too.
# --------------------------------------------------------------------------- #
class _FusedBlockAddNorm2(nn.Module):
    """Like _FusedBlock, but returns (post_ffn_residual, ffn_delta)
    UNCOMBINED -- the caller fuses their add with the NEXT norm (T15's
    cross-layer-chaining pattern, reused verbatim)."""

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


class _FusedTransformerCudaGraph(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        opt_layers = nn.ModuleList()
        for base_block in self.layers:
            blk = _FusedBlockAddNorm2(config.d_model, config.num_heads, config.ffn_dim)
            blk.norm1 = base_block.norm1
            blk.norm2 = base_block.norm2
            blk.ffn_in = base_block.ffn_in
            blk.ffn_out = base_block.ffn_out
            blk.attention.out_proj = base_block.attention.out_proj
            opt_layers.append(blk)
        self.layers = opt_layers
        self._graph = None
        self._static_x = None
        self._static_out = None

    def _forward_impl(self, x, eff_mask):
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
        return out

    def forward(self, x, valid_token_mask=None):
        eff_mask = _effective_mask(self, valid_token_mask)

        if x.is_cuda and eff_mask is None:
            if self._graph is None:
                self._static_x = x.clone()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._forward_impl(self._static_x, None)
                torch.cuda.current_stream().wait_stream(s)

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_out = self._forward_impl(self._static_x, None)

            self._static_x.copy_(x)
            self._graph.replay()
            return self._static_out

        out = self._forward_impl(x, eff_mask)
        if eff_mask is not None:
            out = out.masked_fill(~valid_token_mask[..., None], 0)
        return out


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
    "amp": _AMPTransformer,
    "compile": _CompileTransformer,
    "reduce": _ReduceOverheadTransformer,
    "fused": _FusedTransformer,
    "fusedcg": _FusedTransformerCudaGraph,
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
    (10000, 128, 128, 4): "amp",     # shape 6  -- A100-40: 2.79x vs reduce 2.38x, fused 1.92x (S4)
    (64, 128, 32, 4):    "compile",  # shape 7  -- 3.59x vs reduce 2.79x, best 1.93x, fused 1.99x
    (64, 128, 1024, 4):  "amp",      # shape 8  -- 1.14x vs best 1.09x, compile 1.09x, reduce 1.09x
    (64, 128, 128, 1):   "fusedcg",  # shape 9  -- T17+cudagraph (job 779394): 3.55x vs plain fused's
                                      # 2.23x in the router (+59.1%). Real, profiler-motivated: AddNorm
                                      # fusion alone regressed this shape ~30% (less GPU work, but more
                                      # CPU-side Triton-launch dispatch than the GPU has concurrent work
                                      # to hide behind); CUDA graph capture removes that dispatch cost
                                      # entirely by replaying a captured kernel sequence. See TODO.md T17.
    (64, 128, 128, 2):   "fusedcg",  # shape 10 -- 3.93x vs fused's 2.47x (+59.3%), same mechanism.
    (64, 128, 128, 16):  "fusedcg",  # shape 11 -- 3.52x vs fused's 2.98x (+18.0%).
    (64, 32, 128, 4):    "fusedcg",  # shape 12 -- 8.50x vs fused's 2.47x (+244.5%).
}
_FALLBACK = "compile"


def copy_model_weights(baseline, optimized: "UserOptimizedTransformer") -> None:
    if optimized._impl_name in ("fused", "fusedcg"):
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

