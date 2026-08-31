"""
T10 -- custom Triton kernel: fused Linear + bias + GELU(erf) for the FFN's
first projection.

Current FFN (see v_triton_addnorm.py / best.py): `ffn_out(gelu(ffn_in(x)))`.
`nn.Linear` already fuses bias-add into its GEMM via cuBLAS, so the actual
unfused step is GELU sitting as its own standalone elementwise kernel between
two already-optimized GEMM+bias kernels. This candidate replaces
`ffn_in(x)` + `F.gelu(..., approximate="none")` with ONE Triton kernel that
computes the GEMM, adds the bias, and applies exact (erf) GELU in the same
kernel -- the GEMM's output tile never round-trips through HBM before GELU
is applied to it. `ffn_out` (the second Linear) is left as a plain
`nn.Linear` call: fusing INTO a GEMM (this kernel) is a well-documented,
lower-risk pattern (Triton's own official matmul tutorial covers exactly
this: a block-tiled accumulation loop with a fused epilogue before the
final store); writing a full alternative GEMM to replace `ffn_out` would
not save anything here since nothing follows it inside this kernel's scope
before the residual add.

Deliberately scoped narrower than a full custom GEMM replacement: this
follows Triton's tutorial matmul structure closely rather than attempting
novel tiling, to keep correctness risk contained. Real risk that remains,
stated plainly (see TODO.md T10's own note): this GEMM has to actually
compete with cuBLAS's already-tuned kernel, not just save bandwidth the way
T7's AddNorm did -- a first attempt landing at parity or worse before tuning
helps is a real, expected possibility, not a sign the approach is wrong.

Block sizes below (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) are untuned
starting defaults, not the result of an autotuning sweep -- same caveat as
T7's kernel before T7b. GPU correctness has NOT been verified yet (no CUDA
available while this was written); only the CPU-fallback path (plain
PyTorch ops, used when no CUDA) has been tested.

CPU fallback: Triton requires CUDA. On CPU, falls back to plain
Linear + exact-GELU (identical math, unfused) so correctness/dev tests
still run without a GPU.
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


if _HAS_TRITON:
    # T7b-style autotuning: T10's first GPU attempt used a single fixed
    # config (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) and regressed on shape #8
    # (d_model=1024, 0.84x -- slower than baseline) while winning everywhere
    # else. Our shapes span M from 128 (shape 2) to 1.28M rows (shape 6) and
    # N/K from 32 to 1024 -- one fixed block size cannot be right for all of
    # them. Triton benchmarks every config below against the actual (M,N,K)
    # it's called with and caches the winner per shape (`key=["M","N","K"]`)
    # -- the same mechanism torch.compile's own autotuning uses (see L3 for
    # the one caveat: that mechanism does not dedupe identical shapes
    # appearing at different call sites in one *compiled graph*; that
    # doesn't apply here since this is a single eagerly-called Triton
    # kernel, not something Inductor traces and unrolls).
    _FFN_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ]

    @triton.autotune(configs=_FFN_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_linear_gelu_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Standard Triton-tutorial block-tiled GEMM: C = A @ B + bias, then
        # exact erf GELU applied to the accumulator before the final store.
        # A is [M, K] (the block's input rows), B is [K, N] (ffn_in.weight,
        # transposed to [d_model, ffn_dim] by the caller so this is a plain
        # row-major matmul, no transpose logic inside the kernel).
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k0 < K)
            b_mask = (offs_k[:, None] + k0 < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            # input_precision="ieee" forces full fp32 accumulation. tl.dot's
            # default is TF32 for fp32 inputs -- the exact bug class already
            # found once this session in torch.compile's max-autotune (S1);
            # found again here (T10, journal iter 33) because it wasn't set
            # explicitly. Without this, max_abs sits uniformly at ~0.002-0.0024
            # across nearly every shape (TF32's characteristic ~1e-3 relative
            # error), failing 5/13 shapes outright.
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

        # Exact (erf) GELU -- must match approximate="none" per PROGRAM.md's
        # correctness contract; tanh-approximate is a DIFFERENT function
        # (measured separately, see TODO.md T7's tanh-GELU finding -- close
        # numerically for THIS model, but not used here since exact is free
        # to compute inside a fused kernel and removes that question entirely).
        inv_sqrt2 = 0.7071067811865476
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * inv_sqrt2))

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

    def fused_linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """x: [..., K], weight: [N, K] (nn.Linear layout), bias: [N] -> [..., N]."""
        orig_shape = x.shape
        k = orig_shape[-1]
        x2d = x.reshape(-1, k).contiguous()
        m = x2d.shape[0]
        n = weight.shape[0]
        # nn.Linear computes x @ weight.T; transpose once so the kernel does a
        # plain [M,K] @ [K,N] matmul with no transpose logic inside the loop.
        w_t = weight.t().contiguous()

        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        # BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages are no longer passed
        # here -- @triton.autotune injects the winning config's values (and
        # its own grid, since grid is now a lambda over META below).
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(n, META["BLOCK_N"]))
        _fused_linear_gelu_kernel[grid](
            x2d, w_t, bias, out,
            m, n, k,
            x2d.stride(0), x2d.stride(1),
            w_t.stride(0), w_t.stride(1),
            out.stride(0), out.stride(1),
        )
        return out.view(*orig_shape[:-1], n)
else:
    def fused_linear_gelu(x, weight, bias):
        return F.gelu(F.linear(x, weight, bias), approximate="none")


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
        normed = self.norm2(x)
        hidden = fused_linear_gelu(normed, self.ffn_in.weight, self.ffn_in.bias)
        x = x + self.ffn_out(hidden)
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

    def forward(self, x, valid_token_mask=None):
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())
        eff_mask = valid_token_mask if has_padding else None
        for layer in self.layers:
            x = layer(x, eff_mask, self.config.causal)
        x = self.final_norm(x)
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
