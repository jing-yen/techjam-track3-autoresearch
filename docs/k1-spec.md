# K1 spec — staged block fusion for the S=128 / d=128 family

Status: APPROVED TO ATTEMPT despite the negative literature (user decision,
2026-08-31). Downside is bounded by construction — see "Why this is safe".
Target: shape #2 (B=1, S=128, d=128, H=4, L=4), currently **0.374 ms** via
`reduce-overhead`. Secondary: #3, #4, #12, then the rest of the S=128 family.

## Why this is safe to attempt

`v_router2` dispatches per shape to whichever candidate won that shape. A K1 that
loses is simply never routed to. **It cannot regress the leaderboard** — the only
cost is engineer hours. That is the whole reason the router architecture was worth
building.

## Why it is worth attempting despite the literature

`docs/research-agent-findings.md` closed K1 on A100 megakernel results
(AutoMegaKernel 0.55-0.79x, MPK 1.16x). Two reasons that evidence transfers
weakly to us:

1. **Their baseline is CUDA-graphed cuBLAS on Llama-scale GEMMs**, where cuBLAS is
   heavily tuned. Ours are **128x128x128** — far below cuBLAS's efficient range.
   The bar we must clear is much lower than the bar they failed to clear.
2. **Their workload is batch-1 autoregressive decode.** Ours is a full forward
   pass with S=128 of parallelism available.

The measured headroom is not subtle:

```
one 128x128x128 GEMM     = 4.2 MFLOP = 0.22 us of math at fp32 peak
measured 374 us / ~40 kernels = 9.3 us per kernel
=> ~97% of each kernel's wall time is not arithmetic
```

That is *after* CUDA graphs. Whatever the remaining 97% is — dispatch latency,
HBM round-trips between ops, tail effects on a 108-SM card running a 128x128
problem — a fused kernel that keeps activations in SRAM across stages attacks it
directly.

## Shared-memory budget (A100, 164 KB max per block)

| stage | weights fp16 | + activations fp32 | total | fits |
|--|--|--|--|--|
| QKV (3 mats) | 96 KB | 64 KB | 160 KB | yes, tight |
| out_proj | 32 KB | 64 KB | 96 KB | yes |
| ffn_in | 32 KB | 64 KB | 96 KB | yes |
| ffn_out | 32 KB | 64 KB | 96 KB | yes |
| one head's scores [128,128] fp32 | — | 64 KB | 64 KB | yes |

All weights resident at once is **192 KB — does NOT fit**. Stage them. Attention
must be per-head (all four heads' scores at once is 256 KB).

## Build order — measure after EVERY stage, stop at the first flat result

Do not build the whole block before measuring. Each stage is independently
shippable through the router.

- **K1a — fused FFN block.** `norm2 → ffn_in → GELU → ffn_out → residual add`.
  Five ops to one. **Start here, not with attention.** Highest ratio of
  (ops eliminated) to (implementation risk): no softmax, no causal masking, no
  per-head loop, and `ffn_dim == d_model` on all 14 official shapes so the two
  GEMMs are the same shape. This is T10 scoped down to where cuBLAS is weakest.
- **K1b — fused attention block.** `norm1 → QKV → SDPA → out_proj → residual`.
  Harder: needs the causal mask and an online softmax. Only attempt if K1a wins.
- **K1c — chain both.** One kernel per transformer block, 4 launches per forward
  instead of ~40.

## Correctness invariants — non-negotiable, from PROGRAM.md

1. **GELU must be erf-exact.** Triton has `tl.math.erf`. Do NOT use a tanh
   approximation — it fails the gate. This is exactly why K2 (cuBLASLt) was
   blocked.
2. **LayerNorm reduction in fp32** even when inputs are fp16.
3. **Softmax in fp32**, cast back after (K1b only).
4. **Causal mask is `triu(diagonal=1)`** — strictly above the diagonal (K1b only).
5. **Scale is `1/sqrt(head_dim)`** (K1b only).
6. **Padded query rows zeroed** after attention, after each block, after final
   norm — only when `has_padding`; the no-padding path must skip it entirely.
7. **Parameter names must match the baseline** or provide `copy_model_weights`.

Reference implementation to match, line by line:
`torch_transformer_benchmark.py:140-141` (block) and `:97-118` (attention).

## Kill criteria — agreed in advance, honour them

- **K1a does not beat the routed `reduce` time on shape #2 by >20%** → stop.
  20% because the measured run-to-run noise floor is ±2.7% per shape (S8) and a
  win under 20% will not survive re-measurement convincingly.
- **Any correctness failure that is not fixed within 30 minutes** → stop. The
  gate is absolute; a fast wrong kernel is worth zero.
- **Total time-box: 4 hours from first line of code.** At T-12h the whole
  candidate set freezes regardless.

## If it fails

That is a publishable result for the tech report, not a wasted afternoon:
"we measured 97% non-arithmetic overhead on the small shapes, built a fused
block to attack it, and it did not beat CUDA-graphed cuBLAS — consistent with
AutoMegaKernel's A100 finding." A negative result with a number attached beats
speculation in either direction.

## Limitations

SMEM figures are arithmetic from tensor shapes, not measured occupancy. The 9.3
us/kernel figure divides total time by an estimated kernel count (~40); M2's
profiler run would replace that estimate with a real per-op breakdown, and
**should be run first if the GPU slot allows** — it would also settle K4.

## AI disclosure
Spec produced with AI assistance (Claude Opus 5). Budget arithmetic computed
inline from the shape table; literature context from `docs/research-agent-findings.md`.
