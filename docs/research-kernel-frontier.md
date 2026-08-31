# Beyond flash: what the kernel literature offers THIS workload

Engineering brief · 2026-08-31 · quick mode
Question: is this project "just SDPA + PyTorch edits"? What do
FlashAttention-class, PagedAttention-class, and newer kernel techniques offer,
given our actual 14 shapes?

## First, the premise needs correcting — with our own measurements

**"Flash attention is 3-4x" is true only where sequences are long.** Our shape
catalog is the opposite regime. Computed from the appendix FLOP model:

| shape family | attention share of FLOPs | whole-attention-in-SMEM? |
|--|--|--|
| S=128, d=128 (#1-6,9-11) | **14%** | yes — 56-128 KB vs 164 KB/SM |
| #8 (d=1024) | **2%** | no (224 KB) |
| #13 (S=1024) | 57% | no |
| #14 (S=100k) | 94% | no |

Amdahl on the S=128 family: even an **infinitely fast** attention buys ≤1.16x.
That is why our 3.72x does not come from flash — measured fact: flash was
eligible on **0/14 shapes at fp32** (`docs/sdpa_backend_probe.json`). The gains
came from launch-overhead removal (compile/CUDA graphs), AMP routing, a Triton
AddNorm, and per-shape dispatch. Where attention *does* dominate, we already
collect the flash-class win: #13 at 11.7x, #14 runs at all.

So no — the project is not "using FlashAttention." On 12 of 14 shapes
FlashAttention is nearly irrelevant, and the creative work was elsewhere.

## PagedAttention: honestly inapplicable

PagedAttention (vLLM) virtualizes the **KV cache** so many concurrent
autoregressive requests share fragmented memory. Our task is a single
fixed-shape forward pass with **no KV cache, no decode loop, no concurrent
requests** — the two problems it solves do not exist here. Adopting it would
add its bookkeeping overhead and remove nothing. Rejected on mechanism.

## Where the frontier actually points for OUR shapes

The measured ceiling gap is not in attention:

```
shape 2 today: 0.374 ms for 0.12 GFLOP  =  0.32 TFLOP/s   (0.2% of peak)
shape 1 today: 1.302 ms for 7.5 GFLOP   =  5.8 TFLOP/s    (~4%)
```

The launch-bound family runs at **under 1% of the card's arithmetic peak**. The
literature's answer to exactly this regime is the **megakernel**: fuse the
entire forward pass into ONE persistent kernel launch, eliminating the ~40-60
per-op launches and their pipeline bubbles.

- Hazy Research fused all of Llama-1B into a single kernel; vLLM/SGLang "use at
  most half of H100 bandwidth at low latency" from launch bubbles alone
  ([overview](https://theorempath.com/topics/megakernels),
  [mlops writeup](https://mlops.substack.com/p/one-megakernel-to-rule-llama-1b))
- [AutoMegaKernel, arXiv:2606.09682](https://arxiv.org/html/2606.09682) compiles
  HF models into one persistent cooperative kernel automatically
- [Luminal: compiling models to megakernels](https://blog.luminal.com/p/compiling-models-to-megakernels)

**Our d=128 family is an unusually good megakernel target**, better than
Llama-1B: one layer's ENTIRE weight set (QKV+out+FFN) is 192 KB fp16 — a layer
per SM's shared memory, the whole 4-layer model in 768 KB. Attention per (b,h)
fits in SMEM outright at S=128, no tiling needed.

## Ranked new directions (all falsifiable, all scoped)

- **K1 — Fused whole-block Triton kernel for the S=128/d=128 family.** The
  scoped, achievable slice of the megakernel idea: one Triton kernel per
  transformer block (QKV proj → attention → out proj → AddNorm → FFN →
  AddNorm), one program per (batch-tile, head). At S=128/d=128 every
  intermediate fits in SMEM; weights are 192 KB fp16. 4 launches per forward
  instead of ~40. *Ceiling argument:* shape 2 at 0.32 TFLOP/s has ~100x of
  headroom that is pure overhead. *Risk:* highest-effort item ever proposed
  here; T7 took a day for ONE fused op. *Falsify fast:* build the d=128
  single-block version only; if it does not beat `reduce`'s 0.374 ms on shape 2
  by >20% (beyond the ±2.7% noise floor, S8), stop.
- **K2 — cuBLASLt fused GELU_BIAS epilogue for the FFN.** Ampere exposes
  single-kernel bias+GELU GEMM fusions
  ([cuBLAS 12.0 blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/);
  [torch-cublas-hgemm](https://github.com/aredden/torch-cublas-hgemm) wraps
  exactly `gelu(A@B^T + bias)`). Our eager `fused`/`amp` routes still run
  Linear→GELU→Linear as 3 ops. This is T10's goal without writing a GEMM —
  reuse NVIDIA's. *Caveat:* our GELU must stay erf-exact
  (`approximate="none"`); VERIFY cuBLASLt's GELU variant is erf, not tanh,
  before anything else — tanh fails the gate (PROGRAM.md contract #1).
- **K3 — Flash-decoding-style split-K for #9's occupancy hole.**
  [PyTorch flash-decoding](https://pytorch.org/blog/flash-decoding/): split the
  KV sequence across SMs when batch×heads < SM count. #9 has 64 blocks vs 108
  SMs. *Bounded:* attention is 14% of #9, so max whole-layer gain ~1.09x —
  likely under the noise floor. Log as considered-and-bounded; do not build.

## Recommendation against the clock

K1 is the only creative direction with headroom an order of magnitude above the
noise floor, and it is also the riskiest. With ~25 h and S8/S9 unrun:
**time-box K1 to one attempt (shape-2 target only), only after S8's sweeps are
queued**, K2 as its cheaper sibling if flash-attn-style effort is unavailable.
If neither lands, the honest tech-report line is strong anyway: "we measured
where attention matters, collected the flash-class wins there, and showed the
remaining gap is launch overhead, with the megakernel literature as the path."

## Limitations

Attention-share and SMEM figures are arithmetic from the shape table, not
profiles (M2 unrun). K2's erf-vs-tanh question is unresolved. Megakernel
citations describe H100/MI300X-era results; A100 persistent-kernel gains are
UNVERIFIED here.

## AI disclosure
Produced with AI assistance (Claude Opus 5); measurements from journal/leaderboard, papers as linked.
