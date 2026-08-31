# Three-agent research pass: K1 downgraded, K2 blocked, two new cheap items

Engineering brief · 2026-08-31 · quick mode
Method: three parallel Sonnet research agents, each given the correctness gate up
front so they would reject rather than recommend inadmissible work.
Sources cited inline; anything not read directly is labelled.

## 1. Linear attention (arXiv 2510.21956) — INADMISSIBLE, and its own data points away from us

User-supplied paper: Gerami & Duraiswami, *Transformer Based Linear Attention
with Optimized GPU Kernel Implementation*.

**Decisive:** their kernel is `f(x) = 1 + x` — a first-order Taylor truncation of
`exp(x)`. Paper's wording: *"substituting the exponential ... with a linear
approximation we arrive at Linear Attention."* It never claims exactness, only
"comparable accuracy" and "similar expressivity" — benchmark parity on a trained
1.4B model, not per-element agreement. Same rejection basis as Performer,
Linformer, BigBird, Longformer.

**Three further reasons it would not help even if admissible:**
- The headline **3.3x is linear-attention versus linear-attention** (vs Gated LA,
  Yang et al. 2023) — not versus softmax attention.
- Versus FlashAttention-2 they win only at **N > 3000**. Our N is **128**, 23x
  below their measured crossover — and that crossover was measured at D=128,
  while our D=32 pushes it higher still.
- Measured on an **A6000 in fp32**, never an A100. Their N-sweep never went below
  N=1000. No public code found.

**Salvageable — implementation craft, transfers to any exact kernel:** coalesced
layout with the thread-parallel axis first; shared-memory tiling of reused Q/K
(load once, reuse D times); register-resident accumulators with explicit
register-tiling on overflow; partition so each thread solely owns one accumulator
(**no atomics**) with explicit barriers between phases. Methodology worth
stealing: **profile off-chip bytes moved, not just wall clock** — they found
autograd costing ~100x more data movement than necessary.

## 2. K1 megakernel — LARGELY DEAD ON A100, on the literature's own numbers

| system | A100 result |
|--|--|
| AutoMegaKernel ([arXiv 2606.09682](https://arxiv.org/abs/2606.09682)) | **0.55-0.79x — SLOWER than baseline** |
| Mirage MPK ([arXiv 2512.22219](https://arxiv.org/abs/2512.22219)) | ~1.16x (Qwen3-8B, 14.5→12.5 ms/token) |
| Hazy Research Llama-1B ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [code](https://github.com/HazyResearch/Megakernels)) | never measured — repo hardcodes `GPU ∈ {H100, B200}` |

The 2.5-3.5x headlines are **Hopper-gated**: async TMA copy plus warp-specialized
overlap does most of the work, and A100's `cp.async` is weaker. Ada-MK
([arXiv 2605.11581](https://arxiv.org/abs/2605.11581)) confirms independently —
lacking TMA it needed hand-written PTX and still only reached +24-50% against
Hopper's +150-250%.

**Sharpest detail: AutoMegaKernel's baseline is CUDA-graphed cuBLAS, and
megakernels lose to it on A100.** We already ship CUDA graphs via
`reduce-overhead`. K1 would compete against something we have, on hardware where
the literature says it loses.

**Second problem:** the corpus is nearly all batch-1 autoregressive **decode**.
Our shapes are full forward passes. The transfer is weaker than K1 assumed.

**Effort reality:** Hazy Research is ~1.5K CUDA + 5K Python after weeks of
research; AutoMegaKernel is an agent-driven compiler (~180K CUDA + 928K Python
bytes). None was built in 25 hours.

**Survives:** our 192 KB weights and 56 KB attention working sets fit in shared
memory **without TMA** — the one part of the idea that is not Hopper-gated.
[DITRON](https://arxiv.org/abs/2605.02953) shows native Triton kernels can serve
as megakernel tasks. Recorded as a path, not taken.

## 3. K2 cuBLASLt GELU epilogue — BLOCKED on the erf/tanh gate

**Best evidence says cuBLASLt's GELU epilogue is the tanh approximation, not
erf-exact.** No NVIDIA document states the formula (cuBLAS docs, `cublasLt.h`,
Transformer Engine `gemm.h` are all silent). Convergent circumstantial evidence:

- CUTLASS names the exact-erf functor `GELU` and the tanh functor `GELU_taylor`
  as **distinct** epilogues
  ([activation.h](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/epilogue/thread/activation.h));
  cuBLASLt exposes only one unqualified `GELU_BIAS`.
- An engineer testing CUTLASS-vs-cuBLASLt parity reported *"cublasLt use tanh to
  approximate GELU"* and switched to `GELU_taylor` to match — unchallenged by the
  responding NVIDIA engineer
  ([cutlass discussion #700](https://github.com/NVIDIA/cutlass/discussions/700)).

UNVERIFIED via primary documentation, high-confidence via convergent evidence.
Our contract requires erf-exact (`approximate="none"`, PROGRAM.md #1). **Do not
risk it.** A 15-minute empirical diff against `F.gelu(approximate='none')` would
settle it if anyone wants ground truth.

**This makes K2 moot rather than merely risky — and points at a better route.**

## New items

- **K2' — get the bias+GELU epilogue fused through TRITON, not cuBLASLt.**
  Triton computes `erf` natively, so the numerics are safe by construction.
  `epilogue_fusion=True` is already the Inductor default; the question is whether
  our GEMMs are actually picking the Triton backend or falling back to ATEN.
  Action: drop `ATEN` from `max_autotune_gemm_backends`, or profile to confirm
  Triton already won. Low risk, low cost, and it collapses 1-2 launches per
  Linear+GELU on the twelve launch-bound shapes.
- **K4 — audit CUDA-graph replay for non-static tensor addresses. Cheapest
  high-variance item in the queue.** If parameters are not at static addresses,
  every graph replay silently inserts device-to-device copies.
  `torch._dynamo.mark_static_address` fixes it. **Binary outcome:** either we are
  already clean, or there is a hidden tax on every single call to our
  compiled routes. One profiler trace settles it — and it folds directly into M2,
  which is already open and unrun.
  ([CUDAGraph Trees docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html))

## Closed by this pass, do not revisit

- **CuteDSL / NVGEMM backend** — Inductor config comment: *"CUTEDSL: … SM100-SM109
  only"*. **Blackwell-only, does not run on sm80.**
- **CUTLASS C++ backend** (`max_autotune_gemm_backends="CUTLASS"`) — open,
  triaged, unfixed illegal-memory-access bug
  ([pytorch#171094](https://github.com/pytorch/pytorch/issues/171094)); plus
  `nvcc`-per-candidate compile time. High risk, low expected gain on d=128 GEMMs.
- **`coordinate_descent_tuning`** — open issue that it corrupts its own `do_bench`
  measurements ([pytorch#159525](https://github.com/pytorch/pytorch/issues/159525)).
  Low expected gain, nonzero risk of a misleading autotune result.

## Limitations

Agent-gathered; primary sources linked but not all read end to end by me. The
erf/tanh conclusion is circumstantial, not documented. A100 megakernel figures
come from two papers' reported numbers, not our own measurement. Nothing here has
been run on our cluster.

## AI disclosure
Three parallel Sonnet research agents under Claude Opus 5, each given the
correctness gate as a hard constraint. Findings cross-checked against our ledger.
