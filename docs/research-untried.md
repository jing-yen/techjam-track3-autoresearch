# What we have not tried — audit at 4.02x geomean

Engineering brief · 2026-08-31 · quick mode
Method: full inventory of every technique attempted (52+ journal rows), then a
gap analysis against the current per-shape profile. Arithmetic computed inline.

## The landscape moved, and the queue has not caught up

`fusedcg` landing changed which shape matters. Current shares of remaining
optimized wall clock:

| # | opt ms | share | speedup | achieved | % of fp16 peak (312 TFLOP/s) |
|--|--|--|--|--|--|
| **6** | **56.54** | **83.0%** | 3.29x | 20.8 TFLOP/s | **6.7%** |
| 8 | 4.57 | 6.7% | 1.75x | 92.1 TFLOP/s | 29.5% |
| 1 | 1.21 | 1.8% | 2.17x | 6.2 TFLOP/s | 2.0% |
| 11 | 1.16 | 1.7% | 2.99x | 6.5 TFLOP/s | 2.1% |

**Every optimization item in the current queue targets shapes worth under 7% of
the runtime.** Shape 6 was only admitted to the sweep at S4 and has never been
the subject of a single item.

## Shape 6 is unexplained by either ceiling — and that is the finding

```
compute:    20.8 TFLOP/s achieved  =  6.7% of the fp16 tensor-core peak
bandwidth:  ~12.2 GB of HBM traffic (10 touches/layer x 4 layers, 0.31 GB each)
            at 1935 GB/s  =  6.3 ms floor    vs  56.5 ms measured  =  9x above
```

**Neither compute nor bandwidth explains where 56.5 ms goes.** That gap is the
largest unexplained quantity in the project, and it sits on 83% of the runtime.

M2 profiled shapes 1, 8 and 13. **It did not profile shape 6.**

## U1 — PROFILE SHAPE 6. Highest-value action available, and it is cheap.

`tools/profile_shapes.py` already exists (M2). Point it at shape 6. One trace
answers whether the 9x gap is dtype casting, launch/dispatch, attention
inefficiency at B*H = 40,000, or something nobody has guessed.

Everything below is a hypothesis that this trace either confirms or kills. **Do
not build any of them first.**

## Untried techniques, ranked

- **U2 — L2 cache blocking over the batch (NOT for memory — for locality).**
  Genuinely untried, and distinct from S5's chunking, which existed to fit in
  HBM. Shape 6's activation is **312 MB fp16 against A100's 40 MB L2**, so every
  layer re-reads it from HBM. Chunk the batch and run each chunk through **all
  four layers** while it stays L2-resident:

  | chunk | activation | L2-resident? |
  |--|--|--|
  | 2000 | 62.5 MB | no |
  | **1000** | **31.2 MB** | **yes** |
  | 500 | 15.6 MB | yes |

  HBM traffic would fall from ~12.2 GB to roughly one read + one write of the
  full tensor (~0.6 GB) plus weights — an order of magnitude. Exact, no
  approximation, and the same mechanism S5 already proved correct on #14.
  *Conditional on U1 showing memory movement dominates.*

- **U3 — end-to-end fp16 instead of `autocast`.** M2 measured **17.1% of shape
  8's time in dtype casting** (`aten::to`, `_to_copy`, `copy_`). Autocast casts
  at every op boundary. Casting `x` to fp16 once at block entry and back once at
  exit removes the intermediate churn, while LayerNorm and softmax stay fp32 by
  explicit construction. Distinct from T16, which pre-cast **weights**; this is
  about **activations**. Applies to #6, #8, #13 — i.e. 90% of the runtime.
  *Risk:* moves error, and #8 is already at 88% of the atol budget. Gate on S9.

- **U4 — compile the fused/fusedcg routes.** `compile` and `fused` are separate
  route targets that have never been combined. Recent PyTorch supports custom
  Triton kernels inside compiled regions
  (`torch.library.triton_op` / `capture_triton`), so Inductor could fuse
  *around* the AddNorm kernel instead of stopping at it. T17 reached for manual
  CUDA graphs instead — which worked, but is a different lever.

- **U5 — weight pre-transposition.** `nn.Linear` stores `[out, in]` and computes
  `x @ W.T`. Pre-transposing once at init lets cuBLAS pick an NN kernel variant
  rather than NT. Cheap A/B test, never run. Small, but shape 6 is 86% GEMM.

- **U6 — A100 L2 persistence window.** `cudaStreamSetAttribute` with
  `accessPolicyWindow` pins a chosen address range in L2. Weights are 384 KB
  fp32 per layer, re-read across all four layers and across 10,000 batch items.
  A real Ampere feature that appears nowhere in this project. Reachable from
  PyTorch only via a small extension, so cost is real — list it, do not start it.

## Explicitly closed, do not revisit

Megakernel/K1a (both variants, decisively — iters 46/47), linear attention
(approximation), cuBLASLt (tanh GELU), CUTLASS backend (open IMA bug), CuteDSL
(Blackwell-only), `coordinate_descent_tuning` (open `do_bench` bug), T10 fused
FFN fp16 (loses to cuBLASLt), T16 weight pre-cast, bf16 (fails 13/13), stream
pipelining on #14 (slower), PagedAttention (no KV cache).

## Recommendation with ~18 hours

1. **U1 — profile shape 6.** Minutes. It is 83% of the runtime and has never been
   looked at.
2. Then **at most one** of U2/U3, whichever the profile points to.
3. Freeze at T-12h regardless. `fusedcg` at 4.02x is already a strong, confirmed,
   13/13-correct result — do not risk it for an unmeasured hypothesis.

## Limitations

Shape 6's TFLOP/s uses the appendix FLOP model, not a counter. The "10 touches
per layer" traffic estimate is a rule of thumb, so the 6.3 ms floor is
order-of-magnitude only. The L2-blocking gain is arithmetic, UNVERIFIED — it
assumes the four layers can be restructured to run per-chunk without breaking
the router's shape contract. Shape 13 is absent from the current leaderboard
per-shape table, so its share is excluded here.

## AI disclosure
Produced with AI assistance (Claude Opus 5); shares and ceilings computed inline
from `leaderboard.md` and the appendix shape table.
