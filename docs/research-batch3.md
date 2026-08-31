# Third results batch: T7 lands, and the headline delta is inside the noise

Engineering brief · 2026-08-31 · quick mode
Source: journal iters 27-32, `leaderboard.md`, commits `37a9d06..3d10aa9`

## Headline

**2.99x median / 3.72x geomean, 13/13 shapes** (`leaderboard.md`, iter 30, job
`router2_triton_confirm2`). Up from 2.89x/3.58x. `v_router2` now carries T5
dispatch + B10 mask-cache + T6 AMP on 6/8/13 + **T7 Triton AddNorm** on the
best/amp routes.

**The absolute number is sound. The improvement narrative is not.** See below.

## What landed

**T7 — Triton fused residual-add + LayerNorm.** Survey (iter 27) picked AddNorm
over the fused-FFN epilogue after noting `ffn_dim == d_model` on all 14 official
shapes. Confirmed on real GPU (iter 29): 13/13 correct, standalone geomean
1.88x → 2.02x. Integrated (iter 30) and confirmed under fp16 autocast, which
resolves the previously-untested Triton/autocast interaction.

**L2 — closed as a negative result.** Two-stream CUDA pipelining of the shape-14
chunks is **6.6% slower** (79.6 s vs 74.6 s, iters 27-28). The chunks are already
GPU-resident, so there is no host-device transfer to hide behind compute — the
classic prefetch-while-computing pattern does not apply. Predicted in the
candidate's own docstring before it was run, which is the right order.

## The finding this batch actually turns on

Journal iter 30 records an attribution caveat: **untouched routes — zero code
changes from the prior confirmed run — swung by up to −30% run-to-run** (shape 7:
5.85x → 4.08x). That is real protocol/cluster variance, correctly not attributed
to Triton.

Carried one step further, which the note stops short of:

```
claimed gain, geomean            3.58 → 3.72   = +3.9%
attributable Triton gain (AMP routes only)     = +1.6%
one untouched shape's observed swing, on a
13-shape geomean                               = ±2.7%
```

**The honestly-attributable gain (+1.6%) is smaller than the noise contributed by
a single shape (±2.7%), and the headline delta (+3.9%) is barely above it.**

INFERRED, from the two aggregate figures and the one documented per-shape swing.
A proper variance estimate needs repeat sweeps of an unchanged candidate, and
none exist — every sweep in the ledger changed something.

**What follows for the report.** Quote 2.99x/3.72x — it is a legitimate
official-protocol measurement. Do **not** narrate "Triton bought us +3.9%." State
the attributable per-shape gains, which are consistent and directionally sound
(#6 +11%, #8 +4%, #13 +6%), and disclose the variance. The teammate's caveat
already says most of this; the report should carry it, not bury it.

**Recommended, and cheap:** re-run the current champion unchanged, twice. Two
extra sweeps convert "we suspect ±30% variance" into a measured error bar, and a
measured error bar on your headline is worth more to Technical Execution than
another 0.1x.

## Second risk: the error budget is 88% consumed

```
fp32 SDPA baseline        max_abs ~1e-6
+ Triton AddNorm          max_abs  0.0006 - 0.0013
+ fp16 autocast (#8)      max_abs  0.00176      atol = 0.002
```

Three independent error sources now stack, and the worst case sits at **88% of
the absolute tolerance**.

**Important qualifier, because this is easy to misread.** `max_abs` is *not* the
gate. The gate is per-element `abs <= 0.002` **OR** `rel <= 0.02`
(`torch_transformer_benchmark.py:314-316`), and `passed = failed_elements == 0`
(`:345-350`). An element at 0.00176 passes comfortably on the relative arm if
`|ref|` is not tiny. So this is not a near-miss.

What it is: a **trend**. Every accepted optimization has widened the error, the
harness runs a fixed seed, and no seed sweep has been done. UNVERIFIED whether a
different seed produces a failing element on shape 8.

**Recommended, and cheap:** re-run shape 8 with 2-3 different `--seed` values. If
it holds, that is a strong, defensible correctness claim for the report. If it
does not, better to find out now than in judging.

## Highest-value remaining actions

1. **Variance and seed robustness** (above). Two or three sweeps. Converts the
   two weakest claims in the submission into measured ones.
2. **S6 — `per_shape` is still empty on all 32 journal rows.** The README results
   table remains the pre-AMP 12-shape sweep. `TECH_REPORT.md` §1/§5/§7/§8 were
   synced by hand (commit `3785c09`), the README table was not.
3. **Team names.** 5 placeholders, the only thing blocking the README.

Not recommended with the time left: T7b (autotune) and T9 (SOTA library scope).
Both target gains smaller than the measurement noise documented above.

## Limitations

1. The variance estimate rests on **one** documented per-shape swing. No repeat
   run of an unchanged candidate exists.
2. Geomean sensitivity is computed analytically from that single swing, assuming
   the other twelve shapes are stable — which is exactly what is unverified.
3. Seed robustness is untested; the 88% figure is a single-seed observation.
4. This brief read only the new journal rows, `leaderboard.md`, and the new commit
   messages. Candidate and kernel source were not re-read.

## AI disclosure

Produced with AI assistance (Claude Opus 5). Figures are from `journal.jsonl`
iters 27-32 and `leaderboard.md`; derived quantities are computed inline and
labelled INFERRED.
