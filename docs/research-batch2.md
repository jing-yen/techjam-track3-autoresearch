# Second results batch: shape 14 runs, S2 falsified

Engineering brief · 2026-08-31 05:40 SGT · quick mode
Source: journal iters 15-28, `leaderboard.md`, commits `8bd1e48..a039f0d`

## Headline

**2.89x median / 3.58x geomean, 13/13 shapes** including shape 6
(`leaderboard.md:10`, iter 21, job `step4_confirm_a80`). Up from 2.67x/2.98x.

## S5 succeeded: shape 14 is no longer infeasible

Journal iter 22, `v_chunked14`: **RUNS.** Full forward pass over
`32 x 100000 x 1024`, causal, 2 layers, in **~74.6 s** via chunk=4 exact batch
splitting. Previously OOMed categorically at 73.85 GB / 79.25 GB.

This **overturns the iter-7 "confirmed infeasible" conclusion.** Note what
changed: not the hardware, not the attention algorithm, and not the precision —
only the recognition that B=32 is 32 independent sequences and nothing in the
block couples them.

Correctness is unverifiable directly (`status=baseline_oom`; the reference needs
18.6 TB for its score matrix, so no ground truth can exist). The chunking
*mechanism* was validated on shapes that do have references — the protocol
recommended in `docs/research-shape14.md`.

**Caveat on the timing:** measured under a reduced protocol (warmup 2, repeats 3,
rounds 1) because the official 320-call protocol did not finish inside a 30-minute
limit (job 777216, cancelled with zero progress). The 74.6 s figure is therefore
**not comparable** to the other shapes' official-protocol numbers and must not be
folded into any aggregate. UNVERIFIED as a steady-state measurement.

**Coverage is now 14/14 attempted**: 13 verified against a reference, 1 running
without one.

## S2 was falsified — and the failure mode is the most useful result here

Isolated pairwise test said `reduce` beat `fused` on shapes 9 and 10
(3.65x / 3.98x). The full sweep said the opposite: `reduce` gave **1.81x / 2.01x
versus fused's 2.09x / 2.33x in the same run** (journal iter 21). Reverted.

The hypothesis on record is a `torch.compile` / CUDA-graph memory-pool
interaction between five `reduce` instances compiling concurrently in one
process — invisible in a two-shape isolated test. INFERRED; no profile confirms
the mechanism.

**The transferable finding is methodological: isolated candidate benchmarks do
not predict full-sweep behaviour once `torch.compile` is involved.** Any routing
decision must be validated in the sweep it will ship in. This belongs in the
tech report — it is the kind of result that only appears when you re-measure
something you already "knew".

## T6 (AMP fp16) is where the geomean gain came from

Per `leaderboard.md:25`, routing `torch.autocast(fp16)` onto shapes 6, 8 and 13:

| shape | before | after |
|--|--|--|
| 13 | 4.48x | **10.50x** |
| 8 | 1.28x | 1.68x |
| 6 | 2.81x | 2.89x |

Shape 13's 2.3x jump is consistent with the earlier backend probe finding that
flash SDPA is eligible on all 14 shapes at fp16 and none at fp32 — the
long-sequence shape is where that matters most.

## Shape 8 remains the weakest, and its ceiling is now ambiguous

At 1.68x against a 7.882 ms baseline, INFERRED `opt_ms` ~4.69 ms → ~89.7 TFLOP/s.

| ceiling | position |
|--|--|
| TF32 156 TFLOPS | 57% |
| fp16 tensor core 312 TFLOPS | 29% |

Which applies is genuinely unclear, because `autocast` runs matmuls in fp16 while
keeping reductions in fp32. INFERRED throughput from a ratio, not a measured
`opt_ms`. Headroom plausibly remains, but less than the fp32-era 44% figure
suggested, and the shape is no longer obviously the best use of remaining time.

## Highest-value remaining action

**Not a kernel. The ledger-to-document gap.**

`per_shape` is empty on all 28 journal rows, including every A100 sweep. The only
13-shape data anywhere is prose in `leaderboard.md`, and its per-shape table is
still the **pre-AMP 12-shape sweep** showing #13 at 4.48x and #8 at 1.29x.

Consequence: `README.md` and `TECH_REPORT.md` now carry results tables that
disagree with the leaderboard headline. A submission whose own documents
contradict each other is a worse outcome than one tenth of a speedup.

**Action:** re-emit the iter-21 sweep's per-shape rows into `journal.jsonl` from
the raw harness JSON on the cluster, then regenerate both tables mechanically.
Minutes of work; it is the last thing standing between the measurements and the
deliverable.

## Limitations

1. Shape 8's throughput is derived from a speedup ratio, not a measured `opt_ms`.
2. Shape 14's 74.6 s is a reduced-protocol number and is not comparable to the
   other shapes.
3. The S2 memory-pool explanation is a hypothesis with no supporting profile.
4. This brief read only the new journal rows, `leaderboard.md` and the new commit
   messages, per its scheduling constraints. Candidate source was not re-read.

## AI disclosure

Produced with AI assistance (Claude Opus 5) on a scheduled unattended run.
Figures are from `journal.jsonl` iters 15-28 and `leaderboard.md`; derived
quantities are computed inline and labelled INFERRED.
