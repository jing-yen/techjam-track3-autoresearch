# Round 1 corrections — the fact-checker was right, my brief was not

Pipelined research loop, round 1 · 2026-08-31
Two researchers + one fact-checker verifying the previous round.
**Result: `docs/research-untried.md`'s headline finding and top recommendation
are both retracted.** Corrections below, each independently re-verified by me
before acceptance.

## Retracted

| claim in `research-untried.md` | reality |
|--|--|
| "shape 6 has never been profiled" — the entire basis of U1 | **It was profiled four minutes before I wrote that.** `docs/research-shape6-profile.md`, journal **iter 53** (`profile-shape6`, job 779400) — the last row in the ledger at the time |
| "56.54 ms = **83.0%** of remaining wall clock" | **78.8%.** Shape 13 IS in the leaderboard table (`43.178 → 3.681, 11.73x`). Sum of all 13 = 71.769 ms |
| Limitations: "Shape 13 is absent from the current per-shape table" | **False** — and it was the excuse that produced the arithmetic error |
| "neither compute nor bandwidth explains 56.5 ms — largest unexplained quantity in the project" | **It was already explained.** The trace attributes ~100% of the time |
| U5: "shape 6 is 86% GEMM" | **Unsourced.** No derivation exists anywhere in the repo. The profile shows the fp16 GEMM kernel at **29.57%** of CUDA time |
| pytorch#171094 (CUTLASS backend) "open, triaged, unfixed" | **CLOSED 2026-03-10**, fixed by an NVIDIA engineer |
| pytorch#159525 (`coordinate_descent_tuning`) "open bug" | **CLOSED 2025-07-31**; thread concludes the effect "shouldn't affect results" |
| shapes 1 and 11 measured against the **fp16** 312 TFLOP/s peak | **Scope error.** Those route to `compile`/`fused` — fp32/TF32. Wrong denominator; should be 19.5 or 156 TFLOP/s |

**Root cause:** `TODO.md` carried both a stale "HAS NEVER BEEN PROFILED" header
and a correct "M2-shape6 … UNCLAIMED" entry. I read the stale one and did not
check the ledger's last row before building a brief on top of it.

## Survived verification

1935 GB/s for **A100-80 PCIe** (confirmed against the datasheet's PCIe column —
round 1's other agent "corrected" this to the SXM figure 2039 and was wrong);
312 TFLOP/s fp16 peak; 20.8 TFLOP/s → 6.7%; 312 MiB activation vs 40 MB L2;
17.1% casting on shape 8 (iter 36); the entire linear-attention rejection
verbatim from the PDF; AutoMegaKernel 0.55–0.79× on A100; MPK 14.5→12.5 ms;
Hazy Research's `GPU={H100,B200}`; cuBLASLt tanh GELU; CuteDSL SM100-109 only.
No correctness-gate violations found in U2–U6.

## What the shape-6 profile actually says

Not "unexplained" at all — and it points somewhere specific:

| component | share of CUDA time |
|--|--|
| fp16 GEMM | 29.57% |
| fused AddNorm (T7/T15) | 28.45% |
| **fp32↔fp16 casting** | **19.37%** |
| flash attention | 13.24% |
| LayerNorm | 5.86% |
| GELU | 3.52% |

Plus a **"Command Buffer Full" signal at 39.65% of CPU time** — CPU-side dispatch
pressure, which is exactly what T17's manual CUDA-graph capture already fixed for
shapes 9-12.

## Revised recommendation

- **U1 is dead** — replaced by **U1' : apply T17's CUDA-graph capture to the amp
  route (shapes 6/8/13).** The profile's own dispatch-pressure signal points at
  it, and the mechanism is already built, tested and shipped on the fused route.
  This is the strongest lead in the project and it did not require new research.
- **U3 gains direct support** — 19.37% casting on shape 6, alongside 17.1% on
  shape 8. Same tax on both, ~86% of runtime between them. Note round 1's other
  agent established that autocast **already** keeps LayerNorm/softmax in fp32, so
  the win is removing per-op boundary casts, not changing the precision policy.
- **Re-test, do not assume broken:** CUTLASS backend and
  `coordinate_descent_tuning` are both fixed upstream. Whether the fixes are in
  the pinned torch 2.10.0 is UNVERIFIED — cheap to check, and I had wrongly
  closed both.

## What this says about the workflow

Round 1 of the pipeline caught an error that would have sent the implementation
layer to redo work already finished, with roughly 17 hours left. That is the
entire value proposition, demonstrated once: **the fact-checker is worth its cost
precisely when it contradicts the research it is checking.**

Notable that it also adjudicated a disagreement between the two researchers —
agent A "corrected" the PCIe bandwidth to the SXM figure, and the checker
restored the original by reading the right column of the datasheet.

## AI disclosure
Pipelined loop: two Sonnet researchers plus one Sonnet fact-checker under Claude
Opus 5. Every finding against my own prior brief was independently re-verified by
me (file existence, journal row, arithmetic, `gh issue view`) before acceptance.
