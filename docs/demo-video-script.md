# Track 3 Demo Video — Narration Script

Target runtime: ~2:45-3:00. Timestamps are guides, not hard cuts — let the screen
recording breathe where it needs to (especially the live run).

---

## 1. Hook — 0:00-0:10

**ON SCREEN:** Shape 14 chart (docs/shape14-precision-chart.html), full screen, no
narration for the first 2 seconds — let it sit.

**NARRATION:**
> "This chart is a nine-times speedup. Not from a clever kernel — from figuring out
> what was actually slow. That's what this project is: using AI-assisted tooling to
> optimize a Transformer layer's GPU kernels, and treating every claimed win as
> something to prove, not assume."

---

## 2. The setup — 0:10-0:30

**ON SCREEN:** `torch_transformer_benchmark.py` — scroll to `UserOptimizedTransformer`
stub. Then cut to the 14-shape table (from the spec / `docs/` — batch/seq_len/d_model/
heads columns visible). Then `bench_harness.py --help` or the correctness-gate constants
(rtol=0.02, atol=0.002).

**NARRATION:**
> "The task: replace this one class, `UserOptimizedTransformer`, and make it faster
> without changing what it computes. The organizer specifies fourteen fixed input
> shapes — everything from tiny batches to one shape with a hundred-thousand-token
> sequence length. Every candidate has to pass a real correctness gate against the
> baseline before any speedup number counts — relative error under 0.02, absolute
> error under 0.002."

---

## 3. Run it live — 0:30-1:10

**ON SCREEN:** Terminal, full width. Run:
```
python3 bench_harness.py --candidate candidates/v_router2_autotuned.py \
  --shapes 1,2,3,4,5,6,7,8,9,10,11,12,13 --dtype float32 --device cuda
```
Let it actually run and print. Don't cut away from the correctness/speedup JSON
printing per shape — this is the single most important shot in the video.

**NARRATION (over the run, trail off once output starts):**
> "So here's the current best candidate actually running against real GPU hardware
> — thirteen of the fourteen shapes, correctness checked every single time."

**(let it finish, then, over the final JSON block):**
> "Correctness passed on all thirteen. Median speedup 4.89x, geometric mean 4.88x,
> against the unmodified baseline."

---

## 4. The interesting part — shape 14 — 1:10-2:10

**ON SCREEN:** Quote/screenshot from `journal.jsonl` iter 7 — the line about job
777216 being cancelled at a 30-minute SLURM limit with zero progress. Then cut to the
shape-14 chart again, this time let each bar animate/appear if you want, or just hold
on the finished chart with the annotation visible.

**NARRATION:**
> "Shape fourteen didn't fit this run — and that's the actual story. Its sequence
> length is a hundred thousand tokens. Early on, we tried the standard route —
> torch.compile's max-autotune — and it never finished. We killed the job after
> thirty minutes with zero progress logged. Before this project, this shape was
> effectively broken.
>
> The fix that worked was splitting the batch into small chunks so it fits in
> memory at all. The obvious next question was: how small should the chunks be? So
> we measured it directly, instead of guessing. Chunk size four, chunk size eight,
> both in full precision — barely different, sixty-eight versus seventy-five
> seconds per forward pass. Then we tried switching to sixteen-bit floating point.
>
> Eight point two seconds. Chunk size sixteen, still fp16: eight point three —
> functionally the same as eight. So the chunk count was never the bottleneck.
> Precision was. This shape moves so much data through memory that halving the
> bytes per element mattered nine times more than any amount of clever chunking."

---

## 5. The aggregate win — 2:10-2:30

**ON SCREEN:** Simple stat card or the leaderboard.md diff — before/after numbers
side by side. Keep it to one screen, don't scroll through the whole file.

**NARRATION:**
> "Stacked with an autotuned kernel launch config on top of the existing routing
> table, the aggregate median moved from 4.57x to 4.89x, with correctness holding
> at thirteen out of thirteen — and shape fourteen went from a route that never
> completes to roughly eight seconds a pass."

---

## 6. AI tooling note — 2:30-2:45

**ON SCREEN:** Brief glimpse of the multi-agent git coordination files (`TODO.md`,
`journal.jsonl`, `AGENTS.md`) — enough to show real structure, not a deep dive.

**NARRATION:**
> "This was built with Claude Code driving the actual optimization work — proposing
> changes, dispatching them to GPU hardware, and only accepting a result once it was
> independently verified. Every claim in this video, including the shape-fourteen
> chart, came from a real measured run, not an estimate."

---

## 7. Close — 2:45-3:00

**ON SCREEN:** Back to the shape-14 chart, or a plain end card with the repo link.

**NARRATION:**
> "What's left: applying a pre-transposed weight-layout trick we validated in
> isolation but haven't stacked into the main routing table yet, and giving shape
> thirteen its own explicit route instead of relying on a fallback. Full write-up
> and reproduction steps are in the repo."

---

## Recording notes

- The live-run shot (section 3) is the one piece that can't be faked or skipped —
  it's the evidence the "Technical Execution" criterion is actually looking for.
- Keep section 4 unhurried; it's the strongest "Innovation & Problem Insight"
  material in the whole project — a real dead end, a real measurement, a real
  reversal of assumption (chunk size vs. precision).
- If time is tight, section 6 is the one to trim first — it's a bonus-points line
  in the spec, not a scored criterion on its own.
