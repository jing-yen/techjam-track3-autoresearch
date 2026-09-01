# Track 3 Demo Video — Narration Script

Target runtime: ~3:00. Matches `docs/pitch-deck.html`'s 6 slides one-to-one,
in the deck's current order — the deck's speaker-notes panel carries this
same text per slide (open it with the "notes" button or press `n`).

---

## 1. Team & responsibility — ~0:00-0:15

**ON SCREEN:** Slide 1 — TikTok TechJam lockup, project title, team roster
with photos (name / role / one-line responsibility for each of the three
members).

**NARRATION:**
> "Our team split the work into two layers.
>
> Jing Yen and Brandon built the implementation side — the agent loop, the
> candidate kernels, the benchmarking infrastructure.
>
> Shi Xian led research — problem analysis, the review process, the tech
> report."

---

## 2. Objective — ~0:15-0:29

**ON SCREEN:** Slide 2 — one-sentence task statement, correctness-gate pill,
the standard Transformer block diagram (Vaswani et al., 2017) alongside it.

**NARRATION:**
> "Track 3 asks for a faster GPU implementation of a fixed Transformer
> block.
>
> That's the pattern on the right — self-attention and feed-forward, each
> with a residual connection and LayerNorm.
>
> The rule is simple: match the organizer's reference within a per-element
> tolerance, across fourteen published shapes. Or the speedup doesn't
> count."

---

## 3. Hardware — ~0:29-0:53

**ON SCREEN:** Slide 3 — RunPod A100-SXM4-80GB, real A100 photo, a 2x3 spec
grid (80GB / SXM4 / 312 TFLOPS FP16 / 19.5 TFLOPS FP32 / on-demand /
FlashAttention 0/14→14/14 fp32→fp16).

**NARRATION:**
> "We started on our university's shared A100 cluster.
>
> But the queue became the actual bottleneck — multi-hour waits between
> runs.
>
> So we moved to RunPod. A100-SXM4-80GB instances, on demand, torn down
> after every benchmark.
>
> One spec worth calling out — FlashAttention only turns on at fp16 here.
> Zero of fourteen shapes at fp32. All fourteen once we route through fp16.
> That's exactly why precision routing matters later."

---

## 4. Incremental improvements — ~0:53-1:53

**ON SCREEN:** Slide 4 — a box plot starting from an explicit 1x reference
point, then six milestones (v_router → v_router2 → +autotune/shape14 →
+shape 13 re-routed → +systematic reroute → +full correctness evidence),
each box showing the real min/Q1/median/Q3/max across all 13 shapes, with
the reported geomean overlaid as a separate dashed line.

**NARRATION:**
> "Starting from the unoptimized reference, at 1x.
>
> Six real steps got us to 6.54x geomean.
>
> Each box here is the actual spread across all thirteen shapes — not just
> the headline number. Some shapes gain much more than others. That spread
> is real too.
>
> Per-shape dispatch. Fp16 plus a fused kernel on the heaviest shapes. An
> autotuned kernel config that also fixed a shape that previously never
> finished. One specific shape found badly mis-routed. A systematic
> recheck that found five more shapes on a stale route. And a final re-run
> to attach full correctness evidence to the result."

*Per-point detail (matches the chart's hover tooltips):*
1. Per-shape dispatch across four validated implementations — no new kernel code.
2. Fp16 autocast on the three most compute-heavy shapes, plus a custom fused Triton kernel.
3. Autotuned kernel launch config; shape 14 (100k-token sequence) moved from never finishing to ~8s per pass.
4. Shape 13's fallback route lost 4x to the actual best option — a single mis-routed shape, fixed.
5. A systematic recheck found 5 more of 13 shapes were still on a stale route — all confirmed with a real 6-way comparison.
6. Re-run specifically to capture full per-shape correctness margins — same result, now fully documented.

---

## 5. Summary — ~1:53-2:23

**ON SCREEN:** Slide 5 — the same six milestones from slide 4's chart, now
as a recap table, each tagged by what kind of win it was (S-tier / Unlock /
Rigor / Baseline), plus a 6-card grid of ideas that were tried and didn't
make the cut, and the repo link.

**NARRATION:**
> "Ranking the six steps by what they actually bought us.
>
> The fp16-and-fused-kernel change, and the two re-routing passes — those
> were the real speed wins.
>
> The autotune-and-shape-14 step barely moved the aggregate. But it
> unblocked a shape that never completed before.
>
> The last step wasn't about speed at all — it closed a gap in our own
> evidence.
>
> And six other ideas didn't survive contact with the real system. Most
> rejected outright. One narrowed to just the shapes it actually helps.
>
> All of them understood, not just abandoned."

---

## 6. Agentic architecture — ~2:23-3:08

**ON SCREEN:** Slide 6 — system diagram: research agent (① plan → ②
adversarial review → ③ reconcile) and implementation agent (④ execute → ⑤
verify → ⑥ log), connected only through a shared-memory hub (`TODO.md` /
`journal.jsonl`), with directional read/write arrows both ways, and a
"models used across the loop" badge row (Claude / Codex / Gemini / Doubao).

**NARRATION:**
> "This is a self-improving multi-agent loop.
>
> Six phases, two agents, sharing no state except two git-tracked files.
>
> The research agent plans a direction, an adversarial review checks it,
> the two reconcile. That's phases one through three.
>
> The implementation agent executes one candidate, verifies it against the
> correctness gate, logs the result. Phases four through six.
>
> And the next iteration's plan reads that same log — so the loop actually
> improves itself over time. It doesn't just repeat blind."

---

## Recording notes

- The deck deliberately closes on the architecture slide rather than the
  results — leads with the numbers, ends on how the numbers got made.
- ~430 words total at a natural pace (~150 wpm) runs about 2:50-3:05,
  close to the 3:00 target — don't rush slide 4 (results) or slide 6
  (architecture), the two slides with the most to explain; slides 1-3 stay
  brisk by design.
- Speaker notes in the deck now render with the same paced line breaks as
  this script (toggle with the "notes" button or press `n`).
- The nav controls and notes panel live outside the deck's 16:9 frame on
  purpose — if you're screen-recording just the frame, they won't appear
  in the recording.
