# Track 3 Demo Video — Narration Script

Target runtime: ~3:00. Matches `docs/pitch-deck.html`'s 6 slides one-to-one —
the deck's speaker-notes panel carries this same text per slide.

---

## 1. Team & responsibility — ~0:00-0:15

**ON SCREEN:** Slide 1 — TikTok TechJam lockup, project title, team roster
with photos (name / role / one-line responsibility for each of the three
members).

**NARRATION:**
> "Our team split the work into two layers. Jing Yen and Brandon built the
> implementation side: the agent loop, the candidate kernels, and the
> benchmarking infrastructure. Shi Xian led research: problem analysis, the
> review process, and the tech report."

---

## 2. Objective — ~0:15-0:29

**ON SCREEN:** Slide 2 — one-sentence task statement, correctness-gate pill,
the standard Transformer block diagram (Vaswani et al., 2017) alongside it.

**NARRATION:**
> "Track 3 asks for a faster GPU implementation of a fixed Transformer
> block — the standard pattern on the right, self-attention and
> feed-forward, each with a residual connection and LayerNorm. The rules
> are simple: match the organizer's reference within a per-element
> tolerance, across fourteen published input shapes, or the speedup doesn't
> count."

---

## 3. Hardware — ~0:29-0:49

**ON SCREEN:** Slide 3 — RunPod A100-SXM4-80GB, real A100 photo, a 2x3 spec
grid (80GB / SXM4 / 312 TFLOPS FP16 / 19.5 TFLOPS FP32 / on-demand /
FlashAttention 0/14→14/14 fp32→fp16).

**NARRATION:**
> "We started on our university's shared A100 cluster, but the queue became
> the actual bottleneck — multi-hour waits between runs. We moved to
> RunPod, provisioning A100-SXM4-80GB instances on demand and tearing them
> down after each benchmark. One spec worth calling out: FlashAttention is
> only eligible at fp16 on this card for our shapes — zero of fourteen at
> fp32, all fourteen once we route through fp16 — which is exactly why
> precision routing matters later."

---

## 4. Agentic architecture — ~0:49-1:33

**ON SCREEN:** Slide 4 — system diagram: research agent (① plan → ②
adversarial review → ③ reconcile) and implementation agent (④ execute → ⑤
verify → ⑥ log), connected only through a shared-memory hub (`TODO.md` /
`journal.jsonl`), with directional read/write arrows both ways.

**NARRATION:**
> "This is a self-improving multi-agent loop, six phases across two agents,
> sharing no state except two git-tracked files. The research agent plans a
> direction, an adversarial review checks it, and the two reconcile — that's
> phases one through three. The implementation agent executes one candidate,
> verifies it against the correctness gate, and logs the result — phases
> four through six. The next iteration's plan reads that same log, so the
> loop actually improves itself over time instead of repeating blind."

---

## 5. Incremental improvements — ~1:33-2:33

**ON SCREEN:** Slide 5 — a box plot starting from an explicit 1x reference
point, then six milestones (v_router → v_router2 → +autotune/shape14 →
+shape 13 re-routed → +systematic reroute → +full correctness evidence),
each box showing the real min/Q1/median/Q3/max across all 13 shapes, with
the reported geomean overlaid as a separate dashed line.

**NARRATION:**
> "Starting from the unoptimized reference at 1x, six real steps got us to
> 6.54x geomean. Each box here is the
> actual spread across all thirteen shapes, not just the headline number —
> some shapes gain much more than others, and that spread is real too.
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

## 6. Wrap-up — ~2:33-2:58

**ON SCREEN:** Slide 6 — the same five milestones from slide 5's chart, now
as a recap table, each tagged by what kind of win it was (S-tier / Unlock /
Rigor / Baseline), plus a one-line mention of the two ruled-out ideas and
the repo link.

**NARRATION:**
> "Ranking the five steps by what they actually bought us: the
> fp16-and-fused-kernel change and the systematic re-route were the two real
> speed wins. The autotune-and-shape-14 step barely moved the aggregate but
> unblocked a shape that never completed before. And the last step wasn't
> about speed at all — it closed a gap in our own evidence. Two other ideas
> didn't survive contact with the real system, and both are understood, not
> just abandoned."

---

## Recording notes

- ~360 words total at a natural pace (~150 wpm) runs about 2:25 — under the
  3:00 target, leaving room for slide-transition pauses and not rushing
  slide 4/5 (the two slides with the most to explain).
- Slide 5 is the one piece of hard evidence — don't cut away from it early;
  let each number land before advancing.
- Keep slides 1-3 brisk (they're intentionally light) so slides 4 and 5 get
  the time.
