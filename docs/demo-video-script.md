# Track 3 Demo Video — Narration Script

Target runtime: ~3:00. Matches `docs/pitch-deck.html`'s 6 slides one-to-one —
the deck's speaker-notes panel carries this same text per slide.

---

## 1. Team & responsibility — ~0:00-0:15

**ON SCREEN:** Slide 1 — project title, team roster (name / role / one-line
responsibility for each of the three members).

**NARRATION:**
> "Our team split the work into two layers. Jing Yen built the implementation
> side: the agent loop, the candidate kernels, and the benchmarking
> infrastructure. Shi Xian led research: problem analysis, the review
> process, and the tech report."

---

## 2. Objective — ~0:15-0:29

**ON SCREEN:** Slide 2 — one-sentence task statement, correctness-gate pill.

**NARRATION:**
> "Track 3 asks for a faster GPU implementation of a fixed Transformer
> block. The rules are simple: match the organizer's reference within a
> per-element tolerance, across fourteen published input shapes, or the
> speedup doesn't count."

---

## 3. Hardware — ~0:29-0:49

**ON SCREEN:** Slide 3 — RunPod A100-SXM4-80GB, 4 spec tiles (80GB / SXM4 /
on-demand / per-run lifecycle).

**NARRATION:**
> "We started on our university's shared A100 cluster, but the queue became
> the actual bottleneck — multi-hour waits between runs. We moved to
> RunPod, provisioning A100-SXM4-80GB instances on demand and tearing them
> down after each benchmark."

---

## 4. Agentic architecture — ~0:49-1:33

**ON SCREEN:** Slide 4 — two-layer diagram (research: plan → adversarial
review → reconcile; implementation: write candidate → benchmark + gate →
keep/prune), connected through `TODO.md` / `journal.jsonl`.

**NARRATION:**
> "Rather than hand-tune one kernel, we built two agent layers that only
> talk to each other through git. A research layer proposes a direction, an
> adversarial review checks it, and the two reconcile on what's worth
> testing next. An implementation layer writes one candidate, runs it
> through the benchmark and correctness gate, and keeps or prunes it — the
> next iteration reads back what happened from the same log."

---

## 5. Incremental improvements — ~1:33-2:33

**ON SCREEN:** Slide 5 — a box plot at five milestones (v_router → v_router2
→ +autotune/shape14 → +systematic reroute → +full correctness evidence),
each box showing the real min/Q1/median/Q3/max across all 13 shapes, with
the reported geomean overlaid as a separate dashed line.

**NARRATION:**
> "Five real steps got us from 2.87x to 6.54x geomean. Each box here is the
> actual spread across all thirteen shapes, not just the headline number —
> some shapes gain much more than others, and that spread is real too.
> Per-shape dispatch. Fp16 precision plus a custom fused kernel on the
> heaviest shapes. An autotuned kernel config that also fixed a shape that
> previously never finished. A systematic recheck that found six shapes on
> a stale route. And a final re-run to attach full correctness evidence to
> the result."

*Per-point detail (matches the chart's hover tooltips):*
1. Per-shape dispatch across four validated implementations — no new kernel code.
2. Fp16 autocast on the three most compute-heavy shapes, plus a custom fused Triton kernel.
3. Autotuned kernel launch config; shape 14 (100k-token sequence) moved from never finishing to ~8s per pass.
4. A systematic recheck found 6 of 13 shapes were still on a stale route — all confirmed with a real 6-way comparison.
5. Re-run specifically to capture full per-shape correctness margins — same result, now fully documented.

---

## 6. Wrap-up — ~2:33-2:58

**ON SCREEN:** Slide 6 — two columns: what was ruled out (pretransposed
Linear weights, native fp16 weights) and what's next (selective
pretransposition, an optional cross-check on the original cluster), plus
the repo link.

**NARRATION:**
> "Two ideas looked promising and didn't survive contact with the real
> system — pretransposed weights regressed once stacked on CUDA graphs,
> native fp16 weights failed correctness outright. Both are understood, not
> just abandoned. What's left: give pretransposition a fair test where it
> wasn't tried, and an optional cross-check on our original cluster."

---

## Recording notes

- ~360 words total at a natural pace (~150 wpm) runs about 2:25 — under the
  3:00 target, leaving room for slide-transition pauses and not rushing
  slide 4/5 (the two slides with the most to explain).
- Slide 5 is the one piece of hard evidence — don't cut away from it early;
  let each number land before advancing.
- Keep slides 1-3 brisk (they're intentionally light) so slides 4 and 5 get
  the time.
