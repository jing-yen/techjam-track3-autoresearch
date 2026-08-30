# TODO — Idea backlog & claim board

Claim an `Open` item by moving it to `In progress` with your agent-id (see
`AGENTS.md` §1). Ranked by expected impact. Add follow-ups freely.

**Every item below carries `file:line` evidence. Do not add an item without it.**
Mark anything unmeasured `UNVERIFIED` and say what would settle it.

---

## What actually scores (§3.6) — read before ranking anything

| Criterion | Weight |
|---|---|
| Technical Execution | 35% |
| Innovation & Problem Insight | 20% |
| Impact & Relevance | 20% |
| Feasibility & Practicality | 15% |
| Presentation & Communication | 10% (final event only) |

**There is no line item for raw speedup.** Nothing in the rubric scores
milliseconds. What scores is well-structured code, thoughtful architecture,
sharpness of problem understanding, and a demo that runs reliably.

§3.2: "The use of AI tools is encouraged so that the participants can implement
different kernels for different input shapes in limited time."
§3.5: "Provide a clear tech report including details on the AI skills/tools used
to get bonus points."

- The autoresearch swarm is **not a detour from the task**. It is the strongest
  available answer to Innovation (20%) and to half of Technical Execution (35%),
  and it is explicitly what the organizers asked for. A measured 1.4x with a
  documented agent loop, a correctness ledger and a per-shape dispatch table
  beats an undocumented 2x.
- Corollary: the §3.5 deliverables are **not overhead to do after the kernel
  work. They are most of the score.** Start them in parallel, now.
- §3.3 puts "production-ready deployment" out of scope. Do not gold-plate.

## READ FIRST — three PROGRAM.md claims are false

Verified against the organizer script on 2026-08-30. Fix these beliefs before
writing code against them.

1. **`valid_token_mask` is NEVER `None`.** At `padding_ratio<=0` (the default,
   `torch_transformer_benchmark.py:614`) `generate_random_case` returns an
   **all-True mask**, not `None` (`torch_transformer_benchmark.py:255-259`).
   Every caller passes it: `:391 :392 :472 :494 :504`,
   `bench_harness.py:197 :203`.
   → `candidates/best.py:71` (`if valid_token_mask is None`) is **dead code**;
   `is_causal=True` at `best.py:74` never executes.
   → `PROGRAM.md:35` and `PROGRAM.md:103` ("this is what makes shape #14
   feasible") are **false as implemented**. See B1.

2. **`~40 GB/head` is wrong** (`PROGRAM.md:103`, `bench_harness.py:25`). 37 GB is
   one `(batch,head)` slice of the score matrix. There are 32x16 = 512 of them →
   **18.6 TB** fp32 total.

3. **TF32 is already enabled for both sides** — `allow_tf32` default True,
   `matmul_precision="high"` (`torch_transformer_benchmark.py:638-645, :684-688`;
   `bench_harness.py:288-291, :346-347`). The baseline already runs fp32 matmuls
   on tensor cores. "Switch to tensor cores" is not an available win, and this is
   why the 0.002/0.02 gate is generous.

---

## MEASURED STATE — 2026-08-31, A100-80 (xgph1), fp32, official protocol

`candidates/v_router.py`, **2.27x median / 2.47x geomean, 12/12 correct**
(journal iter 9). Per-shape, with achieved arithmetic throughput computed from
the appendix FLOP model and `attn blocks` = `batch x heads` (one attention
program per (b,h) pair; A100 has **108 SMs**):

| # | speedup | opt ms | TFLOP/s | % of 19.5 fp32 peak | attn blocks | routed to |
|--|--|--|--|--|--|--|
| 2 | 5.07x | 0.374 | — | — | 4 | compile |
| 13 | 4.42x | 14.031 | 8.6 | 44% | 256 | fused |
| 3 | 4.24x | 0.447 | — | — | 16 | compile |
| 7 | 3.59x | 0.527 | — | — | 256 | compile |
| 11 | 2.73x | 1.858 | 4.0 | 21% | 1024 | fused |
| 12 | 2.36x | 0.793 | — | — | 256 | fused |
| 4 | 2.17x | 0.859 | — | — | 64 | best |
| 1 | 2.02x | 1.300 | 5.8 | 30% | 256 | compile |
| **5** | **1.86x** | 2.495 | 6.0 | 31% | 512 | fused |
| **10** | **1.70x** | 1.368 | 5.5 | 28% | 128 | fused |
| **9** | **1.47x** | 1.345 | 5.6 | 29% | 64 | fused |
| **8** | **1.14x** | 26.284 | **16.0** | **82%** | 256 | fused |

**Where the remaining time actually is** (share of the 55.4 ms optimized total):
#8 = **47%**, #13 = 25%, all ten others = 28%. Ranking by speedup ratio hides
this; #8 is half the machine's remaining work.

Hardware ceilings used above, from the vendor datasheet: A100 **FP32 19.5
TFLOPS**, **TF32 Tensor Core 156 TFLOPS**, 108 SMs
(https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf).

---

## Open — SHAPE 6 HAS NEVER BEEN RUN, AND PROBABLY WORKS

- **S4 — Run shape #6. Highest expected value per minute in the queue.**
  **Zero attempts.** No `journal.jsonl` row references shape 6; it has been
  excluded from every sweep since the first, because `bench_harness.py`'s
  `official-safe` set was defined as "everything except the two extreme-memory
  shapes (#6 B=10000, #14 seq=100k)". That definition was written on a MacBook
  before anyone had GPU access, by inspection of which numbers looked large.
  For #14 the guess was correct. **For #6 it is wrong by more than 10x:**

  | | fp32 |
  |--|--|
  | one `[B,S,D]` activation | 0.61 GB |
  | ~7 live tensors | 4.27 GB |
  | baseline `[B,H,S,S]` scores | 2.44 GB |
  | **rough total** | **~6.7 GB** against **79.25 GB** available |

  B=10000 *sounds* extreme, but each sequence is 128 tokens of 128 dims. Ten
  thousand small problems cost far less memory than thirty-two enormous ones
  (#14's single activation alone is 12.2 GB).
  *Why it matters:* #6 is **1,174 GFLOP — the second-largest workload in the
  suite** and one of the 14 shapes §3.7 requires. Reporting 12/14 when it could
  be 13/14 is a self-inflicted gap in Technical Execution, and it is a large,
  batch-parallel shape where our SDPA + compile work should do well.
  *Action:* `python runner.py --candidates candidates/v_router.py --shapes 6`.
  One job. If it passes, add it to every subsequent sweep and to the results
  tables in `README.md` / `TECH_REPORT.md`.
  *Falsify:* it OOMs or fails the gate -> document it beside #14 with the
  measured number, exactly as B5 was documented. Either outcome is a better
  deliverable than silence.
  *Note:* `select_shapes()` at `bench_harness.py:269-271` still hard-codes the
  exclusion. Either pass `--shapes 6` explicitly or widen `official-safe`.

## Open — the four sub-2x shapes, diagnosed

Read `docs/research-sub2x.md` for the full derivation. Summary of the causes,
because three of the four are **not** the same problem:

- **#8 (1.14x) — at the fp32 arithmetic ceiling, not inefficient.** 420.9 GFLOP
  in 26.28 ms is **16.0 TFLOP/s against a 19.5 TFLOP/s fp32 peak, i.e. 82%.**
  There is no meaningful headroom *at this precision*. See S1 — the precision is
  the lever, not the kernel.
- **#9 (1.47x) / #10 (1.70x) — the baseline is unusually good here, not the
  candidate bad.** All of #1/#9/#10/#11 are the same 7.52 GFLOP. The candidate
  is flat across them (1.30/1.35/1.37/1.86 ms); the **baseline** ranges
  1.97 -> 5.07 ms because it materializes `[B,H,S,S]` and does more
  transpose/reshape work as heads increase. The speedup spread across the head
  sweep is therefore a property of the reference, not of our kernel. Genuine but
  smaller levers in S2.
- **#5 (1.86x) — no headroom; the ratio drop is an artifact of scaling.** #5 is
  exactly 2x #1's work. Our candidate scales near-linearly (1.300 -> 2.495 ms,
  1.92x) because it is already efficient; the baseline scales *sub*-linearly
  (2.623 -> 4.653 ms, 1.77x) because at B=64 it was still partly overhead-padded
  and B=128 amortizes that. Two well-behaved curves with different slopes
  produce a falling ratio. **Do not spend GPU time here.** Document it.

- **S1 — RESOLVE THE TF32 QUESTION. Highest value and highest risk in the
  queue.** `candidates/v_router.py:39-42` sets `allow_tf32 = False` and
  `set_float32_matmul_precision("highest")` **at module import, outside any
  function**. These are process-global PyTorch flags, and the harness sets its
  own value earlier (`bench_harness.py:313-314`, default `True`) before
  importing the candidate — so **the candidate's import silently overrides the
  harness and de-TF32s the baseline as well.**
  Two consequences, and they pull in opposite directions:
  1. *Integrity.* The organizer's script defaults to `--allow-tf32` **on**
     (`torch_transformer_benchmark.py:687`, `--matmul-precision high` at `:638`).
     Our 2.47x is therefore measured in a **non-default configuration in which
     both sides are handicapped to ~1/8 of the card's TF32 throughput.** The
     comparison is internally fair, but it is not the organizer's default, and
     on GEMM-heavy shapes a slowed baseline plausibly *inflates* our ratio.
     **This must be disclosed in the tech report whether or not we change it.**
  2. *Opportunity.* #8 at 82% of the fp32 ceiling has ~8x of theoretical
     headroom sitting behind a flag we disabled.
  *Why it was disabled:* commit `8010964` found Inductor's `max-autotune`
  autotuner selecting TF32 GEMM kernels for the **candidate** while the baseline
  used cuBLAS, drifting ~0.005 against `atol=0.002` on 9/12 shapes. That is an
  **asymmetry between two different TF32 implementations**, not evidence that
  TF32 itself fails the gate.
  *Note the routing:* **#8 routes to `fused`, which is not compiled at all.** The
  Inductor problem cannot arise on that path. The global pin is a sledgehammer
  aimed at a compile-path bug, applied to a shape that never touches the
  compiler.
  *Action:* one sweep with `--allow-tf32` left at the harness default and the
  candidate's import-time override **removed**, measuring correctness and speed
  for all 12. Then a second with the override scoped to the compiled path only.
  *Falsify:* if TF32-on fails the gate on the `fused` path for #8 too, the pin
  is justified as-is — record that and close S1.
  *Expect:* #8 absolute time falls sharply; the **ratio** may fall, rise, or hold,
  because the baseline accelerates too. **Report whichever we measure. Do not
  choose the configuration that produces the larger number without saying so.**

- **S2 — Re-test the compiled path on #9 and #10.** Both currently route to
  `fused`. #1, which is the same arithmetic, routes to `compile` and beats them
  (2.02x vs 1.47x/1.70x). The router was built from a sweep taken **before**
  B10's sync was known and before `v_compile_reduce.py` existed, so the
  head-count shapes may be mis-routed. Cheap: they ride along in any sweep.
  *Falsify:* compile does not beat fused on #9/#10 -> routing already optimal,
  close it.

- **S3 — #5 and the large-batch regime: document, do not optimize.** Evidence
  above. Falsify: if a variant beats 1.86x on #5 without regressing #1, this
  reasoning was wrong.

## Open — zero GPU cost, do these first

- **D1 — the §3.5 deliverables. NOW THE TOP ITEM IN THE QUEUE.** A defensible
  2.47x is banked; ~33 hours remain; the write-up is unstarted beyond
  scaffolding. `README.md` and `TECH_REPORT.md` exist with **49 `<FILL>`
  placeholders** guarded by `scripts/check_placeholders.sh`. Every one of them
  is now fillable from `leaderboard.md` and `journal.jsonl` — no new
  measurement is required. Also outstanding: the demo video (public YouTube,
  linked in Devpost) and the Devpost description's five named fields.
  *Blocked on:* per-shape data is prose in `leaderboard.md`, not in
  `journal.jsonl` (`per_shape: []` on iters 5-9), so the results tables must be
  transcribed by hand unless those rows are re-emitted.

- **B10 — remove the per-forward device sync B1 introduced.** `.all()` forces a
  GPU->CPU sync every forward: `best.py:171`, `v_compile.py:124`,
  `v_fused_qkv.py:109`, and **three times** in `v_router.py:124,168,261`.
  Costs most on the shapes with the least work to hide it behind (#2 is 0.374 ms
  total). Fix without a sync: the mask is built by
  `generate_random_case` (`torch_transformer_benchmark.py:255-272`) and is
  all-True whenever `padding_ratio <= 0`; prefer a cached/structural check over
  a value read. *Falsify:* < 1% on #2 -> not worth the risk, close it.

- **AI-attribution gap — worth real points, costs nothing.** Commit `1f99f8d`
  carries `Co-Authored-By: Claude Opus 4.8` and a session link. **None of the
  thirteen commits since do**, including every A100 result. §3.5 awards bonus
  points for the AI skills/tools used and git history is the most credible
  evidence, because it is timestamped rather than written for a judge. Add the
  trailer going forward and state the tooling explicitly in `TECH_REPORT.md` §6.

## Open — the measurement that unblocks the rest

- **M1 — one sweep, three questions.** S1 (TF32 on/off), S2 (#9/#10 routed to
  compile), and **T1** in a single job. `v_amp.py` is CPU-smoke-tested
  only and is the single highest-variance item left: fp16 unlocks flash on
  **all 14 shapes** (vs 0 at fp32) plus tensor cores, but the blanket-cast
  attempt already failed the gate on 11/12. T1 has since landed
  (`v_compile_reduce.py`, 2.29x/2.39x) so it no longer needs a slot.
  *Falsify:* no configuration beats 2.47x geomean -> freeze `v_router`, spend
  every remaining hour on D1.

## Open — shape 14

- **B5 — CLOSED with real evidence, no further work.** OOM during optimized-model
  warmup at **73.85 GB of 79.25 GB** (journal iter 7), confirming plain
  activation memory (32x100000x1024, ~7 live tensors), not an attention-algorithm
  limit — both candidates already use mem-efficient SDPA. `v_compile` never even
  reached the OOM point: killed at the 15-minute limit still inside
  max-autotune's kernel search. Goes in the README limitations section as the
  §3.5 reflection. Do not pursue fp16/H100/chunking (rubric §3.3 puts
  production-grade work out of scope).

## Open — optimization, remaining

- **T6 — IN PROGRESS, and now the most important open lever.** Blanket-cast
  fp16 (`runner.py --dtype float16`) **failed correctness on 11/12** shapes
  (max_abs 0.006-0.009 vs atol 0.002) — expected, it demotes LayerNorm and the
  softmax reduction too. `candidates/v_amp.py` (`torch.autocast(fp16)`, keeping
  norms and reductions in fp32) is the follow-up: **CPU-smoke-tested only, not
  yet run on GPU.**
  *Why it matters more than it looked:* `tools/probe_sdpa_backends.py
  --dtype float16` shows **flash SDPA eligible on all 14 shapes at fp16, versus
  0/14 at fp32** — including #8 at head_dim=256 and #14 at head_dim=64. So fp16
  buys tensor cores **and** the flash backend at once. See S1: this and TF32 are
  the same lever at two different strengths, aimed at the same shape.
- **T4 — memory-layout cleanups** feeding SDPA `[B,H,S,D]`. Small, and #8 at 82%
  of the fp32 ceiling shows layout is not what limits the shape that matters.
- **T7 — custom Triton kernels.** Correctly at the bottom. With ~32 hours left
  and D1 unstarted, **do not start this.**
- **T2, T3 — superseded.** Both landed as `v_compile.py` / `v_fused_qkv.py` and
  are now route targets inside `v_router.py`.

## In progress

_(none yet — claim something above)_

## Done

- **T1 — `torch.compile(mode="reduce-overhead")`** → `candidates/v_compile_reduce.py`.
  A100-80, official protocol: 12/12 correct, **median 2.29x / geomean 2.39x
  standalone**. Beats every existing candidate on shapes 3/4/5. Folded into
  `v_router.py`'s route table as a 4th target (iter 10-11).
- **B2 — RESOLVED by direct measurement** (journal iter 8, `bc67c11`,
  `docs/sdpa_backend_probe.json`). Forcing each SDPA backend per shape on
  A100-80: **flash eligible on 0/14 shapes at fp32**, mem-efficient on 14/14.
  Every speedup in this repo to date came from mem-efficient SDPA, never flash.
  No number changes; the report must simply name it correctly.
- **B7 — CLOSED, claim twice corrected.** Original claim (sm80 caps flash
  head_dim at 128, so #8 is excluded) was wrong twice over. First, Codex cited
  `sdp_utils.cpp` showing the cap is **256**, not 128. Then the fp16 probe showed
  **flash fires on all 14 shapes including #8 at head_dim 256**. The gating
  variable was never head_dim — it is **dtype**. Recorded because it is a clean
  example of a plausible, well-cited claim surviving one review and still being
  wrong until measured.
- **B8 + B9 — DONE** (`164dd97`). Timing now mirrors
  `torch_transformer_benchmark.py:benchmark_models()` (warmup 20 / repeats 100 /
  rounds 3, alternating order); candidate loaded once per sweep. **This pair
  caught a 24% inflation**: iter 5 reported 2.71x median under the old protocol,
  corrected to 2.18x.
- **T8 — ANSWERED without a run.** Baseline wall-clock is ~1.87 ms for #2 (B=1),
  #3 (B=4), #4 (B=16), #7 and #12 alike. Sixteen times the work, same time — the
  reference is launching, not computing. Speedups track it (#2 5.07x, #3 4.24x).
- **TF32 correctness bug — found by running, not reasoning** (`8010964`).
  Inductor's max-autotune selected TF32 GEMM kernels for fp32 inputs, drifting
  ~0.005 against `atol=0.002` on 9/12 shapes. Invisible on CPU and MPS, which
  have no TF32. See S1 — the *fix* is now itself a research question.
- **T5 — Per-shape dispatch** → `candidates/v_router.py`. Routes each shape to
  whichever of best/v_compile/v_fused_qkv empirically won it (measured, not
  the head_dim/backend-table heuristic originally proposed — the three
  candidates' relative strengths didn't reduce to one clean rule across the
  12-shape sample). A100-80, official protocol: 2.27x median / 2.47x
  geomean, beats every single candidate, 12/12 correct. New leaderboard
  best (iter 9). (agent: opus-1)
- **T0 — SDPA seed** → `candidates/best.py`. Correctness verified on CPU on dev
  shapes + `official-safe`, with and without padding. **Caveat:** the CPU run
  cannot expose B1, B2 or B5, and the claimed shape-#14 capability is false (B1).
  GPU speedup still pending the first cluster run. (agent: seed)

## Resolved by the problem statement (27 Aug 2026 revision)

Read directly from TechJam Track 3 §3.1-3.7, not second-hand.

1. **Tolerance** — `relative error < 0.02, abs error < 0.002` (§3.2). See B4.
2. **Shape table** — §3.7 matches `bench_harness.py:70-85` exactly. All 14 rows,
   causal TRUE throughout. Our catalog is correct, no changes needed.
3. **Evaluation GPU and speed aggregation — THERE IS NO ORGANIZER-RUN BENCHMARK.**
   §3.2: "Optimize & test your codes on your own machine. Different methods may
   be used to optimize the codes depending on the machine (GPU cards) you use."
   §3.4: "You can download 1 of these, and run it on your own machine."
   §3.5: the tech report must state "what the environment is (CPU, GPU, DISK, etc)".
   - We choose the GPU. We run the benchmark. We report the numbers.
   - There is no hidden grader, no fixed target hardware, and no cross-shape
     aggregation formula, because nobody re-runs our code on their machine.
   - The correctness gate is self-administered. Report it honestly and in full,
     including the shapes that fail or cannot run.
4. **Shape #14 is no longer a blocker.** §3.5 requires the README to carry "a
   brief reflection on your solution's limitations and what you would improve
   given more time". A documented, quantified OOM with the memory arithmetic
   (~85 GB fp32 for seven live activations) is a legitimate deliverable, not a
   disqualification. Do NOT burn the remaining hours trying to make it run.

## Still worth getting

- **Webinar recording** — 28 Aug, 3:00-3:45pm, linked from the problem statement.
  45 minutes of organizer Q&A that may carry intent not in the written spec.
- **B0 — confirm our script is the 27 Aug 6:25PM revision.** The statement says
  `torch_transformer_benchmark.py` was updated that day. Our copy was committed
  2026-08-30 16:41 (`e13d295`), after the update, and its argparse defaults
  (`:618-619`) already match §3.2, so it is almost certainly current. Diff it
  against the download link once. 30 seconds, removes a silent total-loss risk.

## Seeded by human

_Humans: drop ideas here as free text — an agent will formalize each into a
proper hypothesis + candidate and treat it as high priority._

